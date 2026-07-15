# Copyright The Orthority Contributors.
#
# This file is part of Orthority.
#
# Orthority is free software: you can redistribute it and/or modify it under the terms of the GNU
# Affero General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# Orthority is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with Orthority.
# If not, see <https://www.gnu.org/licenses/>.

"""Utility functions for internal use."""

from __future__ import annotations

import logging
import os
import posixpath
import subprocess
import threading
import time
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime
from inspect import getsourcefile
from io import IOBase
from itertools import product
from multiprocessing import Process
from os import PathLike
from pathlib import Path
from threading import Thread
from typing import IO, Any

import cv2
import fsspec
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.enums import Resampling
from tqdm.auto import tqdm

from orthority import version

try:
    from fsspec.implementations.http import HTTPFileSystem
except ImportError:
    HTTPFileSystem = type('unknown', (), {})

from fsspec.implementations.local import LocalFileSystem
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import DatasetReaderBase, DatasetWriter
from rasterio.windows import Window
from threadpoolctl import ThreadpoolController

from orthority.enums import Compress, Driver, Interp
from orthority.errors import OrthorityError, OrthorityWarning

logger = logging.getLogger(__name__)

_nodata_vals = dict(
    uint8=0, uint16=0, int16=np.iinfo('int16').min, float32=float('nan'), float64=float('nan')
)
"""Nodata values for supported dtypes.  OpenCV remap doesn't support int8 or uint32,
and only supports int32, uint64, int64 with nearest interpolation, so these dtypes are excluded.
"""

_default_out_config = dict(
    driver=Driver.gtiff, write_mask=None, dtype=None, compress=None, build_ovw=True, overwrite=False
)
"""Default configuration values for output images."""


@contextmanager
def suppress_no_georef():
    """Context manager to suppress Rasterio's NotGeoreferencedWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=NotGeoreferencedWarning)
        yield


def expand_window_to_grid(win: Window, expand_pixels: tuple[int, int] = (0, 0)) -> Window:
    """Expand rasterio window extents to the nearest whole numbers."""
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(int(col_off), int(row_off), int(width), int(height))
    return exp_win


def nan_equals(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Compare two numpy objects, returning True where elements of both are nan."""

    def _nan_equals(obj, scalar) -> np.ndarray:
        if np.isnan(scalar):
            return np.isnan(obj)
        else:
            return obj == scalar

    # use _nan_equals() to speed up the special cases where a or b is a scalar
    if np.isscalar(a):
        return _nan_equals(b, a)
    elif np.isscalar(b):
        return _nan_equals(a, b)
    else:
        return (a == b) | (np.isnan(a) & np.isnan(b))


def distort_image(camera, image: np.ndarray, nodata=0, interp=Interp.nearest) -> np.ndarray:
    """Return a distorted image given a frame camera model and source image."""

    if not np.all(np.array(image.shape[-2:][::-1]) == camera.im_size):
        raise ValueError("'image' shape should be the same as the 'camera' image size.")

    # create (j, i) pixel coords for distorted image
    j_range = np.arange(0, camera.im_size[0])
    i_range = np.arange(0, camera.im_size[1])
    j_grid, i_grid = np.meshgrid(j_range, i_range, indexing='xy')
    ji = np.array((j_grid.reshape(-1), i_grid.reshape(-1)))

    # find the corresponding undistorted/ source (j, i) pixel coords
    camera_xyz = camera._pixel_to_camera(ji)
    undist_ji = camera._K_undistort.dot(camera_xyz)[:2].astype('float32')

    def distort_band(src_array: np.ndarray, dst_array: np.ndarray):
        """Distort a 2D band array."""
        cv2.remap(
            src_array,
            undist_ji[0].reshape(image.shape[-2:]),
            undist_ji[1].reshape(image.shape[-2:]),
            Interp[interp].to_cv(),
            dst=dst_array,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    dist_image = np.full(image.shape, dtype=image.dtype, fill_value=nodata)
    for bi in range(image.shape[0]):
        distort_band(image[bi], dist_image[bi])

    return dist_image


def utm_crs_from_latlon(lat: float, lon: float) -> CRS:
    """Return a 2D rasterio UTM CRS for the given (lat, lon) coordinates in degrees."""
    # adapted from https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    zone = int(np.floor((lon + 180) / 6) % 60) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def validate_collection(schema: dict | list, coll: dict | list):
    """
    Validate a nested dict / list of values (``coll``) against a nested dict / list of types,
    tuples of types, and values (``schema``).

    - All items in a ``coll`` dict are validated against the first item in the corresponding
    ``schema`` dict, if it has one item with a type key.  Otherwise, ``coll`` items are validated
    against the same key ``schema`` item.
    - All items in a ``coll`` list are validated against the first item in the corresponding
    ``schema`` list, if it has one item.  Otherwise, ``coll`` items are validated against
    corresponding ``schema`` items.
    - ``coll`` values are not validated against corresponding None values in ``schema``.
    """
    # adapted from https://stackoverflow.com/questions/45812387/how-to-validate-structure-or
    #  -schema-of-dictionary-in-python
    if isinstance(schema, dict) and isinstance(coll, dict):
        # schema is a dict
        first_key = next(iter(schema))
        if len(schema) == 1 and isinstance(first_key, type):
            for k in coll:
                if not isinstance(k, first_key):
                    raise TypeError(f"'{k}' is not an instance of {first_key}.")
                validate_collection(schema[first_key], coll[k])
        else:
            for k in schema:
                if k in coll:
                    validate_collection(schema[k], coll[k])
                else:
                    raise KeyError(f"No key: '{k}'.")
    elif isinstance(schema, list) and isinstance(coll, list) and len(schema) and len(coll):
        # schema is a list
        if len(schema) == 1:
            for item in coll:
                validate_collection(schema[0], item)
        else:
            if len(coll) != len(schema):
                raise ValueError(f'{coll} should have {len(schema)} items.')
            for template_item, coll_item in zip(schema, coll, strict=True):
                validate_collection(template_item, coll_item)
    elif isinstance(schema, type):
        # schema is a type
        if not isinstance(coll, schema):
            raise TypeError(f"'{coll}' is not an instance of {schema}.")
    elif isinstance(schema, tuple) and all([isinstance(item, type) for item in schema]):
        # schema is a tuple of types
        if not isinstance(coll, schema):
            raise TypeError(f"'{coll}' is not an instance of any of {schema}.")
    elif isinstance(schema, (str, int, float)):
        # schema is a value of a basic type
        if not coll == schema:
            raise ValueError(f"'{coll}' does not equal '{schema}'.")
    elif schema is None:
        # don't test
        pass
    else:
        # something else is wrong
        raise ValueError("Invalid collection.")


def get_filename(file: str | PathLike | OpenFile | DatasetReaderBase | IO) -> str:
    """Return a source filename for the given ``file`` object.  If ``file`` is an
    :class:`~fsspec.core.OpenFile` instance, a :class:`~rasterio.io.DatasetReaderBase` instance
    or file object, it should have a ``filename`` attribute i.e. have been created by either
    :class:`Open` or :class:`OpenRaster`.
    """
    if isinstance(file, DatasetReaderBase):
        filename = getattr(file, 'filename', Path(file.name).name)
    elif isinstance(file, OpenFile):
        filename = getattr(file, 'filename', Path(file.path).name)
    elif isinstance(file, IOBase):
        filename = getattr(file, 'filename', Path(getattr(file, 'name', '<file object>')).name)
    else:
        filename = Path(os.fspath(file)).name
    return filename


def join_ofile(
    base: str | PathLike | OpenFile, rel: str, mode: str | None = None, **kwargs
) -> OpenFile:
    """Return an fsspec OpenFile whose path is a join of the ``base`` path with the ``rel`` path."""
    if not isinstance(base, OpenFile):
        base = fsspec.open(os.fspath(base), mode or 'rt')

    joined_path = posixpath.join(base.path, rel)
    return OpenFile(base.fs, joined_path, mode=mode or base.mode, **kwargs)


class OpenRaster:
    """
    Context manager for local or remote Rasterio datasets.

    :param file:
        A path, URI, :class:`~fsspec.core.OpenFile` instance, or open dataset.  If it is an open
        dataset, it is returned unaltered on entering the context, not closed on exiting the
        context, and ``mode`` and ``kwargs`` are ignored.  If is an OpenFile instance, it should
        be open in a binary mode matching ``mode``.
    :param mode:
        Mode in which the dataset is opened.  Either ``'r'`` or ``'w'``.
    :param overwrite:
        Whether to overwrite an existing file in ``'w'`` mode.  Ignored in ``'r'`` mode.
    :param kwargs:
        Keyword arguments to pass to :func:`rasterio.open`.
    """

    def __init__(
        self,
        file: str | PathLike | DatasetReaderBase | OpenFile,
        mode: str = 'r',
        overwrite: bool = False,
        **kwargs,
    ):
        if mode not in ['r', 'w']:
            raise ValueError(f"The 'mode' argument should be either 'r' or 'w', not '{mode}'.")

        self._exit_stack = ExitStack()

        if isinstance(file, DatasetReaderBase):
            if file.closed:
                raise OSError('Dataset is closed.')
            if mode not in file.mode:
                raise OSError(
                    f"Dataset mode: '{file.mode}' not compatible with the mode argument: '{mode}'."
                )
            self._dataset = file

        elif isinstance(file, (str, PathLike, OpenFile)):
            if isinstance(file, OpenFile):
                if mode + 'b' != file.mode:
                    raise OSError(
                        f"OpenFile object mode: '{file.mode}' should be a binary mode matching the "
                        f"mode argument: '{mode}'."
                    )
                ofile = file
            else:
                ofile = fsspec.open(os.fspath(file), mode + 'b')

            if not overwrite and 'w' in mode and ofile.fs.exists(ofile.path):
                raise FileExistsError(f"File exists: '{ofile.path}'")

            if isinstance(ofile.fs, (LocalFileSystem, HTTPFileSystem)):
                # use GDAL internal file system
                try:
                    self._dataset = self._exit_stack.enter_context(
                        rio.open(ofile.path, mode, **kwargs)
                    )
                except RasterioIOError as ex:
                    ex_str = str(ex)
                    if 'no such file or directory' in ex_str.lower():
                        raise FileNotFoundError(ex_str) from ex
                    else:
                        raise
            else:
                # use fsspec file object
                file_obj = self._exit_stack.enter_context(ofile)
                self._dataset = self._exit_stack.enter_context(rio.open(file_obj, mode, **kwargs))

            # store the source filename as a dataset attribute
            self._dataset.filename = get_filename(file)

        else:
            raise TypeError(f"Unsupported 'file' type: {type(file)}")

    def __enter__(self) -> rio.DatasetReader | DatasetWriter:
        if self._dataset.closed:
            raise OSError('Dataset is closed.')
        return self._dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self._exit_stack.close()


class Open:
    """
    Context manager for local or remote file IO.

    :param file:
        A path, URI, :class:`~fsspec.core.OpenFile` instance, or file object.  If it is a file
        object, it is returned unaltered on entering the context, not closed on exiting the
        context, and ``mode`` and ``kwargs`` are ignored.  If is an OpenFile instance, it should
        be opened in ``mode`` (``kwargs`` are ignored).
    :param mode:
        Mode in which the file is opened.
    :param overwrite:
        Whether to overwrite an existing file in `'`w*'`` mode.  Ignored in ``'r*'`` mode.
    :param kwargs:
        Keyword arguments to pass to :func:`fsspec.open`.
    """

    def __init__(
        self,
        file: str | PathLike | IO | OpenFile,
        mode='rt',
        overwrite: bool = False,
        **kwargs,
    ):
        self._exit_stack = ExitStack()
        if isinstance(file, IOBase):
            if file.closed:
                raise OSError('File object is closed.')
            if getattr(file, 'mode', mode) != mode:
                # note: fsspec text mode file objects do not have a mode property
                raise OSError(f"File object mode should match the mode argument: '{mode}'.")
            self._file_obj = file

        elif isinstance(file, (OpenFile, str, PathLike)):
            if isinstance(file, OpenFile):
                if mode != file.mode:
                    raise OSError(
                        f"OpenFile object mode: '{file.mode}', should match the mode argument:"
                        f" '{mode}'."
                    )
                ofile = file
            else:
                ofile = fsspec.open(os.fspath(file), mode, **kwargs)

            # overwrite could be prevented with 'x' modes, but is done this way for consistency
            # with OpenRaster & rasterio which doesn't support 'x'
            if not overwrite and 'w' in mode and ofile.fs.exists(ofile.path):
                raise FileExistsError(f"File exists: '{ofile.path}'")

            self._file_obj = self._exit_stack.enter_context(ofile)
            self._file_obj.filename = get_filename(file)

        else:
            raise TypeError(f"Unsupported 'file' type: {type(file)}")

    def __enter__(self) -> IO:
        return self._file_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self._exit_stack.close()


def create_profile(
    driver: str | Driver,
    shape: Sequence[int],
    dtype: str | np.dtype,
    compress: str | Compress | None = None,
    write_mask: bool | None = None,
    creation_options: dict | None = None,
) -> tuple[dict, bool]:
    """
    Return a partial Rasterio image profile and ``write_mask`` value for an output image given its
    configuration.

    If ``compress`` is ``None`` (the default), JPEG compression is used with the 'uint8'
    ``dtype`` and DEFLATE otherwise.  If ``write_mask`` is ``None`` (the default), it is returned
    as ``True`` when JPEG compression is used, and ``False`` otherwise.  If ``creation_options``
    are supplied, no other creation options are set.
    """
    creation_options = creation_options or {}

    # check dtype support
    dtype = str(dtype)
    if dtype not in _nodata_vals:
        raise OrthorityError(f"Data type '{dtype}' is not supported.")

    # create initial profile
    driver = Driver(driver)
    profile = dict(
        driver=driver.value,
        dtype=dtype,
        width=shape[2],
        height=shape[1],
        count=shape[0],
        bigtiff='if_safer',
    )

    # set creation options
    if len(creation_options) == 0:
        # configure compression
        if compress is None:
            compress = Compress.jpeg if dtype == 'uint8' else Compress.deflate
        else:
            compress = Compress(compress)
            if compress is Compress.jpeg:
                if dtype == 'uint16':
                    warnings.warn(
                        'Attempting a 12 bit JPEG ortho configuration.  Support is rasterio build '
                        'dependent.',
                        category=OrthorityWarning,
                        stacklevel=2,
                    )
                    profile.update(nbits=12)
                elif dtype != 'uint8':
                    raise OrthorityError(
                        "JPEG compression is supported for 'uint8' and 'uint16' data types only."
                    )

        profile.update(compress=compress.value)

        if driver is Driver.gtiff:
            # configure photometric interpretation and tiling
            if compress == Compress.jpeg and shape[0] == 3:
                profile.update(photometric='ycbcr')
            profile.update(tiled=True, blockxsize=512, blockysize=512)
        else:
            # Configure tiling & overviews. Overviews are not created automatically, but copied
            # from any overviews built with DatasetWriter.build_overviews().  GDAL sets
            # photometric internally.
            profile.update(blocksize=512, overviews='force_use_existing')

    else:
        profile.update(**creation_options)

    # resolve auto write_mask (=None) to write masks for jpeg compression
    if write_mask is None:
        write_mask = True if compress == Compress.jpeg else False

    # set nodata to None when writing internal masks to force external tools to use mask,
    # otherwise set by dtype
    nodata = None if write_mask else _nodata_vals[dtype]
    profile.update(nodata=nodata)

    return profile, write_mask


def convert_array_dtype(array: np.ndarray, dtype: str) -> np.array:
    """Return the ``array`` converted to ``dtype``, rounding and clipping in-place when ``dtype``
    is integer.  Adapted from :meth:`homonim.raster_array.RasterArray._convert_array_dtype`.
    """
    unsafe_cast = not np.can_cast(array.dtype, dtype, casting='safe')

    # round if converting from float to integer dtype
    if unsafe_cast and np.issubdtype(array.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        np.round(array, out=array)

    # clip if converting to integer dtype with smaller range than array dtype
    if unsafe_cast and np.issubdtype(dtype, np.integer):
        src_info = (
            np.iinfo(array.dtype)
            if np.issubdtype(array.dtype, np.integer)
            else np.finfo(array.dtype)
        )
        dst_info = np.iinfo(dtype)
        if src_info.min < dst_info.min or src_info.max > dst_info.max:
            # promote array dtype to be able to represent destination dtype exactly (if
            # possible) to clip correctly
            array = array.astype(np.promote_types(array.dtype, dtype))
            np.clip(array, dst_info.min, dst_info.max, out=array)

    # convert dtype (ignoring numpy warnings for float overflow or cast of nan to integer)
    with np.errstate(invalid='ignore', over='ignore'):
        array = array.astype(dtype, copy=False, casting='unsafe')

    return array


def build_overviews(
    im: DatasetWriter,
    max_num_levels: int = 8,
    min_level_pixels: int = 256,
    resampling=Resampling.average,
) -> None:
    """
    Build internal overviews for an open rasterio dataset.  Each overview level is decimated by a
    factor of 2.  The number of overview levels is determined by whichever of the
    ``max_num_levels`` or ``min_level_pixels`` limits is reached first.

    :param im:
        Rasterio dataset opened in 'r+' or 'w' mode.
    :param max_num_levels:
        Maximum number of overview levels.
    :param min_level_pixels:
        Minimum overview width / height in pixels.
    :param resampling:
        Overview resampling method.
    """
    max_ovw_levels = int(np.min(np.log2(im.shape)))
    min_level_shape_pow2 = int(np.log2(min_level_pixels))
    num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
    ovw_levels = [2**m for m in range(1, num_ovw_levels + 1)]
    im.build_overviews(ovw_levels, resampling=resampling)


def get_tqdm_kwargs(**kwargs) -> dict:
    """Return a dictionary of ``tqdm`` progress bar kwargs with a standard ``bar_format``."""
    return dict(
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]',
        dynamic_ncols=True,
        **kwargs,
    )


def block_windows(
    im: DatasetReaderBase | DatasetWriter, block_shape: tuple[int, int] | None = None
) -> Generator[Window]:
    """Block window generator for the given dataset and optional block shape. Blocks are generated
    row-wise, as they would be stored on disk.
    """
    driver = im.driver.lower()
    block_shape = block_shape or (
        (im.profile.get('blocksize', 512),) * 2 if driver == 'cog' else im.block_shapes[0]
    )

    i_range = range(0, im.height, block_shape[0])
    j_range = range(0, im.width, block_shape[1])
    for i_start, j_start in product(i_range, j_range):
        i_stop = min(i_start + block_shape[0], im.height)
        j_stop = min(j_start + block_shape[1], im.width)
        yield Window(j_start, i_start, j_stop - j_start, i_stop - i_start)


@contextmanager
def limit_cv_threads(num_threads: int):
    """Context manager to limit the number of OpenCV threads process-wide.  Not thread-safe."""
    curr_num_threads = cv2.getNumThreads()

    # cv2.setNumThreads() has no effect when OpenCV is built with Apple's GCD (see
    # https://github.com/opencv/opencv/issues/23091)
    limit = 1 if curr_num_threads != 1 else 2
    cv2.setNumThreads(limit)
    if cv2.getNumThreads() != limit:
        warnings.warn(
            'Not limiting OpenCV threads - not supported on this build.',
            category=OrthorityWarning,
            stacklevel=2,
        )

    cv2.setNumThreads(num_threads)
    try:
        yield
    finally:
        cv2.setNumThreads(curr_num_threads)


@contextmanager
def limit_blas_omp_threads(limits=None, user_api=None):
    """Context manager to limit the number of BLAS and OpenMP threads using threadpoolctl.

    Some BLAS / OpenMP libraries use process-wide limits, others use per-thread limits.  This
    context manager sets process-wide limits, and thread limits on the current thread.  It is not
    thread-safe.  The ``initialiser()`` method of the returned object can be used to set
    per-thread limits in other threads.

    Based on https://github.com/joblib/threadpoolctl/issues/208#issuecomment-4745983423.
    """
    # force NumPy's BLAS to load its OpenMP DLL(s) so these can be found by threadpoolctl (for
    # conda-forge NumPy on Windows)
    a = np.ones((3, 3))
    np.dot(a, a)

    with warnings.catch_warnings():
        # ignore the threadpoolctl warning about
        # https://github.com/conda-forge/blas-feedstock/issues/170
        warnings.filterwarnings(
            'ignore',
            message=r'\sFound Intel OpenMP',
            category=RuntimeWarning,
            module='threadpoolctl',
        )

        # create a ThreadpoolController, with retries to work around
        # https://github.com/joblib/threadpoolctl/issues/217
        max_retries = 10
        for retry in range(max_retries + 1):
            try:
                ctrl = ThreadpoolController()
                break
            except OSError as ex:
                if 'GetModuleFileNameEx failed' not in str(ex) or retry == max_retries:
                    raise
                continue

    # threadpoolctl does not support Apple's Accelerate BLAS library (see
    # https://github.com/joblib/threadpoolctl/issues/135)
    if len(ctrl.info()) == 0:
        warnings.warn(
            'Not limiting BLAS / OpenMP threads - no supported libraries found.',
            category=OrthorityWarning,
            stacklevel=2,
        )

    lock = threading.Lock()

    def initialiser():
        """Thread-safe function to set per-thread limits."""
        with lock:
            ctrl.limit(limits=limits, user_api=user_api)

    with ctrl.limit(limits=limits, user_api=user_api) as limiter:
        limiter.initialiser = initialiser
        yield limiter
    pass


def _bench_func(
    func: Callable,
    name: str | None = None,
    loops: int = 1,
    filter_callback: Callable[[Any], bool] | None = None,
    write_pstat: bool = False,
) -> None:
    """Report ``func()`` system utilisation and profile."""
    try:
        import psutil
    except ImportError as ex:
        raise ImportError("'psutil' is required for benchmarking.") from ex
    try:
        import yappi
    except ImportError as ex:
        raise ImportError("'yappi' is required for benchmarking.") from ex

    proc = psutil.Process()
    proc.nice(psutil.HIGH_PRIORITY_CLASS)
    dt = datetime.now()
    yappi.set_clock_type('wall')
    wall_times, cpu_times = [], []
    for _ in range(loops):
        func_gen = func()
        next(func_gen)  # setup
        yappi.start()
        # note that time.process_time() has a resolution of 16ms on Windows, so cpu_time should
        # be >> 16ms for it to be accurate
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        try:
            next(func_gen)  # benchmark
            wall_end, cpu_end = time.perf_counter(), time.process_time()
        finally:
            yappi.stop()
        try:
            next(func_gen)  # teardown
        except StopIteration:
            pass
        wall_times.append(wall_end - wall_start)
        cpu_times.append(cpu_end - cpu_start)

    mem_info = proc.memory_full_info()
    io_info = proc.io_counters()
    proc.threads()
    func_stats = yappi.get_func_stats(filter_callback=filter_callback)
    func_stats = func_stats.strip_dirs().sort('ttot', 'desc')
    name = name or func.__name__
    ttl_cpu_times = sum(cpu_times)
    ttl_wall_times = sum(wall_times)

    print('BENCHMARK\n---------')
    print(f'Name: {name}')
    print(f'Computer: {os.getenv("COMPUTERNAME")}')
    print(f'Date: {dt.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Loops: {loops}')

    print('\nPERFORMANCE')
    print(f'Mean (std) wall time: {np.mean(wall_times):.6f}s ({np.std(wall_times):.6f}s)')
    print(f'Mean (std) CPU time: {np.mean(cpu_times):.6f}s ({np.std(cpu_times):.6f}s)')
    print(f'CPU usage: {(100 / os.cpu_count()) * (ttl_cpu_times / ttl_wall_times):.2f}%')

    print('\nMEMORY')
    print(f'Peak RSS: {tqdm.format_sizeof(mem_info.peak_wset, suffix="B")}')
    print(f'Current RSS: {tqdm.format_sizeof(mem_info.rss, suffix="B")}')
    # TODO: report by major/minor page fault type with p.page_faults() when psutil updates to v8
    print(
        f'Page faults / sec: '
        f'{mem_info.num_page_faults / (1e-9 if ttl_cpu_times == 0 else ttl_cpu_times):.2f}'
    )

    print('\nIO')
    print(f'Read count: {io_info.read_count}')
    print(f'Read bytes: {tqdm.format_sizeof(io_info.read_bytes, suffix="B")}')
    print(f'Write count: {io_info.write_count}')
    print(f'Write bytes: {tqdm.format_sizeof(io_info.write_bytes, suffix="B")}')

    print('\nTHREADS')
    print(f'Num threads: {proc.num_threads()}')

    print('\nPROFILE', end='')
    func_stats.print_all()
    print('\n', flush=True)

    if write_pstat:
        bench_path = Path(getsourcefile(func)).parent
        yappi.get_func_stats().save(bench_path.joinpath(f'{name.lower()}.pstat'), type='pstat')
    yappi.clear_stats()


def bench_func(
    func: Generator,
    name: str | None = None,
    loops: int = 1,
    filter_callback: Callable[[Any], bool] | None = None,
    write_pstat: bool = False,
    container: type[Process | Thread] = Process,
) -> None:
    """
    Report ``func()`` system utilisation and profile.

    :param func:
        A generator that performs any setup, yields, runs the code to benchmark, yields, then
        performs any teardown.
    :param name:
        Benchmark name.  Defaults to the ``func`` name.
    :param loops:
        Number of times to run ``func``.  Wall and CPU times are reported as the mean and standard
        deviation over the runs.
    :param filter_callback:
        A Yappi callback to `filter profiling results
        <https://github.com/sumerc/yappi#different-ways-to-filtersort-stats>`__.
    :param write_pstat:
        Whether to write profiling results to a pstat format file with path
        :file:`{parent of source file containing func()}/{benchmark name}.pstat`
    :param container:
        Type of 'container' in which to run the benchmark.  By default ``Process`` is used to
        separate benchmark memory utilisation from the caller.
    """
    c = container(
        target=_bench_func,
        args=(func,),
        kwargs=dict(
            name=name,
            loops=loops,
            filter_callback=filter_callback,
            write_pstat=write_pstat,
        ),
    )
    c.start()
    c.join()


def run_benchmarks(params: Sequence[dict[str, Any]]) -> None:
    """Run benchmarks defined by a sequence of ``bench_func()`` parameter dictionaries."""
    print(f'Orthority version: {version.__version__}')
    # from https://stackoverflow.com/a/21901260
    git_rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    git_rev = git_rev.decode('utf8').strip()
    print(f'Current git commit: {git_rev}\n', flush=True)
    for param in params:
        bench_func(**param)
