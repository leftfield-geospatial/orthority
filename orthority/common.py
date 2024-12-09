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

import cProfile
import logging
import os
import posixpath
import pstats
import tracemalloc
import warnings
from contextlib import contextmanager, ExitStack
from io import IOBase
from itertools import product
from os import PathLike
from pathlib import Path
from typing import IO, Sequence, Generator

import cv2
import fsspec
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.enums import Resampling

try:
    from fsspec.implementations.http import HTTPFileSystem
except ImportError:
    HTTPFileSystem = type('unknown', (), {})

from fsspec.implementations.local import LocalFileSystem
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import DatasetReaderBase, DatasetWriter
from rasterio.windows import Window

from orthority.enums import Interp, Compress, Driver
from orthority.errors import OrthorityWarning, OrthorityError

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
    """Context manager to suppress rasterio's NotGeoreferencedWarning."""
    # TODO: warnings.catch_warnings is not thread-safe and warnings.simplefilter should rather be
    #  called once in cli.  consider what this does to API doc examples though - perhaps it can
    #  go in __init__.py.
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
    """Compare two numpy objects a & b, returning true where elements of both a & b are nan."""
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


@contextmanager
def profiler():
    """Context manager for profiling in DEBUG log level."""
    if logger.getEffectiveLevel() <= logging.DEBUG:
        proc_profile = cProfile.Profile()
        tracemalloc.start()
        proc_profile.enable()

        yield

        proc_profile.disable()
        # tottime is the total time spent in the function alone. cumtime is the total time spent
        # in the function plus all functions that this function called
        proc_stats = pstats.Stats(proc_profile).sort_stats('cumtime')
        logger.debug(f'Processing times:')
        proc_stats.print_stats(20)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.debug(
            f"Memory usage: current: {current / 10 ** 6:.1f} MB, peak: {peak / 10 ** 6:.1f} MB"
        )
    else:
        yield


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
    ``schema`` list, if it has one item.  Otherwise ``coll`` items are validated against
    corresponding ``schema`` items.
    - ``coll`` values are not validated against corresponding None values in ``schema``.
    """
    # adapted from https://stackoverflow.com/questions/45812387/how-to-validate-structure-or
    #  -schema-of-dictionary-in-python
    if isinstance(schema, dict) and isinstance(coll, dict):
        # schema is a dict
        first_key = [*schema][0]
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
            for template_item, coll_item in zip(schema, coll):
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


def join_ofile(base: str | PathLike | OpenFile, rel: str, mode: str = None, **kwargs) -> OpenFile:
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
                raise IOError('Dataset is closed.')
            if mode not in file.mode:
                raise IOError(
                    f"Dataset mode: '{file.mode}' not compatible with the mode argument: '{mode}'."
                )
            self._dataset = file

        elif isinstance(file, (str, PathLike, OpenFile)):
            # TODO: use the opener arg to rio.open() and pin the rasterio dependency version when
            #  rasterio 1.4 is released, rather than passing an open file object.  this should
            #  allow sidecar files to read / written (test that it does).  also test that files
            #  are not buffered in memory with this option, and currently problematic fsspec
            #  protocols (e.g. github) no longer cause a seg fault.
            if isinstance(file, OpenFile):
                if mode + 'b' != file.mode:
                    raise IOError(
                        f"OpenFile object mode: '{file.mode}' should be a binary mode matching the "
                        f"mode argument: '{mode}'."
                    )
                ofile = file
            else:
                ofile = fsspec.open(os.fspath(file), mode + 'b')

            # TODO: delete sidecar files if overwriting (note that DatasetReader has a files
            #  attribute that lists associated files)
            # TODO: if a file is partially written (e.g. because of an error), rasterio fails when
            #  overwriting it
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
                        raise FileNotFoundError(ex_str)
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
            raise IOError('Dataset is closed.')
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
        Keyword arguments to pass to :func:`fsspec.open` or ``opener`` if it is specified.
    """

    def __init__(
        self,
        file: str | PathLike | IO | OpenFile,
        mode='rt',
        overwrite: bool = False,
        **kwargs,
    ):
        # TODO: text encoding defaults to utf-8, which can't be changed from the CLI
        self._exit_stack = ExitStack()
        if isinstance(file, IOBase):
            if file.closed:
                raise IOError('File object is closed.')
            if getattr(file, 'mode', mode) != mode:
                # note: fsspec text mode file objects do not have a mode property
                raise IOError(f"File object mode should match the mode argument: '{mode}'.")
            self._file_obj = file

        elif isinstance(file, (OpenFile, str, PathLike)):
            if isinstance(file, OpenFile):
                if mode != file.mode:
                    raise IOError(
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
                    )
                    profile.update(nbits=12)
                elif dtype != 'uint8':
                    raise OrthorityError(
                        f"JPEG compression is supported for 'uint8' and 'uint16' data types only."
                    )
        # TODO: pixel interleaving with deflate compression can be much bigger than band
        #  interleaving - does it depend on data being written band-by=band? (see e.g.
        #  geedim/scripts/xee_rioxarray_exp.py)
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
    im: DatasetReaderBase | DatasetWriter, block_shape: tuple[int, int] = None
) -> Generator[Window]:
    """Block window generator for the given image, and optional block shape."""
    driver = im.driver.lower()
    block_shape = block_shape or (
        (im.profile.get('blocksize', 512),) * 2 if driver == 'cog' else im.block_shapes[0]
    )

    xrange = range(0, im.width, block_shape[1])
    yrange = range(0, im.height, block_shape[0])
    for xstart, ystart in product(xrange, yrange):
        xstop = min(xstart + block_shape[1], im.width)
        ystop = min(ystart + block_shape[0], im.height)
        yield Window(xstart, ystart, xstop - xstart, ystop - ystart)
