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
from os import PathLike
from pathlib import Path
from typing import IO, Iterable

import cv2
import fsspec
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import DatasetReaderBase, DatasetWriter
from rasterio.windows import Window

from orthority.enums import Interp

logger = logging.getLogger(__name__)
# TODO: rename this module _utils, & version -> _version
# TODO: use the more general os.PathLike instead of pathlib.Path for all path type specifiers,
#  then use os.fspath to convert paths to str (everywhere).


@contextmanager
def suppress_no_georef():
    """Context manager to suppress rasterio's NotGeoreferencedWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=NotGeoreferencedWarning)
        yield


def expand_window_to_grid(win: Window, expand_pixels: tuple[int, int] = (0, 0)) -> Window:
    """Expand rasterio window extents to the nearest whole numbers."""
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(
        col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int')
    )
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


def validate_collection(template: Iterable, coll: Iterable):
    """
    Validate a nested dict / list of values (``coll``) against a nested dict / list of types, tuples
    of types, and values (``template``).

    All items in a ``coll`` list are validated against the first item in the corresponding
    ``template`` list.
    """
    # adapted from https://stackoverflow.com/questions/45812387/how-to-validate-structure-or-schema-of-dictionary-in-python
    if isinstance(template, dict) and isinstance(coll, dict):
        # struct is a dict of types or other dicts
        for k in template:
            if k in coll:
                validate_collection(template[k], coll[k])
            else:
                raise KeyError(f"No key: '{k}'.")
    elif isinstance(template, list) and isinstance(coll, list) and len(template) and len(coll):
        # struct is list in the form [type or dict]
        for item in coll:
            validate_collection(template[0], item)
    elif isinstance(template, type):
        # struct is the type of conf
        if not isinstance(coll, template):
            raise TypeError(f"'{coll}' is not an instance of {template}.")
    elif isinstance(template, tuple) and all([isinstance(item, type) for item in template]):
        # struct is a tuple of types
        if not isinstance(coll, template):
            raise TypeError(f"'{coll}' is not an instance of any of {template}.")
    elif isinstance(template, object) and template is not None:
        # struct is the value of conf
        if not coll == template:
            raise ValueError(f"'{coll}' does not equal '{template}'.")


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
        Mode in which the dataset is opened.  Either 'r' or 'w'.
    :param overwrite:
        Whether to overwrite an existing file in 'w' mode.  Ignored in 'r' mode.
    :param kwargs:
        Keyword arguments to pass to :func:`rasterio.open`.
    """

    # TODO: using rio.open() with file objects or fsspec OpenFile objects means that sidecar
    #  files are not written or read. We know this is an issue for writing PAM files of projected
    #  CRSs with ellipsoidal height. It is also an issue for reading RPC coefficients from .RPB
    #  or .XML sidecar files.  This should be confirmed and tested with an updated rasterio.  If
    #  necessary, local files should be opened with the native GDAL opener, at least this way,
    #  the sidecar files can be RW locally. This may need a re-work of the CLI's current use of
    #  OpenFile.
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
            # TODO: use the opener arg to rio.open() when that rasterio version is released,
            #  rather than passing an open file object.  test that files are not buffered in
            #  memory with this option, and currently problematic fsspec protocols (e.g. github)
            #  no longer cause a seg fault.
            if isinstance(file, OpenFile):
                if mode + 'b' != file.mode:
                    raise IOError(
                        f"OpenFile object mode: '{file.mode}' should be a binary mode matching the "
                        f"mode argument: '{mode}'."
                    )
                ofile = file
            else:
                ofile = fsspec.open(os.fspath(file), mode + 'b')

            # TODO: delete sidecar files if overwriting
            if not overwrite and 'w' in mode and ofile.fs.exists(ofile.path):
                raise FileExistsError(ofile.path)

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
        Whether to overwrite an existing file in 'w*' mode.  Ignored in 'r*' mode.
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
        # TODO: can text encoding be automatically determined?  previously this worked
        #  with urllib:
        #   req = urllib.urlopen(str(file))
        #   encoding = req.headers.get_content_charset(failobj='utf-8')
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
                raise FileExistsError(ofile.path)

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
