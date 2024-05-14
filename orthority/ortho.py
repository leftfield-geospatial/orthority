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

"""Orthorectification using DEM and camera model input."""
from __future__ import annotations

import logging
import os
import threading
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import ExitStack
from os import PathLike
from typing import Sequence

import cv2
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.crs import CRS
from rasterio.io import DatasetWriter
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling, transform, transform_bounds
from rasterio.windows import Window
from tqdm.std import tqdm, tqdm as std_tqdm

from orthority import utils
from orthority.camera import Camera, FrameCamera
from orthority.enums import Compress, Interp
from orthority.errors import CrsMissingError, OrthorityWarning

logger = logging.getLogger(__name__)


class Ortho:
    """
    Orthorectifier.

    Uses a supplied DEM and camera model to correct a source image for sensor and terrain
    distortion.  The camera model, and a portion of the DEM corresponding to the ortho bounds
    are stored internally.

    :param src_file:
        Source image to be orthorectified. Can be a path or URI string,
        an :class:`~fsspec.core.OpenFile` object in binary mode (``'rb'``), or a dataset reader.
    :param dem_file:
        DEM file covering the source image.  Can be a path or URI string,
        an :class:`~fsspec.core.OpenFile` object in binary mode (``'rb'``), or a dataset reader.
    :param camera:
        Source image camera model.
    :param crs:
        CRS of the ``camera`` world coordinates, and ortho image, as an EPSG, proj4 or WKT string,
        or :class:`~rasterio.crs.CRS` object.  If set to ``None`` (the default), the CRS will be
        read from the source image if possible. If the source image is not projected in the world /
        ortho CRS, ``crs`` should be supplied.
    :param dem_band:
        Index of the DEM band to use (1-based).
    """

    # default configuration values for Ortho.process()
    _default_config = dict(
        dem_band=1,
        resolution=None,
        interp=Interp.cubic,
        dem_interp=Interp.cubic,
        per_band=False,
        write_mask=None,
        dtype=None,
        compress=None,
        build_ovw=True,
        overwrite=False,
    )

    # default ortho (x, y) block size
    _default_blocksize = (512, 512)

    # EGM96/EGM2008 geoid altitude range i.e. minimum and maximum possible vertical difference with
    # the WGS84 ellipsoid (meters)
    _egm_minmax = [-106.71, 82.28]

    # Maximum possible ellipsoidal height i.e. approx. that of Everest (meters)
    _z_max = 8850.0

    # nodata values for supported ortho data types
    _nodata_vals = dict(
        uint8=0,
        uint16=0,
        int16=np.iinfo('int16').min,
        float32=float('nan'),
        float64=float('nan'),
    )

    # default progress bar kwargs
    _default_tqdm_kwargs = dict(
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]',
        dynamic_ncols=True,
        leave=True,
    )

    def __init__(
        self,
        src_file: str | PathLike | OpenFile | rio.DatasetReader,
        dem_file: str | PathLike | OpenFile | rio.DatasetReader,
        camera: Camera,
        crs: str | CRS | None = None,
        dem_band: int = 1,
    ) -> None:
        # TODO: allow dem_file to be specified as a constant height value (in world / ortho
        #  vertical CRS)
        if not isinstance(camera, Camera):
            raise TypeError("'camera' is not a Camera instance.")

        self._src_file = src_file
        self._src_name = utils.get_filename(src_file)
        self._camera = camera
        self._write_lock = threading.Lock()

        self._crs = self._parse_crs(crs)
        self._dem_array, self._dem_transform, self._dem_crs = self._get_init_dem(dem_file, dem_band)
        self._gsd = self._get_gsd()

    @property
    def camera(self) -> Camera | FrameCamera:
        """Source image camera model."""
        return self._camera

    @staticmethod
    def _build_overviews(
        im: DatasetWriter,
        max_num_levels: int = 8,
        min_level_pixels: int = 256,
    ) -> None:
        """Build internal overviews for a given rasterio dataset."""
        max_ovw_levels = int(np.min(np.log2(im.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2**m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)

    def _parse_crs(self, crs: str | CRS) -> CRS:
        """Derive a world / ortho CRS from the ``crs`` parameter and source image."""
        if crs:
            crs = CRS.from_string(crs) if isinstance(crs, str) else crs
        else:
            with utils.suppress_no_georef(), utils.OpenRaster(self._src_file, 'r') as src_im:
                if src_im.crs:
                    crs = src_im.crs
                else:
                    raise CrsMissingError(
                        f"Source image '{self._src_name}' is not projected, 'crs' should be "
                        f"specified."
                    )
        return crs

    def _get_init_dem(
        self, dem_file: str | PathLike | rio.DatasetReader, dem_band: int
    ) -> tuple[np.ndarray, rio.Affine, CRS]:
        """Return an initial DEM array in its own CRS and resolution.  Includes the corresponding
        DEM transform, CRS, and flag indicating ortho and DEM CRS equality in return values.
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), utils.OpenRaster(dem_file, 'r') as dem_im:
            if dem_band <= 0 or dem_band > dem_im.count:
                dem_name = utils.get_filename(dem_file)
                raise ValueError(
                    f"DEM band {dem_band} is invalid for '{dem_name}' with {dem_im.count} band(s)"
                )
            # crs comparison is time-consuming - perform it once here
            crs_equal = self._crs == dem_im.crs

            # find the scale from meters to ortho crs z units
            zs = []
            ref_crs = rio.CRS.from_epsg(4979)
            for z in [0, 1]:
                world_xyz = transform(ref_crs, self._crs, [0], [0], [z])
                zs.append(world_xyz[2][0])
            z_scale = zs[1] - zs[0]
            dem_full_win = Window(0, 0, dem_im.width, dem_im.height)

            def get_win_at_zs(zs: Sequence[float]) -> Window:
                """Return a DEM window that contains the ortho bounds at z values in ``zs``."""
                world_xyz = []
                for z in zs:
                    world_xyz.append(self._camera.world_boundary(z))
                world_xyz = np.column_stack(world_xyz)
                world_bounds = [*np.min(world_xyz[:2], axis=1), *np.max(world_xyz[:2], axis=1)]
                dem_bounds = (
                    transform_bounds(self._crs, dem_im.crs, *world_bounds)
                    if not crs_equal
                    else world_bounds
                )
                dem_win = dem_im.window(*dem_bounds)
                try:
                    dem_win = dem_full_win.intersection(dem_win)
                except rio.errors.WindowError:
                    raise ValueError(
                        f"Ortho for '{self._src_name}' lies outside, or underneath the DEM."
                    )
                return utils.expand_window_to_grid(dem_win)

            # get a dem window containing the ortho bounds at min & max possible altitude, read the
            # window from the dem
            zs = np.array([self._egm_minmax[0], self._z_max + self._egm_minmax[1]]) * z_scale
            dem_win = get_win_at_zs(zs)
            dem_array = dem_im.read(dem_band, window=dem_win, masked=True)
            dem_array_win = dem_win

            # reduce the dem window to contain the ortho bounds at min & max dem altitude,
            # accounting for dem-ortho vertical datum offset & z unit scaling
            zs = []
            for index in [dem_array.argmin(), dem_array.argmax()]:
                ij = np.unravel_index(index, dem_array.shape)
                dem_xyz = (*(dem_im.window_transform(dem_win) * ij[::-1]), dem_array[ij])
                world_xyz = transform(dem_im.crs, self._crs, *[[coord] for coord in dem_xyz])
                zs.append(world_xyz[2][0])

            dem_win = get_win_at_zs(zs)
            dem_win = dem_win.intersection(dem_array_win)  # ensure sub window of dem_array_win

            # crop dem_array to the dem window and find the corresponding transform
            dem_ij_start = (
                dem_win.row_off - dem_array_win.row_off,
                dem_win.col_off - dem_array_win.col_off,
            )
            dem_ij_stop = (dem_ij_start[0] + dem_win.height, dem_ij_start[1] + dem_win.width)
            dem_array = dem_array[
                dem_ij_start[0] : dem_ij_stop[0], dem_ij_start[1] : dem_ij_stop[1]
            ]
            dem_transform = dem_im.window_transform(dem_win)

            # cast dem_array to float32 and set nodata to nan (to persist masking through cv2.remap)
            dem_array = dem_array.astype('float32', copy=False).filled(np.nan)
            return dem_array, dem_transform, dem_im.crs

    def _get_gsd(self) -> float:
        """Return a GSD estimate in units of the world / ortho CRS, that gives approx as many valid
        ortho pixels as valid source pixels.
        """

        def area_poly(coords: np.ndarray) -> float:
            """Area of the polygon defined by (x, y) ``coords`` with (x, y) along 2nd dimension."""
            # uses "shoelace formula": https://en.wikipedia.org/wiki/Shoelace_formula
            return 0.5 * np.abs(
                coords[:, 0].dot(np.roll(coords[:, 1], -1))
                - np.roll(coords[:, 0], -1).dot(coords[:, 1])
            )

        # find image boundary in pixel coordinates, and world coordinates at z=median(DEM),
        # accounting (approximately) for dem-ortho vertical datum offset & z unit scaling
        ji = self._camera.pixel_boundary()
        dem_z = np.nanmedian(self._dem_array)
        dem_ji = (np.array(self._dem_array.shape[::-1]) - 1) / 2
        dem_xyz = (*(self._dem_transform * dem_ji), dem_z)
        world_z = transform(self._dem_crs, self._crs, *[[coord] for coord in dem_xyz])[2][0]
        world_xy = self._camera.world_boundary(world_z)[:2]

        # return the average pixel resolution inside the world boundary
        pixel_area = area_poly(ji.T)
        world_area = area_poly(world_xy.T)
        return np.sqrt(world_area / pixel_area)

    def _reproject_dem(
        self, dem_interp: Interp, resolution: tuple[float, float]
    ) -> tuple[np.ndarray, rio.Affine]:
        """
        Reproject self._dem_array to the world / ortho CRS and ortho resolution, given reprojection
        interpolation and resolution parameters.

        Returns the reprojected DEM array and corresponding transform.
        """
        # return if dem in world / ortho crs and ortho resolution
        dem_res = np.abs((self._dem_transform[0], self._dem_transform[4]))
        if (self._dem_crs == self._crs) and np.all(resolution == dem_res):
            return self._dem_array.copy(), self._dem_transform

        # error check resolution
        init_bounds = array_bounds(*self._dem_array.shape, self._dem_transform)
        ortho_bounds = np.array(transform_bounds(self._dem_crs, self._crs, *init_bounds))
        ortho_size = ortho_bounds[2:] - ortho_bounds[:2]
        if np.any(resolution > ortho_size):
            raise ValueError(f"'resolution' is larger than the ortho size.")

        # find z scaling from dem to world / ortho crs to set MULT_FACTOR_VERTICAL_SHIFT
        # (rasterio does not set it automatically, as GDAL does)
        dem_ji = (np.array(self._dem_array.shape[::-1]) - 1) / 2
        dem_xy = self._dem_transform * dem_ji
        world_zs = []
        for z in [0, 1]:
            world_xyz = transform(self._dem_crs, self._crs, [dem_xy[0]], [dem_xy[1]], [z])
            world_zs.append(world_xyz[2][0])
        z_scale = world_zs[1] - world_zs[0]

        # TODO: rasterio/GDAL sometimes finds bounds for the reprojected dem that lie just inside
        #  the source dem bounds.  This seems suspect, although is unlikely to affect ortho
        #  bounds so am leaving as is for now.

        # reproject dem_array to world / ortho crs and ortho resolution
        dem_array, dem_transform = reproject(
            self._dem_array,
            None,
            src_crs=self._dem_crs,
            src_transform=self._dem_transform,
            src_nodata=float('nan'),
            dst_crs=self._crs,
            dst_resolution=resolution,
            resampling=dem_interp.to_rio(),
            dst_nodata=float('nan'),
            init_dest_nodata=True,
            apply_vertical_shift=True,
            mult_factor_vertical_shift=z_scale,
            num_threads=os.cpu_count(),
        )
        return dem_array.squeeze(), dem_transform

    def _mask_dem(
        self,
        dem_array: np.ndarray,
        dem_transform: rio.Affine,
        dem_interp: Interp,
        crop: bool = True,
        mask: bool = True,
        num_pts: int = 400,
    ) -> tuple[np.ndarray, rio.Affine]:
        """Crop and mask the given DEM to the ortho polygon bounds, returning the adjusted DEM and
        corresponding transform.
        """
        # find ortho boundary polygon
        poly_xy = self._camera.world_boundary(
            dem_array,
            transform=dem_transform,
            interp=dem_interp,
            num_pts=num_pts,
        )[:2, :]

        # find intersection of poly_xy and dem mask, and check dem coverage
        inv_transform = ~(dem_transform * rio.Affine.translation(0.5, 0.5))
        poly_ji = np.round(inv_transform * poly_xy).astype('int')
        poly_mask = np.zeros(dem_array.shape, dtype='uint8')
        poly_mask = cv2.fillPoly(poly_mask, [poly_ji.T], color=(255,)).view(bool)
        dem_mask = poly_mask & ~np.isnan(dem_array)
        dem_mask_sum = dem_mask.sum()

        if dem_mask_sum == 0:
            raise ValueError(
                f"Ortho boundary for '{self._src_name}' lies outside the valid DEM area."
            )
        elif poly_mask.sum() > dem_mask_sum or (
            np.any(np.min(poly_ji, axis=1) < (0, 0))
            or np.any(np.max(poly_ji, axis=1) + 1 > dem_array.shape[::-1])
        ):
            warnings.warn(
                f"Ortho boundary for '{self._src_name}' is not fully covered by the DEM.",
                category=OrthorityWarning,
            )

        if crop:
            # crop dem_mask & dem_array to poly_xy and find corresponding transform
            mask_wheres = [np.where(dem_mask.max(axis=ax))[0] for ax in [1, 0]]
            slices = [slice(mask_where.min(), mask_where.max() + 1) for mask_where in mask_wheres]
            dem_array = dem_array[slices[0], slices[1]]
            dem_mask = dem_mask[slices[0], slices[1]]
            dem_transform = dem_transform * rio.Affine.translation(slices[1].start, slices[0].start)

        if mask:
            # mask the dem
            dem_array[~dem_mask] = np.nan

        return dem_array, dem_transform

    def _create_ortho_profile(
        self,
        src_im: rio.DatasetReader,
        shape: Sequence[int],
        transform: rio.Affine,
        dtype: str,
        compress: str | Compress | None,
        write_mask: bool | None,
    ) -> tuple[dict, bool]:
        """Return a rasterio profile for the ortho image."""
        # Determine dtype, check dtype support
        # (OpenCV remap doesn't support int8 or uint32, and only supports int32, uint64, int64 with
        # nearest interp so these dtypes are excluded).
        ortho_profile = {}
        dtype = dtype or src_im.profile.get('dtype', None)
        if dtype not in Ortho._nodata_vals:
            raise ValueError(f"Data type '{dtype}' is not supported.")

        # setup compression, data interleaving and photometric interpretation
        if compress is None:
            compress = Compress.jpeg if dtype == 'uint8' else Compress.deflate
        else:
            compress = Compress(compress)
            if compress == Compress.jpeg:
                if dtype == 'uint16':
                    warnings.warn(
                        'Attempting a 12 bit JPEG ortho configuration.  Support is rasterio build '
                        'dependent.',
                        category=OrthorityWarning,
                    )
                    ortho_profile.update(nbits=12)
                elif dtype != 'uint8':
                    raise ValueError(
                        f"JPEG compression is supported for 'uint8' and 'uint16' data types only."
                    )

        if compress == Compress.jpeg:
            interleave, photometric = (
                ('pixel', 'ycbcr') if src_im.count == 3 else ('band', 'minisblack')
            )
        else:
            interleave, photometric = ('band', 'minisblack')

        # resolve auto write_mask (=None) to write masks for jpeg compression
        if write_mask is None:
            write_mask = True if compress == Compress.jpeg else False

        # set nodata to None when writing internal masks to force external tools to use mask,
        # otherwise set by dtype
        nodata = None if write_mask else Ortho._nodata_vals[dtype]

        # create ortho profile
        ortho_profile.update(
            driver='GTiff',
            dtype=dtype,
            crs=self._crs,
            transform=transform,
            width=shape[1],
            height=shape[0],
            count=src_im.count,
            tiled=True,
            blockxsize=Ortho._default_blocksize[0],
            blockysize=Ortho._default_blocksize[1],
            nodata=nodata,
            compress=compress.value,
            interleave=interleave,
            photometric=photometric,
        )

        return ortho_profile, write_mask

    def _remap_tile(
        self,
        ortho_im: DatasetWriter,
        src_array: np.ndarray,
        dem_array: np.ndarray,
        tile_win: Window,
        indexes: list[int],
        init_xgrid: np.ndarray,
        init_ygrid: np.ndarray,
        interp: Interp,
        write_mask: bool,
    ) -> None:
        """Thread safe method to map the source image to an ortho tile.  Returns the tile array
        and mask.
        """
        dtype_nodata = self._nodata_vals[ortho_im.profile['dtype']]

        # offset init grids to tile_win
        tile_transform = rio.windows.transform(tile_win, ortho_im.profile['transform'])
        tile_xgrid = init_xgrid[: tile_win.height, : tile_win.width] + (
            tile_transform.xoff - ortho_im.profile['transform'].xoff
        )
        tile_ygrid = init_ygrid[: tile_win.height, : tile_win.width] + (
            tile_transform.yoff - ortho_im.profile['transform'].yoff
        )

        # extract tile_win from dem_array (will be nan outside dem valid area or outside original
        # dem bounds)
        tile_zgrid = dem_array[
            tile_win.row_off : (tile_win.row_off + tile_win.height),
            tile_win.col_off : (tile_win.col_off + tile_win.width),
        ]

        # find mask dilation kernel size for remove blurring with nodata (when needed)
        src_res = np.array((self._gsd, self._gsd))
        ortho_res = np.abs((tile_transform.a, tile_transform.e))
        kernel_size = np.maximum(np.ceil(5 * src_res / ortho_res).astype('int'), (3, 3))

        # remap the source to ortho
        tile_array, tile_mask = self._camera.remap(
            src_array,
            tile_xgrid,
            tile_ygrid,
            tile_zgrid,
            nodata=dtype_nodata,
            interp=interp,
            kernel_size=kernel_size,
        )

        # write tile_array to the ortho image
        with self._write_lock:
            ortho_im.write(tile_array, indexes, window=tile_win)

            if write_mask and (indexes == [1] or len(indexes) == ortho_im.count):
                ortho_im.write_mask(~tile_mask, window=tile_win)

    def _remap(
        self,
        src_im: rio.DatasetReader,
        ortho_im: DatasetWriter,
        dem_array: np.ndarray,
        interp: Interp,
        per_band: bool,
        write_mask: bool,
        progress: bool | std_tqdm,
    ) -> None:
        """Map the source to ortho image by interpolation, given open source and ortho datasets, DEM
        array in the ortho CRS and pixel grid, and configuration parameters.
        """
        # Initialise an (x, y) pixel grid for the first tile here, and offset for remaining tiles
        # in _remap_tile (requires N-up transform).
        # float64 precision is needed for the (x, y) ortho grids in world coordinates for e.g. high
        # resolution drone imagery.
        # gdal / rio geotransform origin refers to the pixel UL corner while OpenCV remap etc.
        # integer pixel coords refer to pixel centers, so the (x, y) coords are offset by half a
        # pixel to account for this.

        j_range = np.arange(0, Ortho._default_blocksize[0])
        i_range = np.arange(0, Ortho._default_blocksize[1])
        init_jgrid, init_igrid = np.meshgrid(j_range, i_range, indexing='xy')
        center_transform = ortho_im.profile['transform'] * rio.Affine.translation(0.5, 0.5)
        init_xgrid, init_ygrid = center_transform * [init_jgrid, init_igrid]

        # create list of band indexes to read, remap & write all bands at once (per_band==False),
        # or per-band (per_band==True)
        if per_band:
            index_list = [[i] for i in range(1, src_im.count + 1)]
        else:
            index_list = [[*range(1, src_im.count + 1)]]

        # create a list of ortho tile windows (assumes all bands configured to same tile shape)
        tile_wins = [tile_win for _, tile_win in ortho_im.block_windows(1)]
        progress.total = len(tile_wins) * len(index_list)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # read, process and write bands, one row of indexes at a time
            for indexes in index_list:
                # read source, and optionally undistort, image band(s) (ortho dtype is
                # required for cv2.remap() to set invalid ortho areas to ortho nodata value)
                dtype_nodata = self._nodata_vals[ortho_im.profile['dtype']]
                src_array = self._camera.read(
                    src_im,
                    indexes=indexes,
                    dtype=ortho_im.profile['dtype'],
                    nodata=dtype_nodata,
                    interp=interp,
                )

                # remap ortho tiles concurrently (tiles are written as they are completed in a
                # possibly non-sequential order, this saves queueing up completed tiles in
                # memory, and is much the same speed as sequential writes)
                futures = [
                    executor.submit(
                        self._remap_tile,
                        ortho_im,
                        src_array,
                        dem_array,
                        tile_win,
                        indexes,
                        init_xgrid,
                        init_ygrid,
                        interp,
                        write_mask,
                    )
                    for tile_win in tile_wins
                ]
                for future in as_completed(futures):
                    future.result()
                    progress.update()
                progress.refresh()

    def process(
        self,
        ortho_file: str | PathLike | OpenFile,
        resolution: tuple[float, float] = _default_config['resolution'],
        interp: str | Interp = _default_config['interp'],
        dem_interp: str | Interp = _default_config['dem_interp'],
        per_band: bool = _default_config['per_band'],
        write_mask: bool | None = _default_config['write_mask'],
        dtype: str = _default_config['dtype'],
        compress: str | Compress | None = _default_config['compress'],
        build_ovw: bool = _default_config['build_ovw'],
        overwrite: bool = _default_config['overwrite'],
        progress: bool | dict = False,
    ) -> None:
        """
        Orthorectify the source image.

        The source image is read and processed band-by-band, or all bands at once, depending on
        the value of ``per_band``.  If necessary, the portion of the DEM stored internally is
        reprojected to the world / ortho CRS and ortho resolution.  Using the camera model and
        DEM, the ortho image is remapped from the source image tile-by-tile.  Up to N ortho tiles
        are processed concurrently, where N is the number of CPUs.

        .. note::

            An occlusion masking option will be added in a future release.

        :param ortho_file:
            Ortho image file to create.  Can be a path or URI string, or an
            :class:`~fsspec.core.OpenFile` object in binary mode (``'wb'``).
        :param resolution:
            Ortho image pixel (x, y) size in units of the world / ortho CRS.  If set to ``None``
            (the default), an approximate ground sampling distance is used as the resolution.
        :param interp:
            Interpolation method for remapping the source to ortho image.
        :param dem_interp:
            Interpolation method for reprojecting the DEM.
        :param per_band:
            Remap the source to ortho image all bands at once (``False``), or band-by-band
            (``True``). ``False`` is faster but requires more memory.
        :param write_mask:
            Mask valid ortho pixels with an internal mask (``True``), or with a nodata value
            based on ``dtype`` (``False``). An internal mask helps remove nodata noise caused by
            lossy compression. If set to ``None`` (the default), the mask will be written when
            JPEG compression is used.
        :param dtype:
            Ortho image data type (``uint8``, ``uint16``, ``float32`` or ``float64``).  If set to
            ``None`` (the default), the source image data type is used.
        :param compress:
            Ortho image compression type (``jpeg`` or ``deflate``).  ``deflate`` can be used with
            any ``dtype``, and ``jpeg`` with the uint8 ``dtype``.  With supporting Rasterio
            builds, ``jpeg`` can also be used with uint16, in which case the ortho is 12 bit JPEG
            compressed. If ``compress`` is set to ``None`` (the default), ``jpeg`` is used for the
            uint8 ``dtype``, and ``deflate`` otherwise.
        :param build_ovw:
            Whether to build overviews for the ortho image.
        :param overwrite:
            Whether to overwrite the ortho image if it exists.
        :param progress:
            Whether to display a progress bar monitoring the portion of ortho tiles written.  Can
            be set to a dictionary of arguments for a custom `tqdm
            <https://tqdm.github.io/docs/tqdm/>`_ bar.
        """
        exit_stack = ExitStack()
        with exit_stack:
            # create the progress bar
            if progress is True:
                progress = tqdm(**Ortho._default_tqdm_kwargs)
            elif progress is False:
                progress = tqdm(disable=True, leave=False)
            else:
                progress = tqdm(**progress)
            progress = exit_stack.enter_context(progress)
            # exit_stack.enter_context(utils.profiler())  # run utils.profiler in DEBUG log level

            # use the GSD for auto resolution if resolution not provided
            if not resolution:
                resolution = (self._gsd, self._gsd)
                res_str = ('{:.4e}' if resolution[0] < 1e-3 else '{:.4f}').format(resolution[0])
                logger.debug('Using auto resolution: ' + res_str)

            # open source image
            env = rio.Env(
                GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False, GDAL_TIFF_INTERNAL_MASK=True
            )
            exit_stack.enter_context(env)
            exit_stack.enter_context(utils.suppress_no_georef())
            src_im = exit_stack.enter_context(utils.OpenRaster(self._src_file, 'r'))

            # warn if source dimensions don't match camera
            if src_im.shape[::-1] != self._camera.im_size:
                warnings.warn(
                    f"Source image '{self._src_name}' size: {src_im.shape[::-1]} does not "
                    f"match camera image size: {self._camera.im_size}."
                )

            # get dem array covering ortho extents in world / ortho crs and ortho resolution
            dem_interp = Interp(dem_interp)
            dem_array, dem_transform = self._reproject_dem(dem_interp, resolution)
            # TODO: Don't mask dem if camera is pinhole or frame with distort=False. Or make dem
            #  masking an option which defaults to not masking with pinhole / frame camera w
            #  distort=False. Note though that dem masking is like occlusion masking for
            #  image edges, which applies to any camera.
            dem_array, dem_transform = self._mask_dem(dem_array, dem_transform, dem_interp)

            # open the ortho image & set write_mask
            ortho_profile, write_mask = self._create_ortho_profile(
                src_im,
                dem_array.shape,
                dem_transform,
                dtype=dtype,
                compress=compress,
                write_mask=write_mask,
            )
            ortho_im = exit_stack.enter_context(
                utils.OpenRaster(ortho_file, 'w', overwrite=overwrite, **ortho_profile)
            )

            # orthorectify
            self._remap(
                src_im,
                ortho_im,
                dem_array,
                interp=Interp(interp),
                per_band=per_band,
                write_mask=write_mask,
                progress=progress,
            )

            if build_ovw:
                # TODO: is it possible to convert to COG here?
                self._build_overviews(ortho_im)


##
