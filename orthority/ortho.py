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
from concurrent.futures import ThreadPoolExecutor
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
from rasterio.warp import reproject, transform, transform_bounds
from rasterio.windows import Window
from tqdm.auto import tqdm

from orthority import common
from orthority.camera import Camera, FrameCamera
from orthority.enums import Compress, Driver, Interp
from orthority.errors import CrsMissingError, OrthorityError, OrthorityWarning

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

    # default algorithm configuration values for Ortho.process()
    _default_alg_config = dict(
        dem_band=1, resolution=None, interp=Interp.cubic, dem_interp=Interp.cubic, per_band=False
    )

    # EGM96/EGM2008 geoid altitude range i.e. minimum and maximum possible vertical difference with
    # the WGS84 ellipsoid (meters)
    _egm_minmax = [-106.71, 82.28]

    # Maximum possible ellipsoidal height i.e. approx. that of Everest (meters)
    _z_max = 8850.0

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
        self._src_name = common.get_filename(src_file)
        self._camera = camera
        self._write_lock = threading.Lock()

        self._crs = self._parse_crs(crs)
        self._dem_array, self._dem_transform, self._dem_crs = self._get_init_dem(dem_file, dem_band)
        self._gsd = self._get_gsd()

    @property
    def camera(self) -> Camera | FrameCamera:
        """Source image camera model."""
        return self._camera

    def _parse_crs(self, crs: str | CRS) -> CRS:
        """Derive a world / ortho CRS from the ``crs`` parameter and source image."""
        if crs:
            crs = CRS.from_string(crs) if isinstance(crs, str) else crs
        else:
            with common.suppress_no_georef(), common.OpenRaster(self._src_file, 'r') as src_im:
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
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUS'), common.OpenRaster(dem_file, 'r') as dem_im:
            if dem_band <= 0 or dem_band > dem_im.count:
                dem_name = common.get_filename(dem_file)
                raise OrthorityError(
                    f"DEM band {dem_band} is invalid for '{dem_name}' with {dem_im.count} band(s)"
                )
            # crs comparison is time-consuming - perform it once here
            crs_equal = self._crs == dem_im.crs

            # find the scale from meters to ortho crs z units (projects from a valid (x,y) in ortho
            # crs so that we stay inside projection domains (#18))
            zs = []
            ref_crs = rio.CRS.from_epsg(4979)
            ji = np.array(self.camera.im_size).reshape(-1, 1) / 2
            world_xyz = self.camera.pixel_to_world_z(ji, 0)
            for z in [0, 1]:
                ref_xyz = transform(self._crs, ref_crs, world_xyz[0], world_xyz[1], [z])
                zs.append(ref_xyz[2][0])
            z_scale = 1 / (zs[1] - zs[0])
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
                    raise OrthorityError(f"Ortho for '{self._src_name}' lies outside the DEM.")
                return common.expand_window_to_grid(dem_win)

            # get a dem window containing the ortho bounds at min & max possible altitude
            zs = np.array([self._egm_minmax[0], self._z_max + self._egm_minmax[1]]) * z_scale
            dem_win = get_win_at_zs(zs)
            # read the window from the dem
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
            raise OrthorityError(
                f"Ortho resolution for '{self._src_name}' is larger than the ortho bounds."
            )

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
        # TODO: option to align the reprojected transform to whole number of pixels from 0 offset
        # TODO: if possible, read (,mask) and reproject the dem from dataset in blocks as the ortho
        #  is written.  or read the dem and src image in parallel, avoiding masked reads

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
            raise OrthorityError(f"Ortho for '{self._src_name}' lies outside the valid DEM area.")
        elif poly_mask.sum() > dem_mask_sum or (
            np.any(np.min(poly_ji, axis=1) < (0, 0))
            or np.any(np.max(poly_ji, axis=1) + 1 > dem_array.shape[::-1])
        ):
            warnings.warn(
                f"Ortho for '{self._src_name}' is not fully covered by the DEM.",
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
        dtype_nodata = common._nodata_vals[ortho_im.profile['dtype']]

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
        progress: tqdm,
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
        block_win = next(common.block_windows(ortho_im))
        j_range = np.arange(0, block_win.width)
        i_range = np.arange(0, block_win.height)
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
        tile_wins = [*common.block_windows(ortho_im)]
        progress.total = len(tile_wins) * len(index_list)

        # TODO: Memory increases ~linearly with number of threads, but does processing speed?
        #  Make number of threads configurable and place a limit on the default value
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # read, process and write bands, one row of indexes at a time
            for indexes in index_list:
                # read source, and optionally undistort, image band(s) (ortho dtype is
                # required for cv2.remap() to set invalid ortho areas to ortho nodata value)
                dtype_nodata = common._nodata_vals[ortho_im.profile['dtype']]
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
                for future in futures:
                    try:
                        future.result()
                    except Exception as ex:
                        # TODO: add cancel_futures=True here and in all executors when min
                        #  supported python >= 3.9
                        executor.shutdown(wait=False)
                        raise RuntimeError('Could not remap tile.') from ex
                    progress.update()
                progress.refresh()

    def process(
        self,
        ortho_file: str | PathLike | OpenFile,
        resolution: tuple[float, float] = _default_alg_config['resolution'],
        interp: str | Interp = _default_alg_config['interp'],
        dem_interp: str | Interp = _default_alg_config['dem_interp'],
        per_band: bool = _default_alg_config['per_band'],
        write_mask: bool | None = common._default_out_config['write_mask'],
        dtype: str = common._default_out_config['dtype'],
        compress: str | Compress | None = common._default_out_config['compress'],
        build_ovw: bool = common._default_out_config['build_ovw'],
        creation_options: dict | None = None,
        driver: str | Driver = common._default_out_config['driver'],
        overwrite: bool = common._default_out_config['overwrite'],
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
            Ortho image data type (``uint8``, ``uint16``, ``int16``, ``float32`` or ``float64``).
            If set to ``None`` (the default), the source image data type is used.
        :param compress:
            Ortho image compression type (``jpeg``, ``deflate`` or ``lzw``).  ``deflate`` and
            ``lzw`` can be used with any ``dtype``, and ``jpeg`` with the uint8 ``dtype``.  With
            supporting Rasterio builds, ``jpeg`` can also be used with uint16, in which case the
            ortho is 12 bit JPEG compressed. If ``compress`` is set to ``None`` (the default),
            ``jpeg`` is used for the uint8 ``dtype``, and ``deflate`` otherwise.
        :param build_ovw:
            Whether to build overviews for the ortho image.
        :param creation_options:
            Ortho image creation options as dictionary of ``name: value`` pairs.  If supplied,
            ``compress`` is ignored.  See the `GDAL docs
            <https://gdal.org/en/latest/drivers/raster/gtiff.html#creation-options>`__ for details.
        :param driver:
            Ortho image driver (``gtiff`` or ``cog``).
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
                progress = common.get_tqdm_kwargs(unit='blocks')
            elif progress is False:
                progress = dict(disable=True, leave=False)
            progress = exit_stack.enter_context(tqdm(**progress))
            # exit_stack.enter_context(common.profiler())  # run common.profiler in DEBUG log level

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
            exit_stack.enter_context(common.suppress_no_georef())
            src_im = exit_stack.enter_context(common.OpenRaster(self._src_file, 'r'))

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

            # open the ortho image & resolve write_mask
            dtype = dtype or src_im.dtypes[0]
            ortho_profile, write_mask = common.create_profile(
                driver=driver,
                shape=(src_im.count, *dem_array.shape),
                dtype=dtype,
                compress=compress,
                write_mask=write_mask,
                creation_options=creation_options,
            )
            ortho_profile.update(crs=self._crs, transform=dem_transform)
            ortho_im = exit_stack.enter_context(
                common.OpenRaster(ortho_file, 'w', overwrite=overwrite, **ortho_profile)
            )

            # copy colorinterp from source to ortho
            ortho_im.colorinterp = src_im.colorinterp

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
                common.build_overviews(ortho_im)


##
