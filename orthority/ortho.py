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

"""Orthrectification using DEM and camera model input."""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.crs import CRS
from rasterio.io import DatasetWriter
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from orthority import utils
from orthority.camera import Camera
from orthority.enums import Compress, Interp
from orthority.errors import CrsMissingError, DemBandError

logger = logging.getLogger(__name__)


class Ortho:
    """
    Orthorectifier.

    Uses a supplied DEM and camera model to correct a source image for sensor and terrain
    distortion.  The camera model, and a portion of the DEM corresponding to the ortho bounds
    are stored internally.

    :param src_file:
        Source image to be orthorectified. Can be a path or URI string,
        an :class:`~fsspec.core.OpenFile` object in binary mode ('rb'), or a dataset reader.
    :param dem_file:
        DEM image covering the source image.  Can be a path or URI string,
        an :class:`~fsspec.core.OpenFile` object in binary mode ('rb'), or a dataset reader.
    :param camera:
        Source image camera model (can be created with :meth:`~orthority.camera.create_camera`).
    :param crs:
        CRS of the ``camera`` world coordinates and ortho image as an EPSG, proj4 or WKT string,
        or :class:`~rasterio.crs.CRS` object.  If set to None (the default), the CRS will be read
        from the source image if possible. If the source image is not projected in the world /
        ortho CRS, ``crs`` should be supplied.
    :param dem_band:
        Index of the DEM band to use (1-based).
    """

    # TODO: what happens if the source image size does not match the camera configuration?
    # default configuration values for Ortho.process()
    _default_config = dict(
        dem_band=1,
        resolution=None,
        interp=Interp.cubic,
        dem_interp=Interp.cubic,
        per_band=False,
        full_remap=True,
        write_mask=None,
        dtype=None,
        compress=None,
        build_ovw=True,
        overwrite=False,
    )

    # default ortho (x, y) block size
    _default_blocksize = (512, 512)

    # Minimum EGM96 geoid altitude i.e. minimum possible vertical difference with the WGS84
    # ellipsoid
    _egm96_min = -106.71

    # nodata values for supported ortho data types
    _nodata_vals = dict(
        uint8=0,
        uint16=0,
        int16=np.iinfo('int16').min,
        float32=float('nan'),
        float64=float('nan'),
    )

    def __init__(
        self,
        src_file: str | Path | OpenFile | rio.DatasetReader,
        dem_file: str | Path | OpenFile | rio.DatasetReader,
        camera: Camera,
        crs: str | CRS | None = None,
        dem_band: int = 1,
    ) -> None:
        if not isinstance(camera, Camera):
            raise TypeError("'camera' is not a Camera instance.")
        if camera._horizon_fov():
            raise ValueError(
                "'camera' has a field of view that includes, or is above, the horizon."
            )

        self._src_file = src_file
        self._src_name = utils.get_filename(src_file)
        self._camera = camera
        self._write_lock = threading.Lock()

        self._crs = self._parse_crs(crs)
        self._dem_array, self._dem_transform, self._dem_crs, self._crs_equal = self._get_init_dem(
            dem_file, dem_band
        )
        self._gsd = self._get_gsd()

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
        if not crs.is_projected:
            raise ValueError(f"'crs' should be a projected system.")
        return crs

    def _get_init_dem(
        self, dem_file: str | rio.DatasetReader, dem_band: int
    ) -> tuple[np.ndarray, rio.Affine, CRS, bool]:
        """Return an initial DEM array in its own CRS and resolution.  Includes the corresponding
        DEM transform, CRS, and flag indicating ortho and DEM CRS equality in return values.
        """

        # TODO: can vertical datums be extracted so we know initial z_min and subsequent offset
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), utils.OpenRaster(dem_file, 'r') as dem_im:
            if dem_band <= 0 or dem_band > dem_im.count:
                dem_name = utils.get_filename(dem_file)
                raise DemBandError(
                    f"DEM band {dem_band} is invalid for '{dem_name}' with {dem_im.count} band(s)"
                )
            # crs comparison is time-consuming - perform it once here, and return result for use
            # elsewhere
            crs_equal = self._crs == dem_im.crs
            dem_full_win = Window(0, 0, dem_im.width, dem_im.height)

            # boundary pixel coordinates of source image
            w, h = np.array(self._camera._im_size) - 1
            src_ji = np.array(
                [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]]
            ).T

            def get_win_at_z_min(z_min: float) -> Window:
                """Return a DEM window corresponding to the ortho bounds at z=z_min."""
                world_xyz = self._camera.pixel_to_world_z(src_ji, z_min)
                # ensure the camera position is included in bounds so that oblique view
                # bounds at z=z_min will include bounds at higher altitudes (z>z_min)
                world_xyz_T = np.column_stack((world_xyz, self._camera._T))
                world_bounds = [*np.min(world_xyz_T[:2], axis=1), *np.max(world_xyz_T[:2], axis=1)]
                dem_bounds = (
                    transform_bounds(self._crs, dem_im.crs, *world_bounds)
                    if not crs_equal
                    else world_bounds
                )
                dem_win = dem_im.window(*dem_bounds)
                try:
                    dem_win = dem_full_win.intersection(dem_win)
                except rio.errors.WindowError:
                    raise ValueError(f"Ortho boundary for '{self._src_name}' lies outside the DEM.")
                return utils.expand_window_to_grid(dem_win)

            # get a dem window corresponding to ortho world bounds at min possible altitude,
            # read the window from the dem & convert to float32 with nodata=nan
            dem_win = get_win_at_z_min(self._egm96_min)
            dem_array = dem_im.read(dem_band, window=dem_win, masked=True)
            dem_array_win = dem_win

            # reduce the dem window to correspond to the ortho world bounds at min dem altitude,
            # accounting for worst case dem-ortho vertical datum offset
            dem_min = dem_array.min()
            dem_win = get_win_at_z_min(dem_min if crs_equal else max(dem_min, 0) + self._egm96_min)
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

            return dem_array, dem_transform, dem_im.crs, crs_equal

    def _get_gsd(self) -> float:
        """Return a GSD estimate that gives approx as many valid ortho pixels as valid source
        pixels.
        """

        def area_poly(coords: np.ndarray) -> float:
            """Area of the polygon defined by (x, y) ``coords`` with (x, y) along 2nd dimension."""
            # uses "shoelace formula": https://en.wikipedia.org/wiki/Shoelace_formula
            return 0.5 * np.abs(
                coords[:, 0].dot(np.roll(coords[:, 1], -1))
                - np.roll(coords[:, 0], -1).dot(coords[:, 1])
            )

        # find (x, y) coords of image boundary in world CRS at z=median(DEM) (note: ignores
        # vertical datum shifts)
        w, h = np.array(self._camera._im_size) - 1
        src_ji = np.array(
            [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]]
        ).T
        world_xy = self._camera.pixel_to_world_z(src_ji, np.nanmedian(self._dem_array))[:2].T

        # return the average pixel resolution inside the world CRS boundary
        pixel_area = np.array(self._camera._im_size).prod()
        world_area = area_poly(world_xy)
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
        if self._crs_equal and np.all(resolution == dem_res):
            return self._dem_array.copy(), self._dem_transform

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
            num_threads=os.cpu_count(),
        )
        return dem_array.squeeze(), dem_transform

    def _get_src_boundary(self, full_remap: bool, num_pts: int) -> np.ndarray:
        """Return a pixel coordinate boundary of the source image with ``num_pts`` points."""

        def im_boundary(im_size: np.ndarray, num_pts: int) -> np.ndarray:
            """Return a rectangular pixel coordinate boundary of ``num_pts`` ~evenly spaced points
            for the given image size ``im_size``.
            """
            br = im_size - 1
            perim = 2 * br.sum()
            cnr_ji = np.array([[0, 0], [br[0], 0], br, [0, br[1]], [0, 0]])
            dist = np.sum(np.abs(np.diff(cnr_ji, axis=0)), axis=1)
            return np.row_stack(
                [
                    np.linspace(
                        cnr_ji[i],
                        cnr_ji[i + 1],
                        np.round(num_pts * dist[i] / perim).astype('int'),
                        endpoint=False,
                    )
                    for i in range(0, 4)
                ]
            ).T

        ji = im_boundary(np.array(self._camera._im_size), num_pts=num_pts)
        if not full_remap:
            # undistort ji to match the boundary of the self._undistort() image
            ji = self._camera.undistort(ji, clip=True)
        return ji

    def _mask_dem(
        self,
        dem_array: np.ndarray,
        dem_transform: rio.Affine,
        dem_interp: Interp,
        full_remap: bool,
        crop: bool = True,
        mask: bool = True,
        num_pts: int = 400,
    ) -> tuple[np.ndarray, rio.Affine]:
        """Crop and mask the given DEM to the ortho polygon bounds, returning the adjusted DEM and
        corresponding transform.
        """

        def inv_transform(transform: rio.Affine, xy: np.ndarray) -> np.ndarray:
            """Return the center (j, i) pixel coords for the given transform and world (x, y)
            coordinates.
            """
            return np.array(~(transform * rio.Affine.translation(0.5, 0.5)) * xy)

        if not crop and not mask:
            return dem_array, dem_transform

        # get polygon of source boundary with 'num_pts' points
        src_ji = self._get_src_boundary(full_remap=full_remap, num_pts=num_pts)

        # find / test dem minimum and maximum, and initialise
        dem_min = np.nanmin(dem_array)
        if dem_min > self._camera._T[2]:
            raise ValueError('The DEM is higher than the camera.')
        # limit dem_max to camera height so that rays go forwards only
        dem_max = min(np.nanmax(dem_array), self._camera._T[2, 0])
        # heuristic limit on ray length to conserve memory
        max_ray_steps = 2 * np.sqrt(np.square(dem_array.shape, dtype='int64').sum()).astype('int')
        poly_xy = np.zeros((2, src_ji.shape[1]))

        # find dem (x, y, z) world coordinate intersections for each (j, i) pixel coordinate in
        # src_ji
        for pi in range(src_ji.shape[1]):
            src_pt = src_ji[:, pi].reshape(-1, 1)

            # create world points along the src_pt ray with (x, y) stepsize <= dem resolution,
            # if num points <= max_ray_steps, else max_ray_steps points
            start_xyz = self._camera.pixel_to_world_z(src_pt, dem_min, distort=full_remap)
            stop_xyz = self._camera.pixel_to_world_z(src_pt, dem_max, distort=full_remap)
            ray_steps = np.abs(
                (stop_xyz - start_xyz)[:2].squeeze() / (dem_transform[0], dem_transform[4])
            )
            ray_steps = min(np.ceil(ray_steps.max()).astype('int') + 1, max_ray_steps)
            ray_z = np.linspace(dem_max, dem_min, ray_steps)
            ray_xyz = self._camera.pixel_to_world_z(src_pt, ray_z, distort=full_remap)

            # find the dem z values corresponding to the ray (dem_z will be nan outside the dem
            # bounds and for already masked / nan dem pixels)
            dem_ji = inv_transform(dem_transform, ray_xyz[:2]).astype('float32', copy=False)
            dem_z = np.full((dem_ji.shape[1],), dtype=dem_array.dtype, fill_value=float('nan'))
            # dem_ji = cv2.convertMaps(*dem_ji, cv2.CV_16SC2)
            cv2.remap(
                dem_array, *dem_ji, dem_interp.to_cv(), dst=dem_z, borderMode=cv2.BORDER_TRANSPARENT
            )

            # store the first ray-dem intersection point if it exists, otherwise the dem_min point
            valid_mask = ~np.isnan(dem_z)
            dem_z = dem_z[valid_mask]
            dem_min_xy = ray_xyz[:2, -1]
            ray_xyz = ray_xyz[:, valid_mask]
            intersection_i = np.nonzero(ray_xyz[2] <= dem_z)[0]
            if len(intersection_i) > 0:
                poly_xy[:, pi] = ray_xyz[:2, intersection_i[0]]
            else:
                poly_xy[:, pi] = dem_min_xy

        # find intersection of poly_xy and dem mask, and check dem coverage
        poly_ji = np.round(inv_transform(dem_transform, poly_xy)).astype('int')
        poly_mask = np.zeros(dem_array.shape, dtype='uint8')
        poly_mask = cv2.fillPoly(poly_mask, [poly_ji.T], color=(255,)).astype('bool', copy=False)
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
            logger.warning(
                f"Ortho boundary for '{self._src_name}' is not fully covered by the DEM."
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
        shape: tuple[int, ...],
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
                    logger.warning(
                        'Attempting a 12 bit JPEG ortho configuration.  Support is rasterio build '
                        'dependent.'
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
        full_remap: bool,
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

        # find the source (j, i) pixel coords corresponding to ortho image (x, y, z) world coords
        # fmt: off
        tile_ji = self._camera.world_to_pixel(
            np.array([tile_xgrid.reshape(-1,), tile_ygrid.reshape(-1,), tile_zgrid.reshape(-1,)]),
            distort=full_remap,
        )
        # fmt: on

        # separate tile_ji into (j, i) grids, converting to float32 for compatibility with
        # cv2.remap (nans are converted to -1 as cv2.remap maps nans to 0 (the first src pixel)
        # on some packages/platforms see
        # https://answers.opencv.org/question/1057/behavior-of-not-a-number-nan-values-in-remap/)
        tile_ji[np.isnan(tile_ji)] = -1
        tile_jgrid = tile_ji[0].reshape(tile_win.height, tile_win.width).astype('float32')
        tile_igrid = tile_ji[1].reshape(tile_win.height, tile_win.width).astype('float32')
        # tile_jgrid, tile_igrid = cv2.convertMaps(tile_jgrid, tile_igrid, cv2.CV_16SC2)

        # initialise ortho tile array
        tile_array = np.full(
            (src_array.shape[0], tile_win.height, tile_win.width),
            dtype=ortho_im.profile['dtype'],
            fill_value=dtype_nodata,
        )

        # remap source image to ortho tile, looping over band(s) (cv2.remap execution time
        # depends on array ordering)
        # TODO: test speed if src and tile are in cv ordering and this done all bands at once
        for oi in range(0, src_array.shape[0]):
            cv2.remap(
                src_array[oi],
                tile_jgrid,
                tile_igrid,
                interp.to_cv(),
                dst=tile_array[oi],
                borderMode=cv2.BORDER_TRANSPARENT,
            )

        # mask of invalid ortho pixels
        tile_mask = np.all(utils.nan_equals(tile_array, dtype_nodata), axis=0)

        # remove cv2.remap blurring with undistort nodata when full_remap=False...
        if (
            not full_remap
            and interp != Interp.nearest
            and not np.isnan(dtype_nodata)
            and self._camera.undistort_maps is not None
        ):
            src_res = np.array((self._gsd, self._gsd))
            ortho_res = np.abs((tile_transform.a, tile_transform.e))
            kernel_size = np.maximum(np.ceil(5 * src_res / ortho_res).astype('int'), (3, 3))
            kernel = np.ones(kernel_size[::-1], np.uint8)
            tile_mask = cv2.dilate(tile_mask.astype(np.uint8, copy=False), kernel)
            tile_mask = tile_mask.astype(bool, copy=False)
            tile_array[:, tile_mask] = dtype_nodata

        # write tile_array to the ortho image
        with self._write_lock:
            ortho_im.write(tile_array, indexes, window=tile_win)

            if write_mask and (indexes == [1] or len(indexes) == ortho_im.count):
                ortho_im.write_mask(~tile_mask, window=tile_win)

    def _undistort(
        self, image: np.ndarray, nodata: float | int, interp: str | Interp
    ) -> np.ndarray:
        """Undistort an image using ``interp`` interpolation and setting invalid pixels to
        ``nodata``.
        """
        if self._camera.undistort_maps is None:
            return image

        def undistort_band(src_array: np.ndarray, dst_array: np.ndarray):
            """Undistort a 2D band array."""
            # equivalent without stored _undistort_maps:
            # return cv2.undistort(band_array, self._K, self._dist_param)
            cv2.remap(
                src_array,
                *self._camera.undistort_maps,
                Interp[interp].to_cv(),
                dst=dst_array,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

        out_image = np.full(image.shape, dtype=image.dtype, fill_value=nodata)
        if image.ndim > 2:
            # undistort by band so that output data stays in the rasterio ordering
            for bi in range(image.shape[0]):
                undistort_band(image[bi], out_image[bi])
        else:
            undistort_band(image, out_image)

        return out_image

    def _remap(
        self,
        src_im: rio.DatasetReader,
        ortho_im: DatasetWriter,
        dem_array: np.ndarray,
        interp: Interp,
        per_band: bool,
        full_remap: bool,
        write_mask: bool,
    ) -> None:
        """Map the source to ortho image by interpolation, given open source and ortho datasets, DEM
        array in the ortho CRS and pixel grid, and configuration parameters.
        """
        # Initialise an (x, y) pixel grid for the first tile here, and offset for remaining tiles
        # in _remap_tile (requires N-up transform).
        # float64 precision is needed for the (x, y) ortho grids in world coordinates for e.g. high
        # resolution drone imagery.
        # gdal / rio geotransform origin refers to the pixel UL corner while OpenCV remap etc
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

        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
        with logging_redirect_tqdm([logging.getLogger(__package__)], tqdm_class=tqdm), tqdm(
            bar_format=bar_format, total=len(tile_wins) * len(index_list), dynamic_ncols=True
        ) as bar:
            # read, process and write bands, one row of indexes at a time
            for indexes in index_list:
                # read source image band(s) (ortho dtype is required for cv2.remap() to set
                # invalid ortho areas to ortho nodata value)
                src_array = src_im.read(indexes, out_dtype=ortho_im.profile['dtype'])

                if not full_remap:
                    # undistort the source image so the distortion model can be excluded from
                    # self._camera.world_to_pixel() in self._remap_tile()
                    dtype_nodata = self._nodata_vals[ortho_im.profile['dtype']]
                    src_array = self._undistort(src_array, nodata=dtype_nodata, interp=interp)

                # remap ortho tiles concurrently (tiles are written as they are completed in a
                # possibly non-sequential order, this saves queueing up completed tiles in
                # memory, and is much the same speed as sequential writes)
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
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
                            full_remap,
                            write_mask,
                        )
                        for tile_win in tile_wins
                    ]
                    for future in as_completed(futures):
                        future.result()
                        bar.update()

    def process(
        self,
        ortho_file: str | Path | OpenFile,
        resolution: tuple[float, float] = _default_config['resolution'],
        interp: str | Interp = _default_config['interp'],
        dem_interp: str | Interp = _default_config['dem_interp'],
        per_band: bool = _default_config['per_band'],
        full_remap: bool = _default_config['full_remap'],
        write_mask: bool | None = _default_config['write_mask'],
        dtype: str = _default_config['dtype'],
        compress: str | Compress | None = _default_config['compress'],
        build_ovw: bool = _default_config['build_ovw'],
        overwrite: bool = _default_config['overwrite'],
    ) -> None:
        """
        Orthorectify the source image.

        The source image is read and processed band-by-band, or all bands at once, depending on
        the value of ``per_band``.  If necessary, the portion of the DEM stored internally is
        reprojected to the ortho CRS and resolution.  If ``full_remap`` is False, the camera
        model is used to undistort the source image.  Using the camera model and DEM, the ortho
        image is remapped from the source or undistorted source image tile-by-tile.  Up to N
        ortho tiles are processed concurrently, where N is the number of CPUs.

        :param ortho_file:
            Ortho image file to create.  Can be a path or URI string, or an
            :class:`~fsspec.core.OpenFile` object in binary mode ('wb').
        :param resolution:
            Ortho image pixel (x, y) size in units of the world / ortho CRS (usually meters).  If
            set to None (the default), an approximate ground sampling distance is used as the
            resolution.
        :param interp:
            Interpolation method to use for remapping the source to ortho image.
        :param dem_interp:
            Interpolation method for reprojecting the DEM.
        :param per_band:
            Remap the source to ortho image all bands at once (False), or band-by-band (True).
            False is faster but requires more memory.
        :param full_remap:
            Orthorectify the source image with the full camera model (True), or the undistorted
            source image with a pinhole camera model (False).  True remaps the source
            image once.  False is faster but remaps the source image twice, so can reduce ortho
            image quality.
        :param write_mask:
            Mask valid ortho pixels with an internal mask (True), or with a nodata value based on
            ``dtype`` (False). An internal mask helps remove nodata noise caused by lossy
            compression. If set to None (the default), the mask will be written when jpeg
            compression is used.
        :param dtype:
            Ortho image data type ('uint8', 'uint16', 'float32' or 'float64').  If set to None (
            the default), the source image dtype is used.
        :param compress:
            Ortho image compression type ('jpeg' or 'deflate').  Deflate can be used with all
            ``dtype``s, and JPEG with the uint8 ``dtype``.  With supporting ``rasterio`` builds,
            JPEG can also be used with uint16, in which case the ortho is 12 bit JPEG compressed.
            If ``compress`` is set to None (the default), JPEG is used for the uint8 ``dtype``,
            and Deflate otherwise.
        :param build_ovw:
            Whether to build overviews for the ortho image.
        :param overwrite:
            Whether to overwrite the ortho image if it exists.
        """
        with utils.profiler():  # run utils.profiler in DEBUG log level
            # use the GSD for auto resolution if resolution not provided
            if not resolution:
                resolution = (self._gsd, self._gsd)
                logger.debug(f'Using auto resolution: {resolution[0]:.4f}')

            env = rio.Env(
                GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False, GDAL_TIFF_INTERNAL_MASK=True
            )
            with env, utils.suppress_no_georef(), utils.OpenRaster(self._src_file, 'r') as src_im:
                # warn if source dimensions don't match camera
                if src_im.shape[::-1] != self._camera._im_size:
                    logger.warning(
                        f"Source image '{self._src_name}' size: {src_im.shape[::-1]} does not "
                        f"match camera image size: {self._camera._im_size}."
                    )

                # get dem array covering ortho extents in world / ortho crs and ortho resolution
                dem_interp = Interp(dem_interp)
                dem_array, dem_transform = self._reproject_dem(dem_interp, resolution)
                # TODO: don't mask dem if pinhole camera, or make dem masking an option which
                #  defaults to not masking with pinhole camera.  note though that dem masking is
                #  like occlusion masking for image edges, which still applies to pinhole camera.
                dem_array, dem_transform = self._mask_dem(
                    dem_array, dem_transform, dem_interp, full_remap=full_remap
                )

                # create ortho profile & set write_mask
                ortho_profile, write_mask = self._create_ortho_profile(
                    src_im,
                    dem_array.shape,
                    dem_transform,
                    dtype=dtype,
                    compress=compress,
                    write_mask=write_mask,
                )
                # TODO: any existing PAM or other sidecar file will not be removed / overwritten
                #  as utils.OpenRaster works currently
                with utils.OpenRaster(
                    ortho_file, 'w', overwrite=overwrite, **ortho_profile
                ) as ortho_im:
                    # orthorectify
                    self._remap(
                        src_im,
                        ortho_im,
                        dem_array,
                        interp=Interp(interp),
                        per_band=per_band,
                        full_remap=full_remap,
                        write_mask=write_mask,
                    )

                    if build_ovw:
                        # TODO: is it possible to convert to COG here?
                        self._build_overviews(ortho_im)


##
