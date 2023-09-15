"""
   Copyright 2021 Dugal Harris - dugalh@gmail.com

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import logging
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Union, Dict, List, Optional
from math import ceil

import cv2
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import Window
from tqdm.auto import tqdm

from simple_ortho.camera import Camera
from simple_ortho.enums import Interp, Compress
from simple_ortho.errors import CrsError, CrsMissingError, DemBandError
from simple_ortho.utils import suppress_no_georef, expand_window_to_grid, nan_equals, profiler

# from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)


class Ortho:
    # default configuration values for Ortho.process()
    _default_config = dict(
        dem_band=1, resolution=None, interp=Interp.cubic, dem_interp=Interp.cubic, per_band=False, full_remap=True,
        write_mask=None, dtype=None, compress=Compress.auto, build_ovw=True, overwrite=False,
    )

    # default ortho (x, y) block size
    _default_blocksize = (512, 512)

    # Minimum EGM96 geoid altitude i.e. minimum possible vertical difference with the WGS84 ellipsoid
    _egm96_min = -106.71

    # nodata values for supported ortho data types
    _nodata_vals = dict(
        uint8=0, uint16=0, int16=np.iinfo('int16').min, float32=float('nan'), float64=float('nan'),
    )

    def __init__(
        self, src_filename: Union[str, Path], dem_filename: Union[str, Path], camera: Camera,
        crs: Union[str, rio.CRS] = None, dem_band: int = 1
    ):
        """
        Class for orthorectifying an image with a specified DEM and camera model.

        Parameters
        ----------
        src_filename: str, pathlib.Path
            Path/URL to a source image to be orthorectified.
        dem_filename: str, pathlib.Path
            Path/URL to a DEM image covering the source image.
        camera: Camera
            Source image camera model (see :meth:`~simple_ortho.camera.create_camera`).
        crs: str, rasterio.CRS, optional
            CRS of the ``camera`` world coordinates and ortho image as an EPSG, proj4 or WKT string.  It should be a projected,
            and not geographic CRS.  Can be omitted if the source image is projected in the ortho CRS.
        dem_band: int, optional
            Index of the DEM band to use (1-based).
        """
        if not Path(src_filename).exists():
            raise FileNotFoundError(f'Source image file {src_filename} does not exist')
        if not Path(dem_filename).exists():
            raise FileNotFoundError(f'DEM image file {dem_filename} does not exist')
        if not isinstance(camera, Camera):
            raise TypeError("'camera' is not a Camera instance.")
        if camera._horizon_fov():
            raise ValueError("'camera' has a field of view that includes, or is above, the horizon.")

        # TODO: refactor so that the camera is guaranteed to be the correct one for src_filename (e.g. the camera has a
        #  src_filename property itself, and src_filename is not passed here?  also the separation of crs from camera
        #  is not ideal, the camera and ortho crs should also be tied together)
        # TODO: make Ortho a context manager that opens the src file once?
        self._src_filename = Path(src_filename)
        self._camera = camera
        self._write_lock = threading.Lock()

        self._ortho_crs = self._parse_crs(crs)
        self._dem_array, self._dem_transform, self._dem_crs, self._crs_equal = self._get_init_dem(
            Path(dem_filename), dem_band
        )

    @staticmethod
    def _build_overviews(im: rio.io.DatasetWriter, max_num_levels: int = 8, min_level_pixels: int = 256):
        """ Build internal overviews for a given rasterio dataset. """
        max_ovw_levels = int(np.min(np.log2(im.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)

    def _parse_crs(self, crs: Union[str, rio.CRS]) -> rio.CRS:
        """ Derive an ortho CRS from the ``crs`` parameter and source image. """
        if crs:
            try:
                ortho_crs = rio.CRS.from_string(crs) if isinstance(crs, str) else crs
            except rio.errors.CRSError as ex:
                raise CrsError(f"Could not interpret 'crs': {crs}. {str(ex)}")
        else:
            with suppress_no_georef(), rio.open(self._src_filename, 'r') as src_im:
                if src_im.crs:
                    ortho_crs = src_im.crs
                else:
                    raise CrsMissingError(f"No source image projection found, 'crs' should be specified.")
        if ortho_crs.is_geographic:
            raise CrsError(f"'crs' should be a projected, not geographic system.")
        return ortho_crs

    def _get_init_dem(self, dem_filename: Path, dem_band: int) -> Tuple[np.ndarray, rio.Affine, rio.CRS, bool]:
        """
        Return an initial DEM array in its own CRS and resolution.  Includes the corresponding DEM transform, CRS,
        and flag indicating ortho and DEM CRS equality in return values.

        The returned DEM array is read to cover the ortho bounds at the z=min(DEM) plane, accounting for worst case
        vertical datum offset between the DEM and ortho CRS, and within the limits of the DEM image bounds.
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(dem_filename, 'r') as dem_im:
            # TODO: can vertical datums be extracted so we know initial z_min and subsequent offset
            if dem_band <= 0 or dem_band > dem_im.count:
                raise DemBandError(
                    f"'dem_band': {dem_band} is invalid for {dem_filename.name} with {dem_im.count} bands"
                )
            # crs comparison is time-consuming - perform it once here, and return result for use elsewhere
            crs_equal = self._ortho_crs == dem_im.crs
            dem_full_win = Window(0, 0, dem_im.width, dem_im.height)

            # corner pixel coordinates of source image
            src_br = self._camera._im_size
            src_ji = np.array([[0, 0], [src_br[0], 0], src_br, [0, src_br[1]]]).T

            def get_win_at_z_min(z_min: float) -> Window:
                """ Return a DEM window corresponding to the ortho bounds at z=z_min. """
                world_xyz = self._camera.pixel_to_world_z(src_ji, z_min)
                # ensure the camera position is included in bounds so that the oblique view bounds at z=z_min will
                # include bounds at higher altitudes (z>z_min)
                world_xyz_T = np.column_stack((world_xyz, self._camera._T))
                world_bounds = [*np.min(world_xyz_T[:2], axis=1), *np.max(world_xyz_T[:2], axis=1)]
                dem_bounds = (
                    transform_bounds(self._ortho_crs, dem_im.crs, *world_bounds) if not crs_equal else world_bounds
                )
                dem_win = dem_im.window(*dem_bounds)
                try:
                    dem_win = dem_full_win.intersection(dem_win)
                except rio.errors.WindowError:
                    raise ValueError(f'Ortho bounds for {self._src_filename.name} lie outside the DEM.')
                return expand_window_to_grid(dem_win)

            # get a dem window corresponding to ortho world bounds at min possible altitude
            dem_win = get_win_at_z_min(self._egm96_min)
            dem_array = dem_im.read(dem_band, window=dem_win, masked=True)
            dem_array_win = dem_win

            # reduce the dem window to correspond to the ortho world bounds at min dem altitude, accounting for worst
            # case dem-ortho vertical datum offset
            dem_min = np.nanmin(dem_array)
            dem_win = get_win_at_z_min(dem_min if crs_equal else max(dem_min, 0) + self._egm96_min)
            dem_win = dem_win.intersection(dem_array_win)  # ensure it is a sub window of dem_array_win

            # crop dem_array to the dem window and find the corresponding transform
            dem_ij_start = (dem_win.row_off - dem_array_win.row_off, dem_win.col_off - dem_array_win.col_off)
            dem_ij_stop = (dem_ij_start[0] + dem_win.height, dem_ij_start[1] + dem_win.width)
            dem_array = dem_array[dem_ij_start[0]:dem_ij_stop[0], dem_ij_start[1]:dem_ij_stop[1]]
            dem_transform = dem_im.window_transform(dem_win)

            # Cast dem_array to float32 and set nodata to nan (to persist masking through cv2.remap)
            dem_array = dem_array.astype('float32', copy=False).filled(np.nan)

            return dem_array, dem_transform, dem_im.crs, crs_equal

    def _get_auto_res(self) -> Tuple[float, float]:
        """ Return an ortho resolution that gives approx. as many valid ortho pixels as valid source pixels. """
        def area_quad(cnrs: np.array) -> float:
            """ Area of the quadrilateral defined by (x, y) corners in `cnrs` (4 x 2). """
            # uses "shoelace formula" - https://en.wikipedia.org/wiki/Shoelace_formula
            return 0.5 * np.abs(cnrs[:, 0].dot(np.roll(cnrs[:, 1], -1)) - np.roll(cnrs[:, 0], -1).dot(cnrs[:, 1]))

        # find (x, y) coords of image corners in world CRS at z=median(DEM) (note: ignores vertical datum shifts)
        src_br = self._camera._im_size
        src_ji = np.array([[0, 0], [src_br[0], 0], src_br, [0, src_br[1]]])
        world_xy = self._camera.pixel_to_world_z(src_ji.T, np.nanmedian(self._dem_array))[:2].T

        # return the average pixel resolution inside the world CRS quadrilateral
        pixel_area = np.array(self._camera._im_size).prod()
        world_area = area_quad(world_xy)
        res = np.sqrt(world_area / pixel_area)
        logger.debug(f'Using auto resolution: {res:.4f}')
        return (res, ) * 2

    def _reproject_dem(self, dem_interp: Interp, resolution: Tuple[float, float]) -> Tuple[np.ndarray, rio.Affine]:
        """
        Reproject self._dem_array to the ortho CRS and resolution, given reprojection interpolation and resolution
        parameters. Returns the reprojected DEM array and corresponding transform.
        """
        # return if dem in ortho crs and resolution
        dem_res = np.abs((self._dem_transform[0], self._dem_transform[4]))
        if self._crs_equal and np.all(resolution == dem_res):
            return self._dem_array.copy(), self._dem_transform

        # reproject dem_array to ortho crs and resolution
        dem_array, dem_transform = reproject(
            self._dem_array, None, src_crs=self._dem_crs, src_transform=self._dem_transform, src_nodata=float('nan'),
            dst_crs=self._ortho_crs, dst_resolution=resolution, resampling=dem_interp.to_rio(), dst_nodata=float('nan'),
            init_dest_nodata=True, apply_vertical_shift=True, num_threads=multiprocessing.cpu_count()
        )
        return dem_array.squeeze(), dem_transform

    def _get_src_boundary(self, full_remap: bool = True, num_pts: int = 400) -> np.ndarray:
        """ Return a pixel coordinate boundary of the source image with `num_pts` points. """

        def im_boundary(im_size: np.array, num_pts: int = 400):
            """
            Return a rectangular pixel coordinate boundary of `num_pts` ~evenly spaced points for the given image
            size `im_size`.
            """
            br = im_size - 1
            perim = 2 * br.sum()
            cnr_ji = np.array([[0, 0], [br[0], 0], br, [0, br[1]], [0, 0]])
            dist = np.sum(np.abs(np.diff(cnr_ji, axis=0)), axis=1)
            return np.row_stack([
                np.linspace(cnr_ji[i], cnr_ji[i+1], np.round(num_pts * dist[i] / perim).astype('int'), endpoint=False)
                for i in range(0, 4)
            ]).T  # yapf: disable

        ji = im_boundary(np.array(self._camera._im_size), num_pts=num_pts)
        if not full_remap:
            # undistort ji to match the boundary of the self._undistort() image
            ji = self._camera.undistort(ji, clip=True)
        return ji

    def _mask_dem(
        self, dem_array: np.ndarray, dem_transform: rio.Affine, dem_interp: Interp, full_remap: bool = True,
        crop: bool = True, mask: bool = True, num_pts: int = 400
    ) -> Tuple[np.ndarray, rio.Affine]:
        """
        Crop and mask the given DEM to the ortho polygon bounds, returning the adjusted DEM and corresponding
        transform.
        """
        # TODO: to exclude occluded areas, this should find the first/highest dem intersection, not the last/lowest
        #  as it currently does.
        def inv_transform(transform: rio.Affine, xy: np.array):
            """ Return the center (j, i) pixel coords for the given transform and world (x, y) coordinates. """
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
        poly_xyz = np.zeros((3, src_ji.shape[1]))

        # find dem (x, y, z) world coordinate intersections for each (j, i) pixel coordinate in src_ji
        for pi in range(src_ji.shape[1]):
            src_pt = src_ji[:, pi].reshape(-1, 1)

            # create world points along the src_pt ray with (x, y) stepsize <= dem resolution, if num points <=
            # max_ray_steps, else max_ray_steps points
            start_xyz = self._camera.pixel_to_world_z(src_pt, dem_min, distort=full_remap)
            stop_xyz = self._camera.pixel_to_world_z(src_pt, dem_max, distort=full_remap)
            ray_steps = np.abs((stop_xyz - start_xyz)[:2].squeeze() / (dem_transform[0], dem_transform[4]))
            ray_steps = min(np.ceil(ray_steps.max()).astype('int') + 1, max_ray_steps)
            ray_z = np.linspace(dem_min, dem_max, ray_steps)
            ray_xyz = self._camera.pixel_to_world_z(src_pt, ray_z, distort=full_remap)

            # find the dem pixel coords, and validity mask for the (x, y) points in ray_xyz
            dem_ji = inv_transform(dem_transform, ray_xyz[:2, :])
            valid_mask = np.logical_and(dem_ji.T >= (0, 0), dem_ji.T < dem_array.shape[::-1]).T
            valid_mask = valid_mask.all(axis=0)

            if not np.any(valid_mask):
                # ray_xyz lies entirely outside the dem (x, y) bounds - store the dem_min point
                poly_xyz[:, pi] = ray_xyz[:, 0]
            else:
                # ray_xyz lies at least partially inside the dem (x, y) bounds, but may not have an intersection. Store
                # its intersection with the dem if it exists, else store the dem_min point.
                dem_ji = dem_ji[:, valid_mask]
                ray_z = ray_z[valid_mask]
                dem_z = np.squeeze(cv2.remap(dem_array, *dem_ji.astype('float32'), dem_interp.to_cv()))

                intersection_i = np.nonzero(ray_z >= dem_z)[0]
                poly_xyz[:, pi] = (
                    ray_xyz[:, valid_mask][:, intersection_i[0] - 1]
                    if len(intersection_i) > 0 and intersection_i[0] > 0 else
                    ray_xyz[:, 0]
                )  # yapf: disable

        # check dem coverage of poly_xy
        poly_xy = poly_xyz[:2]
        dem_bounds = rio.transform.array_bounds(*dem_array.shape, dem_transform)
        poly_min = poly_xy.min(axis=1)
        poly_max = poly_xy.max(axis=1)
        if np.all(poly_min > dem_bounds[-2:]) or np.all(poly_max < dem_bounds[:2]):
            raise ValueError(f'Ortho bounds for {self._src_filename.name} lie outside the DEM.')
        elif np.any(poly_max > dem_bounds[-2:]) or np.any(poly_min < dem_bounds[:2]):
            logger.warning(f'Ortho bounds for {self._src_filename.name} are not fully covered by the DEM.')

        if crop:
            # crop dem_array to poly_xy and find corresponding transform
            dem_cnr_ji = inv_transform(dem_transform, np.array([poly_min, poly_max]).T)
            dem_ul_ji = np.floor(dem_cnr_ji.min(axis=1)).astype('int')
            dem_ul_ji = np.clip(dem_ul_ji, 0, None)
            dem_br_ji = 1 + np.ceil(dem_cnr_ji.max(axis=1)).astype('int')
            dem_br_ji = np.clip(dem_br_ji, None, dem_array.shape[::-1])
            dem_array = dem_array[dem_ul_ji[1]:dem_br_ji[1], dem_ul_ji[0]:dem_br_ji[0]]
            dem_transform = dem_transform * rio.Affine.translation(*dem_ul_ji)

        if mask:
            # mask the dem
            # (poly_ji is intersected with dem_array bounds in cv2.fillPoly)
            poly_ji = np.round(inv_transform(dem_transform, poly_xy)).astype('int')
            mask = np.ones_like(dem_array, dtype='uint8')
            mask = cv2.fillPoly(mask, [poly_ji.T], color=0)
            dem_array[mask.astype('bool', copy=False)] = np.nan

        return dem_array, dem_transform

    def _create_ortho_profile(
        self, src_im: rio.DatasetReader, shape: Tuple[int, int], transform: rio.Affine,
        dtype: Optional[str] = _default_config['dtype'], compress: Compress = _default_config['compress'],
        write_mask: bool = _default_config['write_mask']
    ) -> Tuple[Dict, bool]:
        """ Return a rasterio profile for the ortho image. """
        # Determine dtype, check dtype support
        # (OpenCV remap doesn't support int8 or uint32, and only supports int32, uint64, int64 with nearest
        # interp so these dtypes are excluded).
        dtype = dtype or src_im.profile.get('dtype', None)
        if dtype not in Ortho._nodata_vals:
            raise ValueError(f"Data type '{dtype}' is not supported.")

        # setup compression, data interleaving and photometric interpretation
        if compress == Compress.jpeg and dtype != 'uint8':
            raise ValueError(f"JPEG compression is supported for the 'uint8' data type only.")

        if compress == Compress.auto:
            compress = Compress.jpeg if dtype == 'uint8' else Compress.deflate

        if compress == Compress.jpeg:
            interleave, photometric = ('pixel', 'ycbcr') if src_im.count == 3 else ('band', 'minisblack')
        else:
            interleave, photometric = ('band', 'minisblack')

        # resolve auto write_mask (=None) to write masks for jpeg compression
        if write_mask is None:
            write_mask = True if compress == Compress.jpeg else False

        # set nodata to None when writing internal masks to force external tools to use mask, otherwise set by dtype
        nodata = None if write_mask else Ortho._nodata_vals[dtype]

        # create ortho profile
        ortho_profile = dict(
            driver='GTiff', dtype=dtype, crs=self._ortho_crs, transform=transform, width=shape[1], height=shape[0],
            count=src_im.count, tiled=True, blockxsize=Ortho._default_blocksize[0],
            blockysize=Ortho._default_blocksize[1], nodata=nodata, compress=compress.value, interleave=interleave,
            photometric=photometric,
        )

        return ortho_profile, write_mask

    def _remap_tile(
        self, ortho_im: rio.io.DatasetWriter, src_array, dem_array: np.ndarray, tile_win: Window,
        indexes: List[int], init_xgrid: np.ndarray, init_ygrid: np.ndarray, interp: Interp, full_remap: bool,
        write_mask: bool,
    ):
        """
        Thread safe method to map the source image to an ortho tile, given an open ortho dataset, source image
        array, DEM array in the ortho CRS and grid, tile window into the ortho dataset, band indexes of the source
        array, xy grids for the first ortho tile, and configuration parameters.
        """
        dtype_nodata = self._nodata_vals[ortho_im.profile['dtype']]

        # offset init grids to tile_win
        tile_transform = rio.windows.transform(tile_win, ortho_im.profile['transform'])
        tile_xgrid = (
            init_xgrid[:tile_win.height, :tile_win.width] + (tile_transform.xoff - ortho_im.profile['transform'].xoff)
        )
        tile_ygrid = (
            init_ygrid[:tile_win.height, :tile_win.width] + (tile_transform.yoff - ortho_im.profile['transform'].yoff)
        )

        # extract tile_win from dem_array (will be nan outside dem valid area or outside original dem bounds)
        tile_zgrid = dem_array[
            tile_win.row_off:(tile_win.row_off + tile_win.height),
            tile_win.col_off:(tile_win.col_off + tile_win.width)
        ]  # yapf: disable

        # find the source (j, i) pixel coords corresponding to ortho image (x, y, z) world coords
        tile_ji = self._camera.world_to_pixel(
            np.array([tile_xgrid.reshape(-1, ), tile_ygrid.reshape(-1, ), tile_zgrid.reshape(-1, )]),
            distort=full_remap
        )  # yapf: disable

        # separate tile_ji into (j, i) grids, converting to float32 for compatibility with cv2.remap
        tile_jgrid = tile_ji[0, :].reshape(tile_win.height, tile_win.width).astype('float32')
        tile_igrid = tile_ji[1, :].reshape(tile_win.height, tile_win.width).astype('float32')
        # tile_jgrid, tile_igrid = cv2.convertMaps(tile_jgrid, tile_igrid, cv2.CV_16SC2)

        # initialise ortho tile array
        tile_array = np.full(
            (src_array.shape[0], tile_win.height, tile_win.width), dtype=ortho_im.profile['dtype'],
            fill_value=dtype_nodata
        )

        # remap source image to ortho tile, looping over band(s) (cv2.remap execution time depends on array ordering)
        for oi in range(0, src_array.shape[0]):
            tile_array[oi, :, :] = cv2.remap(
                src_array[oi, :, :], tile_jgrid, tile_igrid, interp.to_cv(), borderMode=cv2.BORDER_CONSTANT,
                borderValue=dtype_nodata,
            )
            # below is the scipy equivalent to cv2.remap - it is slower but doesn't blur with nodata
            # tile_array[oi, :, :] = map_coordinates(
            #     src_array[oi, :, :], (tile_igrid, tile_jgrid), order=2, mode='constant', cval=dtype_nodata,
            #     prefilter=False
            # )

        # mask of invalid ortho pixels
        tile_mask = np.all(nan_equals(tile_array, dtype_nodata), axis=0)

        # remove cv2.remap blurring with nodata at the nodata boundary when necessary
        # @formatter:off
        if (
            interp != Interp.nearest and not np.isnan(dtype_nodata) and
            np.sum(tile_mask) > min(Ortho._default_blocksize[0], Ortho._default_blocksize[1])
        ):  # yapf: disable @formatter:on
            # TODO: to avoid these nodata boundary issues entirely, use dtype=float and
            #  nodata=nan internally, then convert to config dtype and nodata on writing.
            kernel = np.ones((5, 5), np.uint8) if interp == Interp.bilinear else np.ones((9, 9), np.uint8)
            tile_mask = cv2.dilate(tile_mask.astype(np.uint8, copy=False), kernel)
            tile_mask = tile_mask.astype(bool, copy=False)
            tile_array[:, tile_mask] = dtype_nodata

        # write tile_array to the ortho image
        with self._write_lock:
            ortho_im.write(tile_array, indexes, window=tile_win)

            if write_mask and (indexes == [1] or len(indexes) == ortho_im.count):
                ortho_im.write_mask(~tile_mask, window=tile_win)

    def _undistort(
        self, image: np.ndarray, nodata: Union[float, int] = 0, interp: Union[str, Interp] = Interp.bilinear
    ) -> np.ndarray:
        """ Undistort an image using `interp` interpolation and setting invalid pixels to `nodata`. """
        if self._camera._undistort_maps is None:
            return image

        def undistort_band(band: np.array) -> np.array:
            """ Undistort a 2D band array. """
            # equivalent without stored _undistort_maps:
            # return cv2.undistort(band_array, self._K, self._dist_param)
            return cv2.remap(
                band, *self._camera._undistort_maps, Interp[interp].to_cv(), borderMode=cv2.BORDER_CONSTANT,
                borderValue=nodata
            )

        if image.ndim > 2:
            # undistort by band so that output data stays in the rasterio ordering
            out_image = np.full(image.shape, fill_value=nodata, dtype=image.dtype)
            for bi in range(image.shape[0]):
                out_image[bi] = undistort_band(image[bi])
        else:
            out_image = undistort_band(image)

        return out_image

    def _remap(
        self, src_im: rio.DatasetReader, ortho_im: rio.io.DatasetWriter, dem_array: np.ndarray,
        interp: Interp = _default_config['interp'], per_band: bool = _default_config['per_band'],
        full_remap: bool = _default_config['full_remap'], write_mask: bool = _default_config['write_mask'],
    ):
        """
        Map the source to ortho image by interpolation, given open source and ortho datasets, DEM array in the ortho
        CRS and pixel grid, and configuration parameters.
        """
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
        # Initialise an (x, y) pixel grid for the first tile here, and offset for remaining tiles in _remap_tile
        # (requires N-up transform).
        # float64 precision is needed for the (x, y) ortho grids in world coordinates for e.g. high resolution drone
        # imagery.
        # gdal / rio geotransform origin refers to the pixel UL corner while OpenCV remap etc integer pixel coords
        # refer to pixel centers, so the (x, y) coords are offset by half a pixel to account for this.
        # j_range = np.arange(0, ortho_im.profile['width'])
        # i_range = np.arange(0, ortho_im.profile['height'])
        j_range = np.arange(0, Ortho._default_blocksize[0])
        i_range = np.arange(0, Ortho._default_blocksize[1])
        init_jgrid, init_igrid = np.meshgrid(j_range, i_range, indexing='xy')
        center_transform = ortho_im.profile['transform'] * rio.Affine.translation(0.5, 0.5)
        init_xgrid, init_ygrid = center_transform * [init_jgrid, init_igrid]

        # create list of band indexes to read, remap & write all bands at once (per_band==False), or per-band
        # (per_band==True)
        if per_band:
            index_list = [[i] for i in range(1, src_im.count + 1)]
        else:
            index_list = [[*range(1, src_im.count + 1)]]

        # create a list of ortho tile windows (assumes all bands configured to same tile shape)
        ortho_wins = [ortho_win for _, ortho_win in ortho_im.block_windows(1)]

        with tqdm(bar_format=bar_format, total=len(ortho_wins) * len(index_list), dynamic_ncols=True) as bar:
            # read, process and write bands, one row of indexes at a time
            for indexes in index_list:
                # read source image band(s) (ortho dtype is required for cv2.remap() to set invalid ortho areas to
                # ortho nodata value)
                src_array = src_im.read(indexes, out_dtype=ortho_im.profile['dtype'])

                if not full_remap:
                    # undistort the source image so the distortion model can be excluded from
                    # self._camera.world_to_pixel() in self._remap_tile()
                    dtype_nodata = self._nodata_vals[ortho_im.profile['dtype']]
                    src_array = self._undistort(src_array, nodata=dtype_nodata, interp=interp)

                # map ortho tiles concurrently
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    # ortho_win_full = Window(0, 0, ortho_im.width, ortho_im.height)
                    futures = [
                        executor.submit(
                            self._remap_tile, ortho_im, src_array, dem_array, ortho_win, indexes, init_xgrid,
                            init_ygrid, interp, full_remap, write_mask
                        )
                        for ortho_win in ortho_wins
                    ]  # yapf: disable
                    for future in as_completed(futures):
                        future.result()
                        bar.update()

    # TODO: change param names write_mask to internal_mask & full_remap to something friendlier
    def process(
        self, ortho_filename: Union[str, Path],
        resolution: Tuple[float, float] = _default_config['resolution'],
        interp: Union[str, Interp] = _default_config['interp'],
        dem_interp: Union[str, Interp] = _default_config['dem_interp'],
        per_band: bool = _default_config['per_band'],
        full_remap: bool = _default_config['full_remap'],
        write_mask: Optional[bool] = _default_config['write_mask'],
        dtype: str = _default_config['dtype'],
        compress: Union[str, Compress] = _default_config['compress'],
        build_ovw: bool = _default_config['build_ovw'],
        overwrite: bool = _default_config['overwrite'],
    ):  # yaml: disable
        """
        Orthorectify the source image based on the camera model and DEM.

        ortho_filename: str, pathlib.Path
            Path of the ortho image file to create.
        resolution: list of float, optional
            Ortho image pixel (x, y) size in units of the ortho CRS (usually meters).
        interp: str, simple_ortho.enums.Interp, optional
            Interpolation method to use for remapping the source to ortho image.  See
            :class:`~simple_ortho.enums.Interp` for options.  :attr:`~simple_ortho.enums.Interp.nearest` is
            recommended when the ortho and source image resolutions are similar.
        dem_interp: str, simple_ortho.enums.Interp, optional
            Interpolation method for reprojecting the DEM.  See :class:`~simple_ortho.enums.Interp` for options.
            :attr:`~simple_ortho.enums.Interp.cubic` is recommended when the DEM has a coarser resolution than the
            ortho.
        per_band: bool, optional
            Remap the source to the ortho image band-by-band (True), or all bands at once (False).
            False is faster but requires more memory.
        full_remap: bool, optional
            Orthorectify the source image with the full camera model (True), or the undistorted source image with a
            pinhole camera model (False).  False is faster but can reduce the extents and quality of the ortho image.
        write_mask: bool, optional
            Write an internal mask for the ortho image. This helps remove nodata noise caused by lossy compression.
            If None, the mask will be written when jpeg compression is used.
        dtype: str, optional
            Ortho image data type (`uint8`, `uint16`, `float32` or `float64`).  If set to None, the source image
            dtype is used.
        compress: str, Compress, optional
            Ortho image compression type (`deflate`, `jpeg` or `auto`).  See :class:`~simple_ortho.enums.Compress`_
            for option details.
        build_ovw: bool, optional
            Build overviews for the ortho image.
        overwrite: bool, optional
            Overwrite the ortho image if it exists.
        """
        with profiler():  # run profiler in DEBUG log level
            ortho_filename = Path(ortho_filename)
            if ortho_filename.exists():
                if not overwrite:
                    raise FileExistsError(f'Ortho file: {ortho_filename.name} exists.')
                ortho_filename.unlink()
                ortho_filename.with_suffix(ortho_filename.suffix + '.aux.xml').unlink(missing_ok=True)

            # get an auto resolution if resolution not provided
            resolution = resolution or self._get_auto_res()

            # get dem array covering ortho extents in ortho CRS and resolution
            dem_interp = Interp(dem_interp)
            dem_array, dem_transform = self._reproject_dem(dem_interp, resolution)
            dem_array, dem_transform = self._mask_dem(dem_array, dem_transform, dem_interp, full_remap=full_remap)

            env = rio.Env(
                GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False, GDAL_TIFF_INTERNAL_MASK=True,
                ALLOW_ELLIPSOIDAL_HEIGHT_AS_VERTICAL_CRS=True
            )
            with env, suppress_no_georef(), rio.open(self._src_filename, 'r') as src_im:
                # create ortho profile & set write_mask
                ortho_profile, write_mask = self._create_ortho_profile(
                    src_im, dem_array.shape, dem_transform, dtype=dtype, compress=Compress(compress),
                    write_mask=write_mask
                )

                with rio.open(ortho_filename, 'w', **ortho_profile) as ortho_im:
                    # orthorectify
                    self._remap(
                        src_im, ortho_im, dem_array, interp=Interp(interp), per_band=per_band, full_remap=full_remap,
                        write_mask=write_mask,
                    )

            if build_ovw:
                # TODO: move this under the rio multithreaded environment for gdal>=3.8
                # build overviews
                with rio.open(ortho_filename, 'r+') as ortho_im:
                    self._build_overviews(ortho_im)

##
