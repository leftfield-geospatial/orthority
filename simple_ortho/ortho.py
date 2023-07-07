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

import cProfile
import logging
import multiprocessing
import pstats
import sys
import tracemalloc
from pathlib import Path
from typing import Tuple, Union, Dict

import cv2
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window

from simple_ortho.camera import Camera
from simple_ortho.enums import CvInterp
from simple_ortho.utils import suppress_no_georef, expand_window_to_grid, nan_equals

# from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)


class Ortho:
    # default configuration values for Ortho.process()
    _default_config = dict(
        dem_interp='cubic_spline', dem_band=1, interp='bilinear', resolution=[0.5, 0.5], per_band=False, build_ovw=True,
        overwrite=True, write_mask=False, full_remap=True,
    )
    # TODO: check that common source file properties for None profile defaults are compatible with other (set) defaults
    # default ortho profile values for Ortho._create_ortho_profile()
    _default_profile = dict(
        driver='GTiff', dtype=None, nodata=0, blockxsize=512, blockysize=512, compress=None, interleave=None,
        photometric=None,
    )
    # Minimum EGM96 geoid altitude i.e. minimum possible vertical difference with the WGS84 ellipsoid
    egm96_min = -106.71

    def __init__(
        self, src_filename: Union[str, Path], dem_filename: Union[str, Path], camera: Camera,
        crs: Union[str, rio.CRS] = None, dem_band: int = 1
    ):
        """
        Class to orthorectify an image with a specified DEM and camera model.

        Parameters
        ----------
        src_filename: str, pathlib.Path
            Path to a source image to be orthorectified.
        dem_filename: str, pathlib.Path
            Path to a DEM image covering the source image.
        camera: Camera
            Source image camera model (see :meth:`~simple_ortho.camera.create_camera`).
        crs: str, rasterio.CRS, optional
            CRS of the ortho image and ``camera`` world coordinates as an EPSG, proj4 or WKT string.  Can be omitted if
            the source image is projected in the ortho CRS.
        dem_band: int, optional
            DEM image band index to use (1-based).
        """
        if not Path(src_filename).exists():
            raise FileNotFoundError(f'Source image file {src_filename} does not exist')
        if not Path(dem_filename).exists():
            raise FileNotFoundError(f'DEM image file {dem_filename} does not exist')
        if not isinstance(camera, Camera):
            raise TypeError('`camera` is not a Camera instance.')

        self._src_filename = Path(src_filename)
        self._camera = camera

        self._ortho_crs = self._parse_crs(crs)
        self._dem_array, self._dem_transform, self._dem_crs, self._crs_equal = self._get_init_dem(
            Path(dem_filename), dem_band
        )

    def _parse_crs(self, crs: Union[str, rio.CRS]) -> rio.CRS:
        """ Derive an ortho CRS from the ``crs`` parameter and source image. """
        if crs:
            try:
                ortho_crs = rio.CRS.from_string(crs) if isinstance(crs, str) else crs
            except rio.errors.CRSError as ex:
                raise ValueError(f'`crs` not supported: {crs}.\n{ex}')
        else:
            with suppress_no_georef(), rio.open(self._src_filename, 'r') as src_im:
                if src_im.crs:
                    ortho_crs = src_im.crs
                else:
                    raise ValueError(f'`crs` should be specified when the source image has no projection.')
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
                raise ValueError(f'`dem_band`: {dem_band} is invalid for {dem_filename.name} with {dem_im.count} bands')
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
            dem_win = get_win_at_z_min(self.egm96_min)
            dem_array = dem_im.read(dem_band, window=dem_win)
            dem_array_win = dem_win

            # reduce the dem window to correspond to the ortho world bounds at min dem altitude, accounting for worst
            # case dem-ortho vertical datum offset
            # TODO: what if there is an internal mask and not nodata
            dem_min = dem_array[dem_array != dem_im.nodata].min()
            dem_win = get_win_at_z_min(dem_min if crs_equal else max(dem_min, 0) + self.egm96_min)

            # crop dem_array to the dem window and find the corresponding transform
            dem_ij_start = (dem_win.row_off - dem_array_win.row_off, dem_win.col_off - dem_array_win.col_off)
            dem_ij_stop = (dem_ij_start[0] + dem_win.height, dem_ij_start[1] + dem_win.width)
            dem_array = dem_array[dem_ij_start[0]:dem_ij_stop[0], dem_ij_start[1]:dem_ij_stop[1]]
            dem_transform = dem_im.window_transform(dem_win)

            # Cast dem_array to float32 and set nodata to nan (to persist masking through cv2.remap)
            dem_array = dem_array.astype('float32', copy=False)
            if not np.isnan(dem_im.nodata):
                dem_array[dem_array == dem_im.nodata] = np.nan

            return dem_array, dem_transform, dem_im.crs, crs_equal

    def _reproject_dem(self, dem_interp: Resampling, resolution: Tuple[float, float]) -> Tuple[np.ndarray, rio.Affine]:
        """
        Reproject self._dem_array to the ortho CRS and resolution, given reprojection interpolation and resolution
        parameters. Returns the reprojected DEM array and corresponding transform.
        """
        # return if dem in ortho crs and resolution
        dem_res = np.abs((self._dem_transform[0], self._dem_transform[4]))
        if self._crs_equal and np.all(resolution == np.round(dem_res, 3)):
            return self._dem_array, self._dem_transform

        # reproject dem_array to ortho crs and resolution
        dem_array, dem_transform = reproject(
            self._dem_array, None, src_crs=self._dem_crs, src_transform=self._dem_transform, src_nodata=float('nan'),
            dst_crs=self._ortho_crs, dst_resolution=resolution, resampling=dem_interp, dst_nodata=float('nan'),
            init_dest_nodata=True, apply_vertical_shift=True, num_threads=multiprocessing.cpu_count()
        )
        return dem_array.squeeze(), dem_transform

    def _get_ortho_poly(self, dem_array: np.ndarray, dem_transform: rio.Affine, num_pts=400) -> np.ndarray:
        """
        Return a polygon approximating the ortho boundaries in world (x, y, z) coordinates given a DEM array and
        corresponding transform in the ortho CRS and resolution.
        """
        # pixel coordinates of source image borders
        n = int(num_pts / 4)
        im_br = self._camera._im_size - 1
        side_seq = np.linspace(0, 1, n)
        src_ji = np.vstack(
            (np.hstack((side_seq, np.ones(n), side_seq[::-1], np.zeros(n))) * im_br[0],
             np.hstack((np.zeros(n), side_seq, np.ones(n), side_seq[::-1])) * im_br[1],)
        )  # yapf: disable

        # find dem (x, y, z) world coordinate intersections for each (j, i) pixel coordinate in src_ji
        dem_min = np.nanmin(dem_array)
        dem_max = np.nanmax(dem_array)
        xyz = np.zeros((3, src_ji.shape[1]))
        for pi in range(src_ji.shape[1]):
            src_pt = src_ji[:, pi].reshape(-1, 1)

            # create world points along the src_pt ray with (x, y) stepsize <= dem resolution
            start_xyz = self._camera.pixel_to_world_z(src_pt, dem_min)
            stop_xyz = self._camera.pixel_to_world_z(src_pt, dem_max)
            ray_steps = np.abs((stop_xyz - start_xyz)[:2].squeeze() / (dem_transform[0], dem_transform[4]))
            ray_steps = np.ceil(ray_steps.max()).astype('int') + 1
            ray_z = np.linspace(dem_min, dem_max, ray_steps)
            ray_xyz = self._camera.pixel_to_world_z(src_pt, ray_z)

            # find the dem pixel coords, and validity mask for the (x, y) points in ray_xyz
            dem_ji = np.round(~dem_transform * ray_xyz[:2, :]).astype('int')
            mask = np.logical_and(dem_ji.T >= (0, 0), dem_ji.T < dem_array.shape[::-1]).T
            mask = mask.all(axis=0)

            if not np.any(mask):
                # ray_xyz lies entirely outside the dem (x, y) bounds - store the dem_min point
                xyz[:, pi] = ray_xyz[:, 0]
            else:
                # ray_xyz lies at least partially inside the dem (x, y) bounds, but may not have an intersection. Store
                # its intersection with the dem if it exists, else store the dem_min point.
                dem_ji = dem_ji[:, mask]
                ray_z = ray_z[mask]
                dem_z = dem_array[dem_ji[1], dem_ji[0]]
                intersection_i = np.nonzero(ray_z >= dem_z)[0]
                xyz[:, pi] = (
                    ray_xyz[:, mask][:, intersection_i[0] - 1]
                    if len(intersection_i) > 0 and intersection_i[0] > 0 else
                    ray_xyz[:, 0]
                )  # yapf: disable

        return xyz

    def _poly_mask_dem(
        self, dem_array: np.ndarray, dem_transform: rio.Affine, poly_xy: np.ndarray, crop=True, mask=True
    ):
        """
        Return a cropped and masked DEM array and corresponding transform, given an array of polygon (x, y) world
        coordinates to mask.
        """
        # check poly_xy lies partially or fully inside the dem bounds
        dem_bounds = rio.transform.array_bounds(*dem_array.shape, dem_transform)
        poly_min = poly_xy.min(axis=1)
        poly_max = poly_xy.max(axis=1)
        if np.all(poly_min > dem_bounds[-2:]) or np.all(poly_max < dem_bounds[:2]):
            raise ValueError(f'Ortho bounds for {self._src_filename.name} lie outside the DEM.')
        elif np.any(poly_max > dem_bounds[-2:]) or np.any(poly_min < dem_bounds[:2]):
            logger.warning(f'Ortho bounds for {self._src_filename.name} are not fully covered by the DEM.')

        # clip to dem bounds
        poly_xy = np.clip(poly_xy.T, dem_bounds[:2], dem_bounds[-2:]).T

        if crop:
            # crop dem_array to poly_xy and find corresponding transform
            dem_cnr_ji = np.array([~dem_transform * poly_xy.min(axis=1), ~dem_transform * poly_xy.max(axis=1)]).T
            dem_ul_ji = np.floor(dem_cnr_ji.min(axis=1)).astype('int')
            dem_br_ji = np.ceil(dem_cnr_ji.max(axis=1)).astype('int')
            dem_array = dem_array[dem_ul_ji[1]:dem_br_ji[1], dem_ul_ji[0]:dem_br_ji[0]]
            dem_transform = dem_transform * rio.Affine.translation(*dem_ul_ji)

        if mask:
            # mask the dem
            poly_ji = np.round(~dem_transform * poly_xy).astype('int')
            mask = np.ones_like(dem_array, dtype='uint8')
            mask = cv2.fillPoly(mask, [poly_ji.T], color=0)
            dem_array[mask.astype('bool', copy=False)] = np.nan

        return dem_array, dem_transform

    def _create_ortho_profile(
        self, shape: Tuple[int, int], transform: rio.Affine, driver: str = _default_profile['driver'],
        dtype: str = _default_profile['dtype'], nodata: str = _default_profile['nodata'],
        blockxsize: int = _default_profile['blockxsize'], blockysize: int = _default_profile['blockysize'],
        compress: str = _default_profile['compress'], interleave: str = _default_profile['interleave'],
        photometric: str = _default_profile['photometric'],
    ) -> Dict:
        """ Return a rasterio profile for the ortho image. """
        # create an initial profile from the given arguments
        ortho_profile = dict(
            driver=driver, dtype=dtype, nodata=nodata, blockxsize=blockxsize, blockysize=blockysize, compress=compress,
            interleave=interleave, photometric=photometric,
        )

        with suppress_no_georef(), rio.open(self._src_filename, 'r') as src_im:
            # overwrite None values with source image values
            ortho_profile = {k: src_im.profile.get(k, None) if v is None else v for k, v in ortho_profile.items()}
            # TODO: test nodata is not None, or see if an internal mask can be used instead

            # add remaining items
            ortho_profile['crs'] = self._ortho_crs
            ortho_profile['transform'] = transform
            ortho_profile['width'] = shape[1]
            ortho_profile['height'] = shape[0]
            ortho_profile['count'] = src_im.count
            ortho_profile['tiled'] = True

        return ortho_profile

    def _remap_src_to_ortho(
        self, ortho_filename: Path, ortho_profile: Dict, dem_array: np.ndarray,
        per_band: bool = _default_config['per_band'], full_remap: bool = _default_config['full_remap'],
        interp: CvInterp = _default_config['interp'], write_mask: bool = _default_config['write_mask'],
    ):
        """
        Interpolate the ortho image from the source image, given an ortho filename & profile, and DEM array in the ortho
        CRS and pixel grid.
        """
        # Initialise an (x, y) pixel grid for the first tile here, and offset for remaining tiles later (requires N-up
        # transform). float64 precision is needed for the (x, y) ortho grids in world co-ordinates for e.g. high
        # resolution drone imagery.  The smaller range z grid is stored in float32 to save memory.
        # j_range = np.arange(0, ortho_profile['width'])
        # i_range = np.arange(0, ortho_profile['height'])
        j_range = np.arange(0, ortho_profile['blockxsize'])
        i_range = np.arange(0, ortho_profile['blockysize'])
        jgrid, igrid = np.meshgrid(j_range, i_range, indexing='xy')
        xgrid, ygrid = ortho_profile['transform'] * [jgrid, igrid]

        block_count = 0
        env = rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False, GDAL_TIFF_INTERNAL_MASK=True)
        with env, suppress_no_georef(), rio.open(self._src_filename, 'r') as src_im:
            with rio.open(ortho_filename, 'w', **ortho_profile) as ortho_im:
                # create bands as (count x 1) matrix if per_bands==True else as (1 x count) matrix
                bands = np.array([range(1, src_im.count + 1)]).T if per_band else np.array([range(1, src_im.count + 1)])

                ttl_blocks = (
                    np.ceil(ortho_profile['width'] / ortho_profile['blockxsize']) *
                    np.ceil(ortho_profile['height'] / ortho_profile['blockysize'] * bands.shape[0])
                )

                # TODO: add tests for different ortho dtype and nodata values
                # read, process and write one row of `bands` at a time
                for bi in bands.tolist():
                    # read source image band(s) as ortho dtype (required for cv2.remap() to set invalid ortho areas to
                    # ortho nodata value)
                    src_im_array = src_im.read(bi, out_dtype=ortho_profile['dtype'])

                    if not full_remap:
                        # undistort the source image so the distortion model can be excluded from
                        # self._camera.world_to_pixel() below
                        src_im_array = self._camera.undistort(
                            src_im_array, nodata=ortho_profile['nodata'], interp=interp
                        )

                    # ortho_win_full = Window(0, 0, ortho_im.width, ortho_im.height)
                    # for ortho_win in [ortho_win_full]:
                    for ji, ortho_win in ortho_im.block_windows(1):

                        # offset tile grids to ortho_win
                        ortho_win_transform = rio.windows.transform(ortho_win, ortho_im.transform)
                        ortho_xgrid = (
                            xgrid[:ortho_win.height, :ortho_win.width] +
                            (ortho_win_transform.xoff - ortho_im.transform.xoff)
                        )
                        ortho_ygrid = (
                            ygrid[:ortho_win.height, :ortho_win.width] +
                            (ortho_win_transform.yoff - ortho_im.transform.yoff)
                        )

                        # extract ortho_win from dem_array (will be nan outside dem valid area or outside original dem
                        # bounds)
                        ortho_zgrid = dem_array[
                              ortho_win.row_off:(ortho_win.row_off + ortho_win.height),
                              ortho_win.col_off:(ortho_win.col_off + ortho_win.width)
                        ]  # yapf: disable

                        # find the source (j, i) pixel coords corresponding to ortho image (x,y,z) world coords
                        src_ji = self._camera.world_to_pixel(
                            np.array([ortho_xgrid.reshape(-1, ), ortho_ygrid.reshape(-1, ), ortho_zgrid.reshape(-1, )]),
                            distort=full_remap
                        )  # yapf: disable

                        # separate src_ji into (j, i) grids, converting to float32 for compatibility with cv2.remap
                        src_jj = src_ji[0, :].reshape(ortho_win.height, ortho_win.width).astype('float32')
                        src_ii = src_ji[1, :].reshape(ortho_win.height, ortho_win.width).astype('float32')
                        # src_jj, src_ii = cv2.convertMaps(src_jj, src_ii, cv2.CV_16SC2)

                        # interpolate the ortho tile from the source image
                        ortho_im_win_array = np.full(
                            (src_im_array.shape[0], ortho_win.height, ortho_win.width),
                            fill_value=ortho_profile['nodata'], dtype=ortho_profile['dtype']
                        )
                        # loop over band(s) (cv2.remap execution time depends on array ordering)
                        for oi in range(0, src_im_array.shape[0]):
                            # remap setting pixels outside dem or source bounds to nodata
                            ortho_im_win_array[oi, :, :] = cv2.remap(
                                src_im_array[oi, :, :], src_jj, src_ii, interp.value, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=ortho_profile['nodata'],
                            )
                            # below is the scipy equivalent to cv2.remap - it is slower but doesn't blur with
                            # nodata
                            # ortho_im_win_array[oi, :, :] = map_coordinates(  #
                            # src_im_array[oi, :, :], (
                            #   src_ii, src_jj), order=2, mode='constant', cval=self.nodata,  #     prefilter=False  #
                            # )

                        # remove cv2.remap blurring with nodata at the boundary (occurs when interp is not nearest or
                        # nodata is not nan)
                        nodata_mask = np.all(nan_equals(ortho_im_win_array, ortho_profile['nodata']), axis=0)
                        if (
                            interp != CvInterp.nearest and not np.isnan(ortho_profile['nodata'])
                            and np.sum(nodata_mask) > min(ortho_profile['blockxsize'], ortho_profile['blockysize'])
                        ):  # yapf: disable
                            # TODO: to avoid these nodata boundary issues entirely, use dtype=float and
                            #  nodata=nan internally, then convert to config dtype and nodata on writing.
                            # create dilation kernel slightly larger than interpolation kernel size
                            if interp == CvInterp.bilinear:
                                kernel = np.ones((5, 5), np.uint8)
                            else:
                                kernel = np.ones((9, 9), np.uint8)
                            nodata_mask_d = cv2.dilate(nodata_mask.astype(np.uint8, copy=False), kernel)
                            nodata_mask_d = nodata_mask_d.astype(bool, copy=False)
                            ortho_im_win_array[:, nodata_mask_d] = ortho_profile['nodata']
                        else:
                            nodata_mask_d = nodata_mask

                        # write the ortho tile to disk
                        ortho_im.write(ortho_im_win_array, bi, window=ortho_win)

                        if write_mask and np.all(bi == bands[0]):
                            # write mask once for all bands, suppressing np.bool warning
                            with np.testing.suppress_warnings() as sup:
                                sup.filter(DeprecationWarning, '')
                                ortho_im.write_mask(~nodata_mask_d, window=ortho_win)

                        # print progress
                        block_count += 1
                        progress = (block_count / ttl_blocks)
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-50s] %d%%" % ('=' * int(50 * progress), 100 * progress))
                        sys.stdout.flush()

            sys.stdout.write('\n')

    @staticmethod
    def _build_overviews(filename, max_num_levels: int = 8, min_level_pixels: int = 256):
        """ Build internal overviews for given image. """
        if not filename.exists():
            return
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(filename, 'r+') as im:
                max_ovw_levels = int(np.min(np.log2(im.shape)))
                min_level_shape_pow2 = int(np.log2(min_level_pixels))
                num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
                ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
                im.build_overviews(ovw_levels, Resampling.average)

    # TODO: simplify/merge interpolation enums
    # TODO: change param names write_mask to internal_mask, full_remap to something more sensible
    #
    def process(
        self, ortho_filename: Union[str, Path], resolution: Tuple[float, float],
        dem_interp: Union[str, Resampling] = _default_config['dem_interp'],
        interp: Union[str, CvInterp] = _default_config['interp'], per_band: bool = _default_config['per_band'],
        build_ovw: bool = _default_config['build_ovw'], overwrite: bool = _default_config['overwrite'],
        write_mask: bool = _default_config['write_mask'], full_remap: bool = _default_config['full_remap'], **kwargs
    ):  # yaml: disable
        """
        Orthorectify the source image based on the specified camera model and DEM.

        ortho_filename: str, pathlib.Path
            Name of the orthorectified file to create.
        resolution: list of float
            Output pixel (x, y) size in m.
        dem_interp: str, rasterio.enums.Resampling, optional
            Interpolation method for reprojecting the DEM.  See :class:`~rasterio.enums.Resampling` for options.
        interp: str, simple_ortho.enums.CvInterp, optional
            Interpolation method to use for warping source to orthorectified image.  See
            :class:`~simple_ortho.enums.CvInterp` for options.  :class:`~simple_ortho.enums.CvInterp.nearest` is
            recommended when the ortho and source image resolutions are similar.
        per_band: bool, optional
            Remap the source to the ortho image band-by-band (True), or all bands at once (False).
            False is typically faster but requires more memory.
        build_ovw: bool, optional
            Build overviews for the ortho image.
        overwrite: bool, optional
            Overwrite the ortho image if it exists.
        write_mask: bool, optional
            Write an internal mask band for the ortho image. Can help remove noise in nodata area with lossy (e.g.
            jpeg) compression.
        full_remap: bool, optional
            Remap the source to ortho image with full camera model (True), or remap the undistorted source to ortho
            image with a pinhole camera model (False).  False is faster but creates a an with reduced extent and
            quality.
        driver: str, optional
            Ortho image file format - see www.gdal.org/formats_list.html for options.  `GTiff` is recommended.  If
            set to None, the source image driver is used.
        dtype: str, optional
            Ortho image data type (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `float32` or
           `float64`).  If set to None, the source image dtype is used.
        nodata: int, float, optional
            Ortho image nodata value.  If set to None, the source image nodata value is used.
        blockxsize: int, optional
            Ortho image block width in pixels.  If set to None, the source image block width is used.
        blockysize: int, optional
            Ortho image block height in pixels.  If set to None, the source image block height is used.
        compress: str, optional
            Ortho image compression type (`deflate`, `jpeg`, `lzw`, `zstd`, or `none`).  If set to None,
            the source image compression type is used.
        interleave: str, optional
            Ortho image interleaving (`band` or `pixel`).   If set to None, the source image interleaving is used.
        photometric: str, optional
            Ortho image photometric interpretation (`minisblack`, `rgb` or `ycbcr`).  If set to None, the source image
            value is used.
        """

        # init profiling
        if logger.getEffectiveLevel() <= logging.DEBUG:
            tracemalloc.start()
            proc_profile = cProfile.Profile()
            proc_profile.enable()

        ortho_filename = Path(ortho_filename)
        if not overwrite and ortho_filename.exists():
            raise FileExistsError(
                f'Ortho file: {ortho_filename.name} exists and won\'t be overwritten without the `overwrite` option.'
            )

        # get dem array covering ortho extents in ortho CRS and resolution
        dem_array, dem_transform = self._reproject_dem(Resampling[dem_interp], resolution)
        poly_xyz = self._get_ortho_poly(dem_array, dem_transform)
        dem_array, dem_transform = self._poly_mask_dem(dem_array, dem_transform, poly_xyz[:2])

        # create ortho profile
        ortho_profile = self._create_ortho_profile(dem_array.shape, dem_transform, **kwargs)

        # work around an apparent gdal issue with writing masks, building overviews and non-jpeg compression
        if write_mask and ortho_profile['compress'] != 'jpeg':
            write_mask = False
            logger.warning('Setting `write_mask=False`, `write_mask=True` should only be used with `compress=\'jpeg\'`')

        # orthorectify
        self._remap_src_to_ortho(
            ortho_filename, ortho_profile, dem_array, per_band=per_band, full_remap=full_remap, interp=CvInterp[interp],
            write_mask=write_mask,
        )

        if build_ovw:
            # build overviews
            self._build_overviews(ortho_filename)

        if logger.getEffectiveLevel() <= logging.DEBUG:  # print profiling info
            proc_profile.disable()
            # tottime is the total time spent in the function alone. cumtime is the total time spent in the function
            # plus all functions that this function called
            proc_stats = pstats.Stats(proc_profile).sort_stats('cumtime')
            logger.debug(f'Processing time:')
            proc_stats.print_stats(20)

            current, peak = tracemalloc.get_traced_memory()
            logger.debug(f"Memory usage: current: {current / 10 ** 6:.1f} MB, peak: {peak / 10 ** 6:.1f} MB")


##
