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
import os
import pathlib
import pstats
import sys
import tracemalloc
import time
import logging

import cv2
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, Resampling, transform, transform_bounds
from rasterio.windows import Window
from typing import Tuple, Union, Optional

from simple_ortho.camera import Camera
from simple_ortho.enums import CvInterp

# from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)


def expand_window_to_grid(win: Window, expand_pixels: Tuple[int, int] = (0, 0)) -> Window:
    """
    Expand rasterio window extents to the nearest whole numbers.
    """
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int'))
    return exp_win


def nan_equals(a: Union[np.ndarray, float], b: Union[np.ndarray, float]) -> np.ndarray:
    """ Compare two numpy objects a & b, returning true where elements of both a & b are nan. """
    return (a == b) | (np.isnan(a) & np.isnan(b))


class OrthoIm:
    # maximum number of ortho bounds iterations
    ortho_bound_max_iters = 10
    # stop iterating ortho bounds when the difference in min(DEM) estimates goes below this value (m)
    ortho_bound_stop_crit = 1
    default_config = dict(
        crs=None, dem_interp='cubic_spline', dem_band=1, interp='bilinear', resolution=[0.5, 0.5], compress='deflate',
        tile_size=[512, 512], interleave='band', photometric=None, nodata=0, per_band=False, driver='GTiff', dtype=None,
        build_ovw=True, overwrite=True, write_mask=False, full_remap=True,
    )
    # Minimum EGM96 geoid altitude i.e. minimum possible vertical difference with the WGS84 ellipsoid
    egm96_min = -106.71

    def __init__(self, src_im_filename, dem_filename, camera, config=None, ortho_im_filename=None):
        """
        Class to orthorectify image with known DEM and camera model

        Parameters
        ----------
        src_im_filename :   str, pathlib.Path
                            Filename of source image to orthorectified
        dem_filename :      str, pathlib.Path
                            Filename of DEM covering source image
        camera :            Camera
                            camera object relevant to source image
        ortho_im_filename : str, pathlib.Path
                            (optional) specify the filename of the orthorectified image to create.  If not specified,
                            appends '_ORTHO' to the src_im_filename
        config :            dict
                            (optional) dictionary of configuration parameters.  If None, sensible defaults are used.
                            Key, value pairs as follows:
                                crs: Ortho image CRS as an EPSG or WKT string.  If omitted, the ortho will be created in
                                    the source image CRS (if present).  Whether specified by this string or the source
                                    image, the ortho image CRS must be the same as the ``camera`` position CRS
                                    (see :class:`~simple_ortho.camera.Camera`).
                                dem_interp: Interpolation type for resampling DEM (average, bilinear, cubic,
                                    cubic_spline, gauss, lanczos)
                                dem_band: 1-based index of band in DEM raster to use
                                interp: Interpolation type for generating ortho-image (nearest, average, bilinear,
                                    cubic, lanczos)
                                resolution: Output pixel size [x, y] in m
                                compress: GeoTIFF compress type (deflate, jpeg, jpeg2000, lzw, zstd, none)
                                interleave: Interleave by 'pixel' or 'band' (pixel, band)
                                photometric: Photometric interpretation, see https://gdal.org/drivers/raster/gtiff.html
                                    for options (None = same format as source image, recommended)
                                tile_size: Tile/block [x, y] size in pixels  ([512, 512] recommended)
                                nodata: NODATA value
                                per_band: Remap the source raster to the ortho per-band (True), or all bands at once
                                    (False - recommended)
                                driver: Format of ortho raster - see www.gdal.org/formats_list.html (None = same format
                                    as source image)
                                dtype: Data type of ortho raster (e.g. uint8, uint16, float32 etc)  (None = same type as
                                    source image)
                                build_ovw: Build internal overviews
                                overwrite: Overwrite ortho raster if it exists
                                write_mask: True = write an internal mask band, can help remove jpeg noise in nodata
                                    area (False - recommended)
                                full_remap: bool
                                    Remap source to ortho with full camera model (True), or undistorted source to ortho
                                    with pinhole model (False).
        """
        if not os.path.exists(src_im_filename):
            raise Exception(f"Source image file {src_im_filename} does not exist")

        if not os.path.exists(dem_filename):
            raise Exception(f"DEM file {dem_filename} does not exist")

        self._src_im_filename = pathlib.Path(src_im_filename)
        self._dem_filename = pathlib.Path(dem_filename)

        self._camera = camera

        if ortho_im_filename is None:
            self._ortho_im_filename = self._src_im_filename.parent.joinpath(
                self._src_im_filename.stem + '_ORTHO' + self._src_im_filename.suffix
            )
        else:
            self._ortho_im_filename = pathlib.Path(ortho_im_filename)

        # allow for partial or no config
        _config = self.default_config.copy()
        if config is not None:
            _config.update(**config)
        self._parse_config(_config)

        self._check_rasters()

        logger.debug(f'Ortho configuration: {config}')
        logger.debug(f'DEM: {self._dem_filename.parts[-1]}')

    def _check_rasters(self):
        """
        Check that the source image is not 12 bit
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(self._src_im_filename, 'r') as src_im:
                try:
                    # check that we can read the source image
                    tmp_array = src_im.read(1, window=Window(0, 0, 1, 1))
                except Exception as ex:
                    if src_im.profile['compress'] == 'jpeg':  # assume it is a 12bit JPEG
                        raise Exception(
                            f'Could not read {self._src_im_filename.stem}\n'
                            f'    JPEG compression with NBITS==12 is not supported by conda GDAL (and others), \n'
                            f'    you probably need to recompress this file.\n'
                            f'    See the README for details.'
                        )
                    else:
                        raise ex

    def _parse_config(self, config):
        """
        Parse dict config items where necessary

        Parameters
        ----------
        config :  dict
                  e.g. dict(dem_interp='cubic_spline', ortho_interp='bilinear', resolution=[0.5, 0.5],
                          compress='deflate', tile_size=[512, 512])
        """

        for key, value in config.items():
            setattr(self, key, value)

        # TODO: can we drop the use of the source CRS entirely?
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_im_filename, 'r') as src_im:
            src_crs = src_im.crs
            self.count = src_im.profile['count']
            # copy source image values to config attributes that are not set
            attrs_to_copy = ['driver', 'dtype', 'compress', 'interleave', 'photometric']
            for attr in attrs_to_copy:
                setattr(self, attr, getattr(self, attr, None) or src_im.profile.get(attr, None))

        if not src_crs and not self.crs:
            raise ValueError(f'"crs" configuration value must be specified when the source image has no CRS.')
        try:
            # derive crs from configuration string if specified, otherwise use source image crs
            self.crs = rio.CRS.from_string(self.crs) if self.crs else src_crs
        except:
            raise ValueError(f'Unsupported "crs" configuration value: {self.crs}.')
        if (src_crs and self.crs) and (src_crs != self.crs):
            raise ValueError(f'"crs" configuration value ({self.crs}), and source image CRS ({src_crs}) are different.')
        # TODO: must crs be projected not geographic?

        try:
            self.dem_interp = Resampling[self.dem_interp]
        except:
            raise ValueError(f'Unsupported "dem_interp" configuration type: {self.dem_interp}.')

        try:
            self.interp = CvInterp[self.interp]
        except:
            raise ValueError(f'Unsupported "interp" configuration type: {self.interp}.')

        if self._ortho_im_filename.exists():
            if self.overwrite:
                logger.warning(f'Deleting existing ortho file: {self._ortho_im_filename.stem}.')
                os.remove(self._ortho_im_filename)
            else:
                raise FileExistsError(f'Ortho file {self._ortho_im_filename.stem} exists, skipping.')

    def _get_init_dem(self) -> Tuple[np.ndarray, rio.Affine]:
        """
        Get an initial DEM array and corresponding transform, in the ortho CRS and resolution. The DEM array will
        cover the ortho bounds, within the limitations of the DEM bounds.

        Returns
        -------
        array: np.ndarray
            DEM array in the ortho CRS and resolution.
        transform: rio.Affine
            A rasterio transform object for the DEM.
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._dem_filename, 'r') as dem_im:
            # TODO: can vertical datums be extracted so we know initial z_min and subsequent offset
            crs_equal = self.crs == dem_im.crs
            dem_full_win = Window(0, 0, dem_im.width, dem_im.height)
            # corner pixel coordinates of source image
            src_br = self._camera._im_size - 1
            src_ji = np.array([[0, 0], [src_br[0], 0], src_br, [0, src_br[1]]]).T

            def get_win_at_z_min(z_min: float) -> Window:
                """
                Project image corners to world coords at z=z_min, and find corresponding dem bounds & window.
                """
                world_xyz = self._camera.pixel_to_world_z(src_ji, z_min)
                # include camera position in bounds for oblique views whose FOV excludes nadir (world bounds could be
                # projected outside of dem bounds without this)
                world_xyz_T = np.column_stack((world_xyz, self._camera._T))
                world_bounds = [*np.min(world_xyz_T[:2], axis=1), *np.max(world_xyz_T[:2], axis=1)]
                dem_bounds = transform_bounds(self.crs, dem_im.crs, *world_bounds) if not crs_equal else world_bounds
                dem_win = dem_im.window(*dem_bounds)
                try:
                    dem_win = dem_full_win.intersection(dem_win)
                except rio.errors.WindowError:
                    # TODO warn on partial coverage when finding ortho polygon
                    raise ValueError(
                        f'Ortho {self._ortho_im_filename.name} lies outside DEM {self._dem_filename.name}'
                    )
                return expand_window_to_grid(dem_win)

            # read a dem window corresponding to ortho world bounds at min possible altitude
            dem_win = get_win_at_z_min(self.egm96_min)
            dem_array = dem_im.read(self.dem_band, window=dem_win)
            dem_array_win = dem_win
            # reduce the dem window to correspond to the ortho world bounds at min dem altitude, accounting for worst
            # case dem-ortho vertical datum negative offset
            dem_min = dem_array[dem_array!=dem_im.nodata].min()
            dem_win = get_win_at_z_min(dem_min if crs_equal else max(dem_min, 0) + self.egm96_min)

            # crop dem_array to dem_win and find the corresponding transform
            dem_ij_start = (dem_win.row_off - dem_array_win.row_off, dem_win.col_off - dem_array_win.col_off)
            dem_ij_stop = (dem_ij_start[0] + dem_win.height, dem_ij_start[1] + dem_win.width)
            dem_array = dem_array[dem_ij_start[0]:dem_ij_stop[0], dem_ij_start[1]:dem_ij_stop[1]]
            dem_transform = dem_im.window_transform(dem_win)

            # cast dem_array to float (for consistency with (x,y) coords in _remap_src_to_ortho) and set nodata to
            # nan (to persist masking through cv2.remap)
            dem_array = dem_array.astype('float', copy=False)
            if not np.isnan(dem_im.nodata):
                dem_array[dem_array == dem_im.nodata] = np.nan

            # return if dem is in ortho crs and resolution
            if crs_equal and np.all(self.resolution == np.round(dem_im.res, 3)):
                return dem_array, dem_transform

            # reproject dem_array to ortho crs and resolution
            dem_array, dem_transform = reproject(
                dem_array, None, src_crs=dem_im.crs, src_transform=dem_transform, src_nodata=float('nan'),
                dst_crs=self.crs, dst_resolution=self.resolution, resampling=self.dem_interp,
                dst_nodata=float('nan'), init_dest_nodata=True, apply_vertical_shift=True,
                num_threads=multiprocessing.cpu_count()
            )
        return dem_array.squeeze(), dem_transform

    def _remap_src_to_ortho(self, ortho_profile, dem_array):
        """
        Interpolate the ortho image from the source image.
            self.per_band = True: Read/write one band at a time - memory efficient
            self.per_band = False: Read/write all bands at once - processor efficient (recommended)

        Parameters
        ----------
        ortho_profile : dict
                        rasterio profile for ortho image
        dem_array     : numpy.array
                        array of altitude values on corresponding to ortho image i.e. on the same grid
        """

        # Initialise tile grid here once off (save cpu) - to offset later (requires N-up geotransform).
        # Note that numpy is left to its default float64 precision for building the xy ortho grids in geo-referenced
        # co-ordinates.  This precision is needed for e.g. high resolution drone imagery.
        # j_range = np.arange(0, ortho_profile['width'])
        # i_range = np.arange(0, ortho_profile['height'])
        j_range = np.arange(0, self.tile_size[0])
        i_range = np.arange(0, self.tile_size[1])
        jgrid, igrid = np.meshgrid(j_range, i_range, indexing='xy')
        xgrid, ygrid = ortho_profile['transform'] * [jgrid, igrid]

        time_ttl = dict(undistort=0, world_to_pixel=0, remap=0)
        block_count = 0
        with rio.open(self._src_im_filename, 'r') as src_im:
            # src_ji = np.array([[0, 0], [src_im.width, 0], [src_im.width, src_im.height], [0, src_im.height]]).T
            # self._camera.pixel_to_world_dem(src_ji, dem_array, ortho_profile['transform'])

            with rio.open(self._ortho_im_filename, 'w', **ortho_profile) as ortho_im:

                if self.per_band:
                    bands = np.array([range(1, src_im.count + 1)]).T  # RW one row of bands i.e. one band at a time
                else:
                    bands = np.array([range(1, src_im.count + 1)])  # RW one row of bands i.e. all bands at once

                ttl_blocks = (
                    np.ceil(ortho_profile['width'] / ortho_profile['blockxsize']) *
                    np.ceil(ortho_profile['height'] / ortho_profile['blockysize'] * bands.shape[0])
                )

                # TODO: if we want to allow for src nodata, then read masks here, and set src_im_array[mask] = nan below
                # TODO: add tests for different ortho dtype and nodata values

                for bi in bands.tolist():
                    # Read source image band(s).
                    # For cv2.remap() to set invalid ortho areas to self.nodata, the src_im_array dtype must be
                    # able to represent self.nodata.
                    src_im_array = src_im.read(bi, out_dtype=self.dtype)
                    if not self.full_remap:
                        # Undistort the source image so we can exclude the distortion model from the call to
                        # Camera.world_to_pixel() that builds the ortho maps below.
                        s = time.time()
                        src_im_array = self._camera.undistort(src_im_array, nodata=self.nodata, interp=self.interp)
                        time_ttl['undistort'] += time.time() - s

                    # ortho_win_full = Window(0, 0, ortho_im.width, ortho_im.height)
                    # for ortho_win in [ortho_win_full]:
                    for ji, ortho_win in ortho_im.block_windows(1):

                        # offset tile grids to ortho_win
                        ortho_win_transform = rio.windows.transform(ortho_win, ortho_im.transform)
                        ortho_xgrid = (
                            xgrid[:ortho_win.height, :ortho_win.width] + (ortho_win_transform.xoff -
                            ortho_im.transform.xoff)
                        )
                        ortho_ygrid = (
                            ygrid[:ortho_win.height, :ortho_win.width] + (ortho_win_transform.yoff -
                            ortho_im.transform.yoff)
                        )

                        # extract ortho_win from dem_array, will be nan outside dem bounds or in dem nodata
                        ortho_zgrid = dem_array[
                            ortho_win.row_off:(ortho_win.row_off + ortho_win.height),
                            ortho_win.col_off:(ortho_win.col_off + ortho_win.width)
                        ]

                        # find the 2D source image pixel co-ords corresponding to ortho image 3D co-ords
                        s = time.time()
                        src_ji = self._camera.world_to_pixel(
                            np.array([ortho_xgrid.reshape(-1, ), ortho_ygrid.reshape(-1, ), ortho_zgrid.reshape(-1, )]),
                            distort=self.full_remap
                        )
                        time_ttl['world_to_pixel'] += time.time() - s
                        # now that co-rds are in pixel units, their value range is much smaller than world co-ordinates
                        # and they can be converted to float32 for compatibility with cv2.remap
                        src_jj = src_ji[0, :].reshape(ortho_win.height, ortho_win.width).astype('float32')
                        src_ii = src_ji[1, :].reshape(ortho_win.height, ortho_win.width).astype('float32')
                        # src_jj, src_ii = cv2.convertMaps(src_jj, src_ii, cv2.CV_16SC2)

                        # Interpolate the ortho tile from the source image based on warped/unprojected grids.
                        ortho_im_win_array = np.full(
                            (src_im_array.shape[0], ortho_win.height, ortho_win.width), fill_value=self.nodata,
                            dtype=self.dtype
                        )
                        # TODO: check how this works with source nodata.  generally the source image will be rectangular
                        #  with full coverage, but this may not always be the case.  maybe filling src nodata with nan
                        #  would work
                        s = time.time()
                        for oi in range(0, src_im_array.shape[0]):  # for per_band=True, this will loop once only
                            # Ortho pixels outside dem / src bounds or in dem nodata, will be set to
                            # borderValue=self.nodata.
                            # Note that cv2.remap() execution time is sensitive to array ordering.
                            ortho_im_win_array[oi, :, :] = cv2.remap(
                                src_im_array[oi, :, :], src_jj, src_ii, self.interp.value,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=self.nodata,
                            )
                            # below is the scipy equivalent to cv2.remap.  it is ~3x slower but doesn't blur with nodata
                            # ortho_im_win_array[oi, :, :] = map_coordinates(
                            #     src_im_array[oi, :, :], (src_ii, src_jj), order=2, mode='constant', cval=self.nodata,
                            #     prefilter=False
                            # )

                        time_ttl['remap'] += time.time() - s

                        # Remove blurring from cv2.remap interpolation with nodata at the boundary.  This only occurs
                        # if not using nearest interpolation or if nodata is not nan.
                        nodata_mask = np.all(nan_equals(ortho_im_win_array, self.nodata), axis=0)
                        if (
                            self.interp != CvInterp.nearest and not np.isnan(self.nodata) and
                            np.sum(nodata_mask) > np.min(self.tile_size)
                        ):
                            # TODO: to avoid these nodata boundary issues entirely, I could use dtype=float and
                            #  nodata=nan internally, then convert to config dtype and nodata on writing.
                            # create dilation kernel slightly larger than interpolation kernel size
                            if self.interp == CvInterp.bilinear:
                                kernel = np.ones((5, 5), np.uint8)
                            else:
                                kernel = np.ones((9, 9), np.uint8)
                            nodata_mask_d = cv2.dilate(nodata_mask.astype(np.uint8, copy=False), kernel)
                            nodata_mask_d = nodata_mask_d.astype(bool, copy=False)
                            ortho_im_win_array[:, nodata_mask_d] = self.nodata
                        else:
                            nodata_mask_d = nodata_mask

                        # write out the ortho tile to disk
                        ortho_im.write(ortho_im_win_array, bi, window=ortho_win)

                        if self.write_mask and np.all(bi == bands[0]):  # write mask once for all bands
                            with np.testing.suppress_warnings() as sup:
                                sup.filter(DeprecationWarning, "")  # suppress the np.bool warning as it is buggy
                                ortho_im.write_mask(~nodata_mask_d, window=ortho_win)

                        # print progress
                        block_count += 1
                        progress = (block_count / ttl_blocks)
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-50s] %d%%" % ('=' * int(50 * progress), 100 * progress))
                        sys.stdout.flush()

            sys.stdout.write('\n')
            logger.debug('Processing times:')
            for k, v in time_ttl.items():
                logger.debug(f'\t{k}: {v:.5f} s')

    def build_ortho_overviews(self, max_num_levels: int = 8, min_level_pixels: int = 256):
        """
        Builds internal overviews for an existing ortho-image
        """
        if self.build_ovw and self._ortho_im_filename.exists():  # build internal overviews
            with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
                with rio.open(self._ortho_im_filename, 'r+', num_threads='all_cpus') as ortho_im:
                    max_ovw_levels = int(np.min(np.log2(ortho_im.shape)))
                    min_level_shape_pow2 = int(np.log2(min_level_pixels))
                    num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
                    ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
                    ortho_im.build_overviews(ovw_levels, Resampling.average)

    def orthorectify(self):
        """
        Orthorectify the source image based on specified camera model and DEM.
        """
        # init profiling
        if logger.level == logging.DEBUG:
            tracemalloc.start()
            proc_profile = cProfile.Profile()
            proc_profile.enable()

        dem_array, dem_transform = self._get_init_dem()
        attrs_to_copy = ['crs', 'nodata', 'driver', 'dtype', 'compress', 'interleave', 'photometric', 'count']
        ortho_profile = {attr: getattr(self, attr) for attr in attrs_to_copy}
        ortho_profile.update(
            tiled=True, blockxsize=self.tile_size[0], blockysize=self.tile_size[1], transform=dem_transform,
            width=dem_array.shape[1], height=dem_array.shape[0]
        )
        # work around an apparent gdal issue with writing masks, building overviews and non-jpeg compression
        if self.write_mask and ortho_profile['compress'] != 'jpeg':
            self.write_mask = False
            logger.warning('Setting write_mask=False, write_mask=True should only be used with compress=jpeg')

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GDAL_TIFF_INTERNAL_MASK=True):
            self._remap_src_to_ortho(ortho_profile, dem_array)

        if logger.level == logging.DEBUG:  # print profiling info
            proc_profile.disable()
            # tottime is the total time spent in the function alone. cumtime is the total time spent in the function
            # plus all functions that this function called
            proc_stats = pstats.Stats(proc_profile).sort_stats('cumtime')
            logger.debug(f'Processing time:')
            proc_stats.print_stats(20)

            current, peak = tracemalloc.get_traced_memory()
            logger.debug(f"Memory usage: current: {current / 10 ** 6:.1f} MB, peak: {peak / 10 ** 6:.1f} MB")


##
