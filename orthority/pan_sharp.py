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
"""Pan sharpening."""
from __future__ import annotations

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import product
from os import cpu_count, PathLike
from threading import Lock
from typing import Generator, Sequence

import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import intersect, Window
from tqdm.std import tqdm

from orthority import utils
from orthority.enums import Compress, Interp
from orthority.errors import OrthorityWarning
from orthority.ortho import Ortho

logger = logging.getLogger(__name__)


class PanSharpen:
    _working_dtype = 'float32'
    _working_nodata = float('nan')

    def __init__(
        self,
        pan_file: str | PathLike | OpenFile | rio.DatasetReader,
        ms_file: str | PathLike | OpenFile | rio.DatasetReader,
    ):
        self._pan_file = pan_file
        self._ms_file = ms_file
        self._pan_profile, self._ms_profile = self._get_vrt_profiles(pan_file, ms_file)
        self._pan_lock = Lock()
        self._ms_lock = Lock()
        self._out_lock = Lock()

    @staticmethod
    def _block_windows(
        im_shape: tuple[int, int], block_shape: tuple[int, int] = (1024, 1024)
    ) -> Generator[Window]:
        """Block window generator for the given image and optional block shapes."""
        xrange = range(0, im_shape[1], block_shape[1])
        yrange = range(0, im_shape[0], block_shape[0])
        for xstart, ystart in product(xrange, yrange):
            xstop = min(xstart + block_shape[1], im_shape[1])
            ystop = min(ystart + block_shape[0], im_shape[0])
            yield Window(xstart, ystart, xstop - xstart, ystop - ystart)

    def _get_vrt_profiles(self, pan_file, ms_file) -> tuple[dict, dict]:
        """Return pan and MS WarpedVRT profiles for cropping and warping both to a shared pixel
        grid and bounds.  Pan is left on its source grid, and cropped if necessary.  MS is warped
        and cropped (as necessary) to lie on the pan grid.
        """
        # TODO: what if the images have GCPs but not transforms?
        with utils.OpenRaster(pan_file) as pan_ds, utils.OpenRaster(ms_file) as ms_ds:
            if np.any(np.array(pan_ds.res) > ms_ds.res):
                raise ValueError(
                    f'Pan resolution: {pan_ds.res} exceeds multispectral resolution: {ms_ds.res}.'
                )

            # MS bounds, bounding window and resolution in pan coordinates
            ms_bounds_ = np.array(transform_bounds(ms_ds.crs, pan_ds.crs, *ms_ds.bounds))
            ms_win_ = pan_ds.window(*ms_bounds_)
            if not intersect((pan_ds.window(*pan_ds.bounds), ms_win_)):
                raise ValueError('Pan and multispectral extents do not overlap.')
            ms_win_off_ = np.array((ms_win_.col_off, ms_win_.row_off))
            ms_res_ = (ms_bounds_[2:] - ms_bounds_[:2]) / (ms_ds.width, ms_ds.height)

            # ratio between ms and pan resolution in pan coordinates
            res_fact = ms_res_ / pan_ds.res

            if (
                (pan_ds.crs == ms_ds.crs)
                and np.all(ms_win_off_ == ms_win_off_.round())
                and np.all(res_fact == res_fact.round())
            ):
                # The pan and MS pixel grids coincide.  Find common pan and MS bounds that lie on
                # the MS pixel grid.
                pan_win_ = ms_ds.window(*pan_ds.bounds)
                common_win = pan_win_.intersection(ms_ds.window(*ms_ds.bounds))
                assert common_win.round_offsets() == common_win
                common_win = Window(*np.array(common_win.flatten(), dtype='int'))
                common_bounds = ms_ds.window_bounds(common_win)
            else:
                # The pan and MS pixel grids do not coincide.  Find common pan and MS bounds,
                # that lie on the pan and reprojected MS pixel grids.
                logger.debug(
                    'Multispectral image will be reprojected to coincide with the pan pixel grid.'
                )
                res_fact = res_fact.round()
                common_win = pan_ds.window(*pan_ds.bounds).intersection(ms_win_)
                common_win = np.array(common_win.flatten())
                common_win = np.array(
                    (*np.ceil(common_win[:2]), *np.floor(common_win[-2:] / res_fact) * res_fact),
                    dtype='int',
                )
                common_win = Window(*common_win)
                common_bounds = pan_ds.window_bounds(common_win)

            pan_win = pan_ds.window(*common_bounds)
            pan_win = Window(*np.array(pan_win.flatten(), dtype='int'))
            pan_transform = pan_ds.window_transform(pan_win)
            ms_size = (np.array((pan_win.width, pan_win.height)) / res_fact).astype('int')
            ms_transform = pan_transform * rio.Affine.scale(*res_fact)

            # create the pan & MS WarpedVRT profiles
            pan_profile = dict(
                crs=pan_ds.crs, transform=pan_transform, width=pan_win.width, height=pan_win.height
            )
            ms_profile = dict(
                crs=pan_ds.crs, transform=ms_transform, width=ms_size[0], height=ms_size[1]
            )
        return pan_profile, ms_profile

    def _create_out_profile(
        self,
        pan_im: rio.DatasetReader,
        ms_im: rio.DatasetReader,
        dtype: str,
        compress: str | Compress | None,
        write_mask: bool | None,
    ) -> tuple[dict, bool]:
        """Return a rasterio profile for the pan sharpened image."""
        # TODO: make common fn for use here and in Ortho
        # Determine dtype, check dtype support
        # (OpenCV remap doesn't support int8 or uint32, and only supports int32, uint64, int64 with
        # nearest interp so these dtypes are excluded).
        out_profile = dict(**self._pan_profile)
        dtype = dtype or np.promote_types(pan_im.dtypes[0], ms_im.dtypes[0])

        # setup compression, data interleaving and photometric interpretation
        if compress is None:
            compress = Compress.jpeg if dtype == 'uint8' else Compress.deflate
        else:
            compress = Compress(compress)
            if compress == Compress.jpeg:
                if dtype == 'uint16':
                    warnings.warn(
                        'Attempting a 12 bit JPEG output configuration.  Support is rasterio build '
                        'dependent.',
                        category=OrthorityWarning,
                    )
                    out_profile.update(nbits=12)
                elif dtype != 'uint8':
                    raise ValueError(
                        f"JPEG compression is supported for 'uint8' and 'uint16' data types only."
                    )

        if compress == Compress.jpeg:
            interleave, photometric = (
                ('pixel', 'ycbcr') if ms_im.count == 3 else ('band', 'minisblack')
            )
        else:
            interleave, photometric = ('band', 'minisblack')

        # resolve auto write_mask (=None) to write masks for jpeg compression
        if write_mask is None:
            write_mask = True if compress == Compress.jpeg else False

        # set nodata to None when writing internal masks to force external tools to use mask,
        # otherwise set by dtype
        if write_mask:
            nodata = None
        else:
            nodata = float('nan') if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).min

        # create ortho profile
        out_profile.update(
            driver='GTiff',
            dtype=dtype,
            count=ms_im.count,  # TODO: should be len of indexes
            tiled=True,
            blockxsize=Ortho._default_blocksize[0],
            blockysize=Ortho._default_blocksize[1],
            nodata=nodata,
            compress=compress.value,
            interleave=interleave,
            photometric=photometric,
            bigtiff='if_safer',
        )

        return out_profile, write_mask

    def _get_stats(
        self,
        pan_im: rio.DatasetReader,
        ms_im: rio.DatasetReader,
        indexes: Sequence[int] | np.ndarray,
        interp: Resampling,
        progress: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pan / MS means and covariances for the given datasets. Uses the "Numerically
        Stable Parallel Computation of (Co-)Variance" method described in
        https://doi.org/10.1145/3221269.3223036 to aggregate stats over image tiles.
        """
        # working dtype of float64 for aggregating over large images / tiles or large image dtypes
        working_dtype = 'float64'

        def get_tile_stats(
            pan_im: rio.DatasetReader,
            ms_im: rio.DatasetReader,
            indexes: Sequence[int],
            tile_win: Window,
        ) -> tuple[int, np.ndarray, np.ndarray]:
            """Return pan / MS pixel count, mean & sum of deviation products for the given
            ``tile_win``.  Fully masked tiles produce zeros for count, means & products.
            """
            # read pan and MS tiles & tile masks (without masked arrays which don't release GIL)
            with self._pan_lock:
                pan_array = pan_im.read(window=tile_win)
                pan_mask = pan_im.read_masks(window=tile_win).astype('bool', copy=False)
            with self._ms_lock:
                ms_array = ms_im.read(indexes=indexes, window=tile_win)
                ms_mask = ms_im.read_masks(indexes=indexes, window=tile_win).astype(
                    'bool', copy=False
                )

            # mask and combine pan & MS
            mask = pan_mask[0] & ms_mask.all(axis=0)
            pan_ms_array = np.concat(
                (pan_array[0][mask].reshape(1, -1), ms_array[:, mask].reshape(len(indexes), -1)),
                axis=0,
            )

            # find tile mean & sum of deviation products as in eq 12-13 in the paper
            tile_n = mask.sum()
            tile_mean = pan_ms_array.mean(axis=1) if tile_n > 0 else np.zeros(pan_ms_array.shape[0])
            pan_ms_array -= tile_mean.reshape(-1, 1)
            tile_prod = pan_ms_array.dot(pan_ms_array.T)
            return tile_n, tile_mean, tile_prod

        # accumulate pan / MS values over tiles
        with ExitStack() as ex_stack:
            # open MS & downsampled pan WarpedVRTs that lie on a MS resolution grid, and have
            # common bounds
            pan_im = ex_stack.enter_context(
                WarpedVRT(
                    pan_im, **self._ms_profile, resampling=Resampling.average, dtype=working_dtype
                )
            )
            # reproject MS to pan_im grid if necessary
            ms_im = ex_stack.enter_context(
                WarpedVRT(ms_im, **self._ms_profile, resampling=interp, dtype=working_dtype)
            )

            # reference full band mean & cov for testing
            # pan_array = pan_im.read(masked=True)
            # ms_array = ms_im.read(indexes=indexes, masked=True)
            # mask = pan_array.mask[0] | ms_array.mask.any(axis=0)
            # pan_array.mask = ms_array.mask = mask
            # pan_ms_array = np.ma.concatenate(
            #     (pan_array[0].reshape(1, -1), ms_array.reshape(len(indexes), -1)), axis=0
            # )
            # ref_cov = np.ma.cov(pan_ms_array)
            # ref_means = pan_ms_array.mean(axis=1)
            # del pan_array, ms_array, pan_ms_array

            # find tile stats in a thread pool and aggregate
            n = 0
            means = np.zeros(len(indexes) + 1, dtype=working_dtype)
            prod = np.zeros((len(indexes) + 1,) * 2, dtype=working_dtype)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                # TODO: are custom size block windows helpful here & does WarpedVRT dtype affect
                #  threading?
                futures = [
                    ex.submit(get_tile_stats, pan_im, ms_im, indexes, tile_win)
                    for _, tile_win in ms_im.block_windows(indexes[0])
                ]

                for future in tqdm(futures, **progress):
                    tile_n, tile_mean, tile_prod = future.result()

                    if tile_n > 0:
                        # eq 21 from the paper
                        mean_diffs = tile_mean - means
                        mean_diffs_ = mean_diffs.reshape(-1, 1).dot(mean_diffs.reshape(1, -1))
                        prod += tile_prod + mean_diffs_ * (n * tile_n) / (n + tile_n)
                        # eq 17 from the paper
                        n += tile_n
                        means += mean_diffs * tile_n / n

        # convert prod to unbiased covariance (as with numpy.cov default)
        cov = prod / (n - 1)
        return means, cov

    def _get_params(self, means: np.ndarray, cov: np.ndarray, weights: np.ndarray | None) -> dict:
        """Return the Gram-Schmidt pan-sharpening parameters for the given pan / MS means &
        covariances, and optional MS to pan weights.  If MS to pan weights are not provided,
        they are estimated from the data.  Uses the "How to Pan-sharpen Images Using the
        Gram-Schmidt Pan-sharpen Method – A Recipe" method described in
        https://doi.org/10.5194/isprsarchives-XL-1-W1-239-2013.
        """

        def get_weights(cov: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
            """Return normalised MS to pan weights.  If ``weights`` is not provided, weights are
            estimated from the given pan / MS covariance matrix.
            """
            if weights is None:
                # find the LS solution for the weights as described in section 2.2 & eq 6
                weights = np.linalg.lstsq(cov[1:, 1:], cov[0, 1:].reshape(-1, 1), rcond=None)[0]
            else:
                weights = np.array(weights)

            if np.any(weights < 0):
                # TODO: instead of zeroing and normalising, redo the least squares with -ve bands
                #  excluded & set -ve weights to zero.
                warnings.warn(
                    f'Weights contain negative value(s), setting to zero and normalising: '
                    f'{tuple(weights.round(4).tolist())}.',
                    category=OrthorityWarning,
                )

            # set any -ve weights to 0, normalise & return
            weights = weights.flatten().clip(0, None)
            return weights / weights.sum()

        def get_gs_coeffs(cov: np.ndarray, weights: np.ndarray) -> list[np.ndarray]:
            """Return the Gram-Schmidt coefficients for the given MS covariance and MS to pan
            weights.
            """
            a = np.zeros((cov.shape[0], cov.shape[0]))
            coeffs = [np.zeros(k + 1) for k in range(cov.shape[0])]
            e = np.eye(cov.shape[0])
            for k in range(cov.shape[0]):
                # eq 4 from paper
                if k == 0:
                    a[k] = weights
                else:
                    a[k] = e[k - 1] - np.dot(coeffs[k - 1], a[:k])

                # eq 3 from paper
                for l in range(k + 1):
                    num = a[l].dot(cov[k])
                    den = (a[l].reshape(-1, 1).dot(a[l].reshape(1, -1)) * cov).sum()
                    coeffs[k][l] = num / den

            return coeffs

        def get_pan_norm(
            means: np.ndarray, cov: np.ndarray, weights: np.ndarray
        ) -> tuple[float, float]:
            """Return the gain and bias to convert the actual pan to the mean and standard
            deviation of the simulated pan, given the pan / MS means & covariance, and MS to pan
            weights. Note this uses downsampled actual pan stats while the paper
            uses full resolution actual pan stats.  As the simulated pan stats are at MS
            (downsampled) resolution, it seems the correct to use both at same resolution.
            """
            pan_mean, pan_std = means[0], np.sqrt(cov[0, 0])

            # find simulated pan mean and std deviation from weights & MS covariances (no
            # corresponding equation in paper, derived from properties of covariance:
            # https://dlsun.github.io/probability/cov-properties.html)
            # TODO: validate this
            pan_sim_mean = weights.dot(means[1:])
            pan_sim_std = np.sqrt(
                (weights.reshape(-1, 1).dot(weights.reshape(1, -1)) * cov[1:, 1:]).sum()
            )

            # equations 2 & 3 from https://patents.google.com/patent/US6011875A/en
            gain = pan_sim_std / pan_std
            bias = pan_sim_mean - (gain * pan_mean)
            return gain, bias

        weights = get_weights(cov, weights=weights)
        coeffs = get_gs_coeffs(cov[1:, 1:], weights)
        gain, bias = get_pan_norm(means, cov, weights)
        return dict(means=means, coeffs=coeffs, weights=weights, gain=gain, bias=bias)

    def _process_tile_array(
        self,
        pan_array: np.ndarray,
        ms_array: np.ndarray,
        means: np.ndarray,
        coeffs: list[np.ndarray],
        weights: np.ndarray,
        gain: float,
        bias: float,
    ) -> np.ndarray:
        """Return the pan-sharpened tile for the given pan tile, MS tile and Gram-Schmidt
        parameters.  Uses the "Process for Enhancing the Spatial Resolution of Multispectral
        Imagery Using Pan-sharpening" method in https://patents.google.com/patent/US6011875A/en.
        """
        # TODO: document that this finds GS transform of already upsampled MS data, while the paper
        #  finds GS transform then upsamples

        def gs_foward(
            ms_array: np.ndarray,
            means: np.ndarray,
            coeffs: list[np.ndarray],
            weights: np.ndarray,
        ) -> np.ndarray:
            """Forward Gram-Schmidt transform of the given MS array with given parameters."""
            # equations 10-12 of the patent
            gs_array = np.zeros((ms_array.shape[0] + 1, ms_array.shape[1]), dtype=ms_array.dtype)
            gs_array[0] = weights.dot(ms_array)
            for bi in range(0, ms_array.shape[0]):
                phi = coeffs[bi]
                gs_array[bi + 1] = ms_array[bi] - means[bi] - phi.dot(gs_array[: bi + 1])

            return gs_array

        def gs_reverse(
            gs_array: np.ndarray,
            means: np.ndarray,
            coeffs: list[np.ndarray],
            ms_array: np.ndarray | None = None,
        ) -> np.ndarray:
            """Reverse Gram-Schmidt transform of the given Gram-Schmidt array (with substituted
            pan band) with given parameters.  Optionally writes into ``ms_array`` if provided.
            """
            if ms_array is None:
                ms_array = np.zeros(
                    (gs_array.shape[0] - 1, gs_array.shape[1]), dtype=gs_array.dtype
                )

            # equation 14 of the patent
            for bi in range(0, ms_array.shape[0]):
                phi = coeffs[bi]
                ms_array[bi] = gs_array[bi + 1] + means[bi] + phi.dot(gs_array[: bi + 1])

            return ms_array

        gs_array = gs_foward(ms_array, means[1:], coeffs, weights)
        gs_array[0] = (gain * pan_array) + bias  # substitute normalised pan
        ms_array = gs_reverse(gs_array, means[1:], coeffs, ms_array=ms_array)
        return ms_array

    def _process_tile(
        self,
        pan_im: rio.DatasetReader,
        ms_im: rio.DatasetReader,
        tile_win: Window,
        indexes: Sequence[int],
        dtype: str | np.dtype,
        out_im: rio.io.DatasetWriter,
        write_mask: bool,
        **params,
    ) -> None:
        """Thread-safe function to pan-sharpen a MS tile and write it to a dataset."""
        # TODO: working dtype precision checks
        # read pan and MS tiles & tile masks (without masked arrays which don't release GIL)
        with self._pan_lock:
            pan_array = pan_im.read(window=tile_win)
            pan_mask = pan_im.read_masks(window=tile_win).astype('bool', copy=False)
        with self._ms_lock:
            ms_array = ms_im.read(indexes=indexes, window=tile_win)
            ms_mask = ms_im.read_masks(indexes=indexes, window=tile_win).astype('bool', copy=False)

        # mask and reshape pan & MS
        mask = pan_mask[0] & ms_mask.all(axis=0)
        pan_array_ = pan_array[0][mask].reshape(1, -1)
        ms_array_ = ms_array[:, mask].reshape(len(indexes), -1)

        # pan sharpen masked data and write into output mask area
        out_array = np.full(ms_array.shape, fill_value=out_im.nodata or 0, dtype=dtype)
        out_array[:, mask] = self._process_tile_array(pan_array_, ms_array_, **params)

        # write pan sharpened tile
        with self._out_lock:
            # TODO: dtype rounding & clipping, and write mask if it has one
            out_im.write(out_array, window=tile_win, masked=write_mask)

    def process(
        self,
        out_file: str | PathLike | OpenFile,
        indexes: Sequence[int] = None,
        weights: bool | Sequence[float] = None,
        interp: str | Interp = Ortho._default_config['interp'],
        write_mask: bool | None = Ortho._default_config['write_mask'],
        dtype: str = Ortho._default_config['dtype'],
        compress: str | Compress | None = Ortho._default_config['compress'],
        build_ovw: bool = Ortho._default_config['build_ovw'],
        overwrite: bool = Ortho._default_config['overwrite'],
        progress: bool | dict = False,
    ):
        # TODO: allow pan index e.g. Landsat can have all bands incl pan at different resolutions
        #  in a single file

        exit_stack = ExitStack()
        interp = Interp(interp).to_rio()
        with exit_stack:
            # TODO: progress bar labels for reading stats and transforming with internal and user
            #  provided progress params
            if progress is True:
                progress = Ortho._default_tqdm_kwargs
            elif progress is False:
                progress = dict(disable=True, leave=False)
            # else:
            #     progress = tqdm(**progress)
            # progress = exit_stack.enter_context(progress)

            # open pan & MS images
            exit_stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False))
            pan_im = exit_stack.enter_context(utils.OpenRaster(self._pan_file, 'r'))
            ms_im = exit_stack.enter_context(utils.OpenRaster(self._ms_file, 'r'))
            indexes = indexes or ms_im.indexes

            # open output image
            out_profile, write_mask = self._create_out_profile(
                pan_im, ms_im, dtype, compress, write_mask
            )
            out_profile.update(count=len(indexes))
            out_im = exit_stack.enter_context(
                utils.OpenRaster(out_file, 'w', overwrite=overwrite, **out_profile)
            )

            # find pan sharpening parameters from image stats
            means, cov = self._get_stats(pan_im, ms_im, indexes, interp, progress)
            params = self._get_params(means, cov, weights)

            # open pan & upsampled MS WarpedVRTs that lie on the pan resolution grid, and have
            # common bounds
            working_dtype = str(np.promote_types(self._working_dtype, out_profile['dtype']))
            pan_im = exit_stack.enter_context(
                WarpedVRT(pan_im, **self._pan_profile, dtype=working_dtype)
            )
            ms_im = exit_stack.enter_context(
                WarpedVRT(ms_im, **self._pan_profile, resampling=interp, dtype=working_dtype)
            )

            # pan sharpen tiles in a thread pool
            # TODO: use output or custom block windows?
            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures = [
                    executor.submit(
                        self._process_tile,
                        pan_im,
                        ms_im,
                        tile_win,
                        indexes,
                        working_dtype,
                        out_im,
                        write_mask,
                        **params,
                    )
                    # for tile_win in self._block_windows(pan_im.shape)
                    for _, tile_win in out_im.block_windows(1)
                ]
                for future in tqdm(futures, **progress):
                    future.result()

                if build_ovw:
                    # TODO: make utils function
                    Ortho._build_overviews(out_im)
