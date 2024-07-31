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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from itertools import product
from os import PathLike
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

from orthority import common
from orthority.enums import Compress, Interp
from orthority.errors import OrthorityWarning

logger = logging.getLogger(__name__)


class PanSharpen:
    """
    Pan sharpener.

    Pan sharpens a multispectral image with a panchromatic image using the Gram-Schmidt method
    (https://doi.org/10.5194/isprsarchives-XL-1-W1-239-2013).

    :param pan_file:
        Panchromatic image. Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object
        in binary mode (``'rb'``), or a dataset reader.
    :param ms_file:
        Multispectral image. Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object
        in binary mode (``'rb'``), or a dataset reader.
    """

    # dtype for warping and transformations
    _working_dtype = 'float64'

    # default algorithm configuration values for PanSharpen.process()
    _default_alg_config = dict(ms_indexes=None, pan_index=1, weights=None, interp=Interp.cubic)

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
    def _get_vrt_profiles(pan_file, ms_file) -> tuple[dict, dict]:
        """Return pan & MS resolution WarpedVRT profiles that define shared pixel grids and bounds."""
        with common.OpenRaster(pan_file) as pan_ds, common.OpenRaster(ms_file) as ms_ds:
            if not pan_ds.crs or not ms_ds.crs:
                raise ValueError(
                    f'Pan and multispectral images should be georeferenced with a CRS and '
                    f'affine transform.'
                )
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

            # create the pan & MS resolution WarpedVRT profiles
            pan_win = pan_ds.window(*common_bounds)
            pan_win = Window(*np.array(pan_win.flatten(), dtype='int'))
            pan_transform = pan_ds.window_transform(pan_win)
            ms_size = (np.array((pan_win.width, pan_win.height)) / res_fact).astype('int')
            ms_transform = pan_transform * rio.Affine.scale(*res_fact)
            pan_profile = dict(
                crs=pan_ds.crs, transform=pan_transform, width=pan_win.width, height=pan_win.height
            )
            ms_profile = dict(
                crs=pan_ds.crs, transform=ms_transform, width=ms_size[0], height=ms_size[1]
            )
        return pan_profile, ms_profile

    @staticmethod
    def _block_windows(
        im_shape: tuple[int, int], block_shape: tuple[int, int] = (1024, 1024)
    ) -> Generator[Window]:
        """Block window generator for the given image, and optional block shape."""
        xrange = range(0, im_shape[1], block_shape[1])
        yrange = range(0, im_shape[0], block_shape[0])
        for xstart, ystart in product(xrange, yrange):
            xstop = min(xstart + block_shape[1], im_shape[1])
            ystop = min(ystart + block_shape[0], im_shape[0])
            yield Window(xstart, ystart, xstop - xstart, ystop - ystart)

    def _validate_pan_ms_params(
        self,
        pan_im: rio.DatasetReader,
        ms_im: rio.DatasetReader,
        pan_index: int,
        ms_indexes: Sequence[int],
        weights: Sequence[float] | None,
    ) -> tuple[Sequence[int], Sequence[float] | None]:
        """Validate pan / MS indexes and weights."""
        if pan_index <= 0 or pan_index > pan_im.count:
            pan_name = common.get_filename(pan_im)
            raise ValueError(
                f"Pan index {pan_index} is invalid for '{pan_name}' with {pan_im.count} band(s)"
            )

        ms_indexes = ms_im.indexes if ms_indexes is None or len(ms_indexes) == 0 else ms_indexes
        if np.any(ms_indexes_ := np.array(ms_indexes) <= 0) or np.any(ms_indexes_ > ms_im.count):
            ms_name = common.get_filename(ms_im)
            raise ValueError(
                f"Multispectral indexes {tuple(ms_indexes.tolist())} contain invalid values for "
                f"'{ms_name}' with {ms_im.count} band(s)"
            )

        weights = None if weights is None or len(weights) == 0 else weights
        if (weights is not None) and (len(weights) != len(ms_indexes)):
            raise ValueError(
                f"There should be the same number of multispectral to pan weights ({len(weights)}) "
                f"as multispectral indexes ({len(ms_indexes)})."
            )

        return ms_indexes, weights

    def _get_stats(
        self,
        pan_im: rio.DatasetReader,
        ms_im: rio.DatasetReader,
        pan_index: int,
        ms_indexes: Sequence[int] | np.ndarray,
        interp: Resampling,
        progress: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pan / MS means and covariances for the given datasets. Uses the "Numerically
        Stable Parallel Computation of (Co-)Variance" method described in
        https://doi.org/10.1145/3221269.3223036 to aggregate stats over image tiles.
        """

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
                pan_array = pan_im.read(indexes=pan_index, window=tile_win)
                pan_mask = pan_im.read_masks(indexes=pan_index, window=tile_win).astype(
                    'bool', copy=False
                )
            with self._ms_lock:
                ms_array = ms_im.read(indexes=ms_indexes, window=tile_win)
                ms_mask = ms_im.read_masks(indexes=ms_indexes, window=tile_win).astype(
                    'bool', copy=False
                )

            # mask and combine pan & MS
            mask = pan_mask & ms_mask.all(axis=0)
            pan_ms_array = np.concat(
                (pan_array[mask].reshape(1, -1), ms_array[:, mask].reshape(len(indexes), -1)),
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
                    pan_im,
                    **self._ms_profile,
                    resampling=Resampling.average,
                    dtype=self._working_dtype,
                )
            )
            # reproject MS to pan_im grid if necessary
            ms_im = ex_stack.enter_context(
                WarpedVRT(ms_im, **self._ms_profile, resampling=interp, dtype=self._working_dtype)
            )

            # find tile stats in a thread pool and aggregate
            n = 0
            means = np.zeros(len(ms_indexes) + 1, dtype=self._working_dtype)
            prod = np.zeros((len(ms_indexes) + 1,) * 2, dtype=self._working_dtype)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(get_tile_stats, pan_im, ms_im, ms_indexes, tile_win)
                    for tile_win in self._block_windows(ms_im.shape)
                ]

                for future in tqdm(as_completed(futures), **progress, total=len(futures)):
                    # TODO: put this shutdown logic in all thread pools
                    try:
                        tile_n, tile_mean, tile_prod = future.result()
                    except Exception as ex:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError('Could not get tile statistics.') from ex

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
            estimated from the given pan / MS covariance.
            """
            if weights is None:
                # find the LS solution for the weights as described in section 2.2 & eq 6
                weights = np.linalg.lstsq(cov[1:, 1:], cov[0, 1:].reshape(-1, 1), rcond=None)[0].T

                # redo the LS without negatively weighted bands if there are any
                if np.any(weights < 0):
                    ms_indexes_ = np.where(weights > 0)[0] + 1
                    ms_cov = cov[ms_indexes_, :][:, ms_indexes_]
                    pan_ms_cov = cov[0, ms_indexes_].reshape(-1, 1)
                    weights_ = np.linalg.lstsq(ms_cov, pan_ms_cov, rcond=None)[0].T

                    # use the updated weights if they are positive
                    if np.all(weights_ >= 0):
                        weights[ms_indexes_ - 1] = weights_
            else:
                weights = np.array(weights)

            # set any remaining negative weights to 0, normalise & return
            weights = weights.flatten()
            if np.any(weights < 0):
                warnings.warn(
                    f'Weights contain negative value(s), setting to zero and normalising: '
                    f'{tuple(weights.round(4).tolist())}.',
                    category=OrthorityWarning,
                )
                weights = weights.clip(0, None)
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
            deviation of the simulated pan.
            """
            # Note: this uses MS resolution, while the paper uses pan resolution actual pan
            # stats.  As the simulated pan stats are also at the MS resolution, this seems correct.

            pan_mean, pan_std = means[0], np.sqrt(cov[0, 0])

            # find simulated pan mean and std deviation from weights & MS covariances (no
            # corresponding equation in paper, derived from properties of covariance:
            # https://dlsun.github.io/probability/cov-properties.html)
            pan_sim_mean = weights.dot(means[1:])
            pan_sim_std = np.sqrt(
                (weights.reshape(-1, 1).dot(weights.reshape(1, -1)) * cov[1:, 1:]).sum()
            )

            # equations 2 & 3 from https://patents.google.com/patent/US6011875A/en
            gain = pan_sim_std / pan_std
            bias = pan_sim_mean - (gain * pan_mean)
            return gain, bias

        weights = get_weights(cov, weights=weights)
        logger.debug(f"Multispectral to pan weights: {tuple(weights.round(4).tolist())}.")
        coeffs = get_gs_coeffs(cov[1:, 1:], weights)
        gain, bias = get_pan_norm(means, cov, weights)
        logger.debug(f"Simulated pan gain: {gain:.4f}, bias: {bias:.4f}.")
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
        """Return the pan-sharpened tile for the given pan & MS tiles, and Gram-Schmidt
        parameters.  Uses the "Process for Enhancing the Spatial Resolution of Multispectral
        Imagery Using Pan-sharpening" method in https://patents.google.com/patent/US6011875A/en.
        """
        # Note: the Gram-Schmidt transform is applied to upsampled MS data, rather than
        # transforming first then upsampling as in the patent.  This is equivalent and simpler in
        # code.

        def gs_foward(
            ms_array: np.ndarray,
            means: np.ndarray,
            coeffs: list[np.ndarray],
            weights: np.ndarray,
        ) -> np.ndarray:
            """Forward Gram-Schmidt transform of the given MS array with given parameters."""
            # equations 10-12 of the patent
            gs_array = np.zeros(
                (ms_array.shape[0] + 1, ms_array.shape[1]), dtype=self._working_dtype
            )
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
            """Reverse Gram-Schmidt transform of the given Gram-Schmidt array with given
            parameters.  Optionally writes into ``ms_array`` if provided.
            """
            if ms_array is None:
                ms_array = np.zeros(
                    (gs_array.shape[0] - 1, gs_array.shape[1]), dtype=self._working_dtype
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
        pan_index: int,
        ms_indexes: Sequence[int],
        out_im: rio.io.DatasetWriter,
        write_mask: bool,
        **params,
    ) -> None:
        """Thread-safe function to pan-sharpen a MS tile and write it to a dataset."""
        # read pan and MS tiles & tile masks (without masked arrays which don't release GIL)
        with self._pan_lock:
            pan_array = pan_im.read(indexes=pan_index, window=tile_win)
            pan_mask = pan_im.read_masks(indexes=pan_index, window=tile_win).astype(
                'bool', copy=False
            )
        with self._ms_lock:
            ms_array = ms_im.read(indexes=ms_indexes, window=tile_win)
            ms_mask = ms_im.read_masks(indexes=ms_indexes, window=tile_win).astype(
                'bool', copy=False
            )

        # mask and reshape pan & MS
        mask = pan_mask & ms_mask.all(axis=0)
        pan_array_ = pan_array[mask].reshape(1, -1)
        ms_array_ = ms_array[:, mask].reshape(len(ms_indexes), -1)

        # pan sharpen masked data and write into output mask area
        out_array_ = self._process_tile_array(pan_array_, ms_array_, **params)
        out_array_ = common.convert_array_dtype(out_array_, out_im.dtypes[0])
        out_array = np.full(ms_array.shape, fill_value=out_im.nodata or 0, dtype=out_im.dtypes[0])
        out_array[:, mask] = out_array_

        # write pan sharpened tile
        with self._out_lock:
            out_im.write(out_array, window=tile_win)
            if write_mask:
                out_im.write_mask(mask, window=tile_win)

    def process(
        self,
        out_file: str | PathLike | OpenFile,
        pan_index: int = _default_alg_config['pan_index'],
        ms_indexes: Sequence[int] = _default_alg_config['ms_indexes'],
        weights: bool | Sequence[float] = _default_alg_config['weights'],
        interp: str | Interp = _default_alg_config['interp'],
        write_mask: bool | None = common._default_out_config['write_mask'],
        dtype: str = common._default_out_config['dtype'],
        compress: str | Compress | None = common._default_out_config['compress'],
        build_ovw: bool = common._default_out_config['build_ovw'],
        overwrite: bool = common._default_out_config['overwrite'],
        progress: bool | Sequence[dict] = False,
    ):
        """
        Pan-sharpen.

        Image statistics are aggregated tile-by-tile and used to derive the Gramm-Schmidt
        parameters in a first step.  Then the pan-sharpened image is created tile-by-tile in a
        second step.

        :param out_file:
            Output image file to create.  Can be a path or URI string, or an
            :class:`~fsspec.core.OpenFile` object in binary mode (``'wb'``).
        :param pan_index:
            Index of the pan band to use (1-based).
        :param ms_indexes:
            Indexes of the multispectral bands to use (1-based).  If set to ``None`` (the default),
            all multispectral bands are used.
        :param weights:
            Multi-spectral to pan weights.  If set to ``None`` (the default), weights are
            estimated from the images.
        :param interp:
            Interpolation method for resampling the multispectral image.
        :param write_mask:
            Mask valid output pixels with an internal mask (``True``), or with a nodata value
            based on ``dtype`` (``False``). An internal mask helps remove nodata noise caused by
            lossy compression. If set to ``None`` (the default), the mask will be written when
            JPEG compression is used.
        :param dtype:
            Output image data type (``uint8``, ``uint16``, ``int16``, ``float32`` or
            ``float64``).  If set to ``None`` (the default), the source image data type is used.
        :param compress:
            Output image compression type (``jpeg`` or ``deflate``).  ``deflate`` can be used with
            any ``dtype``, and ``jpeg`` with the uint8 ``dtype``.  With supporting Rasterio
            builds, ``jpeg`` can also be used with uint16, in which case the output is 12 bit JPEG
            compressed. If ``compress`` is set to ``None`` (the default), ``jpeg`` is used for the
            uint8 ``dtype``, and ``deflate`` otherwise.
        :param build_ovw:
            Whether to build overviews for the output image.
        :param overwrite:
            Whether to overwrite the output image if it exists.
        :param progress:
            Whether to display a progress bar monitoring the portion of tiles processed in each
            step. Can be set to a sequence of two argument dictionaries defining a custom
            `tqdm <https://tqdm.github.io/docs/tqdm/>`_ bar for each step.
        """
        # set up progress bars
        if progress is True:
            progress = [
                common.get_tqdm_kwargs(desc=desc, unit='blocks')
                for desc in ['Statistics', 'Sharpening']
            ]
        elif progress is False:
            progress = [dict(disable=True, leave=False)] * 2

        exit_stack = ExitStack()
        interp = Interp(interp).to_rio()
        with exit_stack:
            # open pan & MS images
            exit_stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False))
            pan_im = exit_stack.enter_context(common.OpenRaster(self._pan_file, 'r'))
            ms_im = exit_stack.enter_context(common.OpenRaster(self._ms_file, 'r'))

            # validate indexes and weights
            ms_indexes, weights = self._validate_pan_ms_params(
                pan_im, ms_im, pan_index, ms_indexes, weights
            )

            # open output image
            dtype = dtype or np.promote_types(pan_im.dtypes[0], ms_im.dtypes[0])
            out_profile, write_mask = common.create_profile(
                dtype, compress=compress, write_mask=write_mask, colorinterp=ms_im.colorinterp
            )
            out_profile.update(
                crs=self._pan_profile['crs'],
                transform=self._pan_profile['transform'],
                width=self._pan_profile['width'],
                height=self._pan_profile['height'],
                count=len(ms_indexes),
            )
            out_im = exit_stack.enter_context(
                common.OpenRaster(out_file, 'w', overwrite=overwrite, **out_profile)
            )

            # find pan sharpening parameters from image stats
            means, cov = self._get_stats(pan_im, ms_im, pan_index, ms_indexes, interp, progress[0])
            params = self._get_params(means, cov, weights)

            # open pan & upsampled MS WarpedVRTs that lie on the pan resolution grid, and have
            # common bounds
            pan_im = exit_stack.enter_context(
                WarpedVRT(pan_im, **self._pan_profile, dtype=self._working_dtype)
            )
            ms_im = exit_stack.enter_context(
                WarpedVRT(ms_im, **self._pan_profile, resampling=interp, dtype=self._working_dtype),
            )

            # pan sharpen tiles in a thread pool
            # TODO: use output or custom block windows?
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(
                        self._process_tile,
                        pan_im,
                        ms_im,
                        tile_win,
                        pan_index,
                        ms_indexes,
                        out_im,
                        write_mask,
                        **params,
                    )
                    for tile_win in self._block_windows(pan_im.shape, block_shape=(1024, 1024))
                    # for _, tile_win in out_im.block_windows(1)
                ]
                for future in tqdm(as_completed(futures), **progress[1], total=len(futures)):
                    try:
                        future.result()
                    except Exception as ex:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError('Could not process tile.') from ex

                if build_ovw:
                    common.build_overviews(out_im)
