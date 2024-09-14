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
"""Pan-sharpening."""
from __future__ import annotations

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from os import PathLike
from threading import Lock
from typing import Sequence

import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.enums import Resampling, ColorInterp
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import intersect, Window
from rasterio.windows import transform as window_transform
from tqdm.auto import tqdm

from orthority import common
from orthority.enums import Compress, Interp, Driver
from orthority.errors import OrthorityWarning, OrthorityError

logger = logging.getLogger(__name__)


class PanSharpen:
    """
    Pan-sharpener.

    Increases the resolution of a multispectral image to that of a panchromatic image using the
    Gram-Schmidt method (https://doi.org/10.5194/isprsarchives-XL-1-W1-239-2013).

    Panchromatic and multispectral image bounds should overlap if they are georeferenced. If one
    or both of the images are not georeferenced, they are assumed to having matching bounds.

    :param pan_file:
        Panchromatic image. Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object
        in binary mode (``'rb'``), or a dataset reader.
    :param ms_file:
        Multispectral image. Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object
        in binary mode (``'rb'``), or a dataset reader.
    """

    # dtype & nodata for warping and transformations
    _working_dtype = 'float64'
    _working_nodata = common._nodata_vals[_working_dtype]

    # default algorithm configuration values for PanSharpen.process()
    _default_alg_config = dict(ms_indexes=None, pan_index=1, weights=None, interp=Interp.cubic)

    def __init__(
        self,
        pan_file: str | PathLike | OpenFile | rio.DatasetReader,
        ms_file: str | PathLike | OpenFile | rio.DatasetReader,
    ):
        self._pan_file = pan_file
        self._ms_file = ms_file
        self._profiles = self._get_vrt_profiles(pan_file, ms_file)
        self._pan_lock = Lock()
        self._ms_lock = Lock()
        self._out_lock = Lock()

    @staticmethod
    def _get_vrt_profiles(pan_file, ms_file) -> dict[str, dict]:
        """Return a dictionary of WarpedVRT profiles for warping pan/MS to pan and MS grids.  The
        pan and MS grids are cropped versions of the source grids that define shared bounds.
        """
        with ExitStack() as exit_stack:
            exit_stack.enter_context(common.suppress_no_georef())

            # open files
            pan_ds = exit_stack.enter_context(common.OpenRaster(pan_file))
            ms_ds = exit_stack.enter_context(common.OpenRaster(ms_file))
            # maintain source transforms separately as un-georeferenced images get custom
            # transforms
            pan_src_transform = pan_ds.transform
            ms_src_transform = ms_ds.transform

            # find shared pan & MS bounds as pan & MS windows in their own CRSs
            if ms_ds.crs is None and ms_ds.transform.is_identity:
                # MS is not georeferenced and pan may or may not be georeferenced. Assume pan &
                # MS bounds match.
                warnings.warn(
                    'Multispectral image is not georeferenced with a CRS and affine transform - '
                    'assuming its CRS and bounds match the pan image.',
                    category=OrthorityWarning,
                )
                # create a custom MS transform that matches MS to pan bounds
                ms_scale = np.array(pan_ds.shape[::-1]) / ms_ds.shape[::-1]
                ms_src_transform = pan_ds.transform * rio.Affine.scale(*ms_scale)
                # full / matching pan & MS windows
                pan_win = Window(0, 0, pan_ds.width, pan_ds.height)
                ms_win = Window(0, 0, ms_ds.width, ms_ds.height)

            elif pan_ds.crs is None and pan_ds.transform.is_identity:
                # Pan is not georeferenced and MS is georeferenced. Assume pan & MS bounds match.
                warnings.warn(
                    'Pan image is not georeferenced with a CRS and affine transform - assuming '
                    'its CRS and bounds match the multispectral image.',
                    category=OrthorityWarning,
                )
                # create a custom pan transform that matches pan to MS bounds
                pan_scale = np.array(ms_ds.shape[::-1]) / pan_ds.shape[::-1]
                pan_src_transform = ms_ds.transform * rio.Affine.scale(*pan_scale)
                # full / matching pan & MS windows
                pan_win = Window(0, 0, pan_ds.width, pan_ds.height)
                ms_win = Window(0, 0, ms_ds.width, ms_ds.height)

            else:
                # both pan and MS are georeferenced
                # find shared bounds pan window
                ms_bounds_ = transform_bounds(ms_ds.crs, pan_ds.crs, *ms_ds.bounds)
                ms_win_ = pan_ds.window(*ms_bounds_)
                if not intersect((ms_win_, pan_ds.window(*pan_ds.bounds))):
                    raise OrthorityError('Pan and multispectral bounds do not overlap.')
                pan_win = ms_win_.intersection(pan_ds.window(*pan_ds.bounds))
                pan_win = common.expand_window_to_grid(pan_win)

                # find shared bounds MS window
                pan_bounds_ = transform_bounds(pan_ds.crs, ms_ds.crs, *pan_ds.bounds)
                pan_win_ = ms_ds.window(*pan_bounds_)
                ms_win = pan_win_.intersection(ms_ds.window(*ms_ds.bounds))
                ms_win = common.expand_window_to_grid(ms_win)

            # test resolutions
            pan_res = np.abs((pan_src_transform[0], pan_src_transform[4]))
            ms_res = np.abs((ms_src_transform[0], ms_src_transform[4]))
            if np.any(pan_res > ms_res):
                raise OrthorityError(
                    f'Pan resolution: {tuple(pan_res.tolist())} exceeds multispectral resolution: '
                    f'{tuple(ms_res.tolist())}.'
                )

            # create the WarpedVRT profiles ('src_transform' items are required for
            # un-georeferenced images)
            profiles = {}
            pan_transform = window_transform(pan_win, pan_src_transform)
            ms_transform = window_transform(ms_win, ms_src_transform)
            profiles['pan_to_pan'] = dict(
                crs=pan_ds.crs,
                src_transform=pan_src_transform,
                transform=pan_transform,
                width=pan_win.width,
                height=pan_win.height,
            )
            profiles['ms_to_pan'] = profiles['pan_to_pan'].copy()
            profiles['ms_to_pan'].update(src_transform=ms_src_transform)

            profiles['ms_to_ms'] = dict(
                crs=ms_ds.crs,
                src_transform=ms_src_transform,
                transform=ms_transform,
                width=ms_win.width,
                height=ms_win.height,
            )
            profiles['pan_to_ms'] = profiles['ms_to_ms'].copy()
            profiles['pan_to_ms'].update(src_transform=pan_src_transform)

            # add common config items
            for profile in profiles.values():
                profile.update(
                    dtype=PanSharpen._working_dtype,
                    nodata=PanSharpen._working_nodata,
                    num_threads=os.cpu_count(),
                )
        return profiles

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
            raise OrthorityError(
                f"Pan index {pan_index} is invalid for '{pan_name}' with {pan_im.count} band(s)"
            )

        if ms_indexes is None or len(ms_indexes) == 0:
            # default to non-alpha band indexes
            ms_indexes = [
                bi + 1 for bi in range(ms_im.count) if ms_im.colorinterp[bi] != ColorInterp.alpha
            ]

        # test ms_indexes (allows user ms_indexes that are duplicates and or alpha bands)
        ms_indexes_ = np.array(ms_indexes)
        if np.any(ms_indexes_ <= 0) or np.any(ms_indexes_ > ms_im.count):
            ms_name = common.get_filename(ms_im)
            raise OrthorityError(
                f"Multispectral indexes {tuple(ms_indexes)} contain invalid values for '{ms_name}' "
                f"with {ms_im.count} band(s)"
            )

        weights = None if weights is None or len(weights) == 0 else weights
        if weights is not None:
            if len(weights) != len(ms_indexes):
                raise OrthorityError(
                    f"There should be the same number of multispectral to pan weights "
                    f"({len(weights)}) as multispectral indexes ({len(ms_indexes)})."
                )
            if np.any(np.array(weights) < 0):
                raise OrthorityError('Weight values should greater than or equal to 0.')

        return ms_indexes, weights

    def _get_stats(
        self,
        pan_im: rio.DatasetReader,
        ms_im: rio.DatasetReader,
        pan_index: int,
        ms_indexes: Sequence[int] | np.ndarray,
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
            # read pan and MS tiles (without masked arrays which don't release GIL)
            with self._pan_lock:
                pan_array = pan_im.read(indexes=pan_index, window=tile_win)
            with self._ms_lock:
                ms_array = ms_im.read(indexes=ms_indexes, window=tile_win)

            # mask and combine pan & MS
            pan_mask = np.logical_not(common.nan_equals(pan_array, pan_im.nodata))
            ms_mask = np.logical_not(common.nan_equals(ms_array, ms_im.nodata)).all(axis=0)
            mask = pan_mask & ms_mask
            pan_ms_array = np.concatenate(
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
                WarpedVRT(pan_im, **self._profiles['pan_to_ms'], resampling=Resampling.average)
            )
            ms_im = ex_stack.enter_context(WarpedVRT(ms_im, **self._profiles['ms_to_ms']))

            # find tile stats in a thread pool and aggregate
            n = 0
            means = np.zeros(len(ms_indexes) + 1, dtype=self._working_dtype)
            prod = np.zeros((len(ms_indexes) + 1,) * 2, dtype=self._working_dtype)
            executor = ex_stack.enter_context(ThreadPoolExecutor(max_workers=os.cpu_count()))
            futures = [
                executor.submit(get_tile_stats, pan_im, ms_im, ms_indexes, tile_win)
                for tile_win in common.block_windows(ms_im, block_shape=(1024, 1024))
            ]

            for future in tqdm(as_completed(futures), **progress, total=len(futures)):
                try:
                    tile_n, tile_mean, tile_prod = future.result()
                except Exception as ex:
                    executor.shutdown(wait=False)
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
                weights = np.linalg.lstsq(cov[1:, 1:], cov[0, 1:].reshape(-1, 1), rcond=None)[0]
                weights = weights.squeeze()

                # redo the LS without negatively weighted bands if there are any
                if np.any(weights < 0):
                    warnings.warn(
                        f'Weights contain negative value(s): {tuple(weights.round(4).tolist())}, '
                        f're-estimating positive weights.',
                        category=OrthorityWarning,
                    )
                    ms_indexes_ = np.where(weights > 0)[0] + 1
                    ms_cov = cov[ms_indexes_, :][:, ms_indexes_]
                    pan_ms_cov = cov[0, ms_indexes_].reshape(-1, 1)
                    weights_ = np.linalg.lstsq(ms_cov, pan_ms_cov, rcond=None)[0].squeeze()

                    # use the updated weights if they are positive, and set negative weights to 0
                    if np.all(weights_ >= 0):
                        weights = weights.clip(0, None)
                        weights[ms_indexes_ - 1] = weights_
            else:
                weights = np.array(weights)

            # set any remaining negative weights to 0, normalise & return
            weights = weights.flatten()
            if np.any(weights < 0):
                warnings.warn(
                    f'Weights contain negative value(s): {tuple(weights.round(4).tolist())}, '
                    f'setting to zero and normalising.',
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
                    # the 'if' below avoids dividing by zero with canonical weight vectors
                    coeffs[k][l] = num / den if np.any(a[l] != 0) else 0

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
        coeffs = get_gs_coeffs(cov[1:, 1:], weights)
        gain, bias = get_pan_norm(means, cov, weights)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Pan / multispectral means: {means.round(4)}.")
            logger.debug(f"Pan / multispectral covariance: \n{cov.round(4)}.")
            logger.debug(f"Multispectral to pan weights: {weights.round(4).tolist()}.")
            coeffs_str = '\n'.join([str(c.round(4).tolist()) for c in coeffs])
            logger.debug(f"Gram-Schmidt coefficients: \n{coeffs_str}")
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
        # read pan and MS tiles (without masked arrays which don't release GIL)
        with self._pan_lock:
            pan_array = pan_im.read(indexes=pan_index, window=tile_win)
        with self._ms_lock:
            ms_array = ms_im.read(indexes=ms_indexes, window=tile_win)

        # mask and reshape pan & MS
        pan_mask = np.logical_not(common.nan_equals(pan_array, pan_im.nodata))
        ms_mask = np.logical_not(common.nan_equals(ms_array, ms_im.nodata)).all(axis=0)
        mask = pan_mask & ms_mask
        pan_array_ = pan_array[mask].reshape(1, -1)
        ms_array_ = ms_array[:, mask].reshape(len(ms_indexes), -1)

        # pan-sharpen masked data and write into output mask area
        out_array_ = self._process_tile_array(pan_array_, ms_array_, **params)
        out_array_ = common.convert_array_dtype(out_array_, out_im.dtypes[0])
        out_array = np.full(ms_array.shape, fill_value=out_im.nodata or 0, dtype=out_im.dtypes[0])
        out_array[:, mask] = out_array_

        # write pan-sharpened tile
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
        creation_options: dict | None = None,
        driver: str | Driver = common._default_out_config['driver'],
        overwrite: bool = common._default_out_config['overwrite'],
        progress: bool | Sequence[dict] = False,
    ):
        """
        Pan-sharpen.

        The pan-sharpened image is created on the panchromatic pixel grid.  Pan-sharpened image
        bounds are the intersection of the panchromatic and multispectral image bounds.

        Pan-sharpening consists of two steps, both of which operate tile-by-tile:

        1. Derive the Gram-Schmidt parameters from image statistics.
        2. Generate the pan-sharpened image.

        :param out_file:
            Pan-sharpened image file to create.  Can be a path or URI string, or an
            :class:`~fsspec.core.OpenFile` object in binary mode (``'wb'``).
        :param pan_index:
            Index of the panchromatic band to use (1-based).
        :param ms_indexes:
            Indexes of the multispectral bands to use (1-based).  If set to ``None`` (the default),
            all non-alpha multispectral bands are used.
        :param weights:
            Multi-spectral to panchromatic weights (≥0).  If set to ``None`` (the default),
            weights are estimated from the images.
        :param interp:
            Interpolation method for upsampling the multispectral image.
        :param write_mask:
            Mask valid pan-sharpened pixels with an internal mask (``True``), or with a nodata
            value based on ``dtype`` (``False``). An internal mask helps remove nodata noise
            caused by lossy compression. If set to ``None`` (the default), the mask will be
            written when JPEG compression is used.
        :param dtype:
            Pan-sharpened image data type (``uint8``, ``uint16``, ``int16``, ``float32`` or
            ``float64``).  If set to ``None`` (the default), the source image data type is used.
        :param compress:
            Pan-sharpened image compression type (``jpeg``, ``deflate`` or ``lzw``).  ``deflate``
            and ``lzw`` can be used with any ``dtype``, and ``jpeg`` with the uint8 ``dtype``.
            With supporting Rasterio builds, ``jpeg`` can also be used with uint16, in which case
            the ortho is 12 bit JPEG compressed.  If ``compress`` is set to ``None`` (the
            default), ``jpeg`` is used for the uint8 ``dtype``, and ``deflate`` otherwise.
        :param build_ovw:
            Whether to build overviews for the pan-sharpened image.
        :param creation_options:
            Pan-sharpened image creation options as dictionary of ``name: value`` pairs.  If
            supplied, ``compress`` is ignored.  See the `GDAL docs
            <https://gdal.org/en/latest/drivers/raster/gtiff.html#creation-options>`__ for details.
        :param driver:
            Pan-sharpened image driver (``gtiff`` or ``cog``).
        :param overwrite:
            Whether to overwrite the pan-sharpened image if it exists.
        :param progress:
            Whether to display a progress bar monitoring the portion of tiles processed in each
            step. Can be set to a sequence of two argument dictionaries that define a custom
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
            exit_stack.enter_context(common.suppress_no_georef())

            # open pan & MS images
            exit_stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False))
            pan_im = exit_stack.enter_context(common.OpenRaster(self._pan_file, 'r'))
            ms_im = exit_stack.enter_context(common.OpenRaster(self._ms_file, 'r'))

            # validate indexes and weights
            ms_indexes, weights = self._validate_pan_ms_params(
                pan_im, ms_im, pan_index, ms_indexes, weights
            )

            # open output image & resolve write_mask
            dtype = dtype or ms_im.dtypes[0]
            pan_profile = self._profiles['pan_to_pan']
            out_profile, write_mask = common.create_profile(
                driver=driver,
                shape=(len(ms_indexes), pan_profile['height'], pan_profile['width']),
                dtype=dtype,
                compress=compress,
                write_mask=write_mask,
                creation_options=creation_options,
            )
            out_profile.update(crs=pan_profile['crs'], transform=pan_profile['transform'])
            out_im = exit_stack.enter_context(
                common.OpenRaster(out_file, 'w', overwrite=overwrite, **out_profile)
            )

            # copy colorinterp from MS to output
            out_im.colorinterp = [ms_im.colorinterp[mi - 1] for mi in ms_indexes]

            # find pan-sharpening parameters from image stats
            means, cov = self._get_stats(pan_im, ms_im, pan_index, ms_indexes, progress[0])
            params = self._get_params(means, cov, weights)

            # open pan & upsampled MS WarpedVRTs that lie on the pan resolution grid, and have
            # common bounds
            pan_im = exit_stack.enter_context(WarpedVRT(pan_im, **self._profiles['pan_to_pan']))
            ms_im = exit_stack.enter_context(
                WarpedVRT(ms_im, **self._profiles['ms_to_pan'], resampling=interp),
            )

            # pan-sharpen tiles in a thread pool
            executor = exit_stack.enter_context(ThreadPoolExecutor(max_workers=os.cpu_count()))
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
                for tile_win in common.block_windows(out_im)
            ]

            pbar = exit_stack.enter_context(tqdm(**progress[1], total=len(futures)))
            for future in futures:
                try:
                    future.result()
                except Exception as ex:
                    # TODO: add cancel_futures=True to all thread pool shutdowns when supported py
                    #  versions are > 3.8
                    executor.shutdown(wait=False)
                    raise RuntimeError('Could not process tile.') from ex
                pbar.update()
            pbar.refresh()

            if build_ovw:
                common.build_overviews(out_im)
