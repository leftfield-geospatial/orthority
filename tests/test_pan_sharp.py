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

from __future__ import annotations

import logging
import os
from contextlib import ExitStack
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.enums import Resampling, MaskFlags, ColorInterp
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from orthority import common
from orthority.enums import Interp, Compress, Driver
from orthority.errors import OrthorityError, OrthorityWarning
from orthority.pan_sharp import PanSharpen
from tests.conftest import create_profile

logger = logging.getLogger(__name__)


@pytest.fixture(scope='session')
def pan_sharpen(pan_file: Path, ms_file: Path) -> PanSharpen:
    """A pan-sharpener."""
    return PanSharpen(pan_file, ms_file)


@pytest.fixture(scope='session')
def mask_pan_file(
    pan_file: Path, webmerc_crs: str, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Panchromatic image with georeferencing and an internal mask defining a buffer of invalid
    pixels.
    """
    # read pan_file
    with rio.open(pan_file, 'r') as src_im:
        src_array = src_im.read()

    # pad src_array and create georeferenced profile
    pwidth = 100
    pval = 0
    out_array = np.pad(
        src_array, ((0, 0), (pwidth, pwidth), (pwidth, pwidth)), constant_values=pval
    )
    transform = rio.Affine(2, 0, 10, 0, -2, 20)
    out_profile = create_profile(out_array, crs=webmerc_crs, transform=transform)

    # write to output file
    out_file = tmp_path_factory.mktemp('data').joinpath('mask_pan.tif')
    with rio.open(out_file, 'w', **out_profile) as out_im:
        out_im.write(out_array)
        out_im.write_mask((out_array != pval).all(axis=0))
    return out_file


@pytest.fixture(scope='session')
def mask_ms_file(ms_file: Path, webmerc_crs: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Multispectral image with georeferencing and an internal mask defining a buffer of invalid
    pixels.
    """
    # read ms_file
    with rio.open(ms_file, 'r') as src_im:
        src_array = src_im.read()

    # pad src_array  and create georeferenced profile that shares bounds with mask_pan_file
    scale = 4  # scale relative to mask_pan_file
    pwidth = int(100 / scale)
    pval = 0
    out_array = np.pad(
        src_array, ((0, 0), (pwidth, pwidth), (pwidth, pwidth)), constant_values=pval
    )
    transform = rio.Affine(2, 0, 10, 0, -2, 20) * rio.Affine.scale(scale)
    out_profile = create_profile(out_array, crs=webmerc_crs, transform=transform)

    # write to output file
    out_file = tmp_path_factory.mktemp('data').joinpath('mask_ms.tif')
    with rio.open(out_file, 'w', **out_profile) as out_im:
        out_im.write(out_array)
        out_im.write_mask((out_array != pval).all(axis=0))
    return out_file


@pytest.fixture(scope='session')
def ms_neg_band_file(ms_file: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Multispectral image that is ``ms_file`` with a fourth band added that is the inverse of the
    first band.  Also adds a non-standard ``colorinterp``.
    """
    # read ms_file
    with rio.open(ms_file, 'r') as src_im:
        src_array = src_im.read(out_dtype='int16')

    # create 4 band array with 4th band as inverse of first band
    out_array = np.concatenate((src_array, [src_array[0] * -1]), axis=0)

    # write to output file
    out_profile = create_profile(out_array)
    out_file = tmp_path_factory.mktemp('data').joinpath('ms_neg_band.tif')
    with rio.open(out_file, 'w', **out_profile) as out_im:
        out_im.colorinterp = [
            ColorInterp.blue,
            ColorInterp.green,
            ColorInterp.red,
            ColorInterp.undefined,
        ]
        out_im.write(out_array)
    return out_file


def test_init_profiles(pan_file: Path, ms_file: Path):
    """Basic test of pan / MS reprojection profiles."""
    pan_sharp = PanSharpen(pan_file, ms_file)

    req_keys = ['crs', 'src_transform', 'transform', 'width', 'height']
    for profile_key in ['pan_to_pan', 'pan_to_ms', 'ms_to_ms', 'ms_to_pan']:
        assert profile_key in pan_sharp._profiles
        profile = pan_sharp._profiles[profile_key]
        assert all([k in profile for k in req_keys])
        assert profile['dtype'] == PanSharpen._working_dtype
        assert common.nan_equals(profile['nodata'], PanSharpen._working_nodata)
        assert profile['num_threads'] == os.cpu_count()


def test_init_res_error(pan_file: Path, ms_file: Path):
    """Test initialisation raises an error when the pan resolution is larger than the multispectral
    resolution.
    """
    with pytest.raises(OrthorityError) as ex:
        _ = PanSharpen(ms_file, pan_file)
    assert 'resolution' in str(ex.value)


def test_init_bounds_error(odm_dem_file: Path, ngi_dem_file: Path):
    """Test initialisation raises an error when pan and multispectral bounds don't overlap."""
    with pytest.raises(OrthorityError) as ex:
        _ = PanSharpen(odm_dem_file, ngi_dem_file)
    assert 'bounds' in str(ex.value)


def _test_pan_sharp_file(
    out_file: Path, ms_file: Path | rio.DatasetReader, ms_indexes: tuple = None
):
    """Test the pan-sharpened ``out_file`` by reprojecting it to the grid of the given
    multispectral ``ms_file`` and comparing.
    """
    with common.OpenRaster(ms_file, 'r') as ms_im, common.OpenRaster(out_file, 'r') as out_im_:
        # prepare parameters for reprojecting output to MS grid
        if (ms_im.crs is None and ms_im.transform.is_identity) or (
            out_im_.crs is None and out_im_.transform.is_identity
        ):
            # MS or output not georeferenced - assume bounds match
            ms_crs = out_im_.crs
            scale = np.array(out_im_.shape) / ms_im.shape
            ms_transform = out_im_.transform * rio.Affine.scale(*scale[::-1])
            ms_win = Window(0, 0, ms_im.width, ms_im.height)
        else:
            # MS and output georeferenced
            ms_crs = ms_im.crs
            ms_transform = ms_im.transform
            # find MS window corresponding to output bounds
            ms_win = common.expand_window_to_grid(ms_im.window(*out_im_.bounds))

        profile = dict(
            crs=ms_crs,
            transform=ms_transform,
            width=ms_im.width,
            height=ms_im.height,
            resampling=Resampling.average,
        )

        # read reprojected output
        with WarpedVRT(out_im_, **profile) as out_im:
            out_array = out_im.read(window=ms_win, masked=True, out_dtype='float32')

        # read MS window corresponding to output bounds
        ms_indexes = ms_indexes or [
            bi + 1 for bi in range(ms_im.count) if ms_im.colorinterp[bi] != ColorInterp.alpha
        ]
        ms_array = ms_im.read(ms_indexes, window=ms_win, masked=True, out_dtype='float32')

    # compare output to MS
    assert np.all(out_array.mask == ms_array.mask)
    abs_err = np.ma.abs(out_array - ms_array)
    assert np.ma.all(abs_err.mean(axis=(1, 2)) < 1)
    assert np.ma.all(abs_err.std(axis=(1, 2)) < 1)


@pytest.mark.parametrize(
    'pan_profile, ms_profile',
    [
        # pan & MS not georeferenced
        (dict(), dict()),
        # pan georeferenced
        (
            dict(
                crs='EPSG:3857', src_transform=(t := rio.Affine(2, 0, 10, 0, -2, 20)), transform=t
            ),
            dict(),
        ),
        # MS georeferenced
        (
            dict(),
            dict(crs='EPSG:3857', src_transform=t, transform=t),
        ),
        # pan & MS georeferenced with matching bounds
        (
            dict(crs='EPSG:3857', src_transform=t, transform=t),
            dict(crs='EPSG:3857', src_transform=(ms_t := t * rio.Affine.scale(4)), transform=ms_t),
        ),
        # pan & MS georeferenced with different bounds
        (
            dict(crs='EPSG:3857', src_transform=t, transform=t, width=1200, height=800),
            dict(
                crs='EPSG:3857',
                src_transform=ms_t,
                transform=ms_t * rio.Affine.translation(50, 50),
                width=342 - 50,
                height=228 - 50,
            ),
        ),
    ],
)
def test_georef(pan_file: Path, ms_file: Path, pan_profile: dict, ms_profile: dict, tmp_path: Path):
    """Test pan-sharpening using WarpedVRT to simulate different combinations of pan & MS
    georeferencing and bounds (all without masks).
    """
    out_file = tmp_path.joinpath('pan_sharp.tif')
    with rio.open(pan_file, 'r') as pan_im_, rio.open(ms_file, 'r') as ms_im_:
        # set profile widths & heights if they are not set already
        pan_profile['width'] = pan_profile.get('width', pan_im_.width)
        pan_profile['height'] = pan_profile.get('height', pan_im_.height)
        ms_profile['width'] = ms_profile.get('width', ms_im_.width)
        ms_profile['height'] = ms_profile.get('height', ms_im_.height)

        # simulate pan & MS georeferencing, and pan-sharpen
        with WarpedVRT(pan_im_, **pan_profile) as pan_im, WarpedVRT(ms_im_, **ms_profile) as ms_im:
            pan_sharp = PanSharpen(pan_im, ms_im)
            pan_sharp.process(out_file, weights=(1, 1, 1), compress='deflate')
            assert out_file.exists()

            # reproject output to MS grid and compare
            _test_pan_sharp_file(out_file, ms_im)


@pytest.mark.parametrize('vrt_profile', [None, dict(nodata=0), dict(add_alpha=True)])
def test_mask(mask_pan_file: Path, mask_ms_file: Path, tmp_path: Path, vrt_profile: dict):
    """Test masking in the pan-sharpened image with pan & MS images that have internal,
    nodata and alpha masks, that mask invalid pixels inside the common bounds.
    """
    with ExitStack() as exit_stack:
        pan_im = exit_stack.enter_context(rio.open(mask_pan_file, 'r'))
        ms_im = exit_stack.enter_context(rio.open(mask_ms_file, 'r'))

        if vrt_profile:
            pan_im = exit_stack.enter_context(WarpedVRT(pan_im, **vrt_profile))
            ms_im = exit_stack.enter_context(WarpedVRT(ms_im, **vrt_profile))

        pan_sharp = PanSharpen(pan_im, ms_im)
        out_file = tmp_path.joinpath('pan_sharp.tif')
        pan_sharp.process(out_file, weights=(1, 1, 1), compress='deflate')
        assert out_file.exists()

        # reproject output to MS grid and compare
        _test_pan_sharp_file(out_file, ms_im)


def test_stats(pan_file: Path, ms_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Test block-by-block ``PanSharpen._get_stats()`` against full band reference values."""
    pan_sharp = PanSharpen(pan_file, ms_file)

    with rio.open(ms_file, 'r') as ms_im, rio.open(pan_file, 'r') as pan_im_:
        # find reference full band stats
        ms_array = ms_im.read(out_dtype='float64')
        with WarpedVRT(
            pan_im_, **pan_sharp._profiles['pan_to_ms'], resampling=Resampling.average
        ) as pan_im:
            pan_array = pan_im.read()
        pan_ms_array = np.concatenate(
            (pan_array.reshape(1, -1), ms_array.reshape(ms_array.shape[0], -1)), axis=0
        )
        ref_means = pan_ms_array.mean(axis=1)
        ref_cov = np.cov(pan_ms_array)

        # find test block-by-block stats
        pan_sharp = PanSharpen(pan_file, ms_file)
        # patch common.block_windows to use smaller block size (forces >> 1 block)
        monkeypatch.setattr(
            common, 'block_windows', partial(common.block_windows, block_shape=(64, 64))
        )
        test_means, test_cov = pan_sharp._get_stats(pan_im_, ms_im, 1, ms_im.indexes, {})

    assert test_means == pytest.approx(ref_means, abs=1e-6)
    assert test_cov == pytest.approx(ref_cov, abs=1e-6)


def test_weights_auto(pan_file: Path, ms_file: Path):
    """Test estimated MS to pan weights returned by ``PanSharpen._get_params()`` against expected
    values.
    """
    pan_sharp = PanSharpen(pan_file, ms_file)
    with rio.open(ms_file, 'r') as ms_im, rio.open(pan_file, 'r') as pan_im:
        means, cov = pan_sharp._get_stats(pan_im, ms_im, 1, ms_im.indexes, {})
    params = pan_sharp._get_params(means, cov, None)

    # assumes pan == sum(MS bands) exactly
    assert params['weights'] == pytest.approx(1 / 3, abs=0.01)
    assert np.sum(params['weights']) == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize('weights', [(1, 1, 1), (1, 2, 3), (-1, 0, 1)])
def test_weights_user(pan_file: Path, ms_file: Path, weights: tuple | None):
    """Test user provided MS to pan weights are clipped, normalised and passed through
    ``PanSharpen._get_params()``.
    """
    pan_sharp = PanSharpen(pan_file, ms_file)
    with rio.open(ms_file, 'r') as ms_im, rio.open(pan_file, 'r') as pan_im:
        means, cov = pan_sharp._get_stats(pan_im, ms_im, 1, ms_im.indexes, {})
    params = pan_sharp._get_params(means, cov, weights)

    weights = np.array(weights).clip(0, None)
    weights = weights / weights.sum()
    assert np.all(params['weights'] == weights)


def test_weights_negative(pan_file: Path, ms_neg_band_file: Path):
    """Test behaviour of ``PanSharpen._get_params()`` MS to pan weights estimation when estimated
    weights contain negative values.
    """
    pan_sharp = PanSharpen(pan_file, ms_neg_band_file)
    with rio.open(ms_neg_band_file, 'r') as ms_im, rio.open(pan_file, 'r') as pan_im:
        means, cov = pan_sharp._get_stats(pan_im, ms_im, 1, ms_im.indexes, {})

    # get weights, testing a warning is raised
    with pytest.warns(OrthorityWarning) as record:
        params = pan_sharp._get_params(means, cov, None)
    assert len(record) == 1
    assert 'Weights contain negative value(s)' in record[0].message.args[0]

    # test weight of negative band has been clipped to 0
    assert params['weights'][-1] == 0

    # test remaining weights have been re-estimated and normalised
    assert params['weights'][:-1] == pytest.approx(1 / 3, abs=0.01)
    assert np.sum(params['weights']) == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize('weights', [(1, 1, 1), (1, 2, 3)])
def test_gs_coeffs(pan_file: Path, ms_file: Path, weights: tuple):
    """Test ``PanSharpen._get_params()`` Gram-Schmidt coefficients against full band reference
    values.
    """
    working_dtype = PanSharpen._working_dtype
    weights = np.array(weights) / np.sum(weights)

    # find reference coefficients using full band approach
    def gs_foward(ms_array: np.array, weights: np.array) -> list[np.array]:
        """Full band forward GS transform from https://patents.google.com/patent/US6011875A/en."""
        gs_pan = weights.dot(ms_array)
        gs_array = np.zeros((ms_array.shape[0] + 1, ms_array.shape[1]), dtype=working_dtype)
        gs_array[0] = gs_pan
        gs_coeffs = []
        means = ms_array.mean(axis=1)

        for bi in range(0, ms_array.shape[0]):
            c = np.cov(ms_array[bi], gs_array[: bi + 1])
            phi = c[0, 1:] / np.diag(c)[1:]
            gs_array[bi + 1] = ms_array[bi] - means[bi] - phi.dot(gs_array[: bi + 1])
            gs_coeffs.append(phi)

        return gs_coeffs

    with rio.open(ms_file, 'r') as ms_im:
        ms_array = ms_im.read(out_dtype=working_dtype)
    ms_array = ms_array.reshape(ms_array.shape[0], -1)
    ref_coeffs = gs_foward(ms_array, weights)

    # find test coefficients using PanSharpen
    pan_sharp = PanSharpen(pan_file, ms_file)
    with rio.open(ms_file, 'r') as ms_im, rio.open(pan_file, 'r') as pan_im:
        means, cov = pan_sharp._get_stats(pan_im, ms_im, 1, ms_im.indexes, {})
    params = pan_sharp._get_params(means, cov, weights)

    for test_coeff, ref_coeff in zip(params['coeffs'], ref_coeffs):
        assert test_coeff == pytest.approx(ref_coeff, abs=1e-6)


@pytest.mark.parametrize('weights', [(1, 1, 1), (1, 2, 3)])
def test_pan_norm(pan_file: Path, ms_file: Path, weights: tuple):
    """Test ``PanSharpen._get_params()`` pan normalisation parameters against full band reference
    values.
    """
    working_dtype = PanSharpen._working_dtype
    weights = np.array(weights) / np.sum(weights)
    pan_sharp = PanSharpen(pan_file, ms_file)

    # find reference parameters using full band approach
    def get_pan_norm(pan_array: np.ndarray, ms_array: np.ndarray, weights: np.ndarray):
        """Return pan normalisation parameters for given arrays."""
        pan_mean, pan_std = pan_array.mean(), pan_array.std()
        pan_sim_array = np.dot(weights, ms_array)
        pan_sim_mean, pan_sim_std = pan_sim_array.mean(), pan_sim_array.std()
        gain = pan_sim_std / pan_std
        bias = pan_sim_mean - (gain * pan_mean)
        return gain, bias

    with rio.open(ms_file, 'r') as ms_im, rio.open(pan_file, 'r') as pan_im_:
        ms_array = ms_im.read(out_dtype=working_dtype)
        ms_array = ms_array.reshape(ms_array.shape[0], -1)

        with WarpedVRT(
            pan_im_, **pan_sharp._profiles['pan_to_ms'], resampling=Resampling.average
        ) as pan_im:
            pan_array = pan_im.read()
            pan_array = pan_array.reshape(1, -1)

        ref_gain, ref_bias = get_pan_norm(pan_array, ms_array, weights)

        # find test parameters using PanSharpen
        means, cov = pan_sharp._get_stats(pan_im_, ms_im, 1, ms_im.indexes, {})
        params = pan_sharp._get_params(means, cov, weights)

    assert params['gain'] == pytest.approx(ref_gain, abs=1e-6)
    assert params['bias'] == pytest.approx(ref_bias, abs=1e-6)


def test_process_pan_index_error(pan_sharpen: PanSharpen, tmp_path: Path):
    """Test ``PanSharpen.process()`` raises an error when ``pan_index`` is invalid."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    with pytest.raises(OrthorityError) as ex:
        pan_sharpen.process(out_file, pan_index=2)
    assert 'Pan index' in str(ex)


@pytest.mark.parametrize('ms_indexes, weights', [((1, 2, 3), (1, 1, 1)), ((3, 2, 1), (1, 1, 1))])
def test_process_ms_indexes(
    pan_file: Path, ms_file: Path, tmp_path: Path, ms_indexes: tuple, weights: tuple
):
    """Test pan-sharpened bands are correctly defined by the ``PanSharpen.process()``
    ``ms_indexes`` argument.
    """
    # note that for this test to pass, ms_indexes and weights must give a weighted sum of indexed
    # MS bands, that equals the plain sum of MS bands (as pan_file==sum(ms_file bands))
    pan_sharp = PanSharpen(pan_file, ms_file)
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharp.process(out_file, ms_indexes=ms_indexes, weights=weights, compress='deflate')
    assert out_file.exists()

    _test_pan_sharp_file(out_file, ms_file, ms_indexes=ms_indexes)


def test_process_ms_indexes_error(pan_sharpen: PanSharpen, tmp_path: Path):
    """Test ``PanSharpen.process()`` raises an error when ``ms_indexes`` is invalid."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    for ms_indexes in [[0], [4]]:
        with pytest.raises(OrthorityError) as ex:
            pan_sharpen.process(out_file, ms_indexes=ms_indexes)
        assert 'Multispectral indexes' in str(ex)


def test_process_weights(pan_file: Path, ms_file: Path, tmp_path: Path):
    """Test the ``PanSharpen.process()`` ``weights`` argument by comparing pan-sharpened images
    with different weightings.
    """
    out_arrays = []
    for wi, weights in enumerate([(1, 2, 3), (3, 2, 1)]):
        pan_sharp = PanSharpen(pan_file, ms_file)
        out_file = tmp_path.joinpath(f'pan_sharp_{wi}.tif')
        pan_sharp.process(out_file, weights=weights, compress='deflate')
        assert out_file.exists()

        with rio.open(out_file, 'r') as out_im:
            out_arrays.append(out_im.read())

    assert np.any(out_arrays[0] != out_arrays[1])


def test_process_weights_error(pan_sharpen: PanSharpen, tmp_path: Path):
    """Test ``PanSharpen.process()`` raises an error when ``weights`` is invalid."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    with pytest.raises(OrthorityError) as ex:
        pan_sharpen.process(out_file, weights=(1, 1))
    assert 'weights' in str(ex) and 'same number' in str(ex)
    with pytest.raises(OrthorityError) as ex:
        pan_sharpen.process(out_file, weights=(1, 1, -1))
    assert 'Weight values' in str(ex)


@pytest.mark.parametrize('interp', [Interp.bilinear, Interp.cubic, Interp.lanczos])
def test_process_interp(pan_sharpen: PanSharpen, tmp_path: Path, interp: Interp):
    """Test the ``PanSharpen.process()`` ``interp`` argument by comparing with an
    ``interp='nearest'`` image.
    """
    ref_file = tmp_path.joinpath('ref.tif')
    pan_sharpen.process(ref_file, compress='deflate', interp=Interp.nearest)
    test_file = tmp_path.joinpath('test.tif')
    pan_sharpen.process(test_file, compress='deflate', interp=interp)

    with rio.open(ref_file, 'r') as ref_im, rio.open(test_file, 'r') as test_im:
        ref_array = ref_im.read(out_dtype='float32')
        test_array = test_im.read(out_dtype='float32')

    # compare
    assert np.any(test_array != ref_array)
    abs_diff = np.abs(test_array - ref_array)
    assert abs_diff.mean() < 5
    assert abs_diff.std() < 5


@pytest.mark.parametrize('driver', Driver)
def test_process_driver(pan_sharpen: PanSharpen, tmp_path: Path, driver: Driver):
    """Test the ``PanSharpen.process()`` ``driver`` argument."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharpen.process(out_file, driver=driver)
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert out_im.driver.lower() == 'gtiff'
        im_struct = out_im.tags(ns='IMAGE_STRUCTURE')
        if driver is Driver.gtiff:
            assert 'LAYOUT' not in im_struct or im_struct['LAYOUT'].lower() != 'cog'
        else:
            assert im_struct['LAYOUT'].lower() == 'cog'


@pytest.mark.parametrize(
    'write_mask, compress',
    [(None, Compress.jpeg), (None, Compress.deflate), (False, None), (True, None)],
)
def test_process_write_mask(
    pan_sharpen: PanSharpen, tmp_path: Path, write_mask: bool, compress: Compress
):
    """Test the ``PanSharpen.process()`` ``write_mask`` argument and its default value interaction
    with compression.
    """
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharpen.process(out_file, compress=compress, write_mask=write_mask)

    if write_mask is None:
        write_mask = True if compress is Compress.jpeg else False
    mask_flag = MaskFlags.per_dataset if write_mask else MaskFlags.nodata

    with rio.open(out_file, 'r') as out_im:
        assert all([mf[0] == mask_flag for mf in out_im.mask_flag_enums])
        assert (
            (out_im.nodata is None)
            if write_mask
            else common.nan_equals(out_im.nodata, common._nodata_vals[out_im.dtypes[0]])
        )


@pytest.mark.parametrize(
    'ms_dtype, out_dtype',
    [('uint8', None), ('float32', None), *[(None, dt) for dt in common._nodata_vals.keys()]],
)
def test_process_dtype(
    pan_file: Path, ms_file: Path, tmp_path: Path, ms_dtype: str, out_dtype: str
):
    """Test the ``PanSharpen.process()`` ``dtype`` argument and its default value interaction with
    the multispectral image dtype.
    """
    with rio.open(ms_file, 'r') as ms_im_:
        profile = dict(width=ms_im_.width, height=ms_im_.height, dtype=ms_dtype)
        with WarpedVRT(ms_im_, **profile) as ms_im:
            pan_sharp = PanSharpen(pan_file, ms_im)
            out_file = tmp_path.joinpath('pan_sharp.tif')
            pan_sharp.process(out_file, dtype=out_dtype)

        out_dtype = out_dtype or ms_im.dtypes[0]

    with rio.open(out_file, 'r') as out_im:
        assert out_im.profile['dtype'] == out_dtype


@pytest.mark.parametrize(
    'compress, dtype', [(None, 'uint8'), (None, 'float32'), *[(c, None) for c in Compress]]
)
def test_process_compress(pan_sharpen: PanSharpen, tmp_path: Path, compress: Compress, dtype: str):
    """Test the ``PanSharpen.process()`` ``compress`` argument and its default value interaction
    with ``dtype``.
    """
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharpen.process(out_file, dtype=dtype, compress=compress)

    if compress is None:
        compress = Compress.jpeg if dtype == 'uint8' else Compress.deflate

    with rio.open(out_file, 'r') as out_im:
        assert compress.value in out_im.profile['compress'].lower()


@pytest.mark.parametrize('build_ovw, driver', [*product([True, False], Driver)])
def test_process_overview(pan_sharpen: PanSharpen, tmp_path: Path, build_ovw: bool, driver: Driver):
    """Test ``PanSharpen.process()`` creates overview(s) according to the ``build_ovw`` value."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharpen.process(out_file, driver=driver, build_ovw=build_ovw)
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert min(out_im.shape) >= 512
        assert len(out_im.overviews(1)) > 0 if build_ovw else len(out_im.overviews(1)) == 0


@pytest.mark.parametrize('driver', Driver)
def test_process_colorinterp(
    pan_file: Path, ms_neg_band_file: Path, tmp_path: Path, driver: Driver
):
    """Test ``PanSharpen.process()`` copies ``colorinterp`` from source to pan-sharpened image."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharp = PanSharpen(pan_file, ms_neg_band_file)
    pan_sharp.process(out_file)
    assert out_file.exists()

    with rio.open(ms_neg_band_file, 'r') as ms_im, rio.open(out_file, 'r') as out_im:
        assert out_im.colorinterp == ms_im.colorinterp


def test_process_creation_options(pan_sharpen: PanSharpen, tmp_path: Path):
    """Test ``PanSharpen.process()`` configures the pan-sharpened image with ``creation_options``."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharpen.process(
        out_file, creation_options=dict(tiled=True, compress='jpeg', jpeg_quality=50)
    )
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert out_im.profile['tiled'] == True
        assert out_im.profile['compress'].lower() == 'jpeg'
        assert out_im.tags(ns='IMAGE_STRUCTURE')['JPEG_QUALITY'] == '50'


def test_process_progress(pan_sharpen: PanSharpen, tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test ``PanSharpen.process()`` progress bar display."""
    # default bars
    out_file = tmp_path.joinpath('pan_sharp.tif')
    pan_sharpen.process(out_file, progress=True)
    cap = capsys.readouterr()
    progress_words = ['Statistics', 'Sharpening', 'blocks', '100%']
    assert all([w in cap.err for w in progress_words])

    # no bars
    pan_sharpen.process(out_file, overwrite=True, progress=False)
    cap = capsys.readouterr()
    assert not any([w in cap.err for w in progress_words])

    # custom bars
    descs = ['bar1', 'bar2']
    pan_sharpen.process(out_file, overwrite=True, progress=[dict(desc=d) for d in descs])
    cap = capsys.readouterr()
    assert all([w in cap.err for w in descs])
