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

from contextlib import contextmanager
from io import BytesIO, TextIOWrapper
from pathlib import Path

import fsspec
import numpy as np
import pytest
import rasterio as rio
from rasterio.enums import ColorInterp

from orthority import common
from orthority.enums import Compress
from tests.conftest import checkerboard, create_profile


@pytest.mark.parametrize('file', ['odm_image_file', 'odm_image_url'])
def test_get_filename(file: str, request: pytest.FixtureRequest):
    """Test get_filename() returns with different ``file`` objects."""
    file = request.getfixturevalue(file)
    exp_val = Path(file).name
    assert common.get_filename(file) == exp_val
    ofile = fsspec.open(file, 'rb')
    assert common.get_filename(ofile) == exp_val
    with common.Open(file, 'rb') as f:
        assert common.get_filename(f) == exp_val
    with common.OpenRaster(file, 'r') as f:
        assert common.get_filename(f) == exp_val


@pytest.mark.parametrize('file', ['odm_image_file', 'odm_image_url'])
def test_join_ofile(file: str, request: pytest.FixtureRequest):
    """Test join_ofile() returns valid OpenFile instances for existing paths / URIs,
    with different base path / URI types, and with and without trailing slashes on the base path
    / URI.
    """
    file = request.getfixturevalue(file)
    ofile = fsspec.open(file)
    parts = str(file).replace('\\', '/').split('/')

    path_types = [str, fsspec.open, Path] if isinstance(file, Path) else [str, fsspec.open]
    for path_type in path_types:
        for pidx in [1, 2]:
            _base_path = '/'.join(parts[:-pidx])
            rel_path = '/'.join(parts[-pidx:])

            for base_path in map(path_type, [_base_path, _base_path + '/']):
                join_ofile = common.join_ofile(base_path, rel_path)
                assert isinstance(join_ofile, fsspec.core.OpenFile)
                assert join_ofile.path == ofile.path
                assert join_ofile.fs.exists(join_ofile.path)


@pytest.mark.parametrize('raster_file', ['odm_image_file', 'odm_image_url'])
def test_open_raster_read(raster_file: str, request: pytest.FixtureRequest):
    """Test OpenRaster in ``'r'`` mode with path / URI, fsspec OpenFile and dataset objects."""
    raster_file = request.getfixturevalue(raster_file)

    def _test_open_raster_read(_raster_file, test_closed: bool = True):
        """Test OpenRaster for the given ``_raster_file`` object."""
        with common.OpenRaster(_raster_file, 'r') as _ds:
            assert isinstance(_ds, rio.DatasetReader)
            assert not _ds.closed
            assert _ds.filename == Path(raster_file).name
            _ = _ds.read(1, window=_ds.block_window(1, 0, 0))
        if test_closed:
            assert _ds.closed

    _test_open_raster_read(raster_file)
    _test_open_raster_read(fsspec.open(raster_file, 'rb'))
    with common.OpenRaster(raster_file, 'r') as ds:
        _test_open_raster_read(ds, test_closed=False)
    assert ds.closed


def test_open_raster_write(tmp_path: Path):
    """Test OpenRaster in ``'w'`` mode with path / URI, fsspec OpenFile and dataset objects."""
    raster_file = tmp_path.joinpath('test_open_raster_write.tif')
    profile = rio.default_gtiff_profile
    array = np.ones((256, 256), dtype=profile['dtype'])
    profile.update(width=array.shape[1], height=array.shape[0], count=1)

    @contextmanager
    def _test_temp_file(filename: Path):
        """Test ``filename`` exists on exit, then delete."""
        try:
            yield filename
        finally:
            assert filename.exists()
            filename.unlink()

    def _test_open_raster_write(_raster_file, test_closed: bool = True):
        """Test OpenRaster for the given ``_raster_file`` object."""
        with common.OpenRaster(_raster_file, 'w', **profile) as _ds:
            assert isinstance(_ds, rio.io.DatasetWriter)
            assert not _ds.closed
            assert _ds.filename == Path(raster_file).name
            _ds.write(array, indexes=1)
        if test_closed:
            assert _ds.closed

    with _test_temp_file(raster_file):
        _test_open_raster_write(raster_file)
    with _test_temp_file(raster_file):
        _test_open_raster_write(fsspec.open(str(raster_file), 'wb'))
    with _test_temp_file(raster_file):
        with common.OpenRaster(raster_file, 'w', **profile) as ds:
            _test_open_raster_write(ds, test_closed=False)
        assert ds.closed


def test_open_raster_overwrite(tmp_path: Path):
    """Test the OpenRaster ``overwrite`` argument."""
    raster_file = tmp_path.joinpath('test_open_raster_overwrite.tif')
    profile = rio.default_gtiff_profile
    array = np.ones((256, 256), dtype=profile['dtype'])
    profile.update(width=array.shape[1], height=array.shape[0], count=1)
    raster_file.touch()

    def _test_open_raster_overwrite(_raster_file, overwrite: bool):
        with common.OpenRaster(_raster_file, 'w', overwrite=overwrite, **profile) as _ds:
            _ds.write(array, indexes=1)

    # test overwriting an existing file with overwrite=True
    _test_open_raster_overwrite(raster_file, overwrite=True)
    _test_open_raster_overwrite(fsspec.open(str(raster_file), 'wb'), overwrite=True)

    # test writing to an existing file with overwrite=False raises FileExistsError
    with pytest.raises(FileExistsError) as ex:
        _test_open_raster_overwrite(raster_file, overwrite=False)
    assert raster_file.name in str(ex.value)
    with pytest.raises(FileExistsError) as ex:
        _test_open_raster_overwrite(fsspec.open(str(raster_file), 'wb'), overwrite=False)
    assert raster_file.name in str(ex.value)


@pytest.mark.parametrize(
    'raster_file',
    ['unknown/unknown.tif', 'https://un.known/unknown.tif', 'https://github.com/unknown.tif'],
)
def test_open_raster_not_found_error(raster_file: str):
    """Test OpenRaster raises a FileNotFoundError error with non-existing file path / URIs."""
    with pytest.raises(FileNotFoundError):
        with common.OpenRaster(raster_file, 'r'):
            pass
    ofile = fsspec.open(raster_file, 'rb')
    with pytest.raises(FileNotFoundError):
        with common.OpenRaster(ofile, 'r'):
            pass


@pytest.mark.parametrize(
    'file, kwargs',
    [
        ('ngi_oty_int_param_file', dict(newline=None, encoding='utf8')),
        ('ngi_oty_int_param_url', dict()),
    ],
)
def test_open_read(file: str, kwargs: dict, request: pytest.FixtureRequest):
    """Test Open in ``'rt'`` mode with path / URI, fsspec OpenFile and file objects."""
    file = request.getfixturevalue(file)

    def _test_open_read(_file, test_closed: bool = True, **kwargs):
        """Test Open for the given ``_file`` object."""
        with common.Open(_file, 'rt', **kwargs) as _f:
            assert isinstance(_f, TextIOWrapper)
            assert not _f.closed
            assert _f.filename == Path(file).name
            _ = _f.read()
        if test_closed:
            assert _f.closed

    _test_open_read(file)
    _test_open_read(fsspec.open(file, 'rt'))
    with common.Open(file, 'rt', **kwargs) as f:
        _test_open_read(f, test_closed=False, **kwargs)
    assert f.closed


def test_open_write(tmp_path: Path):
    """Test Open in ``'w'`` mode with path / URI, fsspec OpenFile and file objects."""
    file = tmp_path.joinpath('test_open_write.txt')

    @contextmanager
    def _test_temp_file(filename: Path):
        """Test ``filename`` exists on exit, then delete."""
        try:
            yield filename
        finally:
            assert filename.exists()
            assert filename.read_text()
            filename.unlink()

    def _test_open_write(_file, test_closed: bool = True, **kwargs):
        """Test Open for the given ``_file`` object."""
        with common.Open(_file, 'wt', **kwargs) as _f:
            assert isinstance(_f, TextIOWrapper)
            assert not _f.closed
            assert _f.filename == Path(file).name
            _f.write('test')
        if test_closed:
            assert _f.closed

    kwargs = dict(newline=None, encoding='utf8')
    with _test_temp_file(file):
        _test_open_write(file, **kwargs)
    with _test_temp_file(file):
        _test_open_write(fsspec.open(str(file), 'wt'), **kwargs)
    with _test_temp_file(file):
        with common.Open(file, 'wt', **kwargs) as f:
            _test_open_write(f, test_closed=False, **kwargs)
        assert f.closed


def test_open_overwrite(tmp_path: Path):
    """Test the Open ``overwrite`` argument."""
    file = tmp_path.joinpath('test_open_overwrite.txt')
    file.touch()

    def _test_open_overwrite(_file, overwrite: bool):
        with common.Open(_file, 'wt', overwrite=overwrite) as _f:
            _f.write('test')

    # test overwriting an existing file with overwrite=True
    _test_open_overwrite(file, overwrite=True)
    _test_open_overwrite(fsspec.open(str(file), 'wt'), overwrite=True)

    # test writing to an existing file with overwrite=False raises FileExistsError
    with pytest.raises(FileExistsError) as ex:
        _test_open_overwrite(file, overwrite=False)
    assert file.name in str(ex.value)
    with pytest.raises(FileExistsError) as ex:
        _test_open_overwrite(fsspec.open(str(file), 'wt'), overwrite=False)
    assert file.name in str(ex.value)


@pytest.mark.parametrize(
    'file',
    ['unknown/unknown.txt', 'https://un.known/unknown.txt', 'https://github.com/unknown.txt'],
)
def test_open_not_found_error(file: str):
    """Test Open raises a FileNotFoundError error with non-existing file path / URIs."""
    with pytest.raises(FileNotFoundError):
        with common.Open(file, 'rt'):
            pass
    ofile = fsspec.open(file, 'rt')
    with pytest.raises(FileNotFoundError):
        with common.Open(ofile, 'rt'):
            pass


def test_create_profile_non_config_items():
    """Test create_profile() non-configurable items."""
    profile, write_mask = common.create_profile(dtype='uint8')

    exp_profile = {
        'driver': 'GTiff',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'bigtiff': 'if_safer',
    }
    for k, v in exp_profile.items():
        assert profile[k] == v


@pytest.mark.parametrize(
    'dtype, compress, exp_value',
    [
        # compress defaults to 'jpeg' if dtype is uint8, and to 'deflate' otherwise
        ('uint8', None, 'jpeg'),
        ('uint16', None, 'deflate'),
        # compress is copied through as is when provided
        ('uint8', 'deflate', 'deflate'),
        ('uint16', 'jpeg', 'jpeg'),
    ],
)
def test_create_profile_compress(dtype: str, compress: str, exp_value: str):
    """Test create_profile() ``compress`` configuration."""
    profile, write_mask = common.create_profile(dtype=dtype, compress=compress)

    assert profile['dtype'] == dtype
    assert profile['compress'] == exp_value


@pytest.mark.parametrize(
    'compress, colorinterp, exp_values',
    [
        # jpeg compression with any 3 band colorinterp should give 'pixel' / 'ycbcr'
        ('jpeg', [ColorInterp.red, ColorInterp.green, ColorInterp.blue], ('pixel', 'ycbcr')),
        ('jpeg', [ColorInterp.undefined] * 3, ('pixel', 'ycbcr')),
        # any compression with rgb or rgb* colorinterp should give 'band' / 'rgb'
        (
            'jpeg',
            [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha],
            ('band', 'rgb'),
        ),
        ('deflate', [ColorInterp.red, ColorInterp.green, ColorInterp.blue], ('band', 'rgb')),
        (
            'deflate',
            [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined],
            ('band', 'rgb'),
        ),
        # any other configuration should give 'band' / None
        ('jpeg', [ColorInterp.undefined] * 4, ('band', None)),
        ('deflate', [ColorInterp.undefined] * 3, ('band', None)),
    ],
)
def test_create_profile_interleave_photometric(
    compress: str, colorinterp: list[ColorInterp], exp_values: tuple
):
    """Test create_profile() ``interleave`` / ``photometric`` configuration."""
    profile, write_mask = common.create_profile(
        dtype='uint8', compress=compress, colorinterp=colorinterp
    )

    assert profile['dtype'] == 'uint8'
    assert profile['compress'] == compress
    assert profile['interleave'] == exp_values[0]
    assert profile['photometric'] == exp_values[1]


def test_create_profile_12bit_jpeg():
    """Test create_profile() correctly configures a 12bit jpeg profile."""
    # Note: depending on how rasterio is built, it may or may not support reading/writing 12 bit
    # jpeg compression.  This test just checks the profile is correct.
    profile, write_mask = common.create_profile(dtype='uint16', compress=Compress.jpeg)

    assert write_mask
    assert profile['dtype'] == 'uint16'
    assert profile['compress'] == 'jpeg'
    assert 'nbits' in profile and profile['nbits'] == 12


@pytest.mark.parametrize(
    'dtype, write_mask, exp_values',
    [
        # with a uint8 dtype (i.e. jpeg compression) write_mask should default to True and nodata
        # to None
        ('uint8', None, (True, None)),
        # with other dtypes write_mask should default to False and nodata to the dtype min value /
        # nan
        ('uint16', None, (False, 0)),
        ('int16', None, (False, np.iinfo('int16').min)),
        # when write_mask is False, it should be copied through as is, and nodata set to the
        # dtype min value / nan
        ('uint8', False, (False, 0)),
        ('float32', False, (False, float('nan'))),
        ('float64', False, (False, float('nan'))),
        # when write_mask is True, it should be copied through as is, and nodata set to None
        ('uint8', True, (True, None)),
    ],
)
def test_create_profile_write_mask_nodata(dtype: str, write_mask: bool | None, exp_values: tuple):
    """Test create_profile() correctly sets ``write_mask`` and ``nodata``."""
    profile, write_mask = common.create_profile(dtype=dtype, write_mask=write_mask)

    assert write_mask is exp_values[0]
    if profile['nodata'] is None or exp_values[1] is None:
        assert profile['nodata'] is None and exp_values[1] is None
    else:
        assert common.nan_equals(profile['nodata'], exp_values[1])


@pytest.mark.parametrize(
    'src_dtype, dst_dtype',
    [
        ('uint16', 'uint8'),
        ('uint16', 'int16'),
        ('uint16', 'float32'),
        ('float32', 'uint8'),
        ('float32', 'uint16'),
        ('float32', 'int16'),
        ('float32', 'int32'),
        ('float32', 'float32'),
        ('float32', 'float64'),
        ('float64', 'float32'),
        ('float64', 'float64'),
    ],
)
def test_convert_array_dtype(src_dtype: str, dst_dtype: str):
    """Test convert_array_dtype() conversion with combinations covering rounding and clipping
    (with and w/o type promotion).
    """
    src_info = np.iinfo(src_dtype) if np.issubdtype(src_dtype, np.integer) else np.finfo(src_dtype)
    dst_info = np.iinfo(dst_dtype) if np.issubdtype(dst_dtype, np.integer) else np.finfo(dst_dtype)

    # create array that spans the src_dtype range & includes decimals
    array = np.geomspace(1, src_info.max, 50, dtype=src_dtype).reshape(5, 10)
    if src_info.min != 0:
        array = np.concatenate(
            (array, np.geomspace(-1, src_info.min, 50, dtype=src_dtype).reshape(5, 10))
        )

    # convert to dtype
    test_array = common.convert_array_dtype(array, dst_dtype)

    # create rounded & clipped array to test against
    ref_array = array
    if np.issubdtype(dst_dtype, np.integer):
        # promote dtype to clip correctly
        ref_array = ref_array.astype(np.promote_types(array.dtype, dst_dtype))
        ref_array = np.clip(np.round(ref_array), dst_info.min, dst_info.max)
    elif np.issubdtype(src_dtype, np.floating):
        # don't clip float but set out of range vals to +-inf (as np.astype does)
        ref_array[ref_array < dst_info.min] = float('-inf')
        ref_array[ref_array > dst_info.max] = float('inf')
        assert np.any(ref_array % 1 != 0)  # check contains decimals

    assert test_array.dtype == dst_dtype
    # use approx test for case of (expected) precision loss e.g. float64->float32
    assert test_array == pytest.approx(ref_array, rel=1e-6)


def test_build_overviews():
    """Test build_overviews() builds overviews for an open in-memory dataset."""
    # create in-memory dataset
    array = checkerboard((768, 1024)).astype('uint8')
    array = np.stack((array,) * 3, axis=0)
    profile = create_profile(array)
    buf = BytesIO()
    with rio.open(buf, 'w', driver='GTiff', **profile) as im:
        im.write(array)
        # build overviews
        common.build_overviews(im)

    with rio.open(buf, 'r') as im:
        assert len(im.overviews(1)) > 0
