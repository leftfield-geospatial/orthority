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
from io import TextIOWrapper
from pathlib import Path

import fsspec
import numpy as np
import pytest
import rasterio as rio

from orthority import utils


@pytest.mark.parametrize('file', ['odm_image_file', 'odm_image_url'])
def test_get_filename(file: str, request: pytest.FixtureRequest):
    """Test get_filename() returns with different ``file`` objects."""
    file = request.getfixturevalue(file)
    exp_val = Path(file).name
    assert utils.get_filename(file) == exp_val
    ofile = fsspec.open(file, 'rb')
    assert utils.get_filename(ofile) == exp_val
    with utils.Open(file, 'rb') as f:
        assert utils.get_filename(f) == exp_val
    with utils.OpenRaster(file, 'r') as f:
        assert utils.get_filename(f) == exp_val


@pytest.mark.parametrize('file', ['odm_image_file', 'odm_image_url'])
def test_join_ofile(file: str, request: pytest.FixtureRequest):
    """Test join_ofile() returns valid OpenFile instances for existing paths / URIs, with and
    without trailing slashes on the base path / URI.
    """
    file = str(request.getfixturevalue(file))
    parts = file.replace('\\', '/').split('/')
    for pidx in [1, 2]:
        _base_path = '/'.join(parts[:-pidx])
        rel_path = '/'.join(parts[-pidx:])
        for base_path in [_base_path, _base_path + '/']:
            base_ofile = fsspec.open(base_path)
            join_ofile = utils.join_ofile(base_ofile, rel_path)
            assert Path(join_ofile.path).name == Path(file).name
            assert join_ofile.fs.exists(join_ofile.path)


@pytest.mark.parametrize('raster_file', ['odm_image_file', 'odm_image_url'])
def test_open_raster_read(raster_file: str, request: pytest.FixtureRequest):
    """Test OpenRaster in 'r' mode with path / URI, fsspec OpenFile and dataset objects."""
    raster_file = request.getfixturevalue(raster_file)

    def _test_open_raster_read(_raster_file, test_closed: bool = True):
        """Test OpenRaster for the given ``_raster_file`` object."""
        with utils.OpenRaster(_raster_file, 'r') as _ds:
            assert isinstance(_ds, rio.DatasetReader)
            assert not _ds.closed
            assert _ds.filename == Path(raster_file).name
            _ = _ds.read(1, window=_ds.block_window(1, 0, 0))
        if test_closed:
            assert _ds.closed

    _test_open_raster_read(raster_file)
    _test_open_raster_read(fsspec.open(raster_file, 'rb'))
    with utils.OpenRaster(raster_file, 'r') as ds:
        _test_open_raster_read(ds, test_closed=False)
    assert ds.closed


def test_open_raster_write(tmp_path: Path):
    """Test OpenRaster in 'w' mode with path / URI, fsspec OpenFile and dataset objects."""
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
        with utils.OpenRaster(_raster_file, 'w', **profile) as _ds:
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
        with utils.OpenRaster(raster_file, 'w', **profile) as ds:
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
        with utils.OpenRaster(_raster_file, 'w', overwrite=overwrite, **profile) as _ds:
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
        with utils.OpenRaster(raster_file, 'r'):
            pass
    ofile = fsspec.open(raster_file, 'rb')
    with pytest.raises(FileNotFoundError):
        with utils.OpenRaster(ofile, 'r'):
            pass


@pytest.mark.parametrize(
    'file, kwargs',
    [
        ('ngi_oty_int_param_file', dict(newline=None, encoding='utf8')),
        ('ngi_oty_int_param_url', dict()),
    ],
)
def test_open_read(file: str, kwargs: dict, request: pytest.FixtureRequest):
    """Test Open in 'rt' mode with path / URI, fsspec OpenFile and file objects."""
    file = request.getfixturevalue(file)

    def _test_open_read(_file, test_closed: bool = True, **kwargs):
        """Test Open for the given ``_file`` object."""
        with utils.Open(_file, 'rt', **kwargs) as _f:
            assert isinstance(_f, TextIOWrapper)
            assert not _f.closed
            assert _f.filename == Path(file).name
            _ = _f.read()
        if test_closed:
            assert _f.closed

    _test_open_read(file)
    _test_open_read(fsspec.open(file, 'rt'))
    with utils.Open(file, 'rt', **kwargs) as f:
        _test_open_read(f, test_closed=False, **kwargs)
    assert f.closed


def test_open_write(tmp_path: Path):
    """Test Open in 'w' mode with path / URI, fsspec OpenFile and file objects."""
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
        with utils.Open(_file, 'wt', **kwargs) as _f:
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
        with utils.Open(file, 'wt', **kwargs) as f:
            _test_open_write(f, test_closed=False, **kwargs)
        assert f.closed


def test_open_overwrite(tmp_path: Path):
    """Test the Open ``overwrite`` argument."""
    file = tmp_path.joinpath('test_open_overwrite.txt')
    file.touch()

    def _test_open_overwrite(_file, overwrite: bool):
        with utils.Open(_file, 'wt', overwrite=overwrite) as _f:
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
        with utils.Open(file, 'rt'):
            pass
    ofile = fsspec.open(file, 'rt')
    with pytest.raises(FileNotFoundError):
        with utils.Open(ofile, 'rt'):
            pass
