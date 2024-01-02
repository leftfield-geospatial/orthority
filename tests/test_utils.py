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

from io import TextIOWrapper
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
import rasterio as rio

from orthority import utils


@pytest.mark.parametrize(
    'filename, exp_res',
    [
        ('folder/file.tif', False),
        ('C:/folder/file.tif', False),
        ('C:\\folder\\file.tif', False),
        ('https://do.main/file.tif', True),
        ('ftp://do.main/file.tif', True),
        ('file:///folder/file.tif', True),
        (Path('folder/file.tif'), False),
    ],
)
def test_is_url(filename: str | Path, exp_res: bool):
    """Test is_url() gives expected results."""
    assert utils.is_url(filename) == exp_res


@pytest.mark.parametrize(
    'filename, kwargs',
    [('ngi_oty_int_param_file', dict(newline=None)), ('ngi_oty_int_param_url', dict())],
)
def test_open_text(filename: str, kwargs: dict, request: pytest.FixtureRequest):
    """Test open_text() opens an existing file path or URL successfully."""
    filename: str | Path = request.getfixturevalue(filename)
    with utils.open_text(filename, **kwargs) as file_obj:
        assert isinstance(file_obj, TextIOWrapper)
        assert not file_obj.closed
        assert file_obj.encoding is not None
    assert file_obj.closed


@pytest.mark.parametrize(
    'filename',
    ['unknown.tif', 'https://un.known/unknown.tif', 'https://github.com/unknown/unknown.tif'],
)
def test_open_text_error(filename: str):
    """Test open_text() raises an error when the file path or URL doesn't exist."""
    with pytest.raises((FileNotFoundError, URLError, HTTPError)):
        _ = utils.open_text(filename)


def test_text_ctx(ngi_oty_int_param_file: Path):
    """Test text_ctx() re-enters and closes as expected."""
    with utils.text_ctx(ngi_oty_int_param_file, newline=None) as file_obj:
        with utils.text_ctx(file_obj) as file_obj2:
            assert not file_obj2.closed
        assert not file_obj.closed
    assert file_obj.closed


def test_raster_ctx(ngi_image_file: Path):
    """Test raster_ctx() re-enters and closes as expected."""
    with utils.raster_ctx(ngi_image_file) as ds:
        with utils.raster_ctx(ds) as ds2:
            assert isinstance(ds2, rio.DatasetReader)
            assert not ds2.closed
        assert not ds.closed
    assert ds.closed
