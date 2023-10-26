# Copyright The Orthority Contributors.
#
# This file is part of Orthority.
#
# Orthority is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Orthority is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Orthority.  If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path

from orthority.exif import Exif


def test_odm_image(odm_image_file: Path):
    """Test reading an image with valid EXIF & XMP tags."""
    exif = Exif(odm_image_file)
    assert exif.filename == odm_image_file
    for attr in [
        'make',
        'model',
        'serial',
        'focal_len',
        'focal_len_35',
        'im_size',
        'tag_im_size',
        'lla',
        'rpy',
        'dewarp',
        'orientation',
    ]:
        assert getattr(exif, attr) is not None


def test_exif_image(exif_image_file: Path):
    """Test reading an image with EXIF tags including sensor size, and no XMP tags."""
    exif = Exif(exif_image_file)
    assert exif.filename == exif_image_file
    for attr in [
        'make',
        'model',
        'serial',
        'focal_len',
        'focal_len_35',
        'im_size',
        'tag_im_size',
        'sensor_size',
        'lla',
        'orientation',
    ]:
        assert getattr(exif, attr) is not None

    assert exif.rpy is None


def test_ngi_image(ngi_image_file: Path):
    """Test reading an image with no EXIF / XMP tags."""
    exif = Exif(ngi_image_file)
    assert exif.filename == ngi_image_file
    assert exif.im_size is not None
    for attr in [
        'make',
        'model',
        'serial',
        'focal_len',
        'focal_len_35',
        'tag_im_size',
        'lla',
        'rpy',
        'sensor_size',
        'dewarp',
        'orientation',
    ]:
        assert getattr(exif, attr) is None
