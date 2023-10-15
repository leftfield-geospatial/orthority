"""
   Copyright 2023 Dugal Harris - dugalh@gmail.com

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
from pathlib import Path

from simple_ortho.exif import Exif


def test_odm_image(odm_image_file: Path):
    """ Test reading an image with valid EXIF & XMP tags. """
    exif = Exif(odm_image_file)
    assert exif.filename == odm_image_file
    for attr in [
        'make', 'model', 'serial', 'focal_len', 'focal_len_35', 'im_size', 'tag_im_size', 'lla', 'rpy', 'dewarp',
        'orientation'
    ]:
        assert getattr(exif, attr) is not None


def test_exif_image(exif_image_file: Path):
    """ Test reading an image with EXIF tags including sensor size, and no XMP tags. """
    exif = Exif(exif_image_file)
    assert exif.filename == exif_image_file
    for attr in [
        'make', 'model', 'serial', 'focal_len', 'focal_len_35', 'im_size', 'tag_im_size', 'sensor_size', 'lla',
        'orientation'
    ]:
        assert getattr(exif, attr) is not None

    assert exif.rpy is None


def test_ngi_image(ngi_image_file: Path):
    """ Test reading an image with no EXIF / XMP tags. """
    exif = Exif(ngi_image_file)
    assert exif.filename == ngi_image_file
    assert exif.im_size is not None
    for attr in [
        'make', 'model', 'serial', 'focal_len', 'focal_len_35', 'tag_im_size', 'lla', 'rpy', 'sensor_size', 'dewarp',
        'orientation'
    ]:
        assert getattr(exif, attr) is None
