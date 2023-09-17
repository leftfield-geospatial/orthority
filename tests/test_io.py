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
import json
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pytest
import rasterio as rio
from simple_ortho import io
from simple_ortho.utils import validate_collection
from simple_ortho.enums import CameraType, Interp
from simple_ortho.errors import ParamFileError
from tests.conftest import oty_to_osfm_int_param


def test_rw_oty_int_param(mult_int_param_dict: Dict, tmp_path: Path):
    """ Test interior parameter read / write from / to orthority yaml format. """
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, mult_int_param_dict)
    test_dict = io.read_oty_int_param(filename)
    assert test_dict == mult_int_param_dict


@pytest.mark.parametrize('missing_key', ['focal_len', 'sensor_size'])
def test_read_oty_int_param_missing_error(pinhole_int_param_dict: Dict, missing_key: str, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when required keys are missing. """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params.pop(missing_key)
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_oty_int_param(filename)
    assert missing_key in str(ex)


def test_read_oty_int_param_distortion_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when unknown keys are present. """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params['other'] = 0.
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_oty_int_param(filename)
    assert 'other' in str(ex)


def test_read_oty_int_param_cam_type_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error the camera type is unknown. """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params['cam_type'] = Interp.cubic
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_oty_int_param(filename)
    assert 'camera type' in str(ex)


@pytest.mark.parametrize('filename', ['osfm_int_param_file', 'odm_int_param_file'])
def test_read_osfm_int_param(filename: str, mult_int_param_dict: Dict, request: pytest.FixtureRequest):
    """ Test reading interior parameters from ODM / OpenSfM format files. """
    filename: Path = request.getfixturevalue(filename)
    test_dict = io.read_osfm_int_param(filename)
    assert test_dict == mult_int_param_dict


@pytest.mark.parametrize('missing_key', ['focal_x', 'focal_y'])
def test_read_osfm_int_param_missing_error(pinhole_int_param_dict: Dict, missing_key: str, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when required keys are missing. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params.pop(missing_key)
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert missing_key in str(ex)


def test_read_osfm_int_param_distortion_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when unknown keys are present. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['other'] = 0.
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert 'other' in str(ex)


def test_read_osfm_int_param_proj_type_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error the projection type is unsupported. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['projection_type'] = 'other'
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert 'projection type' in str(ex)


def test_read_exif_int_param_dewarp(odm_image_file: Path):
    """ Testing reading EXIF / XMP tag interior parameters from an image with the `DewarpData` XMP tag. """
    int_param_dict = io.read_exif_int_param(odm_image_file)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'brown'
    assert {'k1', 'k2', 'p1', 'p2', 'k3'}.issubset(int_params.keys())


def test_read_exif_int_param_no_dewarp(exif_image_file: Path):
    """ Testing reading EXIF / XMP tag interior parameters from an image without the `DewarpData` XMP tag. """
    int_param_dict = io.read_exif_int_param(exif_image_file)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'pinhole'


def test_read_exif_int_param_error(ngi_image_file: Path):
    """ Testing reading EXIF tag interior parameters from a non EXIF image raises an error. """
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_exif_int_param(ngi_image_file)
    assert 'focal length' in str(ex)


# TODO:
# - Interior:
#   - Test reading different formats is done correctly
#   - Test error conditions reported meaningfully (missing keys?)
#   - Have a test int-param dict / fixture
#   - Fixtures that are the above written to oty, cameras.json & reconstruction.json files
#   - For now, I think just use existing test ngi/odm images for testing exif.  It is not complete, but seems overkill
#     to make code for generating exif test data.
#   - Can there be a single set of interior params that is used for testing io and testing camera & ortho?  We would
#   probably need to change how camera is initialised.
#   - Writing oty & then reading oty params.
#   - We still haven't done checking of distortion params for different cam types...  Use a schema with validate_collection?
#   - Test multiple camera config & legacy format(s)
# Exterior & readers:
#   - Maybe have an exterior param dict fixture, that is used to create different formats, and test reading against.
#   - The "create different formats" is not that trivial though...
#   CSV
#   - Different angle / position formats
#   - With / without header, different delimiters, additional columns, with / without camera id
#   - With / without proj CRS file
#   - Error conditions
