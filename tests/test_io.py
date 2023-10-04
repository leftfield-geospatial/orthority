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
import csv
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import cv2
import numpy as np
import pytest
import rasterio as rio
from rasterio.warp import transform

from simple_ortho import io
from simple_ortho.camera import Camera
from simple_ortho.enums import CameraType, Interp, CsvFormat
from simple_ortho.errors import ParamFileError, CrsMissingError
from tests.conftest import oty_to_osfm_int_param


def _validate_int_param_dict(int_param_dict: Dict):
    """ Basic validation of an internal parameter dictionary. """
    for cam_id, int_params in int_param_dict.items():
        assert isinstance(cam_id, str) or cam_id is None
        req_keys = {'cam_type', 'im_size', 'focal_len', 'sensor_size'}
        assert set(int_params.keys()).issuperset(req_keys)
        cam_type = CameraType(int_params['cam_type'])
        assert len(int_params['im_size']) == 2 and all([isinstance(dim, int) for dim in int_params['im_size']])
        assert len(int_params['sensor_size']) == 2 and all(
            [isinstance(dim, float) for dim in int_params['sensor_size']]
        )
        assert isinstance(int_params['focal_len'], float) or (
            len(int_params['focal_len']) == 2 and all([isinstance(f, float) for f in int_params['focal_len']])
        )
        optional_keys = set(int_params.keys()).difference(req_keys)
        assert set(optional_keys).issubset(io._optional_schema[cam_type])


def _validate_ext_param_dict(ext_param_dict: Dict, cameras: List[str]=None):
    """ Basic validation of an external parameter dictionary. """
    for filename, ext_params in ext_param_dict.items():
        assert set(ext_params.keys()) == {'opk', 'xyz', 'camera'}
        opk, xyz = np.array(ext_params['opk']), np.array(ext_params['xyz'])
        assert len(opk) == 3 and len(xyz) == 3
        # rough check for radians
        assert all(np.abs(opk) <= 2 * np.pi) and any(opk != 0.)
        # rough check for not latitude, longitude, & altitude > 0
        assert all(xyz != 0) and np.abs(xyz[0]) > 180. and np.abs(xyz[1]) > 90. and xyz[2] > 0
        if cameras:
            assert ext_params['camera'] in cameras


def test_rw_oty_int_param(mult_int_param_dict: Dict, tmp_path: Path):
    """ Test interior parameter read / write from / to orthority yaml format. """
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, mult_int_param_dict)
    test_dict = io.read_oty_int_param(filename)
    assert test_dict == mult_int_param_dict


@pytest.mark.parametrize('missing_key', ['focal_len', 'sensor_size', 'im_size'])
def test_read_oty_int_param_missing_error(pinhole_int_param_dict: Dict, missing_key: str, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when required keys are missing. """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params.pop(missing_key)
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_oty_int_param(filename)
    assert missing_key in str(ex)


def test_read_oty_int_param_unknown_error(pinhole_int_param_dict: Dict, tmp_path: Path):
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


@pytest.mark.parametrize('missing_key', ['projection_type', 'width', 'height', 'focal_x', 'focal_y'])
def test_read_osfm_int_param_missing_error(pinhole_int_param_dict: Dict, missing_key: str, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when required keys are missing. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params.pop(missing_key)
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert missing_key in str(ex)


def test_read_osfm_int_param_unknown_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when unknown keys are present. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['other'] = 0.
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert 'other' in str(ex)


def test_read_osfm_int_param_proj_type_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when the projection type is unsupported. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['projection_type'] = 'other'
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert 'projection type' in str(ex)


def test_read_exif_int_param_dewarp(odm_image_file: Path):
    """ Testing reading EXIF / XMP tag interior parameters from an image with the 'DewarpData' XMP tag. """
    int_param_dict = io.read_exif_int_param(odm_image_file)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'brown'
    assert {'k1', 'k2', 'p1', 'p2', 'k3'}.issubset(int_params.keys())
    _validate_int_param_dict(int_param_dict)


def test_read_exif_int_param_no_dewarp(exif_image_file: Path):
    """ Testing reading EXIF / XMP tag interior parameters from an image without the 'DewarpData' XMP tag. """
    int_param_dict = io.read_exif_int_param(exif_image_file)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'pinhole'
    _validate_int_param_dict(int_param_dict)


def test_read_exif_int_param_error(ngi_image_file: Path):
    """ Test reading EXIF tag interior parameters from a non EXIF image raises an error. """
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_exif_int_param(ngi_image_file)
    assert 'focal length' in str(ex)


def test_aa_to_opk(xyz: Tuple, opk: Tuple):
    """ Test _aa_to_opk(). """
    R, _ = Camera._get_extrinsic(xyz, opk)
    aa = cv2.Rodrigues(R.T)[0]
    test_opk = io._aa_to_opk(aa)
    assert test_opk == pytest.approx(opk, 1e-6)


@pytest.mark.parametrize('src_crs, dst_crs', [
    ('wgs84_wgs84_crs', 'utm34n_egm96_crs'),
    ('wgs84_wgs84_crs', 'utm34n_egm2008_crs'),
    ('wgs84_egm96_crs', 'utm34n_wgs84_crs'),
    ('utm34n_egm96_crs', 'wgs84_wgs84_crs'),
    ('utm34n_egm2008_crs', 'wgs84_wgs84_crs'),
    ('utm34n_wgs84_crs', 'wgs84_egm96_crs'),
    ('wgs84_wgs84_crs', 'webmerc_egm96_crs'),
    ('wgs84_wgs84_crs', 'webmerc_egm2008_crs'),
    ('wgs84_egm96_crs', 'webmerc_wgs84_crs'),
])  # yapf: disable
def test_rio_transform_vdatum_both(src_crs: str, dst_crs: str, request: pytest.FixtureRequest):
    """
    Test rasterio.warp.transform adjusts the z coordinate with source and destination CRS vertical datums specified.
    """
    src_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(src_crs))
    dst_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(dst_crs))
    src_xyz = [[10.], [10.], [100.]]

    dst_xyz = transform(src_crs, src_crs, *src_xyz)
    assert dst_xyz[2][0] == pytest.approx(src_xyz[2][0], abs=1e-6)

    dst_xyz = transform(src_crs, dst_crs, *src_xyz)
    assert dst_xyz[2][0] != pytest.approx(src_xyz[2][0], abs=1.)


@pytest.mark.parametrize('src_crs, dst_crs', [
    ('wgs84_crs', 'utm34n_wgs84_crs'),
    ('wgs84_crs', 'utm34n_egm96_crs'),
    ('wgs84_crs', 'utm34n_egm2008_crs'),
    ('wgs84_crs', 'webmerc_wgs84_crs'),
    ('wgs84_crs', 'webmerc_egm96_crs'),
    ('wgs84_crs', 'webmerc_egm2008_crs'),
    ('utm34n_crs', 'wgs84_wgs84_crs'),
    ('utm34n_crs', 'wgs84_egm96_crs'),
    ('utm34n_crs', 'wgs84_egm2008_crs'),
])  # yapf: disable
def test_rio_transform_vdatum_one(src_crs: str, dst_crs: str, request: pytest.FixtureRequest):
    """
    Test rasterio.warp.transform does not adjust the z coordinate with one of the source and destination CRS vertical
    datums specified.
    """
    src_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(src_crs))
    dst_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(dst_crs))
    src_xyz = [[10.], [10.], [100.]]

    dst_xyz = transform(src_crs, dst_crs, *src_xyz)
    assert dst_xyz[2][0] == pytest.approx(src_xyz[2][0], abs=1e-6)


@pytest.mark.parametrize('C_bB', [
    np.array([[0., 1., 0.], [1., 0, 0], [0, 0, -1]]), np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
])
def test_rpy_to_opk(C_bB: np.ndarray):
    """ Test _rpy_to_opk() validity for aligned world and navigation systems. """
    # From https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf:
    # RPY rotates from body to navigation, and OPK from camera to world. If world is a topocentric system, centered on
    # the camera, then world (E) & navigation (n) are aligned with same origin, and C_En = np.array([[ 0, 1, 0], [1, 0,
    # 0], [0, 0, -1]]) (== C_En.T) rotates between them. If body (b) and camera (B) describe a typical drone geometry,
    # then C_bB = C_En (== C_En.T) rotates between them.
    # This test uses the topocentric special case to compare OPK and RPY rotation matrices using:
    # R(o, p, k) = C_EB = C_En * R(o, p, k) * C_bB = C_En * C_nb * C_bB

    n = 100
    llas = np.random.rand(n, 3) * (180, 360, 1000) + (-90, 0, 0)
    rpys = np.random.rand(n, 3) * (4 * np.pi) - (2 * np.pi)
    C_En = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    for lla, rpy in zip(llas, rpys):
        # create orthographic (2D topopcentric) CRS centered on lla
        crs = rio.CRS.from_string(f'+proj=ortho +lat_0={lla[0]:.4f} +lon_0={lla[1]:.4f} +ellps=WGS84')
        opk = io._rpy_to_opk(rpy, lla, crs, C_bB=C_bB, lla_crs=io._default_lla_crs)

        C_nb = io._rpy_to_rotation(rpy)
        R_opk = io._opk_to_rotation(opk)
        assert R_opk == pytest.approx(C_En.dot(C_nb).dot(C_bB), abs=1e-6)
        assert C_En.T.dot(R_opk).dot(C_bB.T) == pytest.approx(C_nb, abs=1e-6)


def test_csv_reader_legacy(ngi_legacy_csv_file: Path, ngi_crs: str, ngi_image_files: Tuple[Path, ...]):
    """ Test reading exterior parameters from a legacy format CSV file. """
    reader = io.CsvReader(ngi_legacy_csv_file, crs=ngi_crs)
    assert reader._fieldnames == io.CsvReader._legacy_fieldnames
    assert reader._format is CsvFormat.xyz_opk
    assert reader.crs == rio.CRS.from_string(ngi_crs)

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.stem for filename in ngi_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    _validate_ext_param_dict(ext_param_dict, cameras=[None])


def test_csv_reader_xyz_opk(ngi_xyz_opk_csv_file: Path, ngi_crs: str, ngi_image_files: Tuple[Path, ...]):
    """ Test reading exterior parameters from an xyz_opk format CSV file with a header. """
    reader = io.CsvReader(ngi_xyz_opk_csv_file, crs=ngi_crs)
    assert reader._fieldnames == io.CsvReader._legacy_fieldnames
    assert reader._format is CsvFormat.xyz_opk
    with open(ngi_xyz_opk_csv_file.with_suffix('.prj')) as f:
        assert reader.crs == rio.CRS.from_string(f.read())

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.stem for filename in ngi_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    _validate_ext_param_dict(ext_param_dict, cameras=[None])


def test_csv_reader_lla_rpy(
    odm_lla_rpy_csv_file: Path, odm_crs: str, odm_image_files: Tuple[Path, ...], osfm_reconstruction_file: Path
):
    """ Test reading exterior parameters from an lla_rpy format CSV file with a header. """
    reader = io.CsvReader(odm_lla_rpy_csv_file, crs=odm_crs)
    assert set(reader._fieldnames) == {
        'filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'camera', 'other'
    }
    assert reader._format is CsvFormat.lla_rpy
    assert reader.crs == rio.CRS.from_string(odm_crs)

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.name for filename in odm_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)

    with open(osfm_reconstruction_file, 'r') as f:
        json_obj = json.load(f)
    cam_id = next(iter(json_obj[0]['cameras'].keys())).strip('v2 ')
    _validate_ext_param_dict(ext_param_dict, cameras=[cam_id])


def test_csv_reader_xyz_opk_prj_crs(ngi_xyz_opk_csv_file: Path):
    """ Test CsvReader initialised with a xyz_* format CSV file and no CRS, reads the CRS from a .prj file. """
    reader = io.CsvReader(ngi_xyz_opk_csv_file, crs=None)
    assert reader._fieldnames == io.CsvReader._legacy_fieldnames
    with open(ngi_xyz_opk_csv_file.with_suffix('.prj')) as f:
        assert reader.crs == rio.CRS.from_string(f.read())


def test_csv_reader_lla_rpy_auto_crs(odm_lla_rpy_csv_file: Path, odm_crs: str):
    """ Test CsvReader initialised with a lla_rpy format CSV file and no CRS generates an auto UTM CRS. """
    reader = io.CsvReader(odm_lla_rpy_csv_file, crs=None)
    assert reader.crs == rio.CRS.from_string(odm_crs)


@pytest.mark.parametrize('fieldnames', [
    ['filename', 'easting', 'northing', 'altitude', 'roll', 'pitch', 'yaw'],
    ['filename', 'latitude', 'longitude', 'altitude', 'omega', 'phi', 'kappa'],
])  # yapf: disable
def test_csv_reader_crs_error(ngi_legacy_csv_file: Path, fieldnames: List):
    """ Test that CsvReader initialised with a xyz_rpy or lla_opk format file and no CRS raises an error. """
    fieldnames = ['filename', 'easting', 'northing', 'altitude', 'roll', 'pitch', 'yaw']
    with pytest.raises(CrsMissingError) as ex:
        reader = io.CsvReader(ngi_legacy_csv_file, fieldnames=fieldnames)
    assert 'crs' in str(ex).lower()


def test_csv_reader_fieldnames(odm_lla_rpy_csv_file: Path):
    """ Test reading exterior parameters from a CSV file with ``fieldnames`` argument. """
    fieldnames = ['filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'camera', 'custom']
    reader = io.CsvReader(odm_lla_rpy_csv_file, fieldnames=fieldnames)
    assert set(reader._fieldnames) == set(fieldnames)
    _ = reader.read_ext_param()


@pytest.mark.parametrize('missing_field', io.CsvReader._legacy_fieldnames)
def test_csv_reader_missing_fieldname_error(ngi_legacy_csv_file: Path, missing_field):
    """ Test that CsvReader intialised with a missing fieldname raises an error. """
    fieldnames = io.CsvReader._legacy_fieldnames.copy()
    fieldnames.remove(missing_field)
    with pytest.raises(ParamFileError) as ex:
        reader = io.CsvReader(ngi_legacy_csv_file, fieldnames=fieldnames)
    assert missing_field in str(ex)


@pytest.mark.parametrize('filename, crs, fieldnames, exp_format', [
    (
        'ngi_xyz_opk_csv_file', 'ngi_crs', ['filename', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'],
        CsvFormat.xyz_opk
    ),
    (
        'ngi_xyz_opk_csv_file', 'ngi_crs', ['filename', 'easting', 'northing', 'altitude', 'roll', 'pitch', 'yaw'],
        CsvFormat.xyz_rpy
    ),
    (
        'odm_lla_rpy_csv_file', 'odm_crs', ['filename', 'latitude', 'longitude', 'altitude', 'omega', 'phi', 'kappa'],
        CsvFormat.lla_opk
    ),
    (
        'odm_lla_rpy_csv_file', 'odm_crs', ['filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw'],
        CsvFormat.lla_rpy
    ),
])  # yapf: disable
def test_csv_reader_format(
    filename: str, crs: str, fieldnames: List, exp_format:CsvFormat, request: pytest.FixtureRequest
):
    """ Test reading exterior parameters from a CSV file in different (simulated) position / orientation formats. """
    filename: Path = request.getfixturevalue(filename)
    crs: str = request.getfixturevalue(crs)

    reader = io.CsvReader(filename, crs=crs, fieldnames=fieldnames)
    assert reader._format == exp_format
    assert reader.crs == rio.CRS.from_string(crs)

    ext_param_dict = reader.read_ext_param()
    assert len(ext_param_dict) > 0
    _validate_ext_param_dict(ext_param_dict, cameras=[None])


@pytest.mark.parametrize('dialect', [
    dict(delimiter=' ', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=' ', lineterminator='\n', quotechar="'", quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=' ', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_ALL),
    dict(delimiter=' ', lineterminator='\n', quotechar="'", quoting=csv.QUOTE_ALL),
    dict(delimiter=' ', lineterminator='\r', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=' ', lineterminator='\r\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=';', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=':', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter='\t', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
])  # yapf: disable
def test_csv_reader_dialect(
    odm_lla_rpy_csv_file: Path, odm_crs: str, odm_image_files: Tuple[Path, ...], osfm_reconstruction_file: Path,
    dialect: Dict, tmp_path: Path
):
    """ Test reading exterior parameters from CSV files in different dialects. """
    # create test CSV file
    test_filename = tmp_path.joinpath('ext-param-test.csv')
    with open(odm_lla_rpy_csv_file, 'r') as fin:
        with open(test_filename, 'w', newline='') as fout:
            reader = csv.reader(fin, delimiter=' ', quotechar='"')
            writer = csv.writer(fout, **dialect)
            for row in reader:
                writer.writerow(row)

    # read test file
    reader = io.CsvReader(test_filename, crs=odm_crs)
    for attr in ['delimiter', 'quotechar']:
        assert getattr(reader._dialect, attr) == dialect[attr]
    ext_param_dict = reader.read_ext_param()

    # validate dict
    file_keys = [filename.name for filename in odm_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    with open(osfm_reconstruction_file, 'r') as f:
        json_obj = json.load(f)
    cam_id = next(iter(json_obj[0]['cameras'].keys())).strip('v2 ')
    _validate_ext_param_dict(ext_param_dict, cameras=[cam_id])


def test_osfm_reader(osfm_reconstruction_file: Path, odm_crs: str):
    """ Test OsfmReader reads internal and external parameters successfully. """
    reader = io.OsfmReader(osfm_reconstruction_file, crs=odm_crs)
    assert reader.crs == rio.CRS.from_string(odm_crs)

    int_param_dict = reader.read_int_param()
    int_cam_ids = set(int_param_dict.keys())
    ext_param_dict = reader.read_ext_param()
    ext_cam_ids = set([ext_param['camera'] for ext_param in ext_param_dict.values()])

    _validate_int_param_dict(int_param_dict)
    _validate_ext_param_dict(ext_param_dict, cameras=int_cam_ids)
    assert ext_cam_ids.issubset(int_cam_ids)


def test_osfm_reader_auto_crs(osfm_reconstruction_file: Path, odm_crs: str):
    """ Test OsfmReader auto determines a UTM CRS correctly. """
    reader = io.OsfmReader(osfm_reconstruction_file, crs=None)
    assert reader.crs == rio.CRS.from_string(odm_crs)


def test_osfm_reader_validity_error(odm_int_param_file: Path):
    """ Test OsfmReader raises an error with an invalid file format. """
    with pytest.raises(ParamFileError) as ex:
        reader = io.OsfmReader(odm_int_param_file, crs=None)
    assert 'valid' in str(ex)


def test_exif_reader(odm_image_files: Tuple[Path, ...], odm_crs: str):
    """ Test ExifReader reads internal and external parameters successfully. """
    reader = io.ExifReader(odm_image_files, crs=odm_crs)
    assert reader.crs == rio.CRS.from_string(odm_crs)

    int_param_dict = reader.read_int_param()
    int_cam_ids = set(int_param_dict.keys())
    ext_param_dict = reader.read_ext_param()
    ext_cam_ids = set([ext_param['camera'] for ext_param in ext_param_dict.values()])

    _validate_int_param_dict(int_param_dict)
    _validate_ext_param_dict(ext_param_dict, cameras=int_cam_ids)
    assert ext_cam_ids.issubset(int_cam_ids)


def test_exif_reader_auto_crs(odm_image_files: Tuple[Path, ...], odm_crs: str):
    """ Test ExifReader auto determines a UTM CRS correctly. """
    reader = io.ExifReader(odm_image_files, crs=None)
    assert reader.crs == rio.CRS.from_string(odm_crs)


def test_exif_reader_empty():
    """ Test ExifReader with empty list of of files. """
    reader = io.ExifReader([], crs=None)
    assert reader.crs is None
    assert reader.read_int_param() == {}
    assert reader.read_ext_param() == {}


def test_oty_reader(ngi_oty_ext_param_file: Path, ngi_crs: str):
    """ Test OtyReader reads external parameters successfully. """
    reader = io.OtyReader(ngi_oty_ext_param_file, crs=ngi_crs)
    assert reader.crs == rio.CRS.from_string(ngi_crs)

    ext_param_dict = reader.read_ext_param()
    ext_cam_ids = set([ext_param['camera'] for ext_param in ext_param_dict.values()])
    assert len(ext_cam_ids) == 1 and isinstance(list(ext_cam_ids)[0], str) and len(list(ext_cam_ids)[0]) > 0
    _validate_ext_param_dict(ext_param_dict, cameras=None)


def test_oty_reader_validity_error(osfm_reconstruction_file: Path):
    """ Test OtyReader raises an error with an invalid file format. """
    with pytest.raises(ParamFileError) as ex:
        _ = io.OtyReader(osfm_reconstruction_file, crs=None)
    assert 'valid' in str(ex)


def test_oty_reader_crs(ngi_oty_ext_param_file: Path, ngi_crs: str):
    """ Test OtyReader reads the crs correctly. """
    reader = io.OtyReader(ngi_oty_ext_param_file, crs=None)
    assert reader.crs == rio.CRS.from_string(ngi_crs)


# TODO: Multi-camera configurations
# TODO: Add config conversions e.g. ODM / Legacy / CSV internal + external -> oty format -> inputs
