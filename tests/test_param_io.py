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

import csv
import json
from pathlib import Path
from typing import Collection

import cv2
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import GroundControlPoint
from rasterio.warp import transform

from orthority import param_io
from orthority.camera import FrameCamera
from orthority.enums import CameraType, CsvFormat, Interp
from orthority.errors import CrsError, CrsMissingError, ParamError
from tests.conftest import oty_to_osfm_int_param, create_profile


def _validate_int_param_dict(int_param_dict: dict):
    """Basic validation of an internal parameter dictionary."""
    assert len(int_param_dict) > 0
    for cam_id, int_params in int_param_dict.items():
        assert isinstance(cam_id, str) or cam_id is None
        req_keys = {'cam_type', 'im_size', 'focal_len'}
        assert set(int_params.keys()).issuperset(req_keys)
        cam_type = CameraType(int_params['cam_type'])
        assert len(int_params['im_size']) == 2 and all(
            [isinstance(dim, int) for dim in int_params['im_size']]
        )
        if 'sensor_size' in int_params:
            assert len(int_params['sensor_size']) == 2 and all(
                [isinstance(dim, float) for dim in int_params['sensor_size']]
            )
        assert isinstance(int_params['focal_len'], float) or (
            len(int_params['focal_len']) == 2
            and all([isinstance(f, float) for f in int_params['focal_len']])
        )
        optional_keys = set(int_params.keys()).difference(req_keys)
        assert set(optional_keys).issubset(param_io._opt_frame_schema[cam_type])


def _validate_ext_param_dict(ext_param_dict: dict, cameras: Collection[str | None] = None):
    """Basic validation of an exterior parameter dictionary."""
    assert len(ext_param_dict) > 0
    for filename, ext_params in ext_param_dict.items():
        assert set(ext_params.keys()) == {'opk', 'xyz', 'camera'}
        opk, xyz = np.array(ext_params['opk']), np.array(ext_params['xyz'])
        assert len(opk) == 3 and len(xyz) == 3
        # rough check for radians
        assert all(np.abs(opk) <= 2 * np.pi) and np.any(opk != 0.0)
        # rough check for not latitude, longitude, & altitude > 0
        assert np.all(xyz != 0) and np.abs(xyz[0]) > 180.0 and np.abs(xyz[1]) > 90.0 and xyz[2] > 0
        if cameras:
            assert ext_params['camera'] in cameras


def test_rw_oty_int_param(mult_int_param_dict: dict, tmp_path: Path):
    """Test interior parameter read / write from / to orthority interior parameter format."""
    filename = tmp_path.joinpath('int_param.yaml')
    param_io.write_int_param(filename, mult_int_param_dict)
    test_dict = param_io.read_oty_int_param(filename)
    assert test_dict == mult_int_param_dict


@pytest.mark.parametrize('missing_key', ['focal_len', 'im_size'])
def test_read_oty_int_param_missing_error(
    pinhole_int_param_dict: dict, missing_key: str, tmp_path: Path
):
    """Test reading orthority format interior parameters raises an error when required keys are
    missing.
    """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params.pop(missing_key)
    filename = tmp_path.joinpath('int_param.yaml')
    param_io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamError) as ex:
        _ = param_io.read_oty_int_param(filename)
    assert (
        'Could not parse' in str(ex.value)
        and missing_key in str(ex.value)
        and filename.name in str(ex.value)
    )


def test_read_oty_int_param_unknown_error(pinhole_int_param_dict: dict, tmp_path: Path):
    """Test reading orthority format interior parameters raises an error when unknown keys are
    present.
    """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params['other'] = 0.0
    filename = tmp_path.joinpath('int_param.yaml')
    param_io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamError) as ex:
        _ = param_io.read_oty_int_param(filename)
    assert (
        'Could not parse' in str(ex.value)
        and 'other' in str(ex.value)
        and filename.name in str(ex.value)
    )


def test_read_oty_int_param_cam_type_error(pinhole_int_param_dict: dict, tmp_path: Path):
    """Test reading orthority format interior parameters raises an error the camera type is
    unknown.
    """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params['cam_type'] = Interp.cubic
    filename = tmp_path.joinpath('int_param.yaml')
    param_io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamError) as ex:
        _ = param_io.read_oty_int_param(filename)
    assert (
        'Could not parse' in str(ex.value)
        and 'camera type' in str(ex.value)
        and filename.name in str(ex.value)
    )


@pytest.mark.parametrize('filename', ['osfm_int_param_file', 'odm_int_param_file'])
def test_read_osfm_int_param(
    filename: str, mult_int_param_dict: dict, request: pytest.FixtureRequest
):
    """Test reading interior parameters from ODM / OpenSfM format files."""
    filename: Path = request.getfixturevalue(filename)
    test_dict = param_io.read_osfm_int_param(filename)

    def compare_dicts(test: dict, ref: dict):
        """Compare interior parameter dicts omitting ``sensor_size``."""
        for k, v in ref.items():
            if isinstance(v, dict):
                compare_dicts(test[k], v)
            elif k != 'sensor_size':
                assert test[k] == v

    compare_dicts(test_dict, mult_int_param_dict)


@pytest.mark.parametrize(
    'missing_key', ['projection_type', 'width', 'height', 'focal_x', 'focal_y']
)
def test_read_osfm_int_param_missing_error(
    pinhole_int_param_dict: dict, missing_key: str, tmp_path: Path
):
    """Test reading ODM / OpenSfM format interior parameters raises an error when required keys are
    missing.
    """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params.pop(missing_key)
    filename = tmp_path.joinpath('int_param.json')
    with filename.open('wt') as f:
        json.dump(osfm_dict, f)

    with pytest.raises(ParamError) as ex:
        _ = param_io.read_osfm_int_param(filename)
    assert (
        'Could not parse' in str(ex.value)
        and missing_key in str(ex.value)
        and filename.name in str(ex.value)
    )


def test_read_osfm_int_param_unknown_error(pinhole_int_param_dict: dict, tmp_path: Path):
    """Test reading ODM / OpenSfM format interior parameters raises an error when unknown keys are
    present.
    """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['other'] = 0.0
    filename = tmp_path.joinpath('int_param.json')
    with filename.open('wt') as f:
        json.dump(osfm_dict, f)

    with pytest.raises(ParamError) as ex:
        _ = param_io.read_osfm_int_param(filename)
    assert (
        'Could not parse' in str(ex.value)
        and 'other' in str(ex.value)
        and filename.name in str(ex.value)
    )


def test_read_osfm_int_param_proj_type_error(pinhole_int_param_dict: dict, tmp_path: Path):
    """Test reading ODM / OpenSfM format interior parameters raises an error when the projection type is
    unsupported.
    """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['projection_type'] = 'other'
    filename = tmp_path.joinpath('int_param.json')
    with filename.open('wt') as f:
        json.dump(osfm_dict, f)

    with pytest.raises(ParamError) as ex:
        _ = param_io.read_osfm_int_param(filename)
    assert (
        'Could not parse' in str(ex.value)
        and 'projection type' in str(ex.value)
        and filename.name in str(ex.value)
    )


def test_read_exif_int_param_dewarp(odm_image_file: Path, odm_reconstruction_file: Path):
    """Test reading EXIF / XMP tag interior parameters from an image with the ``DewarpData`` XMP
    tag.
    """
    int_param_dict = param_io.read_exif_int_param(odm_image_file)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'brown'
    assert {'k1', 'k2', 'p1', 'p2', 'k3'}.issubset(int_params.keys())
    _validate_int_param_dict(int_param_dict)


@pytest.mark.parametrize(
    'filename', ['exif_image_file', 'exif_no_focal_image_file', 'xmp_no_dewarp_image_file']
)
def test_read_exif_int_param_no_dewarp(filename: str, request: pytest.FixtureRequest):
    """Test reading EXIF / XMP tag interior parameters from an image without the ``DewarpData`` XMP
    tag.
    """
    filename: Path = request.getfixturevalue(filename)
    int_param_dict = param_io.read_exif_int_param(filename)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'pinhole'
    _validate_int_param_dict(int_param_dict)


@pytest.mark.parametrize(
    'image_file',
    ['odm_image_file', 'exif_image_file', 'xmp_no_dewarp_image_file', 'exif_no_focal_image_file'],
)
def test_read_exif_int_param_values(
    image_file: str, odm_reconstruction_file: Path, request: pytest.FixtureRequest
):
    """Test EXIF focal length values against those from OsfmReader for images with different tag
    combinations.
    """
    # read EXIF and OpenSfM interior parameters
    image_file: Path = request.getfixturevalue(image_file)
    ref_int_param_dict = param_io.read_osfm_int_param(odm_reconstruction_file)
    ref_int_params = next(iter(ref_int_param_dict.values()))
    test_int_param_dict = param_io.read_exif_int_param(image_file)
    test_int_params = next(iter(test_int_param_dict.values()))

    # normalise EXIF interior parameters and compare to OpenSfM values
    test_focal_len = np.array(test_int_params['focal_len'])
    if 'sensor_size' in test_int_params:
        test_focal_len = test_focal_len / max(test_int_params['sensor_size'])
    test_focal_len = test_focal_len if test_focal_len.size == 1 else test_focal_len[0]
    assert test_focal_len == pytest.approx(ref_int_params['focal_len'], abs=0.01)


def test_read_exif_int_param_error(ngi_image_file: Path):
    """Test reading EXIF tag interior parameters from a non EXIF image raises an error."""
    with pytest.raises(ParamError) as ex:
        _ = param_io.read_exif_int_param(ngi_image_file)
    assert 'focal length' in str(ex.value) and ngi_image_file.name in str(ex.value)


def test_aa_to_opk(xyz: tuple, opk: tuple):
    """Test _aa_to_opk()."""
    R, _ = FrameCamera._get_extrinsic(xyz, opk)
    aa = cv2.Rodrigues(R.T)[0]
    test_opk = param_io._aa_to_opk(aa)
    assert test_opk == pytest.approx(opk, 1e-6)


@pytest.mark.parametrize(
    'src_crs, dst_crs',
    [
        ('wgs84_wgs84_crs', 'utm34n_egm96_crs'),
        ('wgs84_wgs84_crs', 'utm34n_egm2008_crs'),
        ('wgs84_egm96_crs', 'utm34n_wgs84_crs'),
        ('utm34n_egm96_crs', 'wgs84_wgs84_crs'),
        ('utm34n_egm2008_crs', 'wgs84_wgs84_crs'),
        ('utm34n_wgs84_crs', 'wgs84_egm96_crs'),
        ('wgs84_wgs84_crs', 'webmerc_egm96_crs'),
        ('wgs84_wgs84_crs', 'webmerc_egm2008_crs'),
        ('wgs84_egm96_crs', 'webmerc_wgs84_crs'),
    ],
)
def test_rio_transform_vdatum_both(src_crs: str, dst_crs: str, request: pytest.FixtureRequest):
    """Test rasterio.warp.transform adjusts the z coordinate with source and destination CRS
    vertical datums specified.
    """

    src_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(src_crs))
    dst_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(dst_crs))
    src_xyz = [[10.0], [10.0], [100.0]]

    dst_xyz = transform(src_crs, src_crs, *src_xyz)
    assert dst_xyz[2][0] == pytest.approx(src_xyz[2][0], abs=1e-6)

    dst_xyz = transform(src_crs, dst_crs, *src_xyz)
    assert dst_xyz[2][0] != pytest.approx(src_xyz[2][0], abs=5.0)


@pytest.mark.parametrize(
    'src_crs, dst_crs',
    [
        ('wgs84_crs', 'utm34n_wgs84_crs'),
        ('wgs84_crs', 'utm34n_egm96_crs'),
        ('wgs84_crs', 'utm34n_egm2008_crs'),
        ('wgs84_crs', 'webmerc_wgs84_crs'),
        ('wgs84_crs', 'webmerc_egm96_crs'),
        ('wgs84_crs', 'webmerc_egm2008_crs'),
        ('utm34n_crs', 'wgs84_wgs84_crs'),
        ('utm34n_crs', 'wgs84_egm96_crs'),
        ('utm34n_crs', 'wgs84_egm2008_crs'),
    ],
)
@pytest.mark.skipif(rio.get_proj_version() < (9, 1, 1), reason="requires PROJ 9.1.1 or higher")
def test_rio_transform_vdatum_one(src_crs: str, dst_crs: str, request: pytest.FixtureRequest):
    """Test rasterio.warp.transform does not adjust the z coordinate with one of the source and
    destination CRS vertical datums specified.
    """
    src_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(src_crs))
    dst_crs: rio.CRS = rio.CRS.from_string(request.getfixturevalue(dst_crs))
    src_xyz = [[10.0], [10.0], [100.0]]
    dst_xyz = transform(src_crs, dst_crs, *src_xyz)
    # prior proj versions promote 2D->3D with ellipsoidal height
    assert dst_xyz[2][0] == pytest.approx(src_xyz[2][0], abs=1e-6)


@pytest.mark.parametrize(
    'C_bB',
    [
        np.array([[0.0, 1.0, 0.0], [1.0, 0, 0], [0, 0, -1]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    ],
)
def test_rpy_to_opk(C_bB: np.ndarray):
    """Test _rpy_to_opk() validity for aligned world and navigation systems."""
    # From https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf:
    # RPY rotates from body to navigation, and OPK from camera to world. If world is a
    # topocentric system, centered on the camera, then world (E) & navigation (n) are aligned
    # with same origin, and C_En = np.array([[ 0, 1, 0], [1, 0, 0], [0, 0, -1]]) (== C_En.T)
    # rotates between them. If body (b) and camera (B) describe a typical drone geometry,
    # then C_bB = C_En (== C_En.T) rotates between them.
    # This test uses the topocentric special case to compare OPK and RPY rotation matrices using:
    # R(o, p, k) = C_EB = C_En * R(r, p, y) * C_bB = C_En * C_nb * C_bB

    n = 100
    llas = np.random.rand(n, 3) * (180, 360, 1000) + (-90, 0, 0)
    rpys = np.random.rand(n, 3) * (4 * np.pi) - (2 * np.pi)
    C_En = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    for lla, rpy in zip(llas, rpys):
        # create orthographic (2D topopcentric) CRS centered on lla
        crs = rio.CRS.from_proj4(
            f'+proj=ortho +lat_0={lla[0]:.4f} +lon_0={lla[1]:.4f} +ellps=WGS84'
        )
        opk = param_io._rpy_to_opk(rpy, lla, crs, C_bB=C_bB, lla_crs=param_io._default_lla_crs)

        C_nb = param_io._rpy_to_rotation(rpy)
        R_opk = param_io._opk_to_rotation(opk)
        assert R_opk == pytest.approx(C_En.dot(C_nb).dot(C_bB), abs=1e-6)
        assert C_En.T.dot(R_opk).dot(C_bB.T) == pytest.approx(C_nb, abs=1e-6)


def test_csv_reader_legacy(ngi_legacy_csv_file: Path, ngi_crs: str, ngi_image_files: list[Path]):
    """Test reading exterior parameters from a legacy format CSV file."""
    reader = param_io.CsvReader(ngi_legacy_csv_file, crs=ngi_crs)
    assert reader._fieldnames == param_io.CsvReader._legacy_fieldnames
    assert reader._format is CsvFormat.xyz_opk
    assert reader.crs == rio.CRS.from_string(ngi_crs)

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.stem for filename in ngi_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    _validate_ext_param_dict(ext_param_dict, cameras=[None])


def test_csv_reader_xyz_opk(ngi_xyz_opk_csv_file: Path, ngi_crs: str, ngi_image_files: list[Path]):
    """Test reading exterior parameters from an XYZ-OPK format CSV file with a header."""
    reader = param_io.CsvReader(ngi_xyz_opk_csv_file, crs=ngi_crs)
    assert reader._fieldnames == param_io.CsvReader._legacy_fieldnames
    assert reader._format is CsvFormat.xyz_opk
    with open(ngi_xyz_opk_csv_file.with_suffix('.prj')) as f:
        assert reader.crs == rio.CRS.from_string(f.read())

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.stem for filename in ngi_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    _validate_ext_param_dict(ext_param_dict, cameras=[None])


def test_csv_reader_xyz_opk_values(
    odm_xyz_opk_csv_file: Path, odm_crs: str, odm_reconstruction_file: Path
):
    """Test exterior parameter values from CsvReader against those from OsfmReader for an XYZ-OPK
    CSV file.
    """
    ref_reader = param_io.OsfmReader(odm_reconstruction_file, crs=odm_crs)
    ref_ext_param_dict = ref_reader.read_ext_param()
    test_reader = param_io.CsvReader(odm_xyz_opk_csv_file, crs=odm_crs)
    test_ext_param_dict = test_reader.read_ext_param()

    for filename, test_ext_param in test_ext_param_dict.items():
        ref_ext_param = ref_ext_param_dict[Path(filename).stem]
        assert test_ext_param['xyz'] == pytest.approx(ref_ext_param['xyz'], abs=1e-3)
        assert test_ext_param['opk'] == pytest.approx(ref_ext_param['opk'], abs=1e-3)


def test_csv_reader_xyz_opk_radians(
    ngi_xyz_opk_csv_file: Path, ngi_xyz_opk_radians_csv_file: Path, ngi_crs: str
):
    """Test CsvReader(..., radians=True) by comparing exterior parameters from files with angles in
    degrees and radians.
    """
    deg_reader = param_io.CsvReader(ngi_xyz_opk_csv_file, crs=ngi_crs)
    rad_reader = param_io.CsvReader(ngi_xyz_opk_radians_csv_file, crs=ngi_crs, radians=True)
    assert deg_reader.read_ext_param() == rad_reader.read_ext_param()


def test_csv_reader_lla_rpy(
    odm_lla_rpy_csv_file: Path,
    odm_crs: str,
    odm_image_files: list[Path],
    odm_reconstruction_file: Path,
):
    """Test reading exterior parameters from an LLA-RPY format CSV file with a header."""
    reader = param_io.CsvReader(odm_lla_rpy_csv_file, crs=odm_crs)
    assert set(reader._fieldnames) == {
        'filename',
        'latitude',
        'longitude',
        'altitude',
        'roll',
        'pitch',
        'yaw',
        'camera',
        'other',
    }
    assert reader._format is CsvFormat.lla_rpy
    assert reader.crs == rio.CRS.from_string(odm_crs)

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.name for filename in odm_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)

    with open(odm_reconstruction_file, 'r') as f:
        json_obj = json.load(f)
    cam_id = next(iter(json_obj[0]['cameras'].keys())).strip('v2 ')
    _validate_ext_param_dict(ext_param_dict, cameras=[cam_id])


def test_csv_reader_lla_rpy_values(
    odm_lla_rpy_csv_file: Path, odm_crs: str, odm_reconstruction_file: Path
):
    """Test exterior parameter values from CsvReader against those from OsfmReader for an LLA-RPY
    format CSV file.
    """
    ref_reader = param_io.OsfmReader(odm_reconstruction_file, crs=odm_crs)
    ref_ext_param_dict = ref_reader.read_ext_param()
    test_reader = param_io.CsvReader(odm_lla_rpy_csv_file, crs=odm_crs)
    test_ext_param_dict = test_reader.read_ext_param()

    for filename, test_ext_param in test_ext_param_dict.items():
        ref_ext_param = ref_ext_param_dict[Path(filename).stem]
        assert test_ext_param['xyz'] == pytest.approx(ref_ext_param['xyz'], abs=0.1)
        assert test_ext_param['opk'] == pytest.approx(ref_ext_param['opk'], abs=0.1)


@pytest.mark.parametrize('csv_file', ['ngi_xyz_opk_csv_file', 'ngi_xyz_opk_csv_url'])
def test_csv_reader_xyz_opk_prj_crs(csv_file: str, ngi_crs: str, request: pytest.FixtureRequest):
    """Test CsvReader initialised with a xyz_* format CSV file and no CRS, reads the CRS from a .prj
    file path / URI.
    """
    csv_file = str(request.getfixturevalue(csv_file))
    reader = param_io.CsvReader(csv_file, crs=None)
    assert reader._fieldnames == param_io.CsvReader._legacy_fieldnames
    assert reader.crs == rio.CRS.from_string(ngi_crs)


def test_csv_reader_lla_rpy_auto_crs(odm_lla_rpy_csv_file: Path, odm_crs: str):
    """Test CsvReader initialised with a LLA-RPY format CSV file and no CRS generates an auto UTM
    CRS.
    """
    reader = param_io.CsvReader(odm_lla_rpy_csv_file, crs=None)
    assert reader.crs == rio.CRS.from_string(odm_crs)


@pytest.mark.parametrize(
    'fieldnames',
    [
        ['filename', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['filename', 'latitude', 'longitude', 'altitude', 'omega', 'phi', 'kappa'],
    ],
)
def test_csv_reader_crs_missing_error(ngi_legacy_csv_file: Path, fieldnames: list):
    """Test that CsvReader initialised with a XYZ-RPY or LLA-OPK format file and no CRS raises an
    error.
    """
    with pytest.raises(CrsMissingError) as ex:
        _ = param_io.CsvReader(ngi_legacy_csv_file, fieldnames=fieldnames)
    assert 'crs' in str(ex.value).lower() and ngi_legacy_csv_file.name in str(ex.value)


def test_csv_reader_crs_error(ngi_legacy_csv_file: Path):
    """Test CsvReader with a geographic / invalid CRS raises an error."""
    with pytest.raises(CrsError) as ex:
        param_io.CsvReader(ngi_legacy_csv_file, crs='EPSG:4326')
    assert 'projected' in str(ex.value)

    with pytest.raises(CrsError) as ex:
        param_io.CsvReader(ngi_legacy_csv_file, crs='unknown')
    assert 'could not interpret' in str(ex.value).lower()


def test_csv_reader_lla_rpy_lla_crs(odm_lla_rpy_csv_file, odm_crs: str, wgs84_egm2008_crs: str):
    """Test that CsvReader exterior parameters for a LLA-RPY format file are affected by lla_crs as
    expected.
    """
    ref_reader = param_io.CsvReader(odm_lla_rpy_csv_file, crs=odm_crs + '+4326')
    test_reader = param_io.CsvReader(
        odm_lla_rpy_csv_file, crs=odm_crs + '+4326', lla_crs=wgs84_egm2008_crs
    )
    assert test_reader._lla_crs == rio.CRS.from_string(wgs84_egm2008_crs)

    ref_ext_param_dict = ref_reader.read_ext_param()
    test_ext_param_dict = test_reader.read_ext_param()
    for ref_ext_params, test_ext_params in zip(
        ref_ext_param_dict.values(), test_ext_param_dict.values()
    ):
        # test z offset changes and rotation stays same
        assert test_ext_params['xyz'][:2] == pytest.approx(ref_ext_params['xyz'][:2], abs=1e-6)
        assert test_ext_params['xyz'][2] != pytest.approx(ref_ext_params['xyz'][2], abs=1)
        assert test_ext_params['opk'] == pytest.approx(ref_ext_params['opk'], abs=1e-4)


def test_csv_reader_lla_crs_error(ngi_xyz_opk_csv_file: Path):
    """Test CsvReader with a projected / invalid LLA CRS raises an error."""
    with pytest.raises(CrsError) as ex:
        param_io.CsvReader(ngi_xyz_opk_csv_file, lla_crs='EPSG:3857')
    assert 'geographic' in str(ex.value)

    with pytest.raises(CrsError) as ex:
        param_io.CsvReader(ngi_xyz_opk_csv_file, lla_crs='unknown')
    assert 'could not interpret' in str(ex.value).lower()


def test_csv_reader_fieldnames(odm_lla_rpy_csv_file: Path):
    """Test reading exterior parameters from a CSV file with ``fieldnames`` argument."""
    fieldnames = [
        'filename',
        'latitude',
        'longitude',
        'altitude',
        'roll',
        'pitch',
        'yaw',
        'camera',
        'custom',
    ]
    reader = param_io.CsvReader(odm_lla_rpy_csv_file, fieldnames=fieldnames)
    assert set(reader._fieldnames) == set(fieldnames)
    _ = reader.read_ext_param()


@pytest.mark.parametrize('missing_field', param_io.CsvReader._legacy_fieldnames)
def test_csv_reader_missing_fieldname_error(ngi_legacy_csv_file: Path, missing_field):
    """Test that CsvReader initialised with a missing fieldname raises an error."""
    fieldnames = param_io.CsvReader._legacy_fieldnames.copy()
    fieldnames.remove(missing_field)
    with pytest.raises(ParamError) as ex:
        _ = param_io.CsvReader(ngi_legacy_csv_file, fieldnames=fieldnames)
    assert missing_field in str(ex.value) and ngi_legacy_csv_file.name in str(ex.value)


@pytest.mark.parametrize(
    'filename, crs, fieldnames, exp_format',
    [
        (
            'ngi_xyz_opk_csv_file',
            'ngi_crs',
            ['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa'],
            CsvFormat.xyz_opk,
        ),
        (
            'ngi_xyz_opk_csv_file',
            'ngi_crs',
            ['filename', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'],
            CsvFormat.xyz_rpy,
        ),
        (
            'odm_lla_rpy_csv_file',
            'odm_crs',
            ['filename', 'latitude', 'longitude', 'altitude', 'omega', 'phi', 'kappa'],
            CsvFormat.lla_opk,
        ),
        (
            'odm_lla_rpy_csv_file',
            'odm_crs',
            ['filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw'],
            CsvFormat.lla_rpy,
        ),
    ],
)
def test_csv_reader_format(
    filename: str, crs: str, fieldnames: list, exp_format: CsvFormat, request: pytest.FixtureRequest
):
    """Test reading exterior parameters from a CSV file in different (simulated) position /
    orientation formats.
    """
    filename: Path = request.getfixturevalue(filename)
    crs: str = request.getfixturevalue(crs)

    reader = param_io.CsvReader(filename, crs=crs, fieldnames=fieldnames)
    assert reader._format == exp_format
    assert reader.crs == rio.CRS.from_string(crs)

    ext_param_dict = reader.read_ext_param()
    assert len(ext_param_dict) > 0
    _validate_ext_param_dict(ext_param_dict, cameras=[None])


@pytest.mark.parametrize(
    'dialect',
    [
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
    ],
)
def test_csv_reader_dialect(
    odm_lla_rpy_csv_file: Path,
    odm_crs: str,
    odm_image_files: list[Path],
    odm_reconstruction_file: Path,
    dialect: dict,
    tmp_path: Path,
):
    """Test reading exterior parameters from CSV files in different dialects."""
    # create test CSV file
    test_filename = tmp_path.joinpath('ext-param-test.csv')
    with open(odm_lla_rpy_csv_file, 'r') as fin:
        with open(test_filename, 'w', newline='') as fout:
            reader = csv.reader(fin, delimiter=' ', quotechar='"')
            writer = csv.writer(fout, **dialect)
            for row in reader:
                writer.writerow(row)

    # read test file
    reader = param_io.CsvReader(test_filename, crs=odm_crs)
    for attr in ['delimiter', 'quotechar']:
        assert getattr(reader._dialect, attr) == dialect[attr]
    ext_param_dict = reader.read_ext_param()

    # validate dict
    file_keys = [filename.name for filename in odm_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    with open(odm_reconstruction_file, 'r') as f:
        json_obj = json.load(f)
    cam_id = next(iter(json_obj[0]['cameras'].keys())).strip('v2 ')
    _validate_ext_param_dict(ext_param_dict, cameras=[cam_id])


def test_osfm_reader(odm_reconstruction_file: Path, odm_crs: str):
    """Test OsfmReader reads interior and exterior parameters successfully."""
    reader = param_io.OsfmReader(odm_reconstruction_file, crs=odm_crs)
    assert reader.crs == rio.CRS.from_string(odm_crs)

    int_param_dict = reader.read_int_param()
    int_cam_ids = set(int_param_dict.keys())
    ext_param_dict = reader.read_ext_param()
    ext_cam_ids = set([ext_param['camera'] for ext_param in ext_param_dict.values()])

    _validate_int_param_dict(int_param_dict)
    _validate_ext_param_dict(ext_param_dict, cameras=int_cam_ids)
    assert ext_cam_ids.issubset(int_cam_ids)


def test_osfm_reader_auto_crs(odm_reconstruction_file: Path, odm_crs: str):
    """Test OsfmReader auto determines a UTM CRS correctly."""
    reader = param_io.OsfmReader(odm_reconstruction_file, crs=None)
    assert reader.crs == rio.CRS.from_string(odm_crs)


def test_osfm_reader_validity_error(ngi_oty_ext_param_file: Path):
    """Test OsfmReader raises an error with an invalid file format."""
    with pytest.raises(ParamError) as ex:
        _ = param_io.OsfmReader(ngi_oty_ext_param_file, crs=None)
    assert 'Could not parse' in str(ex.value) and ngi_oty_ext_param_file.name in str(ex.value)


def test_exif_reader(odm_image_files: list[Path], odm_crs: str):
    """Test ExifReader reads interior and exterior parameters successfully."""
    reader = param_io.ExifReader(odm_image_files, crs=odm_crs)
    assert reader.crs == rio.CRS.from_string(odm_crs)

    int_param_dict = reader.read_int_param()
    int_cam_ids = set(int_param_dict.keys())
    ext_param_dict = reader.read_ext_param()
    ext_cam_ids = set([ext_param['camera'] for ext_param in ext_param_dict.values()])

    _validate_int_param_dict(int_param_dict)
    _validate_ext_param_dict(ext_param_dict, cameras=int_cam_ids)
    assert ext_cam_ids.issubset(int_cam_ids)


def test_exif_reader_ext_values(odm_image_files: list[Path], odm_crs: str, odm_reconstruction_file):
    """Test exterior parameter values from ExifReader against those from OsfmReader."""
    ref_reader = param_io.OsfmReader(odm_reconstruction_file, crs=odm_crs)
    ref_ext_param_dict = ref_reader.read_ext_param()
    test_reader = param_io.ExifReader(odm_image_files, crs=odm_crs)
    test_ext_param_dict = test_reader.read_ext_param()

    for filename, test_ext_param in test_ext_param_dict.items():
        ref_ext_param = ref_ext_param_dict[Path(filename).stem]
        assert test_ext_param['xyz'] == pytest.approx(ref_ext_param['xyz'], abs=0.1)
        assert test_ext_param['opk'] == pytest.approx(ref_ext_param['opk'], abs=0.1)


def test_exif_reader_auto_crs(odm_image_files: list[Path], odm_crs: str):
    """Test ExifReader auto determines a UTM CRS correctly."""
    reader = param_io.ExifReader(odm_image_files, crs=None)
    assert reader.crs == rio.CRS.from_string(odm_crs)


def test_exif_reader_lla_crs(odm_image_files: list[Path], odm_crs: str, wgs84_egm2008_crs: str):
    """Test ExifReader exterior parameters are affected by lla_crs as expected."""
    ref_reader = param_io.ExifReader(odm_image_files, crs=odm_crs)
    test_reader = param_io.ExifReader(
        odm_image_files, crs=odm_crs + '+4326', lla_crs=wgs84_egm2008_crs
    )
    assert test_reader._lla_crs == rio.CRS.from_string(wgs84_egm2008_crs)

    ref_ext_param_dict = ref_reader.read_ext_param()
    test_ext_param_dict = test_reader.read_ext_param()
    for test_key, test_ext_params in test_ext_param_dict.items():
        # test z offset changes and rotation stays same
        ref_ext_params = ref_ext_param_dict[test_key]
        assert test_ext_params['xyz'][:2] == pytest.approx(ref_ext_params['xyz'][:2], abs=1e-6)
        assert test_ext_params['xyz'][2] != pytest.approx(ref_ext_params['xyz'][2], abs=1)
        assert test_ext_params['opk'] == pytest.approx(ref_ext_params['opk'], abs=1e-4)


def test_exif_reader_empty():
    """Test ExifReader with empty list of files."""
    reader = param_io.ExifReader([], crs=None)
    assert reader.crs is None
    assert reader.read_int_param() == {}
    assert reader.read_ext_param() == {}


def test_exif_reader_progress(odm_image_files: list[Path], capsys: pytest.CaptureFixture):
    """Test ExifReader progress bar display."""
    # default bar
    param_io.ExifReader(odm_image_files, progress=True)
    cap = capsys.readouterr()
    assert 'files' in cap.err and '100%' in cap.err

    # no bar
    param_io.ExifReader(odm_image_files, progress=False)
    cap = capsys.readouterr()
    assert 'files' not in cap.err and '100%' not in cap.err

    # custom bar
    desc = 'custom'
    param_io.ExifReader(odm_image_files, progress=dict(desc=desc))
    cap = capsys.readouterr()
    assert desc in cap.err


def test_oty_rw_ext_param(mult_ext_param_dict: dict, utm34n_crs: str, tmp_path: Path):
    """Test exterior parameter read / write from / to orthority exterior parameter format."""
    ext_param_file = tmp_path.joinpath('ext_param.geojson')
    param_io.write_ext_param(ext_param_file, mult_ext_param_dict, crs=utm34n_crs)
    assert ext_param_file.exists()

    reader = param_io.OtyReader(ext_param_file)
    test_ext_param_dict = reader.read_ext_param()
    assert test_ext_param_dict == mult_ext_param_dict
    assert reader.crs == rio.CRS.from_string(utm34n_crs)


def test_oty_reader_validity_error(odm_reconstruction_file: Path):
    """Test OtyReader raises an error with an invalid file format."""
    with pytest.raises(ParamError) as ex:
        _ = param_io.OtyReader(odm_reconstruction_file)
    assert 'Could not parse' in str(ex.value) and odm_reconstruction_file.name in str(ex.value)


def test_oty_reader_crs(ngi_oty_ext_param_file: Path, ngi_crs: str):
    """Test OtyReader reads the crs correctly."""
    reader = param_io.OtyReader(ngi_oty_ext_param_file)
    assert reader.crs == rio.CRS.from_string(ngi_crs)


def test_rw_oty_rpc_param(rpc_args: dict, tmp_path: Path):
    """Test writing and reading RPC parameters to/from an Orthority RPC file."""
    rpc_param = dict(cam_type=CameraType.rpc, **rpc_args)
    rpc_param_dict = dict(file1=rpc_param, file2=rpc_param)
    param_file = tmp_path.joinpath('rpc_param.yaml')
    param_io.write_rpc_param(param_file, rpc_param_dict)
    assert param_io.read_oty_rpc_param(param_file) == rpc_param_dict


def test_read_oty_rpc_param_error(ngi_oty_int_param_file: Path):
    """Test reading RPC parameters raises an error with an invalid parameter file."""
    with pytest.raises(ParamError) as ex:
        param_io.read_oty_rpc_param(ngi_oty_int_param_file)
    assert 'Could not parse' in str(ex.value) and ngi_oty_int_param_file.name in str(ex.value)


def test_read_im_rpc_param(rpc: dict, tmp_path: Path):
    """Test reading RPC parameters from GeoTIFFs."""
    # create image files with known RPCs, and corresponding RPC parameter dictionary
    im_size = (1, 1)
    im_files = (tmp_path.joinpath('rpc1.tif'), tmp_path.joinpath('rpc2.tif'))
    rpc_param = dict(cam_type=CameraType.rpc, im_size=im_size, rpc=rpc)
    array = np.zeros((1, *im_size[::-1]), dtype='uint8')
    # NB: there is a rio/gdal bug if RPCs are passed as a dict to rio.open()
    profile = create_profile(array, rpcs=rio.transform.RPC(**rpc))
    ref_rpc_param_dict = {}
    for im_file in im_files:
        ref_rpc_param_dict[im_file.name] = rpc_param
        with rio.open(im_file, 'w', **profile) as im:
            im.write(array)

    # read image file rpc params and compare parameter dicts
    def compare_objs(ref_obj, test_obj):
        if isinstance(ref_obj, dict):
            for k, v in ref_obj.items():
                assert k in test_obj, k
                compare_objs(v, test_obj[k])
        else:
            assert ref_obj == pytest.approx(test_obj, rel=0.001), (ref_obj, test_obj)

    test_rpc_param_dict = param_io.read_im_rpc_param(im_files)
    compare_objs(ref_rpc_param_dict, test_rpc_param_dict)


def test_read_im_rpc_param_error(ngi_image_file: Path):
    """Test reading RPC parameters from a GeoTIFF without RPC coefficients raises an error."""
    with pytest.raises(ParamError) as ex:
        param_io.read_im_rpc_param((ngi_image_file,))
    assert 'No RPC parameters' in str(ex.value) and ngi_image_file.name in str(ex.value)


def test_read_im_rpc_param_progress(rpc_image_file: Path, capsys: pytest.CaptureFixture):
    """Test the progress bar display when reading RPC parameters from GeoTIFFs."""
    files = (rpc_image_file,)
    # default bar
    param_io.read_im_rpc_param(files, progress=True)
    cap = capsys.readouterr()
    assert 'files' in cap.err and '100%' in cap.err

    # no bar
    param_io.read_im_rpc_param(files, progress=False)
    cap = capsys.readouterr()
    assert 'files' not in cap.err and '100%' not in cap.err

    # custom bar
    desc = 'custom'
    param_io.read_im_rpc_param(files, progress=dict(desc=desc))
    cap = capsys.readouterr()
    assert desc in cap.err


def test_rw_oty_gcps(im_size: tuple[int, int], xyz: tuple[float, float, float], tmp_path: Path):
    """Test writing and reading GCPs to/from an Orthority GCP file."""
    gcp = dict(ji=im_size, xyz=xyz, id='id', info='info')
    gcp_dict = dict(file1=[gcp, gcp], file2=[gcp])
    gcp_file = tmp_path.joinpath('gcps.geojson')
    param_io.write_gcps(gcp_file, gcp_dict)
    assert param_io.read_oty_gcps(gcp_file) == gcp_dict


def test_read_oty_gcps_error(ngi_oty_ext_param_file: Path):
    """Test reading GCPs raises an error with an invalid parameter file."""
    with pytest.raises(ParamError) as ex:
        param_io.read_oty_gcps(ngi_oty_ext_param_file)
    assert 'Could not parse' in str(ex.value) and ngi_oty_ext_param_file.name


def test_read_im_gcps(tmp_path: Path):
    """Test reading GCPs from GeoTIFFs."""
    # create image files with GCPs and corresponding GCP dictionary (id and info items are
    # omitted as rasterio doesn't write # these)
    im_size = (1, 1)
    im_files = (tmp_path.joinpath('gcp1.tif'), tmp_path.joinpath('gcp2.tif'))
    gcps = [
        dict(ji=(1.0, 2.0), xyz=(10.0, 20.0, 1000.0)),
        dict(ji=(1.11, 2.22), xyz=(11.1, 22.2, 1111.1)),
    ]
    # convert oty GCPs to rasterio format
    rio_gcps = [GroundControlPoint(*gcp['ji'][::-1], *gcp['xyz']) for gcp in gcps]
    array = np.zeros((1, *im_size[::-1]), dtype='uint8')
    profile = create_profile(array, crs='EPSG:4979', gcps=rio_gcps)
    ref_gcp_dict = {}
    for im_file in im_files:
        ref_gcp_dict[im_file.name] = gcps
        with rio.open(im_file, 'w', **profile) as im:
            im.write(array)

    # read image file GCPs and compare (ignoring id and info items)
    def compare_objs(ref_obj, test_obj):
        if isinstance(ref_obj, dict):
            for k, v in ref_obj.items():
                assert k in test_obj, k
                compare_objs(v, test_obj[k])
        elif isinstance(ref_obj, list):
            assert len(ref_obj) == len(test_obj)
            for i in range(len(ref_obj)):
                compare_objs(ref_obj[i], test_obj[i])
        else:
            assert ref_obj == pytest.approx(test_obj, rel=1e-6), (ref_obj, test_obj)

    test_gcp_dict = param_io.read_im_gcps(im_files)
    compare_objs(ref_gcp_dict, test_gcp_dict)


def test_read_im_gcp_crs(xyz: tuple[float, float, float], utm34n_egm2008_crs: str, tmp_path: Path):
    """Test reading GCPs from GeoTIFF transforms the GCP CRS to EPSG:4979."""
    # create an image file with a GCP in non EPSG:4979 CRS
    im_size = (1, 1)
    im_file = tmp_path.joinpath('gcp.tif')
    gcps = [dict(ji=(1.0, 2.0), xyz=xyz)]
    # convert oty GCPs to rasterio format
    rio_gcps = [GroundControlPoint(*gcp['ji'][::-1], *gcp['xyz']) for gcp in gcps]
    array = np.zeros((1, *im_size[::-1]), dtype='uint8')
    profile = create_profile(array, crs=utm34n_egm2008_crs, gcps=rio_gcps)
    with rio.open(im_file, 'w', **profile) as im:
        im.write(array)

    # test GCP coordinate transform
    test_gcp_dict = param_io.read_im_gcps([im_file])
    ref_xyz = transform(utm34n_egm2008_crs, 'EPSG:4979', *[[coord] for coord in xyz])
    ref_xyz = tuple([coord[0] for coord in ref_xyz])
    test_xyz = test_gcp_dict[im_file.name][0]['xyz']
    assert test_xyz == pytest.approx(ref_xyz, abs=1e-6)


def test_read_im_gcps_error(ngi_image_file: Path):
    """Test reading GCPs from a GeoTIFF without GCPs raises an error."""
    with pytest.raises(ParamError) as ex:
        param_io.read_im_gcps((ngi_image_file,))
    assert 'No GCPs' in str(ex.value) and ngi_image_file.name in str(ex.value)


def test_read_im_gcps_progress(rpc_image_file: Path, capsys: pytest.CaptureFixture):
    """Test the progress bar display when reading GCPs from GeoTIFFs."""
    files = (rpc_image_file,)
    # default bar
    param_io.read_im_gcps(files, progress=True)
    cap = capsys.readouterr()
    assert 'files' in cap.err and '100%' in cap.err

    # no bar
    param_io.read_im_gcps(files, progress=False)
    cap = capsys.readouterr()
    assert 'files' not in cap.err and '100%' not in cap.err

    # custom bar
    desc = 'custom'
    param_io.read_im_gcps(files, progress=dict(desc=desc))
    cap = capsys.readouterr()
    assert desc in cap.err
