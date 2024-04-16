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

from pathlib import Path

import pytest
import rasterio as rio

from orthority.camera import BrownCamera, FrameCamera
from orthority.errors import CrsMissingError, ParamError
from orthority.factory import FrameCameras


@pytest.mark.parametrize(
    'int_param_file, ext_param_file',
    [
        ('ngi_oty_int_param_file', 'ngi_oty_ext_param_file'),
        ('ngi_legacy_config_file', 'ngi_xyz_opk_csv_file'),
        ('odm_reconstruction_file', 'odm_reconstruction_file'),
    ],
)
def test_frame_cameras_init(
    int_param_file: str, ext_param_file: str, request: pytest.FixtureRequest
):
    """Test creating a ``FrameCameras`` instance with different parameter files."""
    int_param_file: Path = request.getfixturevalue(int_param_file)
    ext_param_file: Path = request.getfixturevalue(ext_param_file)

    cameras = FrameCameras(int_param_file, ext_param_file)
    assert cameras.crs is not None
    assert cameras.filenames == set(cameras._ext_param_dict.keys())
    assert len(cameras._int_param_dict) > 0
    assert len(cameras._ext_param_dict) > 0


def test_frame_cameras_init_dicts(
    mult_int_param_dict: dict, mult_ext_param_dict: dict, utm34n_crs: str
):
    """Test creating a ``FrameCameras`` instance with parameter dictionaries."""
    io_kwargs = dict(crs=utm34n_crs)
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict, io_kwargs=io_kwargs)
    assert cameras._int_param_dict == mult_int_param_dict
    assert cameras._ext_param_dict == mult_ext_param_dict

    # test crs is passed through
    assert cameras.crs == rio.CRS.from_string(io_kwargs['crs'])


def test_frame_cameras_init_io_kwargs(
    ngi_legacy_config_file: Path, ngi_legacy_csv_file: Path, ngi_crs: str
):
    """Test ``FrameCameras.__init__()`` passes ``io_kwargs`` through to the parameter reader."""
    io_kwargs = dict(crs=ngi_crs)
    cameras = FrameCameras(ngi_legacy_config_file, ngi_legacy_csv_file, io_kwargs=io_kwargs)
    assert cameras.crs == rio.CRS.from_string(ngi_crs)


def test_frame_cameras_init_cam_kwargs(odm_reconstruction_file: Path, odm_image_file: Path):
    """Test ``FrameCameras.__init__()`` passes ``cam_kwargs`` through to the camera(s)."""
    cam_kwargs = dict(distort=False, alpha=0.0)
    cameras = FrameCameras(odm_reconstruction_file, odm_reconstruction_file, cam_kwargs=cam_kwargs)
    camera = cameras.get(odm_image_file)
    assert camera.distort == cam_kwargs['distort']
    assert camera.alpha == cam_kwargs['alpha']


def test_frame_cameras_init_error(ngi_oty_int_param_file: Path, ngi_oty_ext_param_file: Path):
    """Test ``FrameCameras.__init__()`` raises errors with unrecognised parameter file extensions."""
    with pytest.raises(ParamError) as ex:
        FrameCameras(ngi_oty_ext_param_file, ngi_oty_ext_param_file)
    assert 'not supported' in str(ex.value)
    with pytest.raises(ParamError) as ex:
        FrameCameras(ngi_oty_int_param_file, ngi_oty_int_param_file)
    assert 'not supported' in str(ex.value)


def test_frame_cameras_get(odm_reconstruction_file: Path, odm_image_files: tuple[Path, ...]):
    """Test ``FrameCameras.get()`` creates valid cameras for known filenames."""
    cameras = FrameCameras(odm_reconstruction_file, odm_reconstruction_file)
    assert cameras.filenames == {Path(fn).stem for fn in odm_image_files}
    for file in odm_image_files:
        camera = cameras.get(file)
        assert isinstance(camera, BrownCamera)

        # basic test on camera configuration
        ext_param = cameras._ext_param_dict[file.stem]
        int_param = cameras._int_param_dict[ext_param['camera']]
        assert camera.im_size == int_param['im_size']
        assert camera.pos == ext_param['xyz']


def test_frame_cameras_get_mult_camera(
    mult_int_param_dict: dict, mult_ext_param_dict: dict, utm34n_crs: str
):
    """Test ``FrameCameras.get()`` with multiple multiple camera ID interior & exterior parameters."""
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict)
    for file in mult_ext_param_dict.keys():
        camera = cameras.get(file)
        assert isinstance(camera, FrameCamera)


def test_frame_cameras_get_single_camera(pinhole_int_param_dict: dict, xyz: tuple, opk: tuple):
    """Test ``FrameCameras.get()`` with an exterior parameter camera ID of None and a single set of
    interior parameters.
    """
    filename = 'some_file'
    ext_param_dict = {filename: dict(xyz=xyz, opk=opk, camera=None)}
    cameras = FrameCameras(pinhole_int_param_dict, ext_param_dict)
    camera = cameras.get(filename)
    assert isinstance(camera, FrameCamera)


def test_frame_cameras_get_filename_not_found_error(
    pinhole_int_param_dict: dict, xyz: tuple, opk: tuple
):
    """Test ``FrameCameras.get()`` raises an error when there are no exterior parameters for the
    ``filename``.
    """
    filename = 'some_file'
    ext_param_dict = {'unknown': dict(xyz=xyz, opk=opk, camera=None)}
    cameras = FrameCameras(pinhole_int_param_dict, ext_param_dict)

    with pytest.raises(ParamError) as ex:
        cameras.get(filename)
    assert filename in str(ex.value) and 'exterior parameters' in str(ex.value)


def test_frame_cameras_get_camera_not_found_error(
    pinhole_int_param_dict: dict, xyz: tuple, opk: tuple
):
    """Test ``FrameCameras.get()`` raises an error when there are no interior parameters for the
    exterior parameter camera ID.
    """
    filename = 'some_file'
    camera = 'other'
    ext_param_dict = {filename: dict(xyz=xyz, opk=opk, camera=camera)}
    cameras = FrameCameras(pinhole_int_param_dict, ext_param_dict)

    with pytest.raises(ParamError) as ex:
        cameras.get(filename)
    assert camera in str(ex.value) and 'interior parameters' in str(ex.value)


def test_frame_cameras_get_no_cam_id_error(mult_int_param_dict: dict, xyz: tuple, opk: tuple):
    """Test ``FrameCameras.get()`` raises an error when the exterior parameter camera ID is None and
    there are multiple sets of interior parameters.
    """
    filename = 'some_file'
    ext_param_dict = {filename: dict(xyz=xyz, opk=opk, camera=None)}
    cameras = FrameCameras(mult_int_param_dict, ext_param_dict)

    with pytest.raises(ParamError) as ex:
        cameras.get(filename)
    assert filename in str(ex.value) and "'camera' ID" in str(ex.value)


def test_frame_cameras_write_param(
    mult_int_param_dict: dict, mult_ext_param_dict: dict, utm34n_crs: str, tmp_path: Path
):
    """Test ``FrameCameras.write_param()`` creates interior and exterior parameter files."""
    io_kwargs = dict(crs=utm34n_crs)
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict, io_kwargs=io_kwargs)

    cameras.write_param(tmp_path)
    assert tmp_path.joinpath('int_param.yaml').exists()
    assert tmp_path.joinpath('ext_param.geojson').exists()


def test_frame_cameras_write_param_overwrite(
    mult_int_param_dict: dict, mult_ext_param_dict: dict, utm34n_crs: str, tmp_path: Path
):
    """Test ``FrameCameras.write_param()`` raises an error if the destination file(s) exist,
    and overwrites them if ``overwrite=True``.
    """
    io_kwargs = dict(crs=utm34n_crs)
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict, io_kwargs=io_kwargs)

    int_param_file = tmp_path.joinpath('int_param.yaml')
    int_param_file.touch()
    with pytest.raises(FileExistsError) as ex:
        cameras.write_param(tmp_path)
    assert int_param_file.name in str(ex.value)

    cameras.write_param(tmp_path, overwrite=True)
    assert tmp_path.joinpath('ext_param.geojson').exists()


def test_frame_cameras_write_param_crs_error(
    mult_int_param_dict: dict, mult_ext_param_dict: dict, tmp_path: Path
):
    """Test ``FrameCameras.write_param()`` raises an error if there is no world CRS."""
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict)
    with pytest.raises(CrsMissingError):
        cameras.write_param(tmp_path)


def test_frame_cameras_from_images(odm_image_files: tuple[Path, ...]):
    """Test ``FrameCameras.from_images()``."""
    cameras = FrameCameras.from_images(odm_image_files)
    assert cameras.crs is not None
    assert cameras.filenames == {Path(fn).name for fn in odm_image_files}
    assert len(cameras._int_param_dict) > 0
    assert len(cameras._ext_param_dict) > 0


def test_frame_cameras_from_images_kwargs(odm_image_files: tuple[Path, ...], ngi_crs: str):
    """Test ``FrameCameras.from_images()`` passes through ``io_args`` to the ``ExifReader`` and
    ``cam_args`` to the camera(s).
    """
    io_kwargs = dict(crs=ngi_crs)
    cam_kwargs = dict(distort=False, alpha=0.0)
    cameras = FrameCameras.from_images(odm_image_files, io_kwargs=io_kwargs, cam_kwargs=cam_kwargs)
    assert cameras.crs == rio.CRS.from_string(ngi_crs)

    camera = cameras.get(odm_image_files[0])
    assert camera.distort == cam_kwargs['distort']
    assert camera.alpha == cam_kwargs['alpha']
