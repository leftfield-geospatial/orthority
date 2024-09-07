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

import copy
import warnings
from pathlib import Path

import pytest
import rasterio as rio

from orthority import param_io
from orthority.camera import BrownCamera, FrameCamera
from orthority.enums import CameraType, RpcRefine
from orthority.errors import CrsMissingError, ParamError
from orthority.factory import FrameCameras, RpcCameras


@pytest.mark.parametrize(
    'int_param_file, ext_param_file',
    [
        ('ngi_oty_int_param_file', 'ngi_oty_ext_param_file'),
        ('ngi_legacy_config_file', 'ngi_xyz_opk_csv_file'),
        ('odm_reconstruction_file', 'odm_reconstruction_file'),
        ('odm_reconstruction_file', 'odm_lla_rpy_csv_file'),  # test osfm interior / exterior logic
    ],
)
def test_frame_cameras_init(
    int_param_file: str, ext_param_file: str, request: pytest.FixtureRequest
):
    """Test creating a frame camera factory with different parameter files."""
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
    """Test creating a frame camera factory with parameter dictionaries."""
    io_kwargs = dict(crs=utm34n_crs)
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict, io_kwargs=io_kwargs)
    assert cameras._int_param_dict == mult_int_param_dict
    assert cameras._ext_param_dict == mult_ext_param_dict

    # test crs is passed through
    assert cameras.crs == rio.CRS.from_string(io_kwargs['crs'])


def test_frame_cameras_init_io_kwargs(
    ngi_legacy_config_file: Path, ngi_legacy_csv_file: Path, ngi_crs: str
):
    """Test the frame camera factory passes ``io_kwargs`` through to the parameter reader."""
    io_kwargs = dict(crs=ngi_crs)
    cameras = FrameCameras(ngi_legacy_config_file, ngi_legacy_csv_file, io_kwargs=io_kwargs)
    assert cameras.crs == rio.CRS.from_string(ngi_crs)


def test_frame_cameras_init_cam_kwargs(odm_reconstruction_file: Path, odm_image_file: Path):
    """Test the frame camera factory passes ``cam_kwargs`` through to the camera(s)."""
    cam_kwargs = dict(distort=False, alpha=0.0)
    cameras = FrameCameras(odm_reconstruction_file, odm_reconstruction_file, cam_kwargs=cam_kwargs)
    camera = cameras.get(odm_image_file)
    assert camera.distort == cam_kwargs['distort']
    assert camera.alpha == cam_kwargs['alpha']


def test_frame_cameras_init_error(ngi_oty_int_param_file: Path, ngi_oty_ext_param_file: Path):
    """Test the frame camera factory raises errors with unrecognised parameter file extensions."""
    with pytest.raises(ParamError) as ex:
        FrameCameras(ngi_oty_ext_param_file, ngi_oty_ext_param_file)
    assert 'not supported' in str(ex.value)
    with pytest.raises(ParamError) as ex:
        FrameCameras(ngi_oty_int_param_file, ngi_oty_int_param_file)
    assert 'not supported' in str(ex.value)


def test_frame_cameras_from_images(odm_image_files: list[Path]):
    """Test creating a frame camera factory from images."""
    cameras = FrameCameras.from_images(odm_image_files)
    assert cameras.crs is not None
    assert cameras.filenames == {Path(fn).name for fn in odm_image_files}
    assert len(cameras._int_param_dict) > 0
    assert len(cameras._ext_param_dict) > 0


def test_frame_cameras_from_images_kwargs(odm_image_files: list[Path], ngi_crs: str):
    """Test creating a frame camera factory from images passes ``io_kwargs`` and ``cam_kwargs``
    through.
    """
    io_kwargs = dict(crs=ngi_crs)
    cam_kwargs = dict(distort=False, alpha=0.0)
    cameras = FrameCameras.from_images(odm_image_files, io_kwargs=io_kwargs, cam_kwargs=cam_kwargs)
    assert cameras.crs == rio.CRS.from_string(ngi_crs)

    camera = cameras.get(odm_image_files[0])
    assert camera.distort == cam_kwargs['distort']
    assert camera.alpha == cam_kwargs['alpha']


def test_frame_cameras_get(odm_reconstruction_file: Path, odm_image_files: list[Path]):
    """Test ``FrameCameras.get()`` for known filenames."""
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
    """Test ``FrameCameras.get()`` with multiple camera ID interior & exterior parameters."""
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict)
    for file in mult_ext_param_dict.keys():
        camera = cameras.get(file)
        assert isinstance(camera, FrameCamera)


def test_frame_cameras_get_single_camera(pinhole_int_param_dict: dict, xyz: tuple, opk: tuple):
    """Test ``FrameCameras.get()`` with an exterior parameter camera ID of ``None`` and a single
    set of interior parameters.
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
    filename.
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
    """Test ``FrameCameras.get()`` raises an error when the exterior parameter camera ID is
    ``None`` and there are multiple sets of interior parameters.
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
    """Test writing frame camera interior and exterior parameter files."""
    io_kwargs = dict(crs=utm34n_crs)
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict, io_kwargs=io_kwargs)

    cameras.write_param(tmp_path)
    assert tmp_path.joinpath('int_param.yaml').exists()
    assert tmp_path.joinpath('ext_param.geojson').exists()


def test_frame_cameras_write_param_overwrite(
    mult_int_param_dict: dict, mult_ext_param_dict: dict, utm34n_crs: str, tmp_path: Path
):
    """Test writing frame camera parameter files raises an error if the destination file(s) exist,
    and overwrites them when ``overwrite=True``.
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
    """Test writing frame camera parameter files raises an error if there is no world CRS."""
    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict)
    with pytest.raises(CrsMissingError):
        cameras.write_param(tmp_path)


def test_rpc_cameras_init(rpc_param_file: Path):
    """Test creating a RPC camera factory instance from a parameter file."""
    cameras = RpcCameras(rpc_param_file)
    assert cameras.filenames == set(cameras._rpc_param_dict.keys())
    assert len(cameras._rpc_param_dict) == 1


def test_rpc_cameras_init_dict(rpc_args: dict):
    """Test creating a RPC camera factory from a parameter dictionary."""
    rpc_param_dict = {'src_image.tif': dict(cam_type=CameraType.rpc, **rpc_args)}
    cameras = RpcCameras(rpc_param_dict)
    assert cameras._rpc_param_dict == rpc_param_dict


def test_rpc_cameras_init_cam_kwargs(rpc_param_file: Path, ngi_crs: str):
    """Test the RPC camera factory passes ``cam_kwargs`` through to the camera(s)."""
    cam_kwargs = dict(crs=rio.CRS.from_string(ngi_crs))
    cameras = RpcCameras(rpc_param_file, cam_kwargs=cam_kwargs)
    camera = cameras.get(list(cameras.filenames)[0])
    assert camera.crs == cam_kwargs['crs']


def test_rpc_cameras_from_images(rpc_image_file: Path):
    """Test creating a RPC camera factory from images."""
    cameras = RpcCameras.from_images((rpc_image_file,))
    assert cameras.filenames == {rpc_image_file.name}
    assert len(cameras._rpc_param_dict) == 1


def test_rpc_cameras_from_images_kwargs(
    rpc_image_file: Path, ngi_crs: str, capsys: pytest.CaptureFixture
):
    """Test creating a RPC camera factory from images passes ``io_kwargs`` and ``cam_kwargs``
    through.
    """
    desc = 'custom'
    io_kwargs = dict(progress=dict(desc=desc))
    cam_kwargs = dict(crs=rio.CRS.from_string(ngi_crs))
    cameras = RpcCameras.from_images((rpc_image_file,), io_kwargs=io_kwargs, cam_kwargs=cam_kwargs)

    cap = capsys.readouterr()
    assert desc in cap.err

    camera = cameras.get(list(cameras.filenames)[0])
    assert camera.crs == cam_kwargs['crs']


def test_rpc_cameras_get_filename_not_found_error(rpc_param_file: Path):
    """Test ``RpcCameras.get()`` raises an error with an invalid filename."""
    cameras = RpcCameras(rpc_param_file)
    filename = 'unknown.tif'
    with pytest.raises(ParamError) as ex:
        cameras.get(filename)
    assert filename in str(ex.value) and 'could not find' in str(ex.value).lower()


def test_rpc_cameras_refine(rpc_param_file: Path, rpc_image_file: Path, gcp_file: Path):
    """Test refining a RPC camera factory with various sources of GCPs."""

    def test_refine(gcp_arg):
        """Test refining with GCPs in ``gcp_arg`` changes the RPC params."""
        cameras = RpcCameras(rpc_param_file)
        ref_rpc_param = copy.deepcopy(cameras._rpc_param_dict[rpc_image_file.name]['rpc'])
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # fail on warning that no GCPs found
            cameras.refine(gcp_arg)
        assert cameras._rpc_param_dict.keys() == cameras._gcp_dict.keys()
        assert cameras._rpc_param_dict != ref_rpc_param

    test_refine([rpc_image_file])
    test_refine(gcp_file)
    test_refine(param_io.read_oty_gcps(gcp_file))


def test_rpc_cameras_refine_io_kwargs(
    rpc_param_file: Path, rpc_image_file: Path, capsys: pytest.CaptureFixture
):
    """Test refining a RPC camera factory passes through ``io_kwargs`` when the GCPs are read
    from image tags.
    """
    desc = 'custom'
    cameras = RpcCameras(rpc_param_file)
    cameras.refine([rpc_image_file], io_kwargs=dict(progress=dict(desc=desc)))
    cap = capsys.readouterr()
    assert desc in cap.err


def test_rpc_cameras_refine_ref_kwargs(rpc_param_file: Path, rpc_image_file: Path):
    """Test refining a RPC camera factory passes through ``fit_kwargs`` to ``fit.refine_rpc()``."""
    cameras = RpcCameras(rpc_param_file)
    cameras.refine([rpc_image_file], fit_kwargs=dict(method=RpcRefine.shift))
    shift_rpc_param = copy.deepcopy(cameras._rpc_param_dict[rpc_image_file.name]['rpc'])

    cameras = RpcCameras(rpc_param_file)
    cameras.refine([rpc_image_file], fit_kwargs=dict(method=RpcRefine.shift_drift))
    shift_drift_rpc_param = cameras._rpc_param_dict[rpc_image_file.name]['rpc']

    assert shift_drift_rpc_param != shift_rpc_param


def test_rpc_cameras_refine_get(rpc_param_file: Path, rpc_image_file: Path):
    """Test ``RpcCameras.get()`` returns a different camera after refinement."""
    cameras = RpcCameras(rpc_param_file)
    camera = cameras.get(rpc_image_file.name)
    cameras.refine([rpc_image_file])
    refined_camera = cameras.get(rpc_image_file.name)

    assert camera != refined_camera and camera._rpc != refined_camera._rpc
