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

from math import ceil

import cv2
import numpy as np
import pytest
from rasterio.rpc import RPC
from rasterio.warp import transform

from orthority import fit
from orthority.camera import FrameCamera, RpcCamera, create_camera
from orthority.enums import CameraType, RpcRefine
from orthority.errors import OrthorityWarning
from tests.test_param_io import _validate_ext_param_dict, _validate_int_param_dict


@pytest.fixture(scope='session')
def grid_ji(im_size: tuple[int, int]) -> np.ndarray:
    """A 2-by-N array of pixel coordinates lying on a grid inside ``im_size``."""
    step = 10
    j, i = (
        np.arange(step, im_size[0] - step + 1, step),
        np.arange(step, im_size[1] - step + 1, step),
    )
    jgrid, igrid = np.meshgrid(j, i, indexing='xy')
    return np.array((jgrid.reshape(-1), igrid.reshape(-1)))


@pytest.mark.parametrize('shift, drift', [((5.0, 10.0), None), ((5.0, 10.0), (1.2, 0.8))])
def test_refine_rpc(
    rpc: dict, im_size: tuple[int, int], shift: tuple[float, float], drift: tuple[float, float]
):
    """Test ``refine_rpc()`` correctly refines an RPC model by testing it against refinement GCPs."""
    # create affine transform to realise shift & drift
    method = RpcRefine.shift if not drift else RpcRefine.shift_drift
    drift = (1.0, 1.0) if not drift else drift
    refine_tform = np.eye(2, 3)
    refine_tform[:, -1] = shift
    refine_tform[:, :2] = np.diag(drift)

    # generate GCPs randomly spread over the image, with known pixel shift/drift
    n = 10
    camera = RpcCamera(im_size, rpc)
    ji_rpc = (np.random.rand(2, n) - 0.5) * np.array([im_size]).T
    z = rpc['height_off'] / 2 + (np.random.rand(ji_rpc.shape[1]) * rpc['height_off'] / 4)
    xyz = camera.pixel_to_world_z(ji_rpc, z)

    ji_rpc_ = np.vstack((ji_rpc, np.ones(ji_rpc.shape[1])))
    ji_gcp = refine_tform.dot(ji_rpc_)
    gcps = []
    for ji_pt, xyz_pt in zip(ji_gcp.T, xyz.T):
        gcp = dict(ji=tuple(ji_pt.tolist()), xyz=tuple(xyz_pt.tolist()))
        gcps.append(gcp)

    # refine RPC with GCPs
    refined_rpc = fit.refine_rpc(rpc, gcps, method=method)

    # test refined RPC model against original GCPs
    refined_camera = RpcCamera(im_size, refined_rpc)
    xyz_test = refined_camera.pixel_to_world_z(ji_gcp, z)
    assert xyz_test == pytest.approx(xyz, abs=1e-6)


def test_refine_rpc_type(rpc: dict):
    """Test ``refine_rpc()`` works with an RPC dict or :class:`~rasterio.rpc.RPC` object."""
    gcp = dict(
        ji=(rpc['samp_off'], rpc['line_off']),
        xyz=(rpc['long_off'], rpc['lat_off'], rpc['height_off']),
    )
    fit.refine_rpc(rpc, [gcp])
    fit.refine_rpc(RPC(**rpc), [gcp])


@pytest.mark.parametrize('method, min_gcps', [(RpcRefine.shift, 1), (RpcRefine.shift_drift, 2)])
def test_refine_num_gcps(rpc: dict, im_size: tuple[int, int], method: RpcRefine, min_gcps: int):
    """Test ``refine_rpc()`` works with the minimum allowed GCPs and raises an error otherwise."""
    camera = RpcCamera(im_size, rpc)
    gcps = []
    for i in range(min_gcps):
        ji = np.array([rpc['samp_off'], rpc['line_off']]) + np.random.randn(2) * 10
        xyz = camera.pixel_to_world_z(ji.reshape(-1, 1), z=rpc['height_off']).T
        gcp = dict(ji=(*ji,), xyz=(*xyz.squeeze(),))
        gcps.append(gcp)

    fit.refine_rpc(RPC(**rpc), gcps, method=method)
    with pytest.raises(ValueError) as ex:
        fit.refine_rpc(rpc, gcps[:-1], method=method)
    assert 'At least' in str(ex.value)


@pytest.mark.parametrize(
    'cam_type, camera',
    [
        (CameraType.pinhole, 'pinhole_camera'),
        (CameraType.brown, 'brown_camera'),
        (CameraType.opencv, 'opencv_camera'),
        (CameraType.fisheye, 'fisheye_camera'),
    ],
)
def _test_fit_frame_dictionaries(
    cam_type: CameraType, camera: str, grid_ji: np.ndarray, request: pytest.FixtureRequest
):
    """Test fit_frame() returns valid parameter dictionaries."""
    cam: FrameCamera = request.getfixturevalue(camera)

    # create a mock GCP dictionary with multiple images
    xyz = cam.pixel_to_world_z(grid_ji, z=0)
    gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(grid_ji.T, xyz.T)]
    # split gcps over images to avoid fisheye conditioning error
    split_gcps = ceil(len(gcps) / 2)
    gcp_dict = {'file1.ext': gcps[:split_gcps], 'file2.ext': gcps[split_gcps:]}

    # fit parameters
    int_param_dict, ext_param_dict = fit.fit_frame(cam_type, cam.im_size, gcp_dict)

    # test parameter dictionary validity
    assert len(int_param_dict) == 1
    assert ext_param_dict.keys() == gcp_dict.keys()
    _validate_int_param_dict(int_param_dict)
    int_param = next(iter(int_param_dict.values()))
    assert set(int_param.keys()).issuperset(fit._frame_dist_params[cam_type])
    _validate_ext_param_dict(ext_param_dict)


def _test_fit_frame_crs(pinhole_camera: FrameCamera, grid_ji: np.ndarray, utm34n_crs: str):
    """Test fit_frame() crs parameter."""
    # create mock GCP dictionary with coordinates transformed from the reference camera's world
    # CRS (utm34n_crs) to WGS84 geographic
    xyz = pinhole_camera.pixel_to_world_z(grid_ji, z=0)
    lla = np.array(transform(utm34n_crs, 'EPSG:4979', *xyz))
    gcps = [dict(ji=ji_gcp, xyz=lla_gcp) for ji_gcp, lla_gcp in zip(grid_ji.T, lla.T)]
    gcp_dict = {'file.ext': gcps}

    # fit camera params with crs=
    int_param_dict, ext_param_dict = fit.fit_frame(
        CameraType.pinhole, pinhole_camera.im_size, gcp_dict, crs=utm34n_crs
    )

    # create a camera with the fitted parameters and test its world coordinates match those of
    # the reference camera
    int_param = next(iter(int_param_dict.values()))
    ext_param = next(iter(ext_param_dict.values()))
    test_cam = create_camera(**int_param, xyz=ext_param['xyz'], opk=ext_param['opk'])
    test_xyz = test_cam.pixel_to_world_z(grid_ji, z=0)
    assert test_xyz == pytest.approx(xyz, abs=1)


@pytest.mark.parametrize(
    'cam_type, camera',
    [
        (CameraType.pinhole, 'pinhole_camera'),
        (CameraType.brown, 'brown_camera'),
        (CameraType.opencv, 'opencv_camera'),
        (CameraType.fisheye, 'fisheye_camera'),
    ],
)
def _test_fit_frame_min_gcps(
    cam_type: CameraType, camera: str, grid_ji: np.ndarray, request: pytest.FixtureRequest
):
    """Test fit_frame() with the minimum allowed number of GCPs."""
    gcp_cam: FrameCamera = request.getfixturevalue(camera)
    num_params = fit._frame_num_params[cam_type]
    min_gcps = ceil((num_params + 1) / 2)

    # create a grid of at least min_gcps GCPs that approx. covers the image (with a buffer inside
    # the boundary)
    gcps_per_side = ceil(np.sqrt(min_gcps))
    steps = np.array(gcp_cam.im_size) / (gcps_per_side + 1)
    j, i = (
        np.arange(steps[0], gcp_cam.im_size[0], steps[0]),
        np.arange(steps[1], gcp_cam.im_size[1], steps[1]),
    )
    jgrid, igrid = np.meshgrid(j, i, indexing='xy')
    ji = np.array((jgrid.reshape(-1), igrid.reshape(-1)))

    # create a mock GCP dictionary with min_gcps GCPs
    ji = ji[:, :min_gcps]
    xyz = gcp_cam.pixel_to_world_z(ji, z=0)
    gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(ji.T, xyz.T)]
    gcp_dict = {'file.ext': gcps}

    # fit camera params
    int_param_dict, ext_param_dict = fit.fit_frame(cam_type, gcp_cam.im_size, gcp_dict)
    assert len(int_param_dict) == len(ext_param_dict) == 1

    # create a camera with the fitted parameters and test its accuracy
    int_param = next(iter(int_param_dict.values()))
    ext_param = next(iter(ext_param_dict.values()))
    test_cam = create_camera(**int_param, xyz=ext_param['xyz'], opk=ext_param['opk'])

    test_ji = test_cam.world_to_pixel(xyz)
    assert test_ji == pytest.approx(ji, abs=0.5)
    test_xyz = test_cam.pixel_to_world_z(ji, z=0)
    assert test_xyz == pytest.approx(xyz, abs=5)


@pytest.mark.parametrize(
    'cam_type, dist_param',
    [
        (CameraType.pinhole, None),
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
    ],
)
def _test_fit_frame_multiple_images(
    cam_type: CameraType,
    dist_param: str,
    frame_args: dict,
    grid_ji: np.ndarray,
    request: pytest.FixtureRequest,
):
    """Test fit_frame() with multiple image GCPs."""
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}

    # create a mock GCP dictionary with multiple images at different camera positions / orientations
    gcp_dict = {}
    xyz_dict = {}
    gcp_cam = create_camera(cam_type, **frame_args, **dist_param)
    for i, (ext_xyz, ext_opk) in enumerate(
        zip(
            [(2e4, 3e4, 1e3), (3e4, 3e4, 1e3), (3e4, 3e4, 2e3)],
            [(-3.0, 2.0, 10.0), (-15.0, 2.0, 10.0), (-30.0, 20.0, 10.0)],
        )
    ):
        ext_opk = tuple(np.radians(ext_opk))
        gcp_cam.update(xyz=ext_xyz, opk=ext_opk)
        xyz = gcp_cam.pixel_to_world_z(grid_ji, z=0)
        gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(grid_ji.T, xyz.T)]

        key = f'file{i}.ext'
        xyz_dict[key] = xyz
        gcp_dict[key] = gcps

    # fit camera params
    int_param_dict, ext_param_dict = fit.fit_frame(cam_type, gcp_cam.im_size, gcp_dict)

    # configure cameras with the fitted parameters and test accuracy
    int_param = next(iter(int_param_dict.values()))
    test_cam = create_camera(**int_param)
    for filename, ext_param in ext_param_dict.items():
        test_cam.update(xyz=ext_param['xyz'], opk=ext_param['opk'])

        test_ji = test_cam.world_to_pixel(xyz_dict[filename])
        assert test_ji == pytest.approx(grid_ji, abs=0.1)
        test_xyz = test_cam.pixel_to_world_z(grid_ji, z=0)
        assert test_xyz == pytest.approx(xyz_dict[filename], abs=1)


def _test_fit_frame_errors(opencv_camera: FrameCamera, grid_ji: np.ndarray):
    """Test fit_frame() errors and warnings."""
    cam_type = CameraType.opencv
    # create mock GCPs
    xyz = opencv_camera.pixel_to_world_z(grid_ji, z=0)
    gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(grid_ji.T, xyz.T)]

    # test an error is raised with < 4 GCPs in an image
    gcp_dict = {'file1.ext': gcps[:3], 'file2.ext': gcps[3:]}
    with pytest.raises(ValueError, match='At least four'):
        _, _ = fit.fit_frame(cam_type, opencv_camera.im_size, gcp_dict)

    # test an error is raised with less than min number of GCPs (in total) required to fit cam_type
    min_gcps = ceil((1 + fit._frame_num_params[cam_type]) / 2)
    gcp_dict = {'file1.ext': gcps[:4], 'file2.ext': gcps[: min_gcps - 4 - 1]}
    with pytest.raises(ValueError, match='A total of at least'):
        _, _ = fit.fit_frame(cam_type, opencv_camera.im_size, gcp_dict)

    # test a warning is issued with less than the number of GCPs required to globally optimise
    # all cam_type parameters
    gcp_dict = {'file1.ext': gcps[:min_gcps]}
    with pytest.warns(OrthorityWarning, match='will not be globally optimised'):
        try:
            _, _ = fit.fit_frame(cam_type, opencv_camera.im_size, gcp_dict)
        except cv2.error:
            # suppress conditioning error
            pass

    # test an error is raised with non-planar GCPs
    xyz[2] = np.random.randn(1, xyz.shape[1])
    gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(grid_ji.T, xyz.T)]
    gcp_dict = {'file1.ext': gcps}
    with pytest.raises(ValueError, match='should be co-planar'):
        _, _ = fit.fit_frame(cam_type, opencv_camera.im_size, gcp_dict)


def test_fit_frame_exterior_dictionary(
    pinhole_int_param_dict: dict,
    exterior_args: dict,
    grid_ji: np.ndarray,
):
    """Test fit_frame_exterior() returns a valid exterior parameter dictionary."""
    int_param = next(iter(pinhole_int_param_dict.values()))

    # create a mock GCP dictionary with multiple images
    gcp_dict = {}
    gcp_cam = create_camera(**int_param, **exterior_args)
    xyz = gcp_cam.pixel_to_world_z(grid_ji, z=0)
    gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(grid_ji.T, xyz.T)]
    gcp_dict = {'file1.ext': gcps, 'file2.ext': gcps}

    # fit exterior params
    ext_param_dict = fit.fit_frame_exterior(pinhole_int_param_dict, gcp_dict)

    # test dictionary validity
    assert ext_param_dict.keys() == gcp_dict.keys()
    _validate_ext_param_dict(ext_param_dict)


def test_fit_frame_exterior_crs(
    pinhole_int_param_dict: dict, exterior_args: dict, grid_ji: np.ndarray, utm34n_crs: str
):
    """Test fit_frame() crs parameter."""
    # create mock GCP dictionary with coordinates transformed from the reference camera's world
    # CRS (utm34n_crs) to WGS84 geographic
    int_param = next(iter(pinhole_int_param_dict.values()))
    gcp_cam = create_camera(**int_param, **exterior_args)
    xyz = gcp_cam.pixel_to_world_z(grid_ji, z=0)
    lla = np.array(transform(utm34n_crs, 'EPSG:4979', *xyz))
    gcps = [dict(ji=ji_gcp, xyz=lla_gcp) for ji_gcp, lla_gcp in zip(grid_ji.T, lla.T)]
    gcp_dict = {'file.ext': gcps}

    # fit exterior params with crs=
    ext_param_dict = fit.fit_frame_exterior(pinhole_int_param_dict, gcp_dict, crs=utm34n_crs)

    # test the camera position against the reference
    ext_param = next(iter(ext_param_dict.values()))
    assert ext_param['xyz'] == pytest.approx(exterior_args['xyz'], abs=1e-3)


@pytest.mark.parametrize(
    'int_param_dict',
    [
        'pinhole_int_param_dict',
        'brown_int_param_dict',
        'opencv_int_param_dict',
        'fisheye_int_param_dict',
    ],
)
def test_fit_frame_exterior_multiple_images(
    int_param_dict: str,
    exterior_args: dict,
    grid_ji: np.ndarray,
    request: pytest.FixtureRequest,
):
    """Test fit_frame_exterior() with multiple images."""
    int_param_dict: dict = request.getfixturevalue(int_param_dict)
    int_param = next(iter(int_param_dict.values()))

    # create a mock GCP dictionary with multiple images at different camera positions / orientations
    gcp_dict = {}
    ref_ext_param_dict = {}
    gcp_cam = create_camera(**int_param, **exterior_args)
    for i, (ext_xyz, ext_opk) in enumerate(
        zip(
            [(2e4, 3e4, 1e3), (3e4, 3e4, 1e3), (3e4, 3e4, 2e3)],
            [(-3.0, 2.0, 10.0), (-15.0, 2.0, 10.0), (-30.0, 20.0, 10.0)],
        )
    ):
        ext_opk = tuple(np.radians(ext_opk))
        gcp_cam.update(xyz=ext_xyz, opk=ext_opk)
        xyz = gcp_cam.pixel_to_world_z(grid_ji, z=0)
        gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(grid_ji.T, xyz.T)]

        key = f'file{i}.ext'
        ref_ext_param_dict[key] = dict(xyz=ext_xyz, opk=ext_opk)
        gcp_dict[key] = gcps

    # fit exterior params
    test_ext_param_dict = fit.fit_frame_exterior(int_param_dict, gcp_dict)

    # test parameter accuracy
    assert test_ext_param_dict.keys() == ref_ext_param_dict.keys()
    for test_ext_param, ref_ext_param in zip(
        test_ext_param_dict.values(), ref_ext_param_dict.values()
    ):
        assert test_ext_param['xyz'] == pytest.approx(ref_ext_param['xyz'], abs=1e-3)
        assert test_ext_param['opk'] == pytest.approx(ref_ext_param['opk'], abs=1e-5)


def test_fit_frame_exterior_errors(
    pinhole_int_param_dict: dict, exterior_args: dict, grid_ji: np.ndarray
):
    """Test fit_frame_exterior() errors and warnings."""
    # create mock GCPs
    int_param = next(iter(pinhole_int_param_dict.values()))
    cam = create_camera(**int_param, **exterior_args)
    xyz = cam.pixel_to_world_z(grid_ji, z=0)
    gcps = [dict(ji=ji_gcp, xyz=xyz_gcp) for ji_gcp, xyz_gcp in zip(grid_ji.T, xyz.T)]

    # test an error is raised with < 3 GCPs in an image
    gcp_dict = {'file1.ext': gcps[:2], 'file2.ext': gcps[2:]}
    with pytest.raises(ValueError, match='At least three'):
        _ = fit.fit_frame_exterior(pinhole_int_param_dict, gcp_dict)

    # test a warning is issued with >1 camera in the interior parameter dictionary
    int_param_dict = {'cam1': int_param, 'cam2': int_param}
    gcp_dict = {'file.ext': gcps}
    with pytest.warns(OrthorityWarning, match='Refining the first'):
        _ = fit.fit_frame_exterior(int_param_dict, gcp_dict)
