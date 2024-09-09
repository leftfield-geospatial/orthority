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
from typing import Sequence

import cv2
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_bounds

from orthority import common
from orthority.camera import (
    BrownCamera,
    Camera,
    create_camera,
    FisheyeCamera,
    FrameCamera,
    OpenCVCamera,
    PinholeCamera,
    RpcCamera,
)
from orthority.enums import CameraType, Interp
from orthority.errors import CameraInitError, OrthorityWarning, OrthorityError
from tests.conftest import _dem_offset, checkerboard, create_zsurf, ortho_bounds


@pytest.mark.parametrize(
    'cam_type, dist_param, distort, alpha, exp_type',
    [
        (CameraType.pinhole, None, True, 1.0, PinholeCamera),
        (CameraType.brown, 'brown_dist_param', True, 1.0, BrownCamera),
        (CameraType.brown, 'brown_dist_param', False, 0.5, BrownCamera),
        (CameraType.opencv, 'opencv_dist_param', True, 1.0, OpenCVCamera),
        (CameraType.opencv, 'opencv_dist_param', False, 0.5, OpenCVCamera),
        (CameraType.fisheye, 'fisheye_dist_param', True, 1.0, FisheyeCamera),
        (CameraType.fisheye, 'fisheye_dist_param', False, 0.5, FisheyeCamera),
    ],
)
def test_frame_init(
    cam_type: CameraType,
    dist_param: str,
    exp_type: type,
    xyz: tuple,
    opk: tuple,
    im_size: tuple,
    focal_len: float,
    sensor_size: tuple,
    cxy: tuple,
    distort: bool,
    alpha: float,
    request: pytest.FixtureRequest,
):
    """Test ``FrameCamera`` creation."""
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    distort = True
    alpha = 0.0
    camera: FrameCamera = create_camera(
        cam_type,
        im_size,
        focal_len,
        sensor_size=sensor_size,
        cx=cxy[0],
        cy=cxy[1],
        **dist_param,
        xyz=xyz,
        opk=opk,
        distort=distort,
        alpha=alpha,
    )

    im_size = np.array(im_size)
    cxy_pixel = (im_size - 1) / 2 + np.array(cxy) * im_size.max()

    assert type(camera) == exp_type

    assert np.all(camera.pos == xyz)
    assert np.all(camera.im_size == im_size)

    assert camera._K.diagonal()[:2] == pytest.approx(
        np.array(focal_len) * im_size / sensor_size, abs=1e-3
    )
    assert all(camera._K[:2, 2] == cxy_pixel)
    if dist_param:
        assert np.all(camera._dist_param == [*dist_param.values()])

    assert camera.distort == distort
    assert camera.alpha == alpha


@pytest.mark.parametrize(
    'camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera']
)
def test_frame_get_undistort_maps(camera: str, request: pytest.FixtureRequest):
    """Test ``FrameCamera._get_undistort_maps()``."""
    camera: FrameCamera = request.getfixturevalue(camera)
    undistort_maps = camera._get_undistort_maps()
    if type(camera) is PinholeCamera:
        assert undistort_maps is None
    else:
        assert undistort_maps is not None
        im_shape = np.array(camera.im_size[::-1])
        for i in range(2):
            assert all(undistort_maps[i].shape[:2] == im_shape)


def test_frame_update(im_size: tuple, focal_len: float, sensor_size: tuple, xyz: tuple, opk: tuple):
    """Test ``FrameCamera`` exterior parameter update."""
    camera = PinholeCamera(
        im_size, focal_len, sensor_size=sensor_size, xyz=(0, 0, 0), opk=(0, 0, 0)
    )
    camera.update(xyz, opk)

    assert np.all(camera.pos == xyz)
    assert np.all(camera._R != 0)


@pytest.mark.parametrize(
    'cam_type', [CameraType.pinhole, CameraType.opencv, CameraType.brown, CameraType.fisheye]
)
def test_frame_update_error(
    cam_type: CameraType, im_size: tuple, focal_len: float, sensor_size: tuple
):
    """Test an error is raised if a ``FrameCamera`` is used before initialising exterior
    parameters.
    """
    camera = create_camera(cam_type, im_size, focal_len, sensor_size=sensor_size)

    with pytest.raises(CameraInitError):
        camera.world_to_pixel(None)
    with pytest.raises(CameraInitError):
        camera.pixel_to_world_z(None, None)


@pytest.mark.parametrize('crs', ['utm34n_crs', 'wgs84_crs', None])
def test_rpc_init(im_size: tuple, rpc: dict, crs: str, request: pytest.FixtureRequest):
    """Test ``RpcCamera`` creation."""
    crs: str = request.getfixturevalue(crs) if crs else 'EPSG:4979'

    for _rpc in [rpc, rio.transform.RPC(**rpc)]:
        camera: RpcCamera = create_camera(CameraType.rpc, im_size, _rpc, crs=crs)

        assert type(camera) is RpcCamera
        assert camera.im_size == im_size
        assert camera.crs == rio.CRS.from_string(crs)
        __rpc = camera._rpc.to_dict()
        assert all([__rpc[k] == v for k, v in rpc.items()])


@pytest.mark.parametrize('crs', ['utm34n_egm96_crs', 'utm34n_egm2008_crs', 'utm34n_msl_crs'])
def test_rpc_init_crs_error(im_size: tuple, rpc: dict, crs: str, request: pytest.FixtureRequest):
    """Test ``RpcCamera`` creation raises an error when the world / ortho CRS has a vertical CRS."""
    crs: str = request.getfixturevalue(crs)

    with pytest.raises(OrthorityError) as ex:
        _ = RpcCamera(im_size, rpc, crs=crs)

    assert 'crs' in str(ex.value) and 'ellipsoidal' in str(ex.value)


@pytest.mark.parametrize(
    'camera',
    [
        'pinhole_camera',
        'brown_camera',
        'brown_camera_und',
        'opencv_camera',
        'opencv_camera_und',
        'fisheye_camera',
        'fisheye_camera_und',
        'rpc_camera',
        'rpc_camera_proj',
    ],
)
def test_project_points(camera: str, request: pytest.FixtureRequest):
    """Test projection of multiple points between pixel & world coordinates."""
    camera: Camera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera.im_size, (-1, 1))
    z = np.random.rand(1000) * _dem_offset
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz[2] == pytest.approx(z, abs=1e-6)
    assert ji_ == pytest.approx(ji, abs=0.1)

    # test for broadcast type ambiguities where number of pts == number of dimensions
    for sl in (slice(0, 2), slice(0, 3), slice(0, 1)):
        assert xyz[:, sl] == pytest.approx(camera.pixel_to_world_z(ji[:, sl], z[sl]))
        assert ji_[:, sl] == pytest.approx(camera.world_to_pixel(xyz[:, sl]))


@pytest.mark.parametrize(
    'camera',
    [
        'pinhole_camera',
        'brown_camera',
        'brown_camera_und',
        'opencv_camera',
        'opencv_camera_und',
        'fisheye_camera',
        'fisheye_camera_und',
        'rpc_camera',
        'rpc_camera_proj',
    ],
)
def test_project_dims(camera: str, request: pytest.FixtureRequest):
    """Test projection with different pixel & world coordinate dimensionality."""
    camera: Camera = request.getfixturevalue(camera)

    # single point to single z
    ji = np.reshape(camera.im_size, (-1, 1)) / 2
    z = _dem_offset
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz.shape == (3, 1)
    assert ji_.shape == (2, 1)
    assert xyz[2] == pytest.approx(z)
    assert ji_ == pytest.approx(ji, abs=1)

    # single point to multiple z
    z = _dem_offset * np.linspace(0.1, 0.8, 10)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz.shape == (3, z.shape[0])
    assert ji_.shape == (2, z.shape[0])
    assert xyz[2] == pytest.approx(z)
    assert np.allclose(ji, ji_)

    # multiple points to single z
    ji = np.random.rand(2, 10) * np.reshape(camera.im_size, (-1, 1))
    z = _dem_offset
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz.shape == (3, ji.shape[1])
    assert ji_.shape == ji.shape
    assert np.allclose(xyz[2], z)
    assert ji_ == pytest.approx(ji, abs=1)


@pytest.mark.parametrize('camera', ['brown_camera_und', 'opencv_camera_und', 'fisheye_camera_und'])
def test_frame_project_nodistort(camera: str, request: pytest.FixtureRequest):
    """Test ``FrameCamera(distort=False)`` projections match ``PinholeCamera``."""
    camera: FrameCamera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera.im_size, (-1, 1))
    z = np.random.rand(1000) * (camera.pos[2] * 0.8)
    pinhole_xyz = PinholeCamera.pixel_to_world_z(camera, ji, z)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


@pytest.mark.parametrize('camera', ['rpc_camera', 'rpc_camera_proj'])
def test_rpc_project_nans(camera: str, request: pytest.FixtureRequest):
    """Test ``RpcCamera.pixel()_to_world_z()`` and ``RpcCamera.world_to_pixel()`` pass nan
    coordinates through.
    """
    camera: RpcCamera = request.getfixturevalue(camera)

    # test pixel_to_world_z with nans in ji
    nan_mask = np.array([True, False, True, False, True])
    ji = np.random.rand(2, 5) * np.reshape(camera.im_size, (-1, 1))
    z = np.random.rand(5) * _dem_offset
    ji[:, nan_mask] = np.nan
    xyz = camera.pixel_to_world_z(ji, z)
    assert np.all(np.isnan(xyz[:, nan_mask]))
    ji_ = camera.world_to_pixel(xyz)
    assert ji_[:, ~nan_mask] == pytest.approx(ji[:, ~nan_mask], abs=0.1)

    # test pixel_to_world_z with nans in z
    ji = np.random.rand(2, 5) * np.reshape(camera.im_size, (-1, 1))
    z[nan_mask] = np.nan
    xyz = camera.pixel_to_world_z(ji, z)
    assert np.all(np.isnan(xyz[:, nan_mask]))
    ji_ = camera.world_to_pixel(xyz)
    assert ji_[:, ~nan_mask] == pytest.approx(ji[:, ~nan_mask], abs=0.1)

    # test pixel_to_world_z with ji and z all nans
    ji *= np.nan
    z *= np.nan
    xyz = camera.pixel_to_world_z(ji, z)
    assert np.all(np.isnan(xyz))

    # create xyz coords to test world_to_pixel
    ji = np.random.rand(2, 5) * np.reshape(camera.im_size, (-1, 1))
    z = np.random.rand(5) * _dem_offset
    xyz = camera.pixel_to_world_z(ji, z)

    # test world_to_pixel with nan in one dimension of xyz
    nan_mask = np.zeros(ji.shape[1], dtype=bool)
    for row in range(3):
        xyz_ = xyz.copy()
        xyz_[row, nan_mask] = np.nan
        ji_ = camera.world_to_pixel(xyz_)
        assert np.all(np.isnan(ji_[:, nan_mask]))
        assert ji_[:, ~nan_mask] == pytest.approx(ji[:, ~nan_mask], abs=0.1)

    # test world_to_pixel with all nans
    ji = camera.world_to_pixel(np.full((3, 5), fill_value=np.nan))
    assert np.all(np.isnan(ji))


@pytest.mark.parametrize('cam_type', [CameraType.brown, CameraType.opencv])
def test_brown_opencv_zerocoeff(pinhole_camera: Camera, cam_type: CameraType, frame_args: dict):
    """Test ``BrownCamera`` & ``OpenCVCamera`` match ``PinholeCamera`` with zero distortion coeffs
    (``FisheyeCamera`` is excluded as the model distorts with zero distortion coeffs).
    """
    camera = create_camera(cam_type, **frame_args, distort=True)

    ji = np.random.rand(2, 1000) * np.reshape(camera.im_size, (-1, 1))
    z = np.random.rand(1000) * (camera.pos[2] * 0.8)
    pinhole_xyz = pinhole_camera.pixel_to_world_z(ji, z)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


def test_brown_opencv_equiv(frame_args: dict, brown_dist_param: dict):
    """Test ``OpenCVCamera`` and ``BrownCamera`` are equivalent for the Brown distortion parameter
    set."""
    brown_camera = BrownCamera(**frame_args, **brown_dist_param)
    opencv_camera = OpenCVCamera(**frame_args, **brown_dist_param)

    ji = np.random.rand(2, 1000) * np.reshape(brown_camera.im_size, (-1, 1))
    z = np.random.rand(1000) * (brown_camera.pos[2] * 0.8)
    cv_xyz = opencv_camera.pixel_to_world_z(ji, z)
    brown_xyz = brown_camera.pixel_to_world_z(ji, z)

    assert cv_xyz == pytest.approx(brown_xyz, abs=1e-3)


@pytest.mark.parametrize(
    'cam_type, dist_param, scale',
    [
        (CameraType.pinhole, None, 0.5),
        (CameraType.pinhole, None, 2),
        (CameraType.brown, 'brown_dist_param', 0.5),
        (CameraType.brown, 'brown_dist_param', 2),
        (CameraType.opencv, 'opencv_dist_param', 0.5),
        (CameraType.opencv, 'opencv_dist_param', 2),
        (CameraType.fisheye, 'fisheye_dist_param', 0.5),
        (CameraType.fisheye, 'fisheye_dist_param', 2),
    ],
)
def test_frame_project_im_size(
    frame_args: dict,
    cam_type: CameraType,
    dist_param: str,
    scale: float,
    request: pytest.FixtureRequest,
):
    """Test ``FrameCamera`` projection coordinate equivalence for different image sizes with
    same aspect ratio.
    """
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    ref_camera: FrameCamera = create_camera(cam_type, **frame_args, **dist_param)

    test_camera_args = frame_args.copy()
    test_camera_args['im_size'] = tuple(np.array(test_camera_args['im_size']) * scale)
    test_camera: FrameCamera = create_camera(cam_type, **test_camera_args, **dist_param)

    # find reference and test camera coords for world pts corresponding to reference image
    # boundary pts
    w, h = np.array(ref_camera.im_size) - 1
    ref_ji = np.array(
        [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]]
    ).T
    xyz = ref_camera.pixel_to_world_z(ref_ji, 0)
    test_ji = test_camera.world_to_pixel(xyz)
    ref_xy = ref_camera._pixel_to_camera(ref_ji)[:2]
    test_xy = test_camera._pixel_to_camera(test_ji)[:2]

    assert test_xy == pytest.approx(ref_xy, abs=1e-3)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'rpc_camera'])
def test_world_to_pixel_error(camera: str, request: pytest.FixtureRequest):
    """Test ``Camera.world_to_pixel()`` raises a ``ValueError`` with invalid coordinate shapes."""
    camera: Camera = request.getfixturevalue(camera)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros(2))
    assert "'xyz'" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros((2, 1)))
    assert "'xyz'" in str(ex.value)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'rpc_camera'])
def test_pixel_to_world_z_error(camera: Camera, request: pytest.FixtureRequest):
    """Test ``Camera.pixel_to_world_z()`` raises a ValueError with invalid coordinate shapes."""
    camera: Camera = request.getfixturevalue(camera)

    with pytest.raises(ValueError) as ex:
        camera.pixel_to_world_z(np.zeros(2), np.zeros(1))
    assert "'ji'" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        camera.pixel_to_world_z(np.zeros((3, 1)), np.zeros(1))
    assert "'ji'" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        camera.pixel_to_world_z(np.zeros((2, 1)), np.zeros((2, 1)))
    assert "'z'" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        camera.pixel_to_world_z(np.zeros((2, 3)), np.zeros(2))
    assert "'z'" in str(ex.value)


def test_frame_intrinsic_equivalence(
    im_size: tuple[int, int],
    focal_len: float,
    sensor_size: tuple[float, float],
    cxy: tuple[float, float],
    xyz: tuple[float, float, float],
    opk: tuple[float, float, float],
):
    """Test ``FrameCamera`` intrinsic matrix validity for equivalent focal_len & sensor_size
    options.
    """
    ref_camera = PinholeCamera(
        im_size, focal_len, sensor_size=sensor_size, cx=cxy[0], cy=cxy[1], xyz=xyz, opk=opk
    )

    # normalised focal length and no sensor size
    test_camera = PinholeCamera(
        im_size, focal_len / sensor_size[0], cx=cxy[0], cy=cxy[1], xyz=xyz, opk=opk
    )
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)

    # normalised focal length and sensor size
    test_camera = PinholeCamera(
        im_size,
        focal_len / sensor_size[0],
        sensor_size=np.array(sensor_size) / sensor_size[0],
        cx=cxy[0],
        cy=cxy[1],
        xyz=xyz,
        opk=opk,
    )
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)

    # normalised focal length (x, y) tuple and sensor size
    test_camera = PinholeCamera(
        im_size,
        (focal_len, focal_len),
        sensor_size=sensor_size,
        cx=cxy[0],
        cy=cxy[1],
        xyz=xyz,
        opk=opk,
    )
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)


def test_frame_intrinsic_nonsquare_pixels(
    im_size: tuple[int, int],
    focal_len: float,
    sensor_size: tuple[float, float],
    xyz: tuple[float, float, float],
    opk: tuple[float, float, float],
):
    """Test ``FrameCamera`` intrinsic matrix validity for non-square pixels."""
    _sensor_size = (sensor_size[0] * 2, sensor_size[1])
    camera = PinholeCamera(im_size, focal_len, sensor_size=_sensor_size, xyz=xyz, opk=opk)
    assert camera._K[0, 0] == pytest.approx(camera._K[1, 1] / 2, abs=1e-3)


@pytest.mark.parametrize(
    'cam_type, dist_param',
    [
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
    ],
)
def test_frame_horizon_fov(
    cam_type: CameraType,
    dist_param: str,
    frame_args: dict,
    xyz: tuple,
    request: pytest.FixtureRequest,
):
    """Test ``FrameCamera._horizon_fov()`` validity."""
    dist_param: dict = request.getfixturevalue(dist_param)
    camera: FrameCamera = create_camera(cam_type, **frame_args, **dist_param)
    assert not camera._horizon_fov()

    camera.update(xyz, (np.pi / 2, 0, 0))
    assert camera._horizon_fov()
    camera.update(xyz, (0, np.pi, 0))
    assert camera._horizon_fov()


@pytest.mark.parametrize(
    'camera, camera_und',
    [
        ('pinhole_camera', 'pinhole_camera_und'),
        ('brown_camera', 'brown_camera_und'),
        ('opencv_camera', 'opencv_camera_und'),
        ('fisheye_camera', 'fisheye_camera_und'),
    ],
)
def test_frame_undistort_pixel(camera: str, camera_und: str, request: pytest.FixtureRequest):
    """Test ``FrameCamera()._undistort_pixel()`` by comparing ``_undistort_pixel()`` followed by
    ``FrameCamera(distort=False).pixel_to_world_z()`` with
    ``FrameCamera(distort=True).pixel_to_world_z()``.
    """
    camera: FrameCamera = request.getfixturevalue(camera)
    camera_und: FrameCamera = request.getfixturevalue(camera_und)

    ji = np.random.rand(2, 1000) * np.reshape(camera.im_size, (-1, 1))
    z = np.random.rand(1000) * (camera.pos[2] * 0.8)
    xyz = camera.pixel_to_world_z(ji, z)

    ji_undistort = camera_und._undistort_pixel(ji)
    xyz_undistort = camera_und.pixel_to_world_z(ji_undistort, z)

    assert xyz_undistort == pytest.approx(xyz, abs=1e-3)


@pytest.mark.parametrize(
    'camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera']
)
def test_frame_distort_pixel(camera: str, request: pytest.FixtureRequest):
    """Test ``FrameCamera._distort_pixel()`` by comparing undistorted & re-distorted coordinates
    with the original.
    """
    camera: FrameCamera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera.im_size, (-1, 1))
    ji_und = camera._undistort_pixel(ji)
    ji_ = camera._distort_pixel(ji_und)

    assert ji_ == pytest.approx(ji, abs=1e-1)


@pytest.mark.parametrize(
    'cam_type, dist_param, alpha',
    [
        (CameraType.brown, 'brown_dist_param', 0.0),
        (CameraType.brown, 'brown_dist_param', 1.0),
        (CameraType.opencv, 'opencv_dist_param', 0.0),
        (CameraType.opencv, 'opencv_dist_param', 1.0),
        (CameraType.fisheye, 'fisheye_dist_param', 0.0),
        (CameraType.fisheye, 'fisheye_dist_param', 1.0),
    ],
)
def test_frame_undistort_pixel_alpha(
    frame_args: dict,
    cam_type: CameraType,
    dist_param: str,
    alpha: float,
    request: pytest.FixtureRequest,
):
    """Test ``FrameCamera(alpha=0)`` gives undistorted image boundaries outside,
    and ``FrameCamera(alpha=1)`` gives undistorted image boundaries inside the source image.
    """
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    camera: FrameCamera = create_camera(cam_type, **frame_args, **dist_param, alpha=alpha)

    # create boundary coordinates and undistort
    w, h = np.array(camera.im_size) - 1
    ji = np.array(
        [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2], [0, 0]]
    ).T
    undistort_ji = np.round(camera._undistort_pixel(ji), 3)

    # test if points in undistort_ji are on (0), inside (+1), or outside (-1), ji
    inside = np.zeros(undistort_ji.shape[1])
    ji_ = ji.T.astype('float32')
    for pi, und_pt in enumerate(undistort_ji.T):
        inside[pi] = cv2.pointPolygonTest(ji_, und_pt, measureDist=False)

    # test undistorted coords are inside / outside source coords
    if alpha == 1:
        assert np.all(undistort_ji.min(axis=1) >= ji.min(axis=1))
        assert np.all(undistort_ji.max(axis=1) <= ji.max(axis=1))
        assert np.all(inside >= 0) and np.any(inside > 0)
    else:
        assert np.all(undistort_ji.min(axis=1) <= ji.min(axis=1))
        assert np.all(undistort_ji.max(axis=1) >= ji.max(axis=1))
        assert np.all(inside <= 0) and np.any(inside < 0)


@pytest.mark.parametrize(
    'cam_type', [CameraType.pinhole, CameraType.opencv, CameraType.brown, CameraType.fisheye]
)
def test_frame_undistort_pixel_no_ext_init(
    cam_type: CameraType, im_size: tuple, focal_len: float, sensor_size: tuple
):
    """Test ``FrameCamera._undistort_pixel()`` without exterior initialisation."""
    camera = create_camera(cam_type, im_size, focal_len, sensor_size=sensor_size)

    ji = (np.array([im_size]).T - 1) / 2
    ji_ = camera._undistort_pixel(ji)
    assert ji_ == pytest.approx(ji, 1e-3)


@pytest.mark.parametrize('num_pts', [None, 40, 100, 400, 1000, 4000])
def test_pixel_boundary(rpc_camera: Camera, num_pts: int | None):
    """Test ``Camera.pixel_boundary()`` generates a rectangular boundary with the correct
    corners and length.
    """
    # create corner only boundary to test against
    w, h = np.array(rpc_camera.im_size, dtype='float32') - 1
    ref_ji = {(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)}

    # create pixel boundary and simplify to corner only
    ji = rpc_camera.pixel_boundary(num_pts=num_pts).astype('float32')
    test_ji = cv2.approxPolyDP(ji.T, epsilon=1e-6, closed=True)
    test_ji = set([tuple(*pt) for pt in test_ji])

    # test dimensions, and corner only boundaries match
    assert ji.shape == (2, num_pts or 8)
    assert test_ji == ref_ji


@pytest.mark.parametrize(
    'camera, num_pts',
    [
        ('brown_camera_und', 200),
        ('opencv_camera_und', 200),
        ('fisheye_camera_und', 200),
    ],
)
def test_frame_pixel_boundary_undistort(
    camera: str, num_pts: int | None, xyz: tuple[float], request: pytest.FixtureRequest
):
    """Test ``FrameCamera(distort=False).pixel_boundary()`` generates a valid undistorted
    boundary by comparing the boundary mask with an undistorted image mask.
    """
    camera: FrameCamera = request.getfixturevalue(camera)

    # create undistorted boundary, then test its dimensions & bounds
    ji = camera.pixel_boundary(num_pts=num_pts).astype('float32')
    num_pts = num_pts or 8
    assert ji.shape == (2, num_pts)
    assert np.all(ji.T >= (0, 0)) and np.all(ji.T <= (np.array(camera.im_size) - 1))

    # test the boundary does not simplify
    simple_ji = cv2.approxPolyDP(ji.T, epsilon=1e-6, closed=True)[:, 0, :].T
    assert simple_ji.shape == (2, num_pts)

    # convert the undistorted boundary to a mask
    test_mask = np.zeros(camera.im_size[::-1], dtype='uint8')
    ji_ = [np.round(ji.T).astype(int)]
    test_mask = cv2.fillPoly(test_mask, ji_, color=(255,)).astype('bool', copy=False)

    # create undistorted image mask to test against
    maps = [mp.round() for mp in camera._get_undistort_maps()]
    ref_mask = (
        (maps[0] >= 0)
        & (maps[0] <= camera.im_size[0] - 1)
        & (maps[1] >= 0)
        & (maps[1] <= camera.im_size[1] - 1)
    )

    # test mask similarity
    cc = np.corrcoef(test_mask.flatten(), ref_mask.flatten())
    assert cc[0, 1] > 0.95


@pytest.mark.parametrize(
    'camera, num_pts',
    [
        ('pinhole_camera', None),
        ('pinhole_camera', 200),
        ('rpc_camera', None),
        ('rpc_camera', 200),
    ],
)
def test_world_boundary(camera: str, num_pts: int, request: pytest.FixtureRequest):
    """Test ``Camera.world_boundary()`` at scalar and surface z values.  Basic dimensionality,
    z value and sanity testing only.
    """
    camera: Camera = request.getfixturevalue(camera)

    # create reference boundary at scalar z
    z = _dem_offset
    ref_ji = camera.pixel_boundary(num_pts=num_pts)
    ref_xyz = camera.pixel_to_world_z(ref_ji, z)

    # create flat z surface and transform
    bounds = ortho_bounds(camera, z=z)
    surf_z, transform = create_zsurf(bounds, z_off=z, resolution=(5, 5))
    surf_z = surf_z[1]  # flat

    # test boundaries
    kwargs_list = [
        dict(z=z, num_pts=num_pts),
        dict(z=surf_z, num_pts=num_pts, transform=transform),
    ]
    for kwargs in kwargs_list:
        test_xyz = camera.world_boundary(**kwargs)
        assert test_xyz.shape == (3, num_pts or 8)
        assert test_xyz[2] == pytest.approx(z, abs=1e-3)
        assert test_xyz == pytest.approx(ref_xyz, abs=1e-3)


def test_frame_world_boundary_zscalar_clip(pinhole_camera: FrameCamera):
    """Test ``FrameCamera.world_boundary()`` at scalar z clips z values to the camera height."""
    z = pinhole_camera.pos[2] * 1.2
    xyz = pinhole_camera.world_boundary(z)
    assert np.all(xyz[2] == pinhole_camera.pos[2])


def test_frame_world_boundary_zsurf_clip(pinhole_camera: FrameCamera):
    """Test ``FrameCamera.world_boundary()`` at z surface clips z values to the camera height."""
    # create z surface (steps from above to below camera) and corresponding transform
    min_z, max_z = pinhole_camera.pos[2] * 0.5, pinhole_camera.pos[2] * 1.5
    z = np.vstack((max_z * np.ones((5, 20)), min_z * np.ones((15, 20))))
    bounds = ortho_bounds(pinhole_camera, z=min_z)
    transform = from_bounds(*bounds, *z.shape[::-1])

    # test clipping
    xyz = pinhole_camera.world_boundary(z, transform=transform, clip=False)
    assert xyz[2].max() >= pinhole_camera.pos[2]
    xyz = pinhole_camera.world_boundary(z, transform=transform)
    assert xyz[2].max() <= pinhole_camera.pos[2]
    assert xyz[2].min() < pinhole_camera.pos[2]


def _test_world_boundary_zsurf(
    camera: Camera, x: np.ndarray, y: np.ndarray, z: np.ndarray, transform: rio.Affine
):
    """Test ``camera.world_boundary()`` z surface intersection by comparing with the
    ``camera.remap( )`` mask.
    """

    # remap to get reference mask
    nodata = 0
    im_array = np.full((1, *camera.im_size[::-1]), fill_value=127, dtype='uint8')
    _, remap_mask = camera.remap(im_array, x, y, z, nodata=nodata, interp=Interp.nearest)
    ref_mask = ~remap_mask

    # find world boundary and convert to mask
    xyz = camera.world_boundary(z, num_pts=400, transform=transform)
    center_transform = transform * rio.Affine.translation(0.5, 0.5)
    ji = np.array(~center_transform * xyz[:2])
    ji_ = [np.round(ji.T).astype(int)]
    test_mask = np.zeros(ref_mask.shape, dtype='uint8')
    test_mask = cv2.fillPoly(test_mask, ji_, color=(255,)).view(bool)

    # test test_mask contains and is similar to ref_mask
    assert test_mask[ref_mask].sum() / ref_mask.sum() > 0.95
    if not (np.all(test_mask) and np.all(ref_mask)):
        cc = np.corrcoef(test_mask.flatten(), ref_mask.flatten())
        assert cc[0, 1] > 0.9
        print(cc[0, 1])


@pytest.mark.parametrize(
    'xyz_offset, opk_offset',
    [
        # varying rotations starting at ``rotation`` fixture value and keeping FOV below horizon
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (-15, 10, 0)),
        ((0, 0, 0), (-30, 20, 0)),
        ((0, 0, 0), (-45, 20, 0)),
        # varying positions with partial dem coverage
        ((0, 5.5e2, 0), (0, 0, 0)),
        ((0, 0, 1.1e3), (0, 0, 0)),
        ((0, 0, 2.0e3), (0, 0, 0)),
    ],
)
def test_frame_world_boundary_zsurf(
    frame_args: dict,
    xyz_grids: tuple[tuple, rio.Affine],
    xyz_offset: tuple[float],
    opk_offset: tuple[float],
):
    """Test ``Camera.world_boundary()`` z surface intersection by comparing a
    ``PinholeCamera.world_boundary()`` mask with the ``PinholeCamera.remap()`` mask for varying
    camera angles and positions, including partial z surface coverage.
    """
    # Note that these tests should use the pinhole camera model to ensure no artefacts outside
    # the ortho boundary, and z < camera height to ensure no ortho artefacts in z > camera
    # height areas.  While the world boundary excludes occluded pixels, the remap mask
    # does not i.e. to compare these masks, there should be no occlusion.

    # create camera
    _xyz = tuple(np.array(frame_args['xyz']) + xyz_offset)
    _opk = tuple(np.array(frame_args['opk']) + np.radians(opk_offset))
    camera = PinholeCamera(
        frame_args['im_size'],
        frame_args['focal_len'],
        sensor_size=frame_args.get('sensor_size', None),
        xyz=_xyz,
        opk=_opk,
        distort=True,
    )

    # (x, y, z) world coordinate grids
    (x, y, z), transform = xyz_grids
    z = z[0]  # sinusoid

    _test_world_boundary_zsurf(camera, x, y, z, transform)


@pytest.mark.parametrize(
    'xy_offset',
    [(0, 0), (0.005, 0), (0.0, 0.005), (0.005, 0.005)],
)
def test_rpc_world_boundary_zsurf(
    im_size: tuple[int, int],
    rpc: dict,
    utm34n_crs: str,
    xyz_grids: tuple[tuple, rio.Affine],
    xy_offset: tuple[float],
):
    """Test ``Camera.world_boundary()`` z surface intersection by comparing a
    ``RpcCamera.world_boundary()`` mask with the ``RpcCamera.remap()`` mask for varying camera
    angles and positions, including partial z surface coverage.
    """
    _rpc = rpc.copy()
    _rpc['long_off'] += xy_offset[0]
    _rpc['lat_off'] += xy_offset[1]
    camera = RpcCamera(im_size, _rpc, crs=utm34n_crs)

    # (x, y, z) world coordinate grids
    (x, y, z), transform = xyz_grids
    z = z[0]  # sinusoid
    z = z.mean() + (z - z.mean()) * 100  # increase the z "variance"

    _test_world_boundary_zsurf(camera, x, y, z, transform)


@pytest.mark.parametrize(
    'camera, camera_und',
    [
        ('pinhole_camera', 'pinhole_camera_und'),
        ('brown_camera', 'brown_camera_und'),
        ('opencv_camera', 'opencv_camera_und'),
        ('fisheye_camera', 'fisheye_camera_und'),
    ],
)
def test_frame_world_boundary_equiv(camera: str, camera_und: str, request: pytest.FixtureRequest):
    """Test equivalence of ``FrameCamera(distort=True)`` and ``FrameCamera(distort=False)``
    world boundaries.
    """
    camera: FrameCamera = request.getfixturevalue(camera)
    camera_und: FrameCamera = request.getfixturevalue(camera_und)

    xyz = camera.world_boundary(0)
    xyz_und = camera_und.world_boundary(0)

    assert xyz == pytest.approx(xyz_und, abs=1e-6)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'rpc_camera'])
def test_world_boundary_errors(camera: str, request: pytest.FixtureRequest):
    """Test ``Camera.world_boundary()`` error conditions."""
    camera: Camera = request.getfixturevalue(camera)

    # z is 1D array
    with pytest.raises(ValueError) as ex:
        camera.world_boundary(z=np.ones(10))
    assert "'z'" in str(ex.value)

    # z is 2D array but no transform specified
    with pytest.raises(ValueError) as ex:
        camera.world_boundary(z=np.ones((10, 10)), transform=None)
    assert "transform" in str(ex.value)

    # z is 2D array with width and or height > 2**15 - 1
    with pytest.raises(ValueError) as ex:
        camera.world_boundary(z=np.ones((1, 2**15)), transform=rio.Affine.identity())
    assert "'z'" in str(ex.value) and "width" in str(ex.value)


@pytest.mark.parametrize('indexes, dtype', [(None, None), ([1, 2, 3], 'uint8'), (1, 'float32')])
def test_read(
    rgb_byte_src_file: Path, rpc_camera: Camera, indexes: int | Sequence[int], dtype: str
):
    """Test a valid image is returned by ``Camera.read()`` with different band indexes and data
    types.
    """
    test_array = rpc_camera.read(rgb_byte_src_file, indexes=indexes, dtype=dtype)

    with rio.open(rgb_byte_src_file) as im:
        indexes = indexes or im.indexes
        indexes = np.expand_dims(indexes, axis=0) if np.isscalar(indexes) else indexes
        dtype = dtype or im.dtypes[0]
        ref_array = im.read(indexes=indexes, out_dtype=dtype)

    assert test_array.ndim == 3
    assert test_array.shape == ref_array.shape
    assert test_array.dtype == ref_array.dtype
    assert np.all(test_array == ref_array)


@pytest.mark.parametrize(
    'camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera']
)
def test_frame_undistort(camera: str, request: pytest.FixtureRequest):
    """Test ``FrameCamera._undistort_im()`` by comparing source & distorted-undistorted images."""
    nodata = 0
    interp = Interp.cubic
    camera: FrameCamera = request.getfixturevalue(camera)

    # create checkerboard image
    im_array = np.expand_dims(checkerboard(camera.im_size[::-1]), axis=0)

    # distort then undistort
    dist_array = common.distort_image(camera, im_array, nodata=nodata, interp=interp)
    undist_array = camera._undistort_im(dist_array, nodata=nodata, interp=interp)

    # test similarity of source and distorted-undistorted images
    dist_mask = dist_array != nodata
    cc_dist = np.corrcoef(im_array[dist_mask], dist_array[dist_mask])
    undist_mask = undist_array != nodata
    cc = np.corrcoef(im_array[undist_mask], undist_array[undist_mask])
    assert cc[0, 1] > cc_dist[0, 1] or cc[0, 1] == 1
    assert cc[0, 1] > 0.95


def test_frame_undistort_errors(pinhole_camera: FrameCamera):
    """Test ``FrameCamera._undistort_im()`` error conditions."""
    # im_array not 3D
    with pytest.raises(ValueError) as ex:
        pinhole_camera._undistort_im(np.ones((10, 10)))
    assert 'im_array' in str(ex.value) and '3' in str(ex.value)

    # im_array with unsupported dtype
    with pytest.raises(ValueError) as ex:
        pinhole_camera._undistort_im(np.ones((1, 10, 10), dtype='int32'))
    assert 'im_array' in str(ex.value) and 'data type' in str(ex.value)

    # im_array size does not match camera im_size
    with pytest.warns(OrthorityWarning, match='im_size'):
        pinhole_camera._undistort_im(np.ones((1, 10, 10), dtype='uint8'))

    # im_array width and or height > 2**15 - 1
    with pytest.raises(ValueError) as ex:
        pinhole_camera._undistort_im(np.ones((1, 1, 2**15), dtype='uint8'))
    assert "'im_array'" in str(ex.value) and "width" in str(ex.value)


@pytest.mark.parametrize(
    'indexes, dtype, nodata',
    [
        (None, 'uint8', 0),
        (1, 'uint8', 0),
        ([1, 2, 3], 'uint8', 0),
        (1, 'uint16', 2**15),
        (1, 'int16', -123),
        (1, 'float32', float('nan')),
    ],
)
def test_frame_read_undistort(
    rgb_byte_src_file: Path,
    brown_camera_und: FrameCamera,
    indexes: int | Sequence[int] | None,
    dtype: str,
    nodata: int | float,
):
    """Test a valid undistorted image is returned by ``FrameCamera(distort=False).read()`` with
    different band indexes, data types and nodata values.
    """
    interp = Interp.cubic

    # read undistorted image
    test_array = brown_camera_und.read(
        rgb_byte_src_file, indexes=indexes, dtype=dtype, nodata=nodata, interp=interp
    )

    # create a reference undistorted image
    with rio.open(rgb_byte_src_file) as im:
        indexes = indexes or im.indexes
        indexes = np.expand_dims(indexes, 0) if np.isscalar(indexes) else indexes
        dtype = dtype or im.dtypes[0]
        ref_array = im.read(indexes=indexes, out_dtype=dtype)
    ref_array = brown_camera_und._undistort_im(ref_array, nodata=nodata, interp=interp)

    # test dimensions and dtype
    assert test_array.ndim == 3
    assert test_array.shape == ref_array.shape
    assert test_array.dtype == ref_array.dtype == dtype

    # compare to reference
    test_mask = common.nan_equals(test_array, nodata)
    assert np.any(test_mask)
    assert np.all(common.nan_equals(test_array, ref_array))


@pytest.mark.parametrize(
    'indexes, dtype, nodata',
    [
        ([1, 2, 3], 'uint8', 0),
        ([1, 2], 'uint16', 2**15),
        ([1], 'int16', -123),
        ([1], 'float32', float('nan')),
    ],
)
def test_remap(
    rpc_camera_proj: Camera,
    xyz_grids: tuple[tuple, rio.Affine],
    indexes: Sequence[int],
    dtype: str,
    nodata: int | float,
):
    """Test ``Camera.remap()`` with different image dimensions, data types and nodata values.
    Basic dimensionality, dtype and sanity testing only.
    """
    # (x, y, z) world coordinate grids
    (x, y, z), transform = xyz_grids
    z = z[0]  # sinusoid

    # remap checkerboard image (with nearest interp so remapped values can be compared to source)
    im_array = np.expand_dims(checkerboard(rpc_camera_proj.im_size[::-1]), axis=0).astype(dtype)
    remap_array, remap_mask = rpc_camera_proj.remap(
        im_array, x, y, z, nodata=nodata, interp=Interp.nearest
    )

    # mask, dtype and shape tests
    assert np.any(remap_mask)
    assert np.all(remap_mask == np.all(common.nan_equals(remap_array, nodata), axis=0))
    assert remap_array.dtype == dtype
    assert remap_array.ndim == 3
    assert remap_array.shape[0] == im_array.shape[0]
    assert remap_array.shape[-2:] == z.shape

    # basic statistical check on remapped content
    assert np.all(np.unique(im_array) == np.unique(remap_array[:, ~remap_mask]))
    assert im_array.mean() == pytest.approx(remap_array[:, ~remap_mask].mean(), abs=15)
    assert im_array.std() == pytest.approx(remap_array[:, ~remap_mask].std(), abs=15)


@pytest.mark.parametrize(
    'indexes, dtype, nodata',
    [
        ([1, 2, 3], 'uint8', 0),
        ([1], 'uint16', 2**15),
        ([1], 'int16', -123),
    ],
)
def test_frame_remap_mask_dilation(
    brown_camera_und: FrameCamera,
    xyz_grids: tuple[tuple, rio.Affine],
    indexes: Sequence[int],
    dtype: str,
    nodata: int | float,
):
    """Test ``FrameCamera(distort=False).remap()`` mask dilation by comparing dilated and
    undilated remapped masks / images.
    """
    dtype = 'float32'  # for comparison of twice interpolated images
    interp = Interp.cubic

    # (x, y, z) world coordinate grids
    (x, y, z), transform = xyz_grids
    z = z[0]  # sinusoid

    # checkerboard image
    im_array = np.expand_dims(checkerboard(brown_camera_und.im_size[::-1]), axis=0).astype(dtype)

    # reference remap with no mask dilation
    nodata = float('nan')
    und_array = brown_camera_und._undistort_im(im_array, nodata=nodata)
    ref_array, ref_mask = brown_camera_und.remap(
        und_array, x, y, z, nodata=nodata, interp=interp, kernel_size=(5, 5)
    )

    # test remap with mask dilation
    nodata = 0
    und_array = brown_camera_und._undistort_im(im_array, nodata=nodata)
    test_array, test_mask = brown_camera_und.remap(
        und_array, x, y, z, nodata=nodata, interp=interp, kernel_size=(5, 5)
    )

    # test_mask, dtype and shape tests
    assert np.any(test_mask)
    assert np.all(test_mask == np.all(common.nan_equals(test_array, nodata), axis=0))
    assert test_array.dtype == dtype
    assert test_array.ndim == 3
    assert test_array.shape[0] == im_array.shape[0]
    assert test_array.shape[-2:] == z.shape

    # test test_mask is dilated and contained in ref_mask
    assert not np.all(ref_mask[test_mask])
    assert np.all(test_mask[ref_mask])

    # test common unmasked areas of test_array and ref_array are the same
    mask = ~(test_mask | ref_mask)
    assert test_array[:, mask] == pytest.approx(ref_array[:, mask], abs=1)


@pytest.mark.parametrize(
    'camera, camera_und',
    [
        ('pinhole_camera', 'pinhole_camera_und'),
        ('brown_camera', 'brown_camera_und'),
        ('opencv_camera', 'opencv_camera_und'),
        ('fisheye_camera', 'fisheye_camera_und'),
    ],
)
def test_frame_remap_distort(
    xyz_grids: tuple[tuple, rio.Affine],
    camera: str,
    camera_und: str,
    request: pytest.FixtureRequest,
):
    """Test similarity of ``FrameCamera(distort=True)`` and ``FrameCamera(distort=False)``
    remapped images.
    """
    camera: FrameCamera = request.getfixturevalue(camera)
    camera_und: FrameCamera = request.getfixturevalue(camera_und)
    dtype = 'float32'
    nodata = float('nan')  # prevent mask dilation
    interp = Interp.cubic

    # (x, y, z) world coordinate grids
    (x, y, z), transform = xyz_grids
    z = z[0]  # sinusoid

    # remap with distort=True camera
    im_array = np.expand_dims(checkerboard(camera.im_size[::-1]), axis=0).astype(dtype)
    remap_array, remap_mask = camera.remap(im_array, x, y, z, nodata=nodata, interp=interp)

    # remap with distort=False camera
    und_im_array = camera_und._undistort_im(im_array, nodata=nodata, interp=interp)
    und_remap_array, und_remap_mask = camera_und.remap(
        und_im_array, x, y, z, nodata=nodata, interp=interp
    )

    # Compare distort=True/False results. (Note that remap_array / remap_mask can contain
    # artefacts where invalid ortho areas have mapped inside the valid image, and that
    # und_remap_array / und_remap_mask do not have this issue.  This means remap_mask should
    # contain und_remap_mask but not necessarily be similar to it.)
    assert remap_array.shape == und_remap_array.shape
    assert remap_mask[und_remap_mask].sum() / und_remap_mask.sum() > 0.95
    mask = ~(remap_mask | und_remap_mask)
    cc = np.corrcoef(remap_array[:, mask], und_remap_array[:, mask])
    assert cc[0, 1] > 0.99


@pytest.mark.parametrize(
    'cam_type, dist_param',
    [
        (CameraType.pinhole, {}),
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
    ],
)
def test_frame_remap_alpha(
    xyz_grids: tuple[tuple, rio.Affine],
    cam_type: CameraType,
    dist_param: str,
    frame_args: dict,
    request: pytest.FixtureRequest,
):
    """Test similarity of ``FrameCamera(alpha=1)`` and ``FrameCamera(alpha=0.)``
    remapped images.
    """
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    camera_a1 = create_camera(cam_type, **frame_args, **dist_param, alpha=1.0, distort=False)
    camera_a0 = create_camera(cam_type, **frame_args, **dist_param, alpha=0.0, distort=False)
    dtype = 'float32'
    nodata = float('nan')
    interp = Interp.cubic

    # (x, y, z) world coordinate grids
    (x, y, z), transform = xyz_grids
    z = z[0]  # sinusoid

    # checkerboard image
    im_array = np.expand_dims(checkerboard(camera_a1.im_size[::-1]), axis=0).astype(dtype)

    # remap with alpha=1 camera
    a1_im_array = camera_a1._undistort_im(im_array, nodata=nodata, interp=interp)
    a1_remap_array, a1_remap_mask = camera_a1.remap(
        a1_im_array, x, y, z, nodata=nodata, interp=interp
    )

    # remap with alpha=0 camera
    a0_im_array = camera_a0._undistort_im(im_array, nodata=nodata, interp=interp)
    a0_remap_array, a0_remap_mask = camera_a0.remap(
        a0_im_array, x, y, z, nodata=nodata, interp=interp
    )

    # test alpha=0 mask contains alpha=1 mask
    assert a1_remap_mask.shape == a0_remap_mask.shape
    assert a0_remap_mask[a1_remap_mask].sum() / a1_remap_mask.sum() > 0.99
    if cam_type is not CameraType.pinhole:
        assert a0_remap_mask.sum() > a1_remap_mask.sum()

    # test alpha=0 and alpha=1 remap similarity in common unmasked area
    mask = ~(a1_remap_mask | a0_remap_mask)
    cc = np.corrcoef(a1_remap_array[:, mask], a0_remap_array[:, mask])
    assert cc[0, 1] > 0.99


def test_nadir_pinhole_remap(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    im_size: tuple[int, int],
    focal_len: float,
    sensor_size: tuple[float, float],
    xyz: tuple[float, float, float],
):
    """Test nadir ``PinholeCamera().remap()`` result is identical to the source image with a flat
    ``z`` and suitably chosen (``x``, ``y``).
    """
    # create nadir pinhole camera
    camera = PinholeCamera(
        im_size, focal_len, sensor_size=sensor_size, xyz=xyz, opk=(0.0, 0.0, 0.0)
    )

    # construct (x, y) world coordinate grids matching the image size, at a flat z, and with
    # coordinates exactly on source pixel centers
    z = np.ones(im_size[::-1]) * _dem_offset
    bounds = ortho_bounds(camera, z=_dem_offset)
    transform = from_bounds(*bounds, *z.shape[::-1])
    j, i = np.meshgrid(range(0, z.shape[1]), range(0, z.shape[0]), indexing='xy')
    x, y = (transform * rio.Affine.translation(0.5, 0.5)) * [j, i]

    # checkerboard image
    im_array = np.expand_dims(checkerboard(im_size[::-1]), 0)

    # remap and compare
    remap_array, remap_mask = camera.remap(im_array, x, y, z, nodata=0, interp=Interp.nearest)
    assert np.all(remap_array == im_array)


def test_remap_errors(rpc_camera: RpcCamera, xyz_grids: tuple[tuple, rio.Affine]):
    """Test ``Camera.remap()`` error conditions."""
    (x, y, z), transform = xyz_grids

    # im_array not 3D
    with pytest.raises(ValueError) as ex:
        rpc_camera.remap(np.ones((10, 10)), x, y, z)
    assert 'im_array' in str(ex.value) and '3' in str(ex.value)

    # im_array with unsupported dtype
    with pytest.raises(ValueError) as ex:
        rpc_camera.remap(np.ones((1, 10, 10), dtype='int32'), x, y, z)
    assert 'im_array' in str(ex.value) and 'data type' in str(ex.value)

    # 3D z
    with pytest.raises(ValueError) as ex:
        rpc_camera.remap(np.ones((1, 10, 10), dtype='uint8'), x, y, z)
    assert "'z'" in str(ex.value)

    # float32 x/y
    with pytest.raises(ValueError) as ex:
        rpc_camera.remap(np.ones((1, 10, 10), dtype='uint8'), x.astype('float32'), y, z[0])
    assert 'float64' in str(ex.value)

    # im_array width and or height > 2**15 - 1
    with pytest.raises(ValueError) as ex:
        rpc_camera.remap(np.ones((1, 1, 2**15)), x, y, z)
    assert "'im_array'" in str(ex.value) and "width" in str(ex.value)


##
