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

import numpy as np
import pytest

from orthority.camera import (
    BrownCamera,
    Camera,
    create_camera,
    FisheyeCamera,
    FrameCamera,
    OpenCVCamera,
    PinholeCamera,
)
from orthority.enums import CameraType
from orthority.errors import CameraInitError


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
    """Test frame camera creation."""
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    # TODO: add distort and alpha?  or are they tested elsewhere
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
    """Test FrameCamera._get_undistort_maps()."""
    camera: FrameCamera = request.getfixturevalue(camera)
    undistort_maps = camera._get_undistort_maps()
    if type(camera) is PinholeCamera:
        assert undistort_maps is None
    else:
        assert undistort_maps is not None
        im_shape = np.array(camera._im_size[::-1])
        for i in range(2):
            assert all(undistort_maps[i].shape[:2] == im_shape)


def test_frame_update(im_size: tuple, focal_len: float, sensor_size: tuple, xyz: tuple, opk: tuple):
    """Test frame camera exterior parameter update."""
    camera = PinholeCamera(
        im_size, focal_len, sensor_size=sensor_size, xyz=(0, 0, 0), opk=(0, 0, 0)
    )
    camera.update(xyz, opk)

    assert np.all(camera.pos == xyz)
    assert np.all(camera._R != 0)


@pytest.mark.parametrize('cam_type', [*CameraType])
def test_frame_update_error(
    cam_type: CameraType, im_size: tuple, focal_len: float, sensor_size: tuple
):
    """Test an error is raised if a frame camera is used before initialising exterior parameters."""
    camera = create_camera(cam_type, im_size, focal_len, sensor_size=sensor_size)

    with pytest.raises(CameraInitError):
        camera.world_to_pixel(None)
    with pytest.raises(CameraInitError):
        camera.pixel_to_world_z(None, None)


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
    ],
)
def test_project_points(camera: str, request: pytest.FixtureRequest):
    """Test projection of multiple points between pixel & world coordinates."""
    camera: Camera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera.pos[2] * 0.8)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz[2] == pytest.approx(z.squeeze())
    assert ji_ == pytest.approx(ji, abs=1)

    # test for broadcast type ambiguities where number of pts == number of dimensions
    for sl in (slice(0, 2), slice(0, 3)):
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
    ],
)
def test_project_dims(camera: str, request: pytest.FixtureRequest):
    """Test projection with different pixel & world coordinate dimensionality."""
    camera: Camera = request.getfixturevalue(camera)

    # single point to single z
    ji = np.reshape(camera._im_size, (-1, 1)) / 2
    z = camera.pos[2] * 0.5
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz.shape == (3, 1)
    assert ji_.shape == (2, 1)
    assert xyz[2] == pytest.approx(z)
    assert ji_ == pytest.approx(ji, abs=1)

    # single point to multiple z
    z = camera.pos[2] * np.linspace(0.1, 0.8, 10)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz.shape == (3, z.shape[0])
    assert ji_.shape == (2, z.shape[0])
    assert xyz[2] == pytest.approx(z)
    assert np.allclose(ji, ji_)

    # multiple points to single z
    ji = np.random.rand(2, 10) * np.reshape(camera._im_size, (-1, 1))
    z = camera.pos[2] * 0.5
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz.shape == (3, ji.shape[1])
    assert ji_.shape == ji.shape
    assert np.allclose(xyz[2], z)
    assert ji_ == pytest.approx(ji, abs=1)


@pytest.mark.parametrize('camera', ['brown_camera_und', 'opencv_camera_und', 'fisheye_camera_und'])
def test_frame_project_nodistort(camera: str, request: pytest.FixtureRequest):
    """Test frame camera projected points with ``distort=False`` match pinhole camera."""
    camera: FrameCamera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera.pos[2] * 0.8)
    pinhole_xyz = PinholeCamera.pixel_to_world_z(camera, ji, z)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


@pytest.mark.parametrize('cam_type', [CameraType.brown, CameraType.opencv])
def test_brown_opencv_zerocoeff(pinhole_camera: Camera, cam_type: CameraType, frame_args: dict):
    """Test Brown & OpenCV cameras match pinhole camera with zero distortion coeffs (
    Fisheye is excluded as the model distorts with zero distortion coeffs).
    """
    camera = create_camera(cam_type, **frame_args, distort=True)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera.pos[2] * 0.8)
    pinhole_xyz = pinhole_camera.pixel_to_world_z(ji, z)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


def test_brown_opencv_equiv(frame_args: dict, brown_dist_param: dict):
    """Test OpenCV and Brown cameras are equivalent for the Brown distortion parameter set."""
    brown_camera = BrownCamera(**frame_args, **brown_dist_param)
    opencv_camera = OpenCVCamera(**frame_args, **brown_dist_param)

    ji = np.random.rand(2, 1000) * np.reshape(brown_camera._im_size, (-1, 1))
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
    """Test frame camera coordinate equivalence for different image sizes."""
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    ref_camera: FrameCamera = create_camera(cam_type, **frame_args, **dist_param)

    test_camera_args = frame_args.copy()
    test_camera_args['im_size'] = tuple(np.array(test_camera_args['im_size']) * scale)
    test_camera: FrameCamera = create_camera(cam_type, **test_camera_args, **dist_param)

    # find reference and test camera coords for world pts corresponding to reference image
    # boundary pts
    w, h = np.array(ref_camera._im_size) - 1
    ref_ji = np.array(
        [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]]
    ).T
    xyz = ref_camera.pixel_to_world_z(ref_ji, 0)
    test_ji = test_camera.world_to_pixel(xyz)
    ref_xy = ref_camera._pixel_to_camera(ref_ji)[:2]
    test_xy = test_camera._pixel_to_camera(test_ji)[:2]

    assert test_xy == pytest.approx(ref_xy, abs=1e-3)


@pytest.mark.parametrize(
    'camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera']
)
def test_world_to_pixel_error(camera: str, request: pytest.FixtureRequest):
    """Test world_to_pixel raises a ValueError with invalid coordinate shapes."""
    camera: Camera = request.getfixturevalue(camera)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros(2))
    assert "'xyz'" in str(ex)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros((2, 1)))
    assert "'xyz'" in str(ex)


def test_pixel_to_world_z_error(pinhole_camera: Camera):
    """Test pixel_to_world_z raises a ValueError with invalid coordinate shapes."""
    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros(2), np.zeros(1))
    assert "'ji'" in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((3, 1)), np.zeros(1))
    assert "'ji'" in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((2, 1)), np.zeros((2, 1)))
    assert "'z'" in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((2, 3)), np.zeros(2))
    assert "'z'" in str(ex)


def test_frame_intrinsic_equivalence(
    im_size: tuple,
    focal_len: float,
    sensor_size: tuple,
    cxy: tuple,
    xyz: tuple,
    opk: tuple,
):
    """Test frame camera intrinsic matrix validity for equivalent focal_len & sensor_size
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
    """Test frame camera intrinsic matrix validity for non-square pixels."""
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
    """Test FrameCamera._horizon_fov() validity."""
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
def test_frame_undistort(camera: str, camera_und: str, request: pytest.FixtureRequest):
    """Test frame camera undistort() + pixel_to_world_z() with ``distort=False`` matches
    pixel_to_world_z() with ``distort=True``.
    """
    camera: FrameCamera = request.getfixturevalue(camera)
    camera_und: FrameCamera = request.getfixturevalue(camera_und)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera.pos[2] * 0.8)
    xyz = camera.pixel_to_world_z(ji, z)

    ji_undistort = camera_und.undistort_pixel(ji)
    xyz_undistort = camera_und.pixel_to_world_z(ji_undistort, z)

    assert xyz_undistort == pytest.approx(xyz, abs=1e-3)


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
def test_frame_undistort_alpha(
    frame_args: dict,
    cam_type: CameraType,
    dist_param: str,
    alpha: float,
    request: pytest.FixtureRequest,
):
    """Test ``alpha=0`` gives undistorted image boundaries outside, and ``alpha=1`` gives
    undistorted image boundaries inside the source image, for frame cameras.
    """
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    camera: FrameCamera = create_camera(cam_type, **frame_args, **dist_param, alpha=alpha)

    # create boundary coordinates and undistort
    w, h = np.array(camera._im_size) - 1
    ji = np.array(
        [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2], [0, 0]]
    ).T
    undistort_ji = np.round(camera.undistort_pixel(ji), 3)

    def inside_outside(ji: np.array, undistort_ji: np.array, inside=True):
        """Test if ``undistort_ji`` lies inside (``inside=True``) or outside (``inside=False``)
        ``ji``.
        """
        # setup indexes to extract bottom i, right j, top i, left j edge values
        edge_idxs = [1, 0, 1, 0]
        edge_slices = [slice(0, 3), slice(2, 5), slice(4, 7), slice(6, 9)]

        # edge - undistorted edge comparison functions
        if inside:
            edge_cmps = [np.less_equal, np.greater_equal, np.greater_equal, np.less_equal]
        else:
            edge_cmps = [np.greater_equal, np.less_equal, np.less_equal, np.greater_equal]

        # extract bottom, right, top, left edges and compare
        for edge_idx, edge_slice, edge_cmp in zip(edge_idxs, edge_slices, edge_cmps):
            undistort_edge = undistort_ji[edge_idx, edge_slice]
            edge = ji[edge_idx, edge_slice]
            assert np.all(edge_cmp(edge, undistort_edge))

    # test undistorted coords are inside / outside source coords
    if alpha == 1:
        assert np.all(undistort_ji.min(axis=1) >= ji.min(axis=1))
        assert np.all(undistort_ji.max(axis=1) <= ji.max(axis=1))
        inside_outside(ji, undistort_ji, inside=True)
    else:
        assert np.all(undistort_ji.min(axis=1) <= ji.min(axis=1))
        assert np.all(undistort_ji.max(axis=1) >= ji.max(axis=1))
        inside_outside(ji, undistort_ji, inside=False)


@pytest.mark.parametrize(
    'cam_type', [CameraType.pinhole, CameraType.opencv, CameraType.brown, CameraType.fisheye]
)
def test_frame_undistort_no_ext_init(
    cam_type: CameraType, im_size: tuple, focal_len: float, sensor_size: tuple
):
    """Test frame camera undistorting without exterior initialisation."""
    camera = create_camera(cam_type, im_size, focal_len, sensor_size=sensor_size)

    ji = (np.array([im_size]).T - 1) / 2
    ji_ = camera.undistort_pixel(ji)
    assert ji_ == pytest.approx(ji, 1e-3)
