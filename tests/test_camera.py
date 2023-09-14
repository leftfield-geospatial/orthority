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
from typing import Tuple, Dict

import numpy as np
import pytest

from simple_ortho.camera import Camera, PinholeCamera, BrownCamera, OpenCVCamera, FisheyeCamera, create_camera
from simple_ortho.enums import CameraType
from simple_ortho.errors import CameraInitError


# @formatter:off
@pytest.mark.parametrize(
    'cam_type, dist_param, exp_type', [
        (CameraType.pinhole, None, PinholeCamera),
        (CameraType.brown, 'brown_dist_param', BrownCamera),
        (CameraType.opencv, 'opencv_dist_param', OpenCVCamera),
        (CameraType.fisheye, 'fisheye_dist_param', FisheyeCamera),
    ],
)  # yapf: disable  # @formatter:on
def test_init(
    cam_type: CameraType, dist_param: str, exp_type: type, xyz: Tuple, opk: Tuple, focal_len: float,
    im_size: Tuple, sensor_size: Tuple, cxy: Tuple, request: pytest.FixtureRequest
):
    """ Test camera creation. """
    dist_param: Dict = request.getfixturevalue(dist_param) if dist_param else {}

    camera = create_camera(
        cam_type, focal_len, im_size, sensor_size=sensor_size, cx=cxy[0], cy=cxy[1], **dist_param, xyz=xyz, opk=opk
    )
    im_size = np.array(im_size)
    cxy_pixel = (im_size - 1) / 2 + np.array(cxy) * im_size.max()

    assert isinstance(camera, exp_type)
    assert np.all(camera._T.flatten() == xyz)
    assert camera._K.diagonal()[:2] == pytest.approx(np.array(focal_len) * im_size / sensor_size, abs=1e-3)
    assert all(camera._K[:2, 2] == cxy_pixel)
    if dist_param:
        assert np.all(camera._dist_param == [*dist_param.values()])


def test_update(focal_len: float, im_size: Tuple, sensor_size: Tuple, xyz: Tuple, opk: Tuple):
    """ Test exterior parameter update. """
    camera = PinholeCamera(focal_len, im_size, sensor_size=sensor_size, xyz=(0, 0, 0), opk=(0, 0, 0))
    camera.update(xyz, opk)

    assert np.all(camera._T.flatten() == xyz)
    assert np.all(camera._R != 0)


@pytest.mark.parametrize('cam_type', [*CameraType])
def test_update_error(cam_type: CameraType, focal_len: float, im_size: Tuple, sensor_size: Tuple):
    """ Test an error is raised if the camera is used before intialising exterior parameters. """
    camera = create_camera(cam_type, focal_len, im_size, sensor_size=sensor_size)

    with pytest.raises(CameraInitError):
        camera.world_to_pixel(None)
    with pytest.raises(CameraInitError):
        camera.pixel_to_world_z(None, None)
    with pytest.raises(CameraInitError):
        camera.undistort(None)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera', ])
def test_project_points(camera: str, request: pytest.FixtureRequest):
    """ Test projection of multiple points between pixel & world coordinates. """
    camera: Camera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz[2] == pytest.approx(z.squeeze())
    assert ji_ == pytest.approx(ji, abs=1)

    # test for broadcast type ambiguities where number of pts == number of dimensions
    for sl in (slice(0, 2), slice(0, 3)):
        assert xyz[:, sl] == pytest.approx(camera.pixel_to_world_z(ji[:, sl], z[sl]))
        assert ji_[:, sl] == pytest.approx(camera.world_to_pixel(xyz[:, sl]))


# @formatter:off
@pytest.mark.parametrize(
    'camera, distort', [
        ('pinhole_camera', True),
        ('brown_camera', True),
        ('brown_camera', False),
        ('opencv_camera', True),
        ('opencv_camera', False),
        ('fisheye_camera', True),
        ('fisheye_camera', False),
    ]
)  # yapf:disable  # @formatter:on
def test_project_dims(camera: str, distort: bool, request: pytest.FixtureRequest):
    """ Test projection with different pixel & world coordinate dimensionality. """
    camera: Camera = request.getfixturevalue(camera)

    # single point to single z
    ji = np.reshape(camera._im_size, (-1, 1)) / 2
    z = camera._T[2] * .5
    xyz = camera.pixel_to_world_z(ji, z, distort=distort)
    ji_ = camera.world_to_pixel(xyz, distort=distort)

    assert xyz.shape == (3, 1)
    assert ji_.shape == (2, 1)
    assert xyz[2] == pytest.approx(z)
    assert ji_ == pytest.approx(ji, abs=1)

    # single point to multiple z
    z = camera._T[2] * np.linspace(0.1, 0.8, 10)
    xyz = camera.pixel_to_world_z(ji, z, distort=distort)
    ji_ = camera.world_to_pixel(xyz, distort=distort)

    assert xyz.shape == (3, z.shape[0])
    assert ji_.shape == (2, z.shape[0])
    assert xyz[2] == pytest.approx(z)
    assert np.allclose(ji, ji_)

    # multiple points to single z
    ji = np.random.rand(2, 10) * np.reshape(camera._im_size, (-1, 1))
    z = camera._T[2] * 0.5
    xyz = camera.pixel_to_world_z(ji, z, distort=distort)
    ji_ = camera.world_to_pixel(xyz, distort=distort)

    assert xyz.shape == (3, ji.shape[1])
    assert ji_.shape == ji.shape
    assert np.allclose(xyz[2], z)
    assert ji_ == pytest.approx(ji, abs=1)


@pytest.mark.parametrize('camera', ['brown_camera', 'opencv_camera', 'fisheye_camera'], )
def test_project_points_nodistort(camera: str, request: pytest.FixtureRequest):
    """ Test projected points with distort==False match pinhole camera. """
    camera: Camera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    pinhole_xyz = PinholeCamera.pixel_to_world_z(camera, ji, z, distort=False)
    xyz = camera.pixel_to_world_z(ji, z, distort=False)
    ji_ = camera.world_to_pixel(xyz, distort=False)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


@pytest.mark.parametrize('cam_type', [CameraType.brown, CameraType.opencv], )
def test_brown_opencv_zerocoeff(pinhole_camera: Camera, cam_type: CameraType, camera_args: Dict):
    """ Test Brown & OpenCV cameras match pinhole camera with zero distortion coeffs. """
    camera: Camera = create_camera(cam_type, **camera_args)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    pinhole_xyz = pinhole_camera.pixel_to_world_z(ji, z)
    xyz = camera.pixel_to_world_z(ji, z, distort=True)
    ji_ = camera.world_to_pixel(xyz, distort=True)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


def test_brown_opencv_equiv(camera_args: Dict, brown_dist_param: Dict):
    """ Test OpenCV and Brown cameras are equivalent for the Brown distortion parameter set. """
    brown_camera = BrownCamera(**camera_args, **brown_dist_param)
    opencv_camera = OpenCVCamera(**camera_args, **brown_dist_param)

    ji = np.random.rand(2, 1000) * np.reshape(brown_camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (brown_camera._T[2] * .8)
    cv_xyz = opencv_camera.pixel_to_world_z(ji, z)
    brown_xyz = brown_camera.pixel_to_world_z(ji, z)

    assert cv_xyz == pytest.approx(brown_xyz, abs=1e-3)


@pytest.mark.parametrize(
    'cam_type, dist_param, scale', [
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
def test_project_im_size(
    camera_args: Dict, cam_type: CameraType, dist_param: str, scale: float, request: pytest.FixtureRequest
):
    """ Test camera coordinate equivalence for different image sizes. """
    dist_param: Dict = request.getfixturevalue(dist_param) if dist_param else {}
    ref_camera = create_camera(cam_type, **camera_args, **dist_param)

    test_camera_args = camera_args.copy()
    test_camera_args['im_size'] = tuple(np.array(test_camera_args['im_size']) * scale)
    test_camera = create_camera(cam_type, **test_camera_args, **dist_param)

    # find reference and test camera coords for world pts corresponding to reference image corners
    ref_br = np.array(ref_camera._im_size) - 1
    ref_ji = np.array([[0, 0], [ref_br[0], 0], ref_br, [0, ref_br[1]]]).T
    xyz = ref_camera.pixel_to_world_z(ref_ji, 0)
    test_ji = test_camera.world_to_pixel(xyz)
    ref_xy = ref_camera._pixel_to_camera(ref_ji)[:2]
    test_xy = test_camera._pixel_to_camera(test_ji)[:2]

    assert test_xy == pytest.approx(ref_xy, abs=1e-3)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera', ])
def test_world_to_pixel_error(camera: str, request: pytest.FixtureRequest):
    """ Test world_to_pixel raises a ValueError with invalid coordinate shapes. """
    camera: Camera = request.getfixturevalue(camera)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros(2))
    assert '`xyz`' in str(ex)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros((2, 1)))
    assert '`xyz`' in str(ex)


def test_pixel_to_world_z_error(pinhole_camera: Camera):
    """ Test pixel_to_world_z raises a ValueError with invalid coordinate shapes. """
    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros(2), np.zeros(1))
    assert '`ji`' in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((3, 1)), np.zeros(1))
    assert '`ji`' in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((2, 1)), np.zeros((2, 1)))
    assert '`z`' in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((2, 3)), np.zeros(2))
    assert '`z`' in str(ex)


def test_intrinsic_equivalence(
    focal_len: float, im_size: Tuple, sensor_size: Tuple, cxy: Tuple, xyz: Tuple, opk: Tuple,
):
    """ Test intrinsic matrix validity for equivalent focal_len & sensor_size options. """
    ref_camera = PinholeCamera(focal_len, im_size, sensor_size=sensor_size, cx=cxy[0], cy=cxy[1], xyz=xyz, opk=opk)

    # normalised focal length and no sensor size
    test_camera = PinholeCamera(focal_len / sensor_size[0], im_size, cx=cxy[0], cy=cxy[1], xyz=xyz, opk=opk)
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)

    # normalised focal length and sensor size
    test_camera = PinholeCamera(
        focal_len / sensor_size[0], im_size, sensor_size=np.array(sensor_size) / sensor_size[0], cx=cxy[0], cy=cxy[1],
        xyz=xyz, opk=opk
    )
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)

    # normalised focal length (x, y) tuple and sensor size
    test_camera = PinholeCamera(
        np.ones(2) * focal_len, im_size, sensor_size=sensor_size, cx=cxy[0], cy=cxy[1], xyz=xyz, opk=opk
    )
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)


def test_instrinsic_nonsquare_pixels(
    focal_len: float, im_size: Tuple, sensor_size: Tuple, xyz: Tuple, opk: Tuple,
):
    """ Test intrinsic matrix validity for non-square pixels. """
    sensor_size = np.array(sensor_size)
    sensor_size[0] *= 2
    camera = PinholeCamera(focal_len, im_size, sensor_size=sensor_size, xyz=xyz, opk=opk)
    assert camera._K[0, 0] == pytest.approx(camera._K[1, 1] / 2, abs=1e-3)


@pytest.mark.parametrize(
    'cam_type, dist_param', [
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
    ],
)
def test_horizon_fov(
    cam_type: CameraType, dist_param: str, camera_args: Dict, xyz: Tuple, request: pytest.FixtureRequest
):
    """ Test Camera._horizon_fov() validity.  """
    dist_param: Dict = request.getfixturevalue(dist_param)
    camera = create_camera(cam_type, **camera_args, **dist_param)
    assert not camera._horizon_fov()

    camera.update(xyz, (np.pi / 2, 0, 0))
    assert camera._horizon_fov()
    camera.update(xyz, (0, np.pi, 0))
    assert camera._horizon_fov()


@pytest.mark.parametrize(
    'cam_type, dist_param, alpha', [
        (CameraType.brown, 'brown_dist_param', 0.),
        (CameraType.brown, 'brown_dist_param', 1.),
        (CameraType.opencv, 'opencv_dist_param', 0.),
        (CameraType.opencv, 'opencv_dist_param', 1.),
        (CameraType.fisheye, 'fisheye_dist_param', 0.),
        (CameraType.fisheye, 'fisheye_dist_param', 1.),
    ],
)  # yapf: disable
def test_undistort_alpha(
    camera_args: Dict, cam_type: CameraType, dist_param: str, alpha: float, request: pytest.FixtureRequest
):
    """
    Test alpha=0 gives undistorted image boundaries outside, and alpha=1 gives undistorted image boundaries inside
    the source image.
    """
    dist_param: Dict = request.getfixturevalue(dist_param) if dist_param else {}
    camera = create_camera(cam_type, **camera_args, **dist_param, alpha=alpha)

    # create boundary coordinates and undistort
    w, h = np.array(camera._im_size) - 1
    ji = np.array(
        [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2], [0, 0]]
    ).T  # yapf: disable
    undistort_ji = np.round(camera.undistort(ji), 3)

    def inside_outside(ji: np.array, undistort_ji: np.array, inside=True):
        """ Test if `undistort_ji` lies inside (`inside`=True) or outside (`inside`=False) `ji`. """
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
