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

import pytest
import numpy as np
from simple_ortho.camera import Camera, PinholeCamera, BrownCamera, OpenCVCamera, FisheyeCamera, create_camera
from simple_ortho.enums import CameraType, Interp
from simple_ortho.utils import distort_image
from tests.conftest import checkerboard


@pytest.mark.parametrize(
    'cam_type, dist_coeff, exp_type', [
        (CameraType.pinhole, None, PinholeCamera),
        (CameraType.brown, 'brown_dist_coeff', BrownCamera),
        (CameraType.opencv, 'opencv_dist_coeff', OpenCVCamera),
        (CameraType.fisheye, 'fisheye_dist_coeff', FisheyeCamera),
    ],
)
def test_init(
    cam_type: CameraType, dist_coeff: str, exp_type: type, position: Tuple, nadir_rotation,
    focal_len: float, im_size: Tuple, sensor_size: Tuple, request: pytest.FixtureRequest
):
    """ Test camera creation. """
    dist_coeff: Dict = request.getfixturevalue(dist_coeff) if dist_coeff else {}

    camera = create_camera(cam_type, position, nadir_rotation, focal_len, im_size, sensor_size=sensor_size, **dist_coeff)

    assert isinstance(camera, exp_type)
    assert np.all(camera._T.flatten() == position)
    assert camera._K.diagonal()[:2] == pytest.approx(np.array(focal_len) * im_size / sensor_size, abs=1e-3)
    if dist_coeff:
        assert np.all(camera._dist_coeff == [*dist_coeff.values()])


@pytest.mark.parametrize(
    'camera', ['nadir_pinhole_camera', 'nadir_brown_camera', 'nadir_opencv_camera', 'nadir_fisheye_camera',]
)
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


@pytest.mark.parametrize(
    'camera, distort', [
        ('nadir_pinhole_camera', True),
        ('nadir_brown_camera', True),
        ('nadir_brown_camera', False),
        ('nadir_opencv_camera', True),
        ('nadir_opencv_camera', False),
        ('nadir_fisheye_camera', True),
        ('nadir_fisheye_camera', False),
    ]
)  # yapf:disable
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


@pytest.mark.parametrize(
    'camera', ['nadir_brown_camera', 'nadir_opencv_camera', 'nadir_fisheye_camera'],
)
def test_project_points_nodistort(nadir_pinhole_camera, camera: str, request: pytest.FixtureRequest):
    """ Test projected points with distort==False match pinhole camera. """
    camera: Camera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    pinhole_xyz = nadir_pinhole_camera.pixel_to_world_z(ji, z)
    xyz = camera.pixel_to_world_z(ji, z, distort=False)
    ji_ = camera.world_to_pixel(xyz, distort=False)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


@pytest.mark.parametrize(
    'cam_type', [CameraType.brown, CameraType.opencv],
)
def test_brown_opencv_zerocoeff(nadir_pinhole_camera, cam_type: CameraType, nadir_camera_args: Dict):
    """ Test Brown & OpenCV cameras match pinhole camera with zero distortion coeffs. """
    camera: Camera = create_camera(cam_type, *nadir_camera_args.values())

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    pinhole_xyz = nadir_pinhole_camera.pixel_to_world_z(ji, z)
    xyz = camera.pixel_to_world_z(ji, z, distort=True)
    ji_ = camera.world_to_pixel(xyz, distort=True)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-3)


def test_brown_opencv_equiv(nadir_camera_args: Dict, brown_dist_coeff: Dict):
    """ Test OpenCV and Brown cameras are equivalent for the (cx, cy) == (0, 0) special case. """
    brown_camera = BrownCamera(**nadir_camera_args, **brown_dist_coeff)
    opencv_camera = OpenCVCamera(**nadir_camera_args, **brown_dist_coeff)

    ji = np.random.rand(2, 1000) * np.reshape(brown_camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (brown_camera._T[2] * .8)
    cv_xyz = opencv_camera.pixel_to_world_z(ji, z)
    brown_xyz = brown_camera.pixel_to_world_z(ji, z)

    assert cv_xyz == pytest.approx(brown_xyz, abs=1e-3)


@pytest.mark.parametrize(
    'camera', ['nadir_pinhole_camera', 'nadir_brown_camera', 'nadir_opencv_camera', 'nadir_fisheye_camera',]
)
def test_world_to_pixel_error(camera: str, request: pytest.FixtureRequest):
    """ Test world_to_pixel raises a ValueError with invalid coordinate shapes. """
    camera: Camera = request.getfixturevalue(camera)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros(2))
    assert '`xyz`' in str(ex)

    with pytest.raises(ValueError) as ex:
        camera.world_to_pixel(np.zeros((2, 1)))
    assert '`xyz`' in str(ex)


def test_pixel_to_world_z_error(nadir_pinhole_camera: Camera):
    """ Test pixel_to_world_z raises a ValueError with invalid coordinate shapes. """
    with pytest.raises(ValueError) as ex:
        nadir_pinhole_camera.pixel_to_world_z(np.zeros(2), np.zeros(1))
    assert '`ji`' in str(ex)

    with pytest.raises(ValueError) as ex:
        nadir_pinhole_camera.pixel_to_world_z(np.zeros((3, 1)), np.zeros(1))
    assert '`ji`' in str(ex)

    with pytest.raises(ValueError) as ex:
        nadir_pinhole_camera.pixel_to_world_z(np.zeros((2, 1)), np.zeros((2, 1)))
    assert '`z`' in str(ex)

    with pytest.raises(ValueError) as ex:
        nadir_pinhole_camera.pixel_to_world_z(np.zeros((2, 3)), np.zeros(2))
    assert '`z`' in str(ex)


def test_instrinsic_equivalence(
    position: Tuple, nadir_rotation, focal_len: float, im_size: Tuple, sensor_size: Tuple
):
    """ Test intrinsic matrix validity for equivalent focal_len and sensor_size options. """
    ref_camera = PinholeCamera(position, nadir_rotation, focal_len, im_size, sensor_size)

    # normalised focal length and no sensor size
    test_camera = PinholeCamera(position, nadir_rotation, focal_len / sensor_size[0], im_size)
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)

    # normalised focal length and sensor size
    test_camera = PinholeCamera(
        position, nadir_rotation, focal_len / sensor_size[0], im_size, np.array(sensor_size) / sensor_size[0]
    )
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)

    # normalised focal length (x, y) tuple and sensor size
    test_camera = PinholeCamera(position, nadir_rotation, np.ones(2) * focal_len, im_size, sensor_size)
    assert test_camera._K == pytest.approx(ref_camera._K, abs=1e-3)


def test_instrinsic_nonsquare_pixels(
    position: Tuple, nadir_rotation, focal_len: float, im_size: Tuple, sensor_size: Tuple
):
    """ Test intrinsic matrix validity for non-square pixels. """
    sensor_size = np.array(sensor_size)
    sensor_size[0] *= 2
    camera = PinholeCamera(position, nadir_rotation, focal_len, im_size, sensor_size)
    assert camera._K[0, 0] == pytest.approx(camera._K[1, 1] / 2, abs=1e-3)


@pytest.mark.parametrize(
    'camera', ['nadir_pinhole_camera', 'nadir_brown_camera', 'nadir_opencv_camera', 'nadir_fisheye_camera'],
)
def test_undistort(camera: str, request: pytest.FixtureRequest):
    """ Test undistort method by comparing source & distorted-undistorted checkerboard images. """
    nodata = 0
    interp = Interp.bilinear
    camera: Camera = request.getfixturevalue(camera)

    # create checkerboard source image
    image = checkerboard(camera._im_size[::-1])

    # distort then undistort
    dist_image = distort_image(camera, image, nodata=nodata, interp=interp)
    undist_image = camera.undistort(dist_image, nodata=nodata, interp=interp)

    # test similarity of source and distorted-undistorted images
    dist_mask = dist_image != nodata
    cc_dist = np.corrcoef(image[dist_mask], dist_image[dist_mask])
    undist_mask = undist_image != nodata
    cc = np.corrcoef(image[undist_mask], undist_image[undist_mask])
    assert cc[0, 1] > cc_dist[0, 1] or cc[0, 1] == 1
    assert cc[0, 1] > 0.95


