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
from typing import Tuple
import pytest
import numpy as np
from simple_ortho.camera import Camera, BrownCamera, create_camera
from simple_ortho.enums import CameraType


@pytest.mark.parametrize(
    'camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera',]
)
def test_project_points(camera: str, request: pytest.FixtureRequest):
    """ Test projection of multiple points between pixel & world coordinates. """
    camera: Camera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    xyz = camera.pixel_to_world_z(ji, z)
    ji_ = camera.world_to_pixel(xyz)

    assert xyz[2] == pytest.approx(z.squeeze())
    assert ji_ == pytest.approx(ji, abs=1e-4)

    # test for broadcast type ambiguities where number of pts == number of dimensions
    for sl in (slice(0, 2), slice(0, 3)):
        assert xyz[:, sl] == pytest.approx(camera.pixel_to_world_z(ji[:, sl], z[sl]))
        assert ji_[:, sl] == pytest.approx(camera.world_to_pixel(xyz[:, sl]))


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
)  # yapf:disable
def test_project_point(camera: str, distort: bool, request: pytest.FixtureRequest):
    """ Test projection of single point between pixel & world coordinates. """
    camera: Camera = request.getfixturevalue(camera)

    ji = np.reshape(camera._im_size, (-1, 1)) / 2
    z = camera._T[2] * .5
    xyz = camera.pixel_to_world_z(ji, z, distort=distort)
    ji_ = camera.world_to_pixel(xyz, distort=distort)

    assert xyz.shape == (3, 1)
    assert ji_.shape == (2, 1)
    assert xyz[2] == pytest.approx(z)
    assert ji_ == pytest.approx(ji, abs=1e-4)


@pytest.mark.parametrize(
    'camera', ['brown_camera', 'opencv_camera', 'fisheye_camera'],
)
def test_project_points_nodistort(pinhole_camera: Camera, camera: str, request: pytest.FixtureRequest):
    """ Test projected points with distort==False match pinhole camera. """
    camera: Camera = request.getfixturevalue(camera)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    pinhole_xyz = pinhole_camera.pixel_to_world_z(ji, z)
    xyz = camera.pixel_to_world_z(ji, z, distort=False)
    ji_ = camera.world_to_pixel(xyz, distort=False)

    assert pinhole_xyz == pytest.approx(xyz)
    assert ji_ == pytest.approx(ji, abs=1e-4)


@pytest.mark.parametrize(
    'cam_type', [CameraType.brown, CameraType.opencv],
)
def test_brown_opencv_zerocoeff(
    pinhole_camera: Camera, cam_type: CameraType, camera_args: Tuple, request: pytest.FixtureRequest
):
    """ Test Brown & OpenCV cameras match pinhole camera with zero distortion coeffs. """
    camera: Camera = create_camera(cam_type, *camera_args)

    ji = np.random.rand(2, 1000) * np.reshape(camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (camera._T[2] * .8)
    pinhole_xyz = pinhole_camera.pixel_to_world_z(ji, z)
    xyz = camera.pixel_to_world_z(ji, z, distort=True)
    ji_ = camera.world_to_pixel(xyz, distort=True)

    assert pinhole_xyz == pytest.approx(xyz, abs=1e-3)
    assert ji_ == pytest.approx(ji, abs=1e-4)


def test_brown_opencv_equiv(opencv_camera: Camera, camera_args):
    """ Test OpenCV and Brown cameras are equivalent for the (cx, cy) == (0, 0) special case. """
    brown_camera = BrownCamera(*camera_args, *opencv_camera._dist_coeff)

    ji = np.random.rand(2, 1000) * np.reshape(brown_camera._im_size, (-1, 1))
    z = np.random.rand(1000) * (brown_camera._T[2] * .8)
    cv_xyz = opencv_camera.pixel_to_world_z(ji, z)
    brown_xyz = brown_camera.pixel_to_world_z(ji, z)

    assert cv_xyz == pytest.approx(brown_xyz)


@pytest.mark.parametrize(
    'camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera',]
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


def test_pixel_to_world_z_error(pinhole_camera):
    """ Test pixel_to_world_z raises a ValueError with invalid coordinate shapes. """

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros(2), np.zeros(1))
    assert '`ji`' in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((3, 1)), np.zeros(1))
    assert '`ji`' in str(ex)

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((2, 1)), np.zeros((2, 1)))

    with pytest.raises(ValueError) as ex:
        pinhole_camera.pixel_to_world_z(np.zeros((2, 3)), np.zeros(2))
    assert '`z`' in str(ex)
