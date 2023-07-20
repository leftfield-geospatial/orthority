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
import pathlib
from typing import Dict, List, Tuple

import pytest
from simple_ortho.camera import Camera, PinholeCamera, BrownCamera, OpenCVCamera, FisheyeCamera


@pytest.fixture
def position() -> Tuple[float, float, float]:
    """ Example camera (Easting, Northing, altitude) position (m). """
    return (363646.4512, 6243245.1684, 1098.3802)


@pytest.fixture
def rotation() -> Tuple[float, float, float]:
    """ Example camera (omega, phi, kappa) rotation (degrees). """
    return (-6.6512, -1.0879, -79.6693)


@pytest.fixture
def focal_len() -> float:
    """ Example camera focal length (mm). """
    return 4.88


@pytest.fixture
def im_size() -> Tuple[int, int]:
    """ Example camera image size (pixels). """
    return (400, 300)   # (4608, 3456)


@pytest.fixture
def sensor_size() -> Tuple[float, float]:
    """ Example camera sensor size (mm). """
    return (6.17471716, 4.63103787)


@pytest.fixture
def camera_args(position, rotation, focal_len, im_size, sensor_size) -> Tuple:
    """ Example positional arguments for Camera.__init__(). """
    return (position, rotation, focal_len, im_size, sensor_size)


@pytest.fixture
def brown_dist_coeff() -> Dict:
    """ Example BrownCamera distortion coefficients. """
    # k1=-0.0093, k2=0.0075, p1=-0.0004, p2=-0.0004, k3=0.0079
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=0.1)


@pytest.fixture
def opencv_dist_coeff() -> Dict:
    """ Example OpenCVCamera distortion coefficients. """
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=0.1, k4=-0.001, k5=0.001, k6=-0.001)


@pytest.fixture
def fisheye_dist_coeff() -> Dict:
    """ Example FisheyeCamera distortion coefficients. """
    return dict(k1=-0.25, k2=0.1)


@pytest.fixture
def pinhole_camera(camera_args) -> Camera:
    """ Pinhole camera. """
    return PinholeCamera(*camera_args)


@pytest.fixture
def brown_camera(camera_args, brown_dist_coeff) -> Camera:
    """ Brown camera. """
    # cx = -0.0049, cy = 0.0011,
    return BrownCamera(*camera_args, **brown_dist_coeff, cx=-0.01, cy=0.02)


@pytest.fixture
def opencv_camera(camera_args, opencv_dist_coeff) -> Camera:
    """ OpenCV camera. """
    return OpenCVCamera(*camera_args, **opencv_dist_coeff)


@pytest.fixture
def fisheye_camera(camera_args, fisheye_dist_coeff) -> Camera:
    """ Fisheye camera. """
    return FisheyeCamera(*camera_args, **fisheye_dist_coeff)

