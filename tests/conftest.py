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
    return (363730.5706, 6243364.7372, 1108.8894)


@pytest.fixture
def rotation() -> Tuple[float, float, float]:
    return (-4.1245, 4.2025, -106.5994)


@pytest.fixture
def focal_len() -> float:
    return 0.8790


@pytest.fixture
def im_size() -> Tuple[int, int]:
    return (640, 1152)


@pytest.fixture
def sensor_size() -> Tuple[float, float]:
    return (1, 0.75)


@pytest.fixture
def camera_args(position, rotation, focal_len, im_size, sensor_size) -> Tuple:
    return (position, rotation, focal_len, im_size, sensor_size)


@pytest.fixture
def pinhole_camera(camera_args) -> Camera:
    """ Pinhole camera. """
    return PinholeCamera(*camera_args)


@pytest.fixture
def brown_camera(camera_args) -> Camera:
    """ Brown camera. """
    return BrownCamera(*camera_args, k1=-0.0093, k2=0.0075, p1=-0.0004, p2=-0.0004, k3=0.0079, cx=-0.0049, cy=0.0011,)


@pytest.fixture
def opencv_camera(camera_args) -> Camera:
    """ OpenCV camera. """
    return OpenCVCamera(*camera_args, k1=-0.0093, k2=0.0075, p1=-0.0004, p2=-0.0004, k3=0.0079,)


@pytest.fixture
def fisheye_camera(camera_args) -> Camera:
    """ Fisheye camera. """
    return FisheyeCamera(*camera_args, k1=-0.0525, k2=-0.0098,)

