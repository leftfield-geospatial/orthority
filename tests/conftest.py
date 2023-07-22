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
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin, from_bounds
from rasterio.warp import transform_bounds

from simple_ortho.camera import Camera, PinholeCamera, BrownCamera, OpenCVCamera, FisheyeCamera
from simple_ortho.ortho import Ortho


def checkerboard(shape, square: int=25, vals: np.ndarray = np.array([1, 255], dtype=np.uint8)):
    """ Return a checkerboard image given an image shape. """
    # from https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    coords = np.ogrid[0:shape[0], 0:shape[1]]
    idx = (coords[0] // square + coords[1] // square) % 2
    return vals[idx]


def sinusoidal(shape: Tuple):
    """ Return a sinusoidal surface with z vals 0..1, given an array shape. """
    x = np.linspace(-7, 7, shape[0])
    y = np.linspace(-5, 5, shape[1])
    x, y = np.meshgrid(x, y)

    array = np.sin(x + y) + np.sin(2 * x - y) + np.cos(3 * x + 4 * y)
    return 0.5*(array + 1)


def ortho_bounds(camera: Camera, dem_min: float = Ortho.egm96_min, include_camera: bool = False) -> Tuple:
    """ Return ortho bounds for the given camera at z=dem_min. """
    size = camera._im_size
    ji = np.array([[0, 0], [0, size[1]], [*size], [size[0], 0]]).T
    xyz = camera.pixel_to_world_z(ji, dem_min)
    if include_camera:
        xyz = np.column_stack((xyz, camera._T))

    return (*xyz[:2].min(axis=1), *xyz[:2].max(axis=1))


def dem_params(
    camera: Camera, crs: str, resolution: Tuple = (30, 30), dtype: str = 'float32', include_camera=False
) -> Tuple[Dict, np.ndarray]:
    """ Return a DEM profile and array for the given parameters. """
    bounds = np.array(ortho_bounds(camera, include_camera=include_camera))
    size = 1 + np.ceil((bounds[2:] - bounds[:2]) / resolution).astype('int')
    transform = from_origin(bounds[0], bounds[3], *resolution)
    profile = dict(
        crs=crs, transform=transform, dtype=dtype, width=size[0], height=size[1], count=1
    )
    array = sinusoidal(size[::-1]).astype(dtype) * 20 + 800

    return profile, array


@pytest.fixture(scope='session')
def position() -> Tuple[float, float, float]:
    """ Example camera (Easting, Northing, altitude) position (m). """
    # return (363646.4512, 6243245.1684, 1098.3802)
    return (2e4, 3e4, 1e3)


@pytest.fixture(scope='session')
def nadir_rotation() -> Tuple[float, float, float]:
    """ Example camera (omega, phi, kappa) rotation (degrees). """
    # return (-6.6512, -1.0879, -79.6693)
    return tuple(np.radians((-3., 2., 10.)))


@pytest.fixture(scope='session')
def oblique_rotation() -> Tuple[float, float, float]:
    """ Example camera (omega, phi, kappa) rotation (degrees). """
    # return (-6.6512, -1.0879, -79.6693)
    return tuple(np.radians((-30., 20., 10.)))


@pytest.fixture(scope='session')
def focal_len() -> float:
    """ Example camera focal length (mm). """
    return 4.88


@pytest.fixture(scope='session')
def im_size() -> Tuple[int, int]:
    """ Example camera image size (pixels). """
    return (400, 300)   # (4608, 3456)


@pytest.fixture(scope='session')
def sensor_size() -> Tuple[float, float]:
    """ Example camera sensor size (mm). """
    return (6.17471716, 4.63103787)


@pytest.fixture(scope='session')
def nadir_camera_args(position, nadir_rotation, focal_len, im_size, sensor_size) -> Tuple:
    """ Example positional arguments for Camera.__init__(). """
    return (position, nadir_rotation, focal_len, im_size, sensor_size)


@pytest.fixture(scope='session')
def oblique_camera_args(position, oblique_rotation, focal_len, im_size, sensor_size) -> Tuple:
    """ Example positional arguments for Camera.__init__(). """
    return (position, oblique_rotation, focal_len, im_size, sensor_size)


@pytest.fixture(scope='session')
def brown_dist_coeff() -> Dict:
    """ Example BrownCamera distortion coefficients. """
    # k1=-0.0093, k2=0.0075, p1=-0.0004, p2=-0.0004, k3=0.0079
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=0.1)


@pytest.fixture(scope='session')
def opencv_dist_coeff() -> Dict:
    """ Example OpenCVCamera distortion coefficients. """
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=0.1, k4=-0.001, k5=0.001, k6=-0.001)


@pytest.fixture(scope='session')
def fisheye_dist_coeff() -> Dict:
    """ Example FisheyeCamera distortion coefficients. """
    return dict(k1=-0.25, k2=0.1, k3=0.01, k4=-0.01)


@pytest.fixture(scope='session')
def nadir_pinhole_camera(nadir_camera_args) -> Camera:
    """ Pinhole camera with nadir orientation. """
    return PinholeCamera(*nadir_camera_args)


@pytest.fixture(scope='session')
def oblique_pinhole_camera(oblique_camera_args) -> Camera:
    """ Pinhole camera with oblique orientation. """
    return PinholeCamera(*oblique_camera_args)


@pytest.fixture(scope='session')
def nadir_brown_camera(nadir_camera_args, brown_dist_coeff) -> Camera:
    """ Brown camera with nadir orientation. """
    return BrownCamera(*nadir_camera_args, **brown_dist_coeff, cx=-0.01, cy=0.02)


@pytest.fixture(scope='session')
def oblique_brown_camera(oblique_camera_args, brown_dist_coeff) -> Camera:
    """ Brown camera with oblique orientation. """
    return BrownCamera(*oblique_camera_args, **brown_dist_coeff, cx=-0.01, cy=0.02)


@pytest.fixture(scope='session')
def nadir_opencv_camera(nadir_camera_args, opencv_dist_coeff) -> Camera:
    """ OpenCV camera with nadir orientation. """
    return OpenCVCamera(*nadir_camera_args, **opencv_dist_coeff)


@pytest.fixture(scope='session')
def oblique_opencv_camera(oblique_camera_args, opencv_dist_coeff) -> Camera:
    """ OpenCV camera with oblique orientation. """
    return OpenCVCamera(*oblique_camera_args, **opencv_dist_coeff)


@pytest.fixture(scope='session')
def nadir_fisheye_camera(nadir_camera_args, fisheye_dist_coeff) -> Camera:
    """ Fisheye camera with nadir orientation. """
    return FisheyeCamera(*nadir_camera_args, **fisheye_dist_coeff)


@pytest.fixture(scope='session')
def oblique_fisheye_camera(oblique_camera_args, fisheye_dist_coeff) -> Camera:
    """ Fisheye camera with oblique orientation. """
    return FisheyeCamera(*oblique_camera_args, **fisheye_dist_coeff)


@pytest.fixture(scope='session')
def ortho_crs_no_vdatum() -> str:
    """ Ortho / world CRS with no vertical datum. """
    return 'EPSG:32634'


@pytest.fixture(scope='session')
def ortho_crs_wgs84_vdatum() -> str:
    """ Ortho / world CRS with WGS84 vertical datum. """
    return 'EPSG:32634+4326'


@pytest.fixture(scope='session')
def ortho_crs_egm96_vdatum() -> str:
    """ Ortho / world CRS with EGM96 vertical datum. """
    return 'EPSG:32634+5773'


@pytest.fixture(scope='session')
def ortho_crs_egm2008_vdatum() -> str:
    """ Ortho / world CRS with EGM2008 vertical datum. """
    return 'EPSG:32634+3855'


@pytest.fixture(scope='session')
def src_file_rgb_byte(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ An RGB byte source file with no CRS. """
    profile = dict(
        crs=None, transform=None, dtype='uint8', width=im_size[0], height=im_size[1], count=3,
    )
    src_filename = Path(tmpdir_factory.mktemp('data').join('src-rgb-byte.tif'))
    src_array = checkerboard(im_size[::-1])
    src_array = np.stack((src_array,) * profile['count'], axis=0)

    with rio.open(src_filename, 'w', **profile) as src_im:
        src_im.write(src_array)
    return src_filename


@pytest.fixture(scope='session')
def src_file_float(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ A single band float source file with no CRS. """
    profile = dict(
        crs=None, transform=None, dtype='float64', width=im_size[0], height=im_size[1], count=1,
    )
    src_filename = Path(tmpdir_factory.mktemp('data').join('src-1band-float.tif'))
    src_array = checkerboard(im_size[::-1], vals=np.array([0, 1], dtype=profile['dtype']))

    with rio.open(src_filename, 'w', **profile) as src_im:
        src_im.write(src_array)
    return src_filename


@pytest.fixture(scope='session')
def src_file_rgb_byte_crs(
    tmpdir_factory: pytest.TempdirFactory, nadir_pinhole_camera: Camera, ortho_crs_no_vdatum
) -> Path:
    """ An RGB byte source file with ortho CRS and bounds 100m below nadir_pinhole_camera. """
    # find bounds & transform
    size = nadir_pinhole_camera._im_size
    bounds = ortho_bounds(nadir_pinhole_camera, dem_min=nadir_pinhole_camera._T[2] - 100)
    transform = from_bounds(*bounds, *size)

    # create file
    profile = dict(
        crs=ortho_crs_no_vdatum, transform=transform, dtype='uint8', width=size[0], height=size[1], count=3,
    )
    filename = Path(tmpdir_factory.mktemp('data').join('src-rgb-byte-crs.tif'))
    array = checkerboard(size[::-1])
    array = np.stack((array,) * 3, axis=0)
    with rio.open(filename, 'w', **profile) as src_im:
        src_im.write(array, (1, 2, 3))

    return filename


@pytest.fixture(scope='session')
def nadir_dem_30m_float_no_vdatum(
    tmpdir_factory: pytest.TempdirFactory, nadir_pinhole_camera, oblique_rotation: Tuple, ortho_crs_no_vdatum
) -> Path:
    """ A DEM file with nadir coverage, 30m resolution, float data type and no vertical datum. """
    # def get_test_dem(size: Tuple):
    #     """Return a surface with a peak and depression."""
    #     # adapted from matplotlib
    #     x = np.linspace(-3, 3, size[0])
    #     y = np.linspace(-3, 3, size[1])
    #     X, Y = np.meshgrid(x, y)
    #
    #     Z1 = np.exp(-(X ** 2 + Y ** 2) / 2) / (2 * np.pi)
    #     Z2 = (np.exp(-(((X - 1) / 1.5) ** 2 + ((Y - 1) / 0.5) ** 2) / 2) / (2 * np.pi * 0.5 * 1.5))
    #     Z = Z2 - Z1
    #     return Z * 500
    #
    # def get_test_dem2(size: Tuple, alpha: int, eta=1, purity=1):
    #     """ Multiphoton shrodinger cat. eta is the fidelity, alpha the number of photons"""
    #     # adapted from mayavi example
    #     x = np.linspace(-3, 3, size[0])
    #     y = np.linspace(-3, 3, size[1])
    #     x, y = np.meshgrid(x, y)
    #
    #     return (1 + eta * (
    #         np.exp(-x ** 2 - (y - alpha) ** 2) + np.exp(-x ** 2 - (y + alpha) ** 2) +
    #         2 * purity * np.exp(-x ** 2 - y ** 2) * np.cos(2 * alpha * x)) / (2 * (1 + np.exp(- alpha ** 2)))
    #     ) / 2   # yapf: disable

    profile, array = dem_params(nadir_pinhole_camera, ortho_crs_no_vdatum, resolution=(30, 30), dtype='float64')
    filename = Path(tmpdir_factory.mktemp('data').join('nadir-dem-float-no_vdatum.tif'))

    with rio.open(filename, 'w', **profile) as dem_im:
        dem_im.write(array, 1)
    return filename


@pytest.fixture(scope='session')
def nadir_dem_30m_float_wgs84_vdatum(
    tmpdir_factory: pytest.TempdirFactory, nadir_pinhole_camera, oblique_rotation: Tuple, ortho_crs_wgs84_vdatum
) -> Path:
    """ A DEM file with nadir coverage, 30m resolution, float data type and EGM96 vertical datum. """
    profile, array = dem_params(nadir_pinhole_camera, ortho_crs_wgs84_vdatum, resolution=(30, 30), dtype='float64')
    filename = Path(tmpdir_factory.mktemp('data').join('nadir-dem-float-wgs84_vdatum.tif'))

    with rio.open(filename, 'w', **profile) as dem_im:
        dem_im.write(array, 1)
    return filename


@pytest.fixture(scope='session')
def nadir_dem_30m_float_egm96_vdatum(
    tmpdir_factory: pytest.TempdirFactory, nadir_pinhole_camera, oblique_rotation: Tuple, ortho_crs_egm96_vdatum
) -> Path:
    """ A DEM file with nadir coverage, 30m resolution, float data type and EGM96 vertical datum. """
    profile, array = dem_params(nadir_pinhole_camera, ortho_crs_egm96_vdatum, resolution=(30, 30), dtype='float64')
    filename = Path(tmpdir_factory.mktemp('data').join('nadir-dem-float-egm96_vdatum.tif'))

    with rio.open(filename, 'w', **profile) as dem_im:
        dem_im.write(array, 1)
    return filename


@pytest.fixture(scope='session')
def oblique_dem_30m_float_no_vdatum(
    tmpdir_factory: pytest.TempdirFactory, nadir_pinhole_camera, oblique_rotation: Tuple, ortho_crs_no_vdatum
) -> Path:
    """ A DEM file with oblique coverage, 30m resolution, float data type and no vertical datum. """
    profile, array = dem_params(
        nadir_pinhole_camera, ortho_crs_no_vdatum, resolution=(30, 30), dtype='float64', include_camera=True
    )
    filename = Path(tmpdir_factory.mktemp('data').join('oblique-dem-float-no_vdatum.tif'))

    with rio.open(filename, 'w', **profile) as dem_im:
        dem_im.write(array, 1)
    return filename


@pytest.fixture(scope='session')
def nadir_dem_30m_uint16_no_vdatum(
    tmpdir_factory: pytest.TempdirFactory, nadir_pinhole_camera, oblique_rotation: Tuple, ortho_crs_no_vdatum
) -> Path:
    """ A DEM file with nadir coverage, 30m resolution, uint16 data type and no vertical datum. """
    profile, array = dem_params(nadir_pinhole_camera, ortho_crs_no_vdatum, resolution=(30, 30), dtype='uint16')
    filename = Path(tmpdir_factory.mktemp('data').join('nadir-dem-float-no_vdatum.tif'))

    with rio.open(filename, 'w', **profile) as dem_im:
        dem_im.write(array, 1)
    return filename


@pytest.fixture(scope='session')
def nadir_dem_30m_float_wgs84_wgs84_vdatum(
    tmpdir_factory: pytest.TempdirFactory, nadir_pinhole_camera, oblique_rotation: Tuple, ortho_crs_wgs84_vdatum
) -> Path:
    """ A DEM file in WGS84 with nadir coverage, 30m resolution, float data type and WGS84 vertical datum. """
    dtype = 'float32'
    dem_crs = rio.CRS.from_string('EPSG:4326+4326')
    bounds = np.array(ortho_bounds(nadir_pinhole_camera))
    wgs84_bounds = transform_bounds(rio.CRS.from_string(ortho_crs_wgs84_vdatum), dem_crs, *bounds)
    size = 1 + np.ceil((bounds[2:] - bounds[:2]) / (30, 30)).astype('int')
    transform = from_bounds(*wgs84_bounds, *size)
    profile = dict(
        crs=dem_crs, transform=transform, dtype=dtype, width=size[0], height=size[1], count=1
    )
    array = sinusoidal(size[::-1]).astype(dtype) * 20 + 800

    filename = Path(tmpdir_factory.mktemp('data').join('nadir-dem-float-no_vdatum.tif'))
    with rio.open(filename, 'w', **profile) as dem_im:
        dem_im.write(array, 1)
    return filename
