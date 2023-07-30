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

_dem_resolution = (30, 30)
""" Default DEM resolution (m). """


def checkerboard(shape, square: int = 25, vals: np.ndarray = None):
    """ Return a checkerboard image given an image shape. """
    # adapted from https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    vals = np.array([1, 255], dtype=np.uint8) if vals is None else vals
    coords = np.ogrid[0:shape[0], 0:shape[1]]
    idx = (coords[0] // square + coords[1] // square) % 2
    return vals[idx]


def sinusoidal(shape: Tuple):
    """ Return a sinusoidal surface with z vals 0..1, given an array shape. """
    # adapted from https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    x = np.linspace(-4 * np.pi, 4 * np.pi, shape[1])
    y = np.linspace(-4 * np.pi, 4 * np.pi, shape[0]) * shape[0] / shape[1]
    x, y = np.meshgrid(x, y)

    array = np.sin(x + y) + np.sin(2 * x - y) + np.cos(3 * x + 4 * y)
    array -= array.min()
    array /= array.max()
    return array


def ortho_bounds(camera: Camera, dem_min: float = Ortho._egm96_min, include_camera: bool = False) -> Tuple:
    """ Return ortho bounds for the given `camera` at z=`dem_min`. """
    size = camera._im_size
    ji = np.array([[0, 0], [0, size[1]], [*size], [size[0], 0]]).T
    xyz = camera.pixel_to_world_z(ji, dem_min)
    if include_camera:
        xyz = np.column_stack((xyz, camera._T))
    return (*xyz[:2].min(axis=1), *xyz[:2].max(axis=1))


def create_dem(
    filename: Path, camera: Camera, camera_crs: str, dem_crs: str = None, resolution: Tuple = _dem_resolution,
    dtype: str = 'float32', include_camera=False,
):
    """
    Create a 2 band DEM file that covers the ortho bounds of the given `camera`.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    # wgs84_bounds = transform_bounds(rio.CRS.from_string(ortho_crs_wgs84_vdatum), dem_crs, *bounds)
    # size = 1 + np.ceil((bounds[2:] - bounds[:2]) / (30, 30)).astype('int')
    # transform = from_bounds(*wgs84_bounds, *size)

    bounds = np.array(ortho_bounds(camera, include_camera=include_camera))
    size = 1 + np.ceil((bounds[2:] - bounds[:2]) / resolution).astype('int')
    array = np.stack(
        (sinusoidal(size[::-1]) * 250 + camera._T[2] - 200, np.ones(size[::-1]) * (250 / 2) + camera._T[2] - 200,),
        axis=0
    ).astype(dtype)  # yapf: disable

    if dem_crs is not None:
        camera_crs = rio.CRS.from_string(camera_crs)
        dem_crs = rio.CRS.from_string(dem_crs)
        bounds = transform_bounds(camera_crs, dem_crs, *bounds)
        transform = from_bounds(*bounds, *size)
    else:
        dem_crs = camera_crs
        transform = from_origin(bounds[0], bounds[3], *resolution)

    profile = dict(crs=dem_crs, transform=transform, dtype=dtype, width=size[0], height=size[1], count=2)

    with rio.open(filename, 'w', **profile) as dem_im:
        dem_im.write(array)


def create_src(
    filename: Path, size: Tuple, dtype: str = 'uint8', count: int = 3, camera: Camera = None, crs: str = None
):
    """ Create a source checkerboard file with optional CRS where `camera` & `crs` are specified.  """
    profile = dict(crs=None, transform=None, dtype=dtype, width=size[0], height=size[1], count=count)

    if camera is not None:
        # find bounds & transform
        size = camera._im_size
        bounds = ortho_bounds(camera, dem_min=camera._T[2] - 100)
        transform = from_bounds(*bounds, *size)
        profile.update(crs=crs, transform=transform)

    array = checkerboard(size[::-1])
    array = np.stack((array, ) * count, axis=0).astype(dtype)

    with rio.open(filename, 'w', **profile) as src_im:
        src_im.write(array)


@pytest.fixture(scope='session')
def position() -> Tuple[float, float, float]:
    """ Example camera UTM (Easting, Northing, altitude) position (m). """
    return (2e4, 3e4, 1e3)


@pytest.fixture(scope='session')
def rotation() -> Tuple[float, float, float]:
    """ Example camera (omega, phi, kappa) rotation (radians). """
    return tuple(np.radians((-3., 2., 10.)))


@pytest.fixture(scope='session')
def focal_len() -> float:
    """ Example camera focal length (mm). """
    return 5


@pytest.fixture(scope='session')
def im_size() -> Tuple[int, int]:
    """ Example camera image size (pixels). """
    return (200, 150)


@pytest.fixture(scope='session')
def sensor_size() -> Tuple[float, float]:
    """ Example camera sensor size (mm). """
    return (6.0, 4.5)


@pytest.fixture(scope='session')
def camera_args(position, rotation, focal_len, im_size, sensor_size) -> Dict:
    """ A dictionary of positional arguments for `Camera.__init__()`. """
    return dict(position=position, rotation=rotation, focal_len=focal_len, im_size=im_size, sensor_size=sensor_size)


@pytest.fixture(scope='session')
def brown_dist_coeff() -> Dict:
    """ Example `BrownCamera` distortion coefficients. """
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=-0.1)


@pytest.fixture(scope='session')
def opencv_dist_coeff() -> Dict:
    """ Example `OpenCVCamera` distortion coefficients. """
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=-0.1, k4=0.001, k5=0.001, k6=-0.001)


@pytest.fixture(scope='session')
def fisheye_dist_coeff() -> Dict:
    """ Example `FisheyeCamera` distortion coefficients. """
    return dict(k1=-0.25, k2=0.1, k3=0.01, k4=-0.01)


@pytest.fixture(scope='session')
def pinhole_camera(camera_args) -> Camera:
    """ Example `PinholeCamera` object with near-nadir orientation. """
    return PinholeCamera(**camera_args)


@pytest.fixture(scope='session')
def brown_camera(camera_args, brown_dist_coeff: Dict) -> Camera:
    """ Example `BrownCamera` object with near-nadir orientation. """
    return BrownCamera(**camera_args, **brown_dist_coeff, cx=-0.01, cy=0.02)


@pytest.fixture(scope='session')
def opencv_camera(camera_args, opencv_dist_coeff: Dict) -> Camera:
    """ Example `OpenCVCamera` object with near-nadir orientation. """
    return OpenCVCamera(**camera_args, **opencv_dist_coeff)


@pytest.fixture(scope='session')
def nadir_fisheye_camera(camera_args, fisheye_dist_coeff: Dict) -> Camera:
    """  Example `FisheyeCamera` object with near-nadir orientation.  """
    return FisheyeCamera(**camera_args, **fisheye_dist_coeff)


@pytest.fixture(scope='session')
def utm34n_crs() -> str:
    """ CRS string for UTM zone 34N with no vertical datum. """
    return 'EPSG:32634'


@pytest.fixture(scope='session')
def utm34n_wgs84_crs() -> str:
    """ CRS string for UTM zone 34N with WGS84 ellipsoid vertical datum. """
    return 'EPSG:32634+4326'


@pytest.fixture(scope='session')
def utm34n_egm96_crs() -> str:
    """ CRS string for UTM zone 34N with EGM96 geoid vertical datum. """
    return 'EPSG:32634+5773'


@pytest.fixture(scope='session')
def utm34n_egm2008_crs() -> str:
    """ CRS string for UTM zone 34N with EGM2008 geoid vertical datum. """
    return 'EPSG:32634+3855'


@pytest.fixture(scope='session')
def rgb_byte_src_file(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ An RGB byte checkerboard image with no CRS. """
    src_filename = Path(tmpdir_factory.mktemp('data').join('rgb_byte_src.tif'))
    create_src(src_filename, im_size, dtype='uint8', count=3)
    return src_filename


@pytest.fixture(scope='session')
def rgb_float_src_file(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ An RGB float32 checkerboard image with no CRS. """
    src_filename = Path(tmpdir_factory.mktemp('data').join('rgb_float_src.tif'))
    create_src(src_filename, im_size, dtype='float32', count=3)
    return src_filename


@pytest.fixture(scope='session')
def float_src_file(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ A single band float64 checkerboard image with no CRS. """
    src_filename = Path(tmpdir_factory.mktemp('data').join('float_src.tif'))
    create_src(src_filename, im_size, dtype='float32', count=1)
    return src_filename


@pytest.fixture(scope='session')
def rgb_byte_utm34n_src_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_crs) -> Path:
    """ An RGB byte checkerboard image with UTM zone 34N CRS and bounds 100m below `pinhole_camera`. """
    src_filename = Path(tmpdir_factory.mktemp('data').join('rgb_byte_src.tif'))
    create_src(src_filename, pinhole_camera._im_size, dtype='uint8', count=3, camera=pinhole_camera, crs=utm34n_crs)
    return src_filename


@pytest.fixture(scope='session')
def float_utm34n_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_crs) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with no vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data').join('float_utm34n_dem.tif'))
    create_dem(filename, pinhole_camera, utm34n_crs, resolution=_dem_resolution, dtype='float32')
    return filename


@pytest.fixture(scope='session')
def float_utm34n_wgs84_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_wgs84_crs) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with WGS84 ellipsoid vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data').join('float_utm34n_wgs84_dem.tif'))
    create_dem(filename, pinhole_camera, utm34n_wgs84_crs, resolution=_dem_resolution, dtype='float32')
    return filename


@pytest.fixture(scope='session')
def float_utm34n_egm96_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_egm96_crs) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with EGM96 geoid vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data').join('float_utm34n_egm96_dem.tif'))
    create_dem(filename, pinhole_camera, utm34n_egm96_crs, resolution=_dem_resolution, dtype='float32')
    return filename


@pytest.fixture(scope='session')
def float_wgs84_wgs84_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_wgs84_crs) -> Path:
    """
    A 2 band float DEM file in WGS84 with WGS84 ellipsoid vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data').join('float_wgs84_wgs84_dem.tif'))
    create_dem(
        filename, pinhole_camera, utm34n_wgs84_crs, resolution=_dem_resolution, dtype='float32',
        dem_crs='EPSG:4326+4326'
    )
    return filename


@pytest.fixture(scope='session')
def rgb_pinhole_utm34n_ortho(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str
) -> Ortho:
    """
    An Ortho object initialised with with RGB byte source image, float DEM in UTM zone 34N (no vertical datum),
    pinhole camera, and UTM zone 34N CRS (no vertical datum).
    """
    return Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs)
