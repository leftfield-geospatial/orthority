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
from collections import namedtuple

import json
import os
import yaml
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin, from_bounds
from rasterio.warp import transform_bounds, transform

from simple_ortho.enums import CameraType
from simple_ortho.camera import Camera, PinholeCamera, BrownCamera, OpenCVCamera, FisheyeCamera
from simple_ortho.ortho import Ortho
from simple_ortho.exif import Exif

_dem_resolution = (30., 30.)
""" Default DEM resolution (m). """


if '__file__' in globals():
    root_path = Path(__file__).absolute().parents[1]
else:
    root_path = Path(os.getcwd())


def checkerboard(shape, square: int = 25, vals: np.ndarray = None):
    """ Return a checkerboard image given an image shape. """
    # adapted from https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    vals = np.array([127, 255], dtype=np.uint8) if vals is None else vals
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
    # TODO: add partial dem fixture
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


def oty_to_osfm_int_param(int_param_dict: Dict) -> Dict:
    """ Return equivalent OpenSfM / ODM format interior parameters for given orthority format parameters. """
    osfm_dict = {}
    for cam_id, int_params in int_param_dict.items():
        osfm_params = int_params.copy()
        cam_type = osfm_params.pop('cam_type')
        osfm_params['projection_type'] = 'perspective' if cam_type == 'brown' else cam_type
        im_size = osfm_params.pop('im_size')
        osfm_params['width'] = im_size[0]
        osfm_params['height'] = im_size[1]

        sensor_size = osfm_params.pop('sensor_size')
        osfm_params['focal_x'] = osfm_params['focal_y'] = osfm_params.pop('focal_len') / sensor_size[0]

        for from_key, to_key in zip(['cx', 'cy'], ['c_x', 'c_y']):
            if from_key in osfm_params:
                osfm_params[to_key] = osfm_params.pop(from_key)
        osfm_dict[cam_id] = osfm_params
    return osfm_dict


@pytest.fixture(scope='session')
def xyz() -> Tuple[float, float, float]:
    """ Example camera (easting, northing, altitude) position (m). """
    return 2e4, 3e4, 1e3


@pytest.fixture(scope='session')
def opk() -> Tuple[float, float, float]:
    """ Example camera (omega, phi, kappa) rotation (radians). """
    return tuple(np.radians((-3., 2., 10.)).tolist())


@pytest.fixture(scope='session')
def focal_len() -> float:
    """ Example camera focal length (mm). """
    return 5


@pytest.fixture(scope='session')
def im_size() -> Tuple[int, int]:
    """ Example camera image size (pixels). """
    return 200, 150


@pytest.fixture(scope='session')
def sensor_size() -> Tuple[float, float]:
    """ Example camera sensor size (mm). """
    return 6.0, 4.5


@pytest.fixture(scope='session')
def cxy() -> Tuple[float, float]:
    """ Example principal point offset (normalised image coordinates). """
    return -0.01, 0.02


@pytest.fixture(scope='session')
def interior_args(focal_len, im_size, sensor_size, cxy) -> Dict:
    """ A dictionary of interior parameters for `Camera.__init__()`. """
    return dict(
        im_size=im_size, focal_len=focal_len / sensor_size[0], sensor_size=(1, sensor_size[1] / sensor_size[0]),
        cx=cxy[0], cy=cxy[1]
    )


@pytest.fixture(scope='session')
def exterior_args(xyz: Tuple, opk: Tuple) -> Dict:
    """ A dictionary of exterior parameters for `Camera.__init__()` / `Camera.update()`. """
    return dict(xyz=xyz, opk=opk)


@pytest.fixture(scope='session')
def camera_args(interior_args: Dict, exterior_args: Dict) -> Dict:
    """ A dictionary of interior and exterior parameters for `Camera.__init__()`. """
    return dict(**interior_args, **exterior_args)


@pytest.fixture(scope='session')
def brown_dist_param() -> Dict:
    """ Example `BrownCamera` distortion coefficients. """
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=-0.1)


@pytest.fixture(scope='session')
def opencv_dist_param() -> Dict:
    """ Example `OpenCVCamera` distortion coefficients. """
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=-0.1, k4=0.001, k5=0.001, k6=-0.001)


@pytest.fixture(scope='session')
def fisheye_dist_param() -> Dict:
    """ Example `FisheyeCamera` distortion coefficients. """
    return dict(k1=-0.25, k2=0.1, k3=0.01, k4=-0.01)


@pytest.fixture(scope='session')
def pinhole_camera(camera_args) -> Camera:
    """ Example `PinholeCamera` object with near-nadir orientation. """
    return PinholeCamera(**camera_args)


@pytest.fixture(scope='session')
def brown_camera(camera_args, brown_dist_param: Dict) -> Camera:
    """ Example `BrownCamera` object with near-nadir orientation. """
    return BrownCamera(**camera_args, **brown_dist_param)


@pytest.fixture(scope='session')
def opencv_camera(camera_args, opencv_dist_param: Dict) -> Camera:
    """ Example `OpenCVCamera` object with near-nadir orientation. """
    return OpenCVCamera(**camera_args, **opencv_dist_param)


@pytest.fixture(scope='session')
def fisheye_camera(camera_args, fisheye_dist_param: Dict) -> Camera:
    """  Example `FisheyeCamera` object with near-nadir orientation.  """
    return FisheyeCamera(**camera_args, **fisheye_dist_param)


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
def webmerc_crs() -> str:
    """ CRS string for web mercator with no vertical datum. """
    return '+proj=webmerc +datum=WGS84'


@pytest.fixture(scope='session')
def webmerc_wgs84_crs() -> str:
    """ CRS string for web mercator with WGS84 ellipsoid vertical datum. """
    return '+proj=webmerc +datum=WGS84 +ellps=WGS84 +vunits=m'


@pytest.fixture(scope='session')
def webmerc_egm96_crs() -> str:
    """ CRS string for web mercator with EGM96 geoid vertical datum. """
    return '+proj=webmerc +datum=WGS84 +geoidgrids=egm96_15.gtx +vunits=m'


@pytest.fixture(scope='session')
def webmerc_egm2008_crs() -> str:
    """ CRS string for web mercator with EGM2008 geoid vertical datum. """
    return '+proj=webmerc +datum=WGS84 +geoidgrids=egm08_25.gtx +vunits=m'


@pytest.fixture(scope='session')
def rgb_byte_src_file(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ An RGB byte checkerboard image with no CRS. """
    src_filename = Path(tmpdir_factory.mktemp('data')).joinpath('rgb_byte_src.tif')
    create_src(src_filename, im_size, dtype='uint8', count=3)
    return src_filename


@pytest.fixture(scope='session')
def rgb_float_src_file(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ An RGB float32 checkerboard image with no CRS. """
    src_filename = Path(tmpdir_factory.mktemp('data')).joinpath('rgb_float_src.tif')
    create_src(src_filename, im_size, dtype='float32', count=3)
    return src_filename


@pytest.fixture(scope='session')
def float_src_file(tmpdir_factory: pytest.TempdirFactory, im_size: Tuple) -> Path:
    """ A single band float64 checkerboard image with no CRS. """
    src_filename = Path(tmpdir_factory.mktemp('data')).joinpath('float_src.tif')
    create_src(src_filename, im_size, dtype='float32', count=1)
    return src_filename


@pytest.fixture(scope='session')
def rgb_byte_utm34n_src_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_crs) -> Path:
    """ An RGB byte checkerboard image with UTM zone 34N CRS and bounds 100m below `pinhole_camera`. """
    src_filename = Path(tmpdir_factory.mktemp('data')).joinpath('rgb_byte_src.tif')
    create_src(src_filename, pinhole_camera._im_size, dtype='uint8', count=3, camera=pinhole_camera, crs=utm34n_crs)
    return src_filename


@pytest.fixture(scope='session')
def float_utm34n_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_crs) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with no vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data')).joinpath('float_utm34n_dem.tif')
    create_dem(filename, pinhole_camera, utm34n_crs, resolution=_dem_resolution, dtype='float32')
    return filename


@pytest.fixture(scope='session')
def float_utm34n_wgs84_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_wgs84_crs) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with WGS84 ellipsoid vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data')).joinpath('float_utm34n_wgs84_dem.tif')
    create_dem(filename, pinhole_camera, utm34n_wgs84_crs, resolution=_dem_resolution, dtype='float32')
    return filename


@pytest.fixture(scope='session')
def float_utm34n_egm96_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_egm96_crs) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with EGM96 geoid vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data')).joinpath('float_utm34n_egm96_dem.tif')
    create_dem(filename, pinhole_camera, utm34n_egm96_crs, resolution=_dem_resolution, dtype='float32')
    return filename


@pytest.fixture(scope='session')
def float_wgs84_wgs84_dem_file(tmpdir_factory: pytest.TempdirFactory, pinhole_camera, utm34n_wgs84_crs) -> Path:
    """
    A 2 band float DEM file in WGS84 with WGS84 ellipsoid vertical datum.
    Band 1 is a sinusoidal surface, and band 2, a planar surface.
    """
    filename = Path(tmpdir_factory.mktemp('data')).joinpath('float_wgs84_wgs84_dem.tif')
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


@pytest.fixture(scope='session')
def pinhole_int_param_dict(interior_args: Dict) -> Dict:
    """ A pinhole camera interior parameter dictionary. """
    return {'pinhole test camera': dict(cam_type=CameraType.pinhole, **interior_args)}


@pytest.fixture(scope='session')
def opencv_int_param_dict(interior_args: Dict, opencv_dist_param: Dict) -> Dict:
    """ An opencv camera interior parameter dictionary. """
    return {'cv test camera': dict(cam_type=CameraType.opencv, **interior_args, **opencv_dist_param)}


@pytest.fixture(scope='session')
def brown_int_param_dict(interior_args: Dict, brown_dist_param: Dict) -> Dict:
    """ A brown camera interior parameter dictionary. """
    return {'brown test camera': dict(cam_type=CameraType.brown, **interior_args, **brown_dist_param)}


@pytest.fixture(scope='session')
def fisheye_int_param_dict(interior_args: Dict, fisheye_dist_param: Dict) -> Dict:
    """ A fisheye camera interior parameter dictionary. """
    return {'fisheye test camera': dict(cam_type=CameraType.opencv, **interior_args, **fisheye_dist_param)}


@pytest.fixture(scope='session')
def mult_int_param_dict(
    pinhole_int_param_dict: Dict, brown_int_param_dict: Dict, opencv_int_param_dict: Dict, fisheye_int_param_dict: Dict
) -> Dict:
    """ An interior parameter dictionary consisting of multiple cameras. """
    return dict(**pinhole_int_param_dict, **brown_int_param_dict, **opencv_int_param_dict, **fisheye_int_param_dict)


@pytest.fixture(scope='session')
def oty_int_param_file(tmpdir_factory: pytest.TempdirFactory, mult_int_param_dict: Dict,) -> Path:
    """ An interior parameter file in orthority yaml format. """
    filename = Path(tmpdir_factory.mktemp('data')).joinpath('oty_int_param_file.yaml')
    with open(filename, 'w') as f:
        yaml.dump(mult_int_param_dict, f)
    return filename


@pytest.fixture(scope='session')
def odm_int_param_file(tmpdir_factory: pytest.TempdirFactory, mult_int_param_dict: Dict) -> Path:
    """ An interior parameter file in ODM cameras.json format. """
    filename = Path(tmpdir_factory.mktemp('data')).joinpath('odm_int_param_file.json')
    int_param = oty_to_osfm_int_param(mult_int_param_dict)
    with open(filename, 'w') as f:
        json.dump(int_param, f)
    return filename


@pytest.fixture(scope='session')
def osfm_int_param_file(tmpdir_factory: pytest.TempdirFactory, mult_int_param_dict: Dict) -> Path:
    """ An interior parameter file in OpenSfM reconstruction.json format. """
    filename = Path(tmpdir_factory.mktemp('data')).joinpath('osfm_int_param_file.json')
    int_param = oty_to_osfm_int_param(mult_int_param_dict)
    int_param = [dict(cameras=int_param)]
    with open(filename, 'w') as f:
        json.dump(int_param, f)
    return filename


@pytest.fixture(scope='session')
def odm_proj_dir() -> Path:
    """ ODM project directory. """
    return root_path.joinpath('tests', 'data', 'odm')


@pytest.fixture(scope='session')
def odm_image_files(odm_proj_dir: Path) -> Tuple[Path, ...]:
    """ ODM drone image files. """
    return tuple([fn for fn in odm_proj_dir.joinpath('images').glob('*.tif')])


@pytest.fixture(scope='session')
def odm_image_file(odm_proj_dir: Path) -> Path:
    """ ODM drone image file. """
    return next(iter(odm_proj_dir.joinpath('images').glob('*.tif')))


@pytest.fixture(scope='session')
def odm_dem_file(odm_proj_dir: Path) -> Path:
    """ ODM DEM file. """
    return odm_proj_dir.joinpath('odm_dem', 'dsm.tif')


@pytest.fixture(scope='session')
def osfm_reconstruction_file(odm_proj_dir: Path) -> Path:
    """ ODM / OpenSfM reconstruction file. """
    return odm_proj_dir.joinpath('opensfm', 'reconstruction.json')


@pytest.fixture(scope='session')
def odm_crs(odm_dem_file) -> str:
    """ CRS string for ODM exterior parameters & orthos in EPSG format. """
    with rio.open(odm_dem_file, 'r') as im:
        crs = im.crs
    return f'EPSG:{crs.to_epsg()}'


@pytest.fixture(scope='session')
def ngi_image_files() -> Tuple[Path, ...]:
    """ NGI image files. """
    return tuple([fn for fn in root_path.joinpath('tests', 'data', 'ngi').glob('*RGB.tif')])


@pytest.fixture(scope='session')
def ngi_image_file() -> Path:
    """ NGI aerial image file. """
    return next(iter(root_path.joinpath('tests', 'data', 'ngi').glob('*RGB.tif')))


@pytest.fixture(scope='session')
def ngi_dem_file() -> Path:
    """ NGI DEM file. """
    return root_path.joinpath('tests', 'data', 'ngi', 'dem.tif')


@pytest.fixture(scope='session')
def ngi_crs(ngi_image_file) -> str:
    """ CRS string for NGI exterior parameters & orthos in proj4 format. """
    with rio.open(ngi_image_file, 'r') as im:
        crs = im.crs
    return crs.to_proj4()

# TODO: move fixtures used by a single module to that module
@pytest.fixture(scope='session')
def ngi_legacy_config_file() -> Path:
    """ Legacy format configuration file for NGI test data. """
    return root_path.joinpath('tests', 'data', 'ngi', 'config.yaml')


@pytest.fixture(scope='session')
def ngi_oty_int_param_file() -> Path:
    """ Orthority format interior parameter file for NGI test data. """
    return root_path.joinpath('tests', 'data', 'io', 'ngi_int_param.yaml')


@pytest.fixture(scope='session')
def ngi_legacy_csv_file() -> Path:
    """ Legacy format exterior parameter CSV file for NGI test data. """
    return root_path.joinpath('tests', 'data', 'ngi', 'camera_pos_ori.txt')


@pytest.fixture(scope='session')
def ngi_oty_ext_param_file() -> Path:
    """ Orthority format exterior parameter file for NGI test data. """
    return root_path.joinpath('tests', 'data', 'io', 'ngi_ext_param.geojson')


@pytest.fixture(scope='session')
def exif_image_file() -> Path:
    """ An ODM image file without the 'DewarpData' XMP tag. """
    return root_path.joinpath('tests', 'data', 'io', '100_0005_0140.tif')


@pytest.fixture(scope='session')
def ngi_xyz_opk_csv_file() -> Path:
    """
    Exterior parameters for NGI data in (easting, northing, altitude), (omega, phi, kappa) CSV format. Includes
    a header and .proj file.
    """
    return root_path.joinpath('tests', 'data', 'io', 'ngi_xyz_opk.csv')


@pytest.fixture(scope='session')
def odm_lla_rpy_csv_file() -> Path:
    """
    Exterior parameters for ODM data in (latitude, longitude, altitude), (roll, pitch, yaw) CSV format. Includes
    a header.
    """
    return root_path.joinpath('tests', 'data', 'io', 'odm_lla_rpy.csv')

