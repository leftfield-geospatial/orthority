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

import csv
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import array_bounds, transform, transform_bounds

from orthority.camera import (
    BrownCamera, Camera, FisheyeCamera, FrameCamera, OpenCVCamera, PinholeCamera, RpcCamera,
)
from orthority.enums import CameraType
from orthority.ortho import Ortho

_dem_resolution = (30.0, 30.0)
"""Default DEM resolution (m)."""
_dem_offset = 825.0
"""Default DEM average height (m)."""
_dem_gain = 25.0
"""Default DEM absolute height variation (m)."""

if '__file__' in globals():
    root_path = Path(__file__).absolute().parents[1]
else:
    root_path = Path(os.getcwd())


def checkerboard(shape: tuple[int, int], square: int = 25, vals: np.ndarray = None) -> np.ndarray:
    """Return a checkerboard image given an image ``shape``."""
    # adapted from https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    vals = np.array([127, 255], dtype=np.uint8) if vals is None else vals
    coords = np.ogrid[0 : shape[0], 0 : shape[1]]
    idx = (coords[0] // square + coords[1] // square) % 2
    return vals[idx]


def sinusoid(shape: tuple[int, int]) -> np.ndarray:
    """Return a sinusoidal surface with z vals -1..1, given an array ``shape``."""
    # adapted from https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#surf
    x = np.linspace(-4 * np.pi, 4 * np.pi, shape[1])
    y = np.linspace(-4 * np.pi, 4 * np.pi, shape[0]) * shape[0] / shape[1]
    x, y = np.meshgrid(x, y)

    array = np.sin(x + y) + np.sin(2 * x - y) + np.cos(3 * x + 4 * y)
    array -= array.mean()
    array /= np.max(np.abs((array.min(), array.max())))
    return array


def ortho_bounds(camera: Camera, z: float = Ortho._egm_minmax[0]) -> tuple:
    """Return (left, bottom, right, top) ortho bounds for the given ``camera`` at z=``z``."""
    w, h = np.array(camera.im_size) - 1
    ji = np.array(
        [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]]
    ).T
    xyz = camera.pixel_to_world_z(ji, z)
    if isinstance(camera, FrameCamera) and camera.pos:
        xyz = np.column_stack((xyz, camera.pos))
    return *xyz[:2].min(axis=1), *xyz[:2].max(axis=1)


def create_zsurf(
    bounds: tuple[float],
    z_off: float = _dem_offset,
    z_gain: float = _dem_gain,
    resolution: tuple[float, float] = _dem_resolution,
) -> tuple[np.ndarray, rio.Affine]:
    """
    Return a z surface (DEM array) and corresponding transform, such that the surface covers the
    given ``bounds``, at the given ``resolution``.

    The returned array has 2 bands.  Band 1 is a sinusoidal surface with average height ``z_off``
    and ~amplitude ``z_gain`` (approx.).  Band 2 is a horizontal plane at height ``z_off``.
    """
    bounds = np.array(bounds)
    shape = np.ceil((bounds[2:] - bounds[:2]) / resolution).astype('int')[::-1]

    array = np.stack((sinusoid(shape) * z_gain + z_off, np.ones(shape) * z_off), axis=0)
    array = array.astype('float32')
    transform = from_origin(bounds[0], bounds[3], *resolution)
    return array, transform


def create_profile(
    array: np.ndarray,
    transform: rio.Affine = None,
    crs: str | rio.CRS = None,
    nodata: int | float = None,
) -> dict:
    """Return a Rasterio profile for the given parameters."""
    if array.ndim != 2 and array.ndim != 3:
        raise ValueError("'array' should be 2D or 3D.")
    shape = (1, *array.shape) if array.ndim == 2 else array.shape
    return dict(
        crs=crs,
        transform=transform,
        dtype=array.dtype,
        width=shape[2],
        height=shape[1],
        count=shape[0],
        nodata=nodata,
    )


def oty_to_osfm_int_param(int_param_dict: dict) -> dict:
    """Return equivalent OpenSfM / ODM format interior parameters for given orthority format
    parameters.
    """
    osfm_dict = {}
    for cam_id, int_params in int_param_dict.items():
        osfm_params = int_params.copy()
        cam_type = osfm_params.pop('cam_type')
        osfm_params['projection_type'] = 'perspective' if cam_type == 'brown' else cam_type
        im_size = osfm_params.pop('im_size')
        osfm_params['width'] = im_size[0]
        osfm_params['height'] = im_size[1]

        if 'sensor_size' in osfm_params:
            sensor_size = osfm_params.pop('sensor_size')
            osfm_params['focal_x'] = osfm_params['focal_y'] = (
                osfm_params.pop('focal_len') / sensor_size[0]
            )
        else:
            osfm_params['focal_x'] = osfm_params['focal_y'] = osfm_params.pop('focal_len')

        for from_key, to_key in zip(['cx', 'cy'], ['c_x', 'c_y']):
            if from_key in osfm_params:
                osfm_params[to_key] = osfm_params.pop(from_key)
        osfm_dict[cam_id] = osfm_params
    return osfm_dict


@pytest.fixture(scope='session')
def runner():
    """Click runner for command line execution."""
    return CliRunner()


@pytest.fixture(scope='session')
def xyz() -> tuple[float, float, float]:
    """Example camera (x, y, z) position (m)."""
    return 2e4, 3e4, 1e3


@pytest.fixture(scope='session')
def opk() -> tuple[float, float, float]:
    """Example camera (omega, phi, kappa) rotation (radians)."""
    return tuple(np.radians((-3.0, 2.0, 10.0)).tolist())


@pytest.fixture(scope='session')
def focal_len() -> float:
    """Example camera focal length (mm)."""
    return 5


@pytest.fixture(scope='session')
def im_size() -> tuple[int, int]:
    """Example camera image size (pixels)."""
    return 200, 150


@pytest.fixture(scope='session')
def sensor_size() -> tuple[float, float]:
    """Example camera sensor size (mm)."""
    return 6.0, 4.5


@pytest.fixture(scope='session')
def cxy() -> tuple[float, float]:
    """Example principal point offset (normalised image coordinates)."""
    return -0.01, 0.02


@pytest.fixture(scope='session')
def rpc(xyz: tuple[float, float, float], utm34n_crs: str, im_size: tuple[int, int]) -> dict:
    """Example RPC dictionary."""
    xyz_ = transform(utm34n_crs, 'EPSG:4326', *[[coord] for coord in xyz])
    lat_scale = 0.005 / 2  # frame camera fixture ortho bounds are ~0.01 deg
    long_scale = lat_scale * im_size[0] / im_size[1]
    line_num_coeff = [0.0] * 20
    line_num_coeff[:5] = [-0.005, -0.033, -1.042, 0.008, -0.001]
    line_den_coeff = [0.0] * 20
    line_den_coeff[:3] = [1.0, -0.001, -0.002]
    samp_num_coeff = [0.0] * 20
    samp_num_coeff[:5] = [0.008, 1.016, 0.002, 0.013, 0.001]
    samp_den_coeff = [0.0] * 20
    samp_den_coeff[:3] = [1.0, -0.002, -0.001]

    rpc = dict(
        height_off=_dem_offset,
        height_scale=_dem_offset,
        lat_off=xyz_[1][0],
        lat_scale=lat_scale,
        long_off=xyz_[0][0],
        long_scale=long_scale,
        line_off=(im_size[1] - 1) / 2,
        line_scale=im_size[1] / 2,
        samp_off=(im_size[0] - 1) / 2,
        samp_scale=im_size[0] / 2,
        line_num_coeff=line_num_coeff,
        line_den_coeff=line_den_coeff,
        samp_num_coeff=samp_num_coeff,
        samp_den_coeff=samp_den_coeff,
        err_bias=-1.0,  # unknown
        err_rand=-1.0,  # unknown
    )
    return rpc


@pytest.fixture(scope='session')
def interior_args(focal_len, im_size, sensor_size, cxy) -> dict:
    """A dictionary of interior parameters for ``FrameCamera.__init__()``."""
    return dict(
        im_size=im_size,
        focal_len=focal_len / sensor_size[0],
        sensor_size=(1, sensor_size[1] / sensor_size[0]),
        cx=cxy[0],
        cy=cxy[1],
    )


@pytest.fixture(scope='session')
def exterior_args(xyz: tuple, opk: tuple) -> dict:
    """A dictionary of exterior parameters for ``FrameCamera.__init__()`` /
    ``FrameCamera.update()``.
    """
    return dict(xyz=xyz, opk=opk)


@pytest.fixture(scope='session')
def frame_args(interior_args: dict, exterior_args: dict) -> dict:
    """A dictionary of interior and exterior parameters for ``FrameCamera.__init__()``."""
    return dict(**interior_args, **exterior_args)


@pytest.fixture(scope='session')
def rpc_args(im_size: tuple[int, int], rpc: dict) -> dict:
    """A dictionary of parameters for ``RpcCamera.__init__()``."""
    return dict(im_size=im_size, rpc=rpc)


@pytest.fixture(scope='session')
def brown_dist_param() -> dict:
    """Example ``BrownCamera`` distortion coefficients."""
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=-0.1)


@pytest.fixture(scope='session')
def opencv_dist_param() -> dict:
    """Example ``OpenCVCamera`` distortion coefficients."""
    return dict(k1=-0.25, k2=0.2, p1=0.01, p2=0.01, k3=-0.1, k4=0.001, k5=0.001, k6=-0.001)


@pytest.fixture(scope='session')
def fisheye_dist_param() -> dict:
    """Example ``FisheyeCamera`` distortion coefficients."""
    return dict(k1=-0.25, k2=0.1, k3=0.01, k4=-0.01)


@pytest.fixture(scope='session')
def pinhole_camera(frame_args: dict) -> FrameCamera:
    """Example ``PinholeCamera`` object with near-nadir orientation."""
    return PinholeCamera(**frame_args)


@pytest.fixture(scope='session')
def pinhole_camera_und(frame_args: dict) -> FrameCamera:
    """Example ``PinholeCamera`` object with near-nadir orientation and ``distort=False``."""
    # TODO: use parameterised fixtures for these and similar?
    #  https://docs.pytest.org/en/latest/example/parametrize.html#indirect-parametrization
    return PinholeCamera(**frame_args, distort=False)


@pytest.fixture(scope='session')
def brown_camera(frame_args: dict, brown_dist_param: dict) -> FrameCamera:
    """Example ``BrownCamera`` object with near-nadir orientation."""
    return BrownCamera(**frame_args, **brown_dist_param)


@pytest.fixture(scope='session')
def brown_camera_und(frame_args: dict, brown_dist_param: dict) -> FrameCamera:
    """Example ``BrownCamera`` object with near-nadir orientation and ``distort=False``."""
    return BrownCamera(**frame_args, **brown_dist_param, distort=False)


@pytest.fixture(scope='session')
def opencv_camera(frame_args: dict, opencv_dist_param: dict) -> FrameCamera:
    """Example ``OpenCVCamera`` object with near-nadir orientation."""
    return OpenCVCamera(**frame_args, **opencv_dist_param)


@pytest.fixture(scope='session')
def opencv_camera_und(frame_args: dict, opencv_dist_param: dict) -> FrameCamera:
    """Example ``OpenCVCamera`` object with near-nadir orientation and ``distort=False``."""
    return OpenCVCamera(**frame_args, **opencv_dist_param, distort=False)


@pytest.fixture(scope='session')
def fisheye_camera(frame_args: dict, fisheye_dist_param: dict) -> FrameCamera:
    """Example ``FisheyeCamera`` object with near-nadir orientation."""
    return FisheyeCamera(**frame_args, **fisheye_dist_param)


@pytest.fixture(scope='session')
def fisheye_camera_und(frame_args: dict, fisheye_dist_param: dict) -> FrameCamera:
    """Example ``FisheyeCamera`` object with near-nadir orientation and ``distort=False``."""
    return FisheyeCamera(**frame_args, **fisheye_dist_param, distort=False)


@pytest.fixture(scope='session')
def rpc_camera(rpc_args: dict) -> RpcCamera:
    """Example ``RpcCamera`` object with geographic world coordinates."""
    return RpcCamera(**rpc_args)


@pytest.fixture(scope='session')
def rpc_camera_proj(rpc_args: dict, utm34n_crs: str) -> RpcCamera:
    """Example ``RpcCamera`` object with projected world coordinates."""
    return RpcCamera(**rpc_args, crs=utm34n_crs)


@pytest.fixture(scope='session')
def xyz_grids(
    pinhole_camera: Camera,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], rio.Affine]:
    """(x, y, z) coordinate grids and transform at 5m resolution.  Z grid has 2 bands: band 1 is
    a sinusoidal surface, and band 2, a horizontal plane.
    """
    bounds = ortho_bounds(pinhole_camera)
    z, transform = create_zsurf(bounds, resolution=(5, 5))
    j, i = np.meshgrid(range(0, z.shape[2]), range(0, z.shape[1]), indexing='xy')
    x, y = (transform * rio.Affine.translation(0.5, 0.5)) * [j, i]
    return (x, y, z), transform


@pytest.fixture(scope='session')
def utm34n_crs() -> str:
    """CRS string for UTM zone 34N with no vertical CRS."""
    return 'EPSG:32634'


@pytest.fixture(scope='session')
def utm34n_wgs84_crs() -> str:
    """CRS string for UTM zone 34N with height above WGS84 ellipsoid vertical CRS."""
    return 'EPSG:32634+4326'


@pytest.fixture(scope='session')
def utm34n_egm96_crs() -> str:
    """CRS string for UTM zone 34N with height above EGM96 geoid vertical CRS."""
    return 'EPSG:32634+5773'


@pytest.fixture(scope='session')
def utm34n_egm2008_crs() -> str:
    """CRS string for UTM zone 34N with height above EGM2008 geoid vertical CRS."""
    return 'EPSG:32634+3855'


@pytest.fixture(scope='session')
def utm34n_msl_crs() -> str:
    """CRS string for UTM zone 34N with height above MSL (feet) vertical CRS."""
    return 'EPSG:32634+8050'


@pytest.fixture(scope='session')
def webmerc_crs() -> str:
    """CRS string for web mercator with no vertical CRS."""
    return '+proj=webmerc +datum=WGS84'


@pytest.fixture(scope='session')
def webmerc_wgs84_crs() -> str:
    """CRS string for web mercator with height above WGS84 ellipsoid vertical CRS."""
    return '+proj=webmerc +datum=WGS84 +ellps=WGS84 +vunits=m'


@pytest.fixture(scope='session')
def webmerc_egm96_crs() -> str:
    """CRS string for web mercator with height above EGM96 geoid vertical CRS."""
    return '+proj=webmerc +datum=WGS84 +geoidgrids=egm96_15.gtx +vunits=m'


@pytest.fixture(scope='session')
def webmerc_egm2008_crs() -> str:
    """CRS string for web mercator with height above EGM2008 geoid vertical CRS."""
    return '+proj=webmerc +datum=WGS84 +geoidgrids=egm08_25.gtx +vunits=m'


@pytest.fixture(scope='session')
def wgs84_crs() -> str:
    """CRS string for WGS84 with no vertical CRS."""
    return 'EPSG:4326'


@pytest.fixture(scope='session')
def wgs84_wgs84_crs() -> str:
    """CRS string for WGS84 with height above WGS84 ellipsoid vertical CRS."""
    return 'EPSG:4979'


@pytest.fixture(scope='session')
def wgs84_egm96_crs() -> str:
    """CRS string for WGS84 with height above EGM96 geoid vertical CRS."""
    return 'EPSG:4326+5773'


@pytest.fixture(scope='session')
def wgs84_egm2008_crs() -> str:
    """CRS string for WGS84 with height above EGM2008 geoid vertical CRS."""
    return 'EPSG:4326+3855'


@pytest.fixture(scope='session')
def rgb_byte_src_file(tmp_path_factory: pytest.TempPathFactory, im_size: tuple) -> Path:
    """An RGB byte checkerboard image with no CRS."""
    array = checkerboard(im_size[::-1]).astype('uint8')
    array = np.stack((array,) * 3, axis=0)
    profile = create_profile(array)
    src_filename = tmp_path_factory.mktemp('data').joinpath('rgb_byte_src.tif')
    with rio.open(src_filename, 'w', **profile) as im:
        im.write(array)
    return src_filename


@pytest.fixture(scope='session')
def float_src_file(tmp_path_factory: pytest.TempPathFactory, im_size: tuple) -> Path:
    """A single band float64 checkerboard image with no CRS."""
    array = np.expand_dims(checkerboard(im_size[::-1]).astype('float64'), axis=0)
    profile = create_profile(array)
    src_filename = tmp_path_factory.mktemp('data').joinpath('float_src.tif')
    with rio.open(src_filename, 'w', **profile) as im:
        im.write(array)
    return src_filename


@pytest.fixture(scope='session')
def rgb_byte_utm34n_src_file(
    tmp_path_factory: pytest.TempPathFactory, pinhole_camera: Camera, utm34n_crs: str
) -> Path:
    """An RGB byte checkerboard image with UTM zone 34N CRS and bounds 100m below
    ``pinhole_camera``.
    """
    array = checkerboard(pinhole_camera.im_size[::-1]).astype('uint8')
    array = np.stack((array,) * 3, axis=0)
    bounds = ortho_bounds(pinhole_camera)
    transform = from_bounds(*bounds, *pinhole_camera.im_size)
    profile = create_profile(array, crs=utm34n_crs, transform=transform, nodata=0)

    src_filename = tmp_path_factory.mktemp('data').joinpath('rgb_byte_utm34n_src.tif')
    with rio.open(src_filename, 'w', **profile) as im:
        im.write(array)
    return src_filename


@pytest.fixture(scope='session')
def float_utm34n_dem_file(
    tmp_path_factory: pytest.TempPathFactory, pinhole_camera: Camera, utm34n_crs: str
) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with no vertical CRS.

    Band 1 is a sinusoidal surface, and band 2, a horizontal plane.
    """
    bounds = ortho_bounds(pinhole_camera)
    array, transform = create_zsurf(bounds)
    profile = create_profile(array, transform=transform, crs=utm34n_crs, nodata=float('nan'))
    filename = tmp_path_factory.mktemp('data').joinpath('float_utm34n_dem.tif')
    with rio.open(filename, 'w', **profile) as im:
        im.write(array)
    return filename


@pytest.fixture(scope='session')
def float_utm34n_wgs84_dem_file(
    tmp_path_factory: pytest.TempPathFactory, pinhole_camera: Camera, utm34n_wgs84_crs: str
) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with height above WGS84 ellipsoid vertical CRS.

    Band 1 is a sinusoidal surface, and band 2, a horizontal plane.
    """
    bounds = ortho_bounds(pinhole_camera)
    array, transform = create_zsurf(bounds)
    profile = create_profile(array, transform=transform, crs=utm34n_wgs84_crs, nodata=float('nan'))

    filename = tmp_path_factory.mktemp('data').joinpath('float_utm34n_wgs84_dem.tif')
    with rio.open(filename, 'w', **profile) as im:
        im.write(array)
    return filename


@pytest.fixture(scope='session')
def float_utm34n_egm96_dem_file(
    tmp_path_factory: pytest.TempPathFactory, pinhole_camera: Camera, utm34n_egm96_crs: str
) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with height above EGM96 geoid vertical CRS.

    Band 1 is a sinusoidal surface, and band 2, a horizontal plane.
    """
    bounds = ortho_bounds(pinhole_camera)
    array, transform = create_zsurf(bounds)
    profile = create_profile(array, transform=transform, crs=utm34n_egm96_crs, nodata=float('nan'))

    filename = tmp_path_factory.mktemp('data').joinpath('float_utm34n_egm96_dem.tif')
    with rio.open(filename, 'w', **profile) as im:
        im.write(array)
    return filename


@pytest.fixture(scope='session')
def float_utm34n_egm2008_dem_file(
    tmp_path_factory: pytest.TempPathFactory, pinhole_camera: Camera, utm34n_egm2008_crs: str
) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with height above EGM2008 geoid vertical CRS.

    Band 1 is a sinusoidal surface, and band 2, a horizontal plane.
    """
    bounds = ortho_bounds(pinhole_camera)
    array, transform = create_zsurf(bounds)
    profile = create_profile(
        array, transform=transform, crs=utm34n_egm2008_crs, nodata=float('nan')
    )

    filename = tmp_path_factory.mktemp('data').joinpath('float_utm34n_egm96_dem.tif')
    with rio.open(filename, 'w', **profile) as im:
        im.write(array)
    return filename


@pytest.fixture(scope='session')
def float_wgs84_wgs84_dem_file(
    tmp_path_factory: pytest.TempPathFactory,
    pinhole_camera: Camera,
    utm34n_crs: str,
    wgs84_wgs84_crs: str,
) -> Path:
    """
    A 2 band float DEM file in WGS84 with height above WGS84 ellipsoid vertical CRS.

    Band 1 is a sinusoidal surface, and band 2, a horizontal plane.
    """
    bounds = ortho_bounds(pinhole_camera)
    array, transform = create_zsurf(bounds)
    bounds = array_bounds(*array.shape[-2:], transform)
    bounds = transform_bounds(utm34n_crs, wgs84_wgs84_crs, *bounds)
    transform = from_bounds(*bounds, *array.shape[1:][::-1])
    profile = create_profile(array, transform=transform, crs=wgs84_wgs84_crs, nodata=float('nan'))

    filename = tmp_path_factory.mktemp('data').joinpath('float_wgs84_wgs84_dem.tif')
    with rio.open(filename, 'w', **profile) as im:
        im.write(array)
    return filename


@pytest.fixture(scope='session')
def float_utm34n_msl_dem_file(
    tmp_path_factory: pytest.TempPathFactory,
    pinhole_camera: Camera,
    utm34n_msl_crs: str,
) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with height above MSL (feet) vertical CRS.

    Band 1 is a sinusoidal surface, and band 2, a horizontal plane.
    """
    bounds = ortho_bounds(pinhole_camera)
    array, transform = create_zsurf(bounds)
    array *= 3.28084  # meters to feet
    profile = create_profile(array, transform=transform, crs=utm34n_msl_crs, nodata=float('nan'))

    filename = tmp_path_factory.mktemp('data').joinpath('float_utm34n_msl_dem.tif')
    with rio.open(filename, 'w', **profile) as im:
        im.write(array)
    return filename


@pytest.fixture(scope='session')
def float_utm34n_partial_dem_file(
    tmp_path_factory: pytest.TempPathFactory, pinhole_camera: Camera, utm34n_crs: str
) -> Path:
    """
    A 2 band float DEM file in UTM zone 34N with no vertical CRS.

    Pixels above the diagonal are nodata. Band 1 is a sinusoidal surface, and band 2,
    a horizontal plane.
    """
    nodata = float('nan')
    bounds = ortho_bounds(pinhole_camera)
    array, transform = create_zsurf(bounds)
    mask = np.fliplr(np.tril(np.ones(array.shape, dtype='bool'), k=1))
    array[mask] = nodata
    profile = create_profile(array, transform=transform, crs=utm34n_crs, nodata=nodata)

    filename = tmp_path_factory.mktemp('data').joinpath('float_utm34n_dem.tif')
    with rio.open(filename, 'w', **profile) as im:
        im.write(array)
    return filename


@pytest.fixture(scope='session')
def rgb_pinhole_utm34n_ortho(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_crs: str,
) -> Ortho:
    """An Ortho object initialised with RGB byte source image, float DEM in UTM zone 34N (no
    vertical CRS), pinhole camera, and UTM zone 34N CRS (no vertical CRS).
    """
    return Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs)


@pytest.fixture(scope='session')
def github_root_url() -> str:
    """URL of github repository root."""
    # TODO: change to /main
    return r'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/feature_docs/'


@pytest.fixture(scope='session')
def odm_dataset_dir() -> Path:
    """ODM dataset directory."""
    return root_path.joinpath('tests', 'data', 'odm')


@pytest.fixture(scope='session')
def odm_image_files(odm_dataset_dir: Path) -> tuple[Path, ...]:
    """ODM drone image files."""
    return tuple([fn for fn in odm_dataset_dir.joinpath('images').glob('*.tif')])


@pytest.fixture(scope='session')
def odm_image_file(odm_dataset_dir: Path) -> Path:
    """ODM drone image file."""
    return next(iter(odm_dataset_dir.joinpath('images').glob('*.tif')))


@pytest.fixture(scope='session')
def odm_image_url(github_root_url: str, odm_image_file: Path) -> str:
    """ODM drone image URL."""
    return github_root_url + odm_image_file.relative_to(root_path).as_posix()


@pytest.fixture(scope='session')
def odm_dem_file(odm_dataset_dir: Path) -> Path:
    """ODM DEM file."""
    return odm_dataset_dir.joinpath('odm_dem', 'dsm.tif')


@pytest.fixture(scope='session')
def odm_reconstruction_file(odm_dataset_dir: Path) -> Path:
    """ODM reconstruction file."""
    return odm_dataset_dir.joinpath('opensfm', 'reconstruction.json')


@pytest.fixture(scope='session')
def odm_reconstruction_url(github_root_url: str, odm_reconstruction_file: Path) -> str:
    """ODM reconstruction URL."""
    return github_root_url + odm_reconstruction_file.relative_to(root_path).as_posix()


@pytest.fixture(scope='session')
def odm_crs(odm_dem_file) -> str:
    """CRS string for ODM exterior parameters & orthos in EPSG format."""
    with rio.open(odm_dem_file, 'r') as im:
        crs = im.crs
    return f'EPSG:{crs.to_epsg()}'


@pytest.fixture(scope='session')
def ngi_image_files() -> tuple[Path, ...]:
    """NGI image files."""
    return tuple([fn for fn in root_path.joinpath('tests/data/ngi').glob('*RGB.tif')])


@pytest.fixture(scope='session')
def ngi_image_file() -> Path:
    """NGI aerial image file."""
    return next(iter(root_path.joinpath('tests/data/ngi').glob('*RGB.tif')))


@pytest.fixture(scope='session')
def ngi_image_url(github_root_url: str, ngi_image_file: Path) -> str:
    """NGI aerial image URL."""
    return github_root_url + ngi_image_file.relative_to(root_path).as_posix()


@pytest.fixture(scope='session')
def ngi_dem_file() -> Path:
    """NGI DEM file."""
    return root_path.joinpath('tests/data/ngi/dem.tif')


@pytest.fixture(scope='session')
def ngi_dem_url(github_root_url: str, ngi_dem_file: Path) -> str:
    """NGI DEM URL."""
    return github_root_url + ngi_dem_file.relative_to(root_path).as_posix()


@pytest.fixture(scope='session')
def ngi_crs(ngi_image_file) -> str:
    """CRS string for NGI exterior parameters & orthos in proj4 format."""
    with rio.open(ngi_image_file, 'r') as im:
        crs = im.crs
    return crs.to_proj4()


@pytest.fixture(scope='session')
def ngi_legacy_config_file() -> Path:
    """Legacy format configuration file for NGI test data."""
    return root_path.joinpath('tests/data/ngi/config.yaml')


@pytest.fixture(scope='session')
def ngi_oty_int_param_file() -> Path:
    """Orthority format interior parameter file for NGI test data."""
    return root_path.joinpath('tests/data/io/ngi_int_param.yaml')


@pytest.fixture(scope='session')
def ngi_oty_int_param_url(github_root_url: str, ngi_oty_int_param_file: Path) -> str:
    """Orthority format interior parameter URL for NGI test data."""
    return github_root_url + ngi_oty_int_param_file.relative_to(root_path).as_posix()


@pytest.fixture(scope='session')
def ngi_legacy_csv_file() -> Path:
    """Legacy format exterior parameter CSV file for NGI test data."""
    return root_path.joinpath('tests/data/ngi/camera_pos_ori.txt')


@pytest.fixture(scope='session')
def ngi_oty_ext_param_file() -> Path:
    """Orthority format exterior parameter file for NGI test data."""
    return root_path.joinpath('tests/data/io/ngi_ext_param.geojson')


@pytest.fixture(scope='session')
def ngi_oty_ext_param_url(github_root_url: str, ngi_oty_ext_param_file: Path) -> str:
    """Orthority format exterior parameter URL for NGI test data."""
    return github_root_url + ngi_oty_ext_param_file.relative_to(root_path).as_posix()


@pytest.fixture(scope='session')
def ngi_xyz_opk_csv_file() -> Path:
    """
    Exterior parameter file path for NGI data in (x, y, z), (omega, phi, kappa) CSV format.

    Includes a header and .proj file.
    """
    return root_path.joinpath('tests/data/io/ngi_xyz_opk.csv')


@pytest.fixture(scope='session')
def ngi_xyz_opk_csv_url(github_root_url: str, ngi_xyz_opk_csv_file: Path) -> str:
    """
    Exterior parameter file URL for NGI data in (x, y, z), (omega, phi, kappa) CSV format.

    Includes a header and .proj file.
    """
    return github_root_url + ngi_xyz_opk_csv_file.relative_to(root_path).as_posix()


@pytest.fixture(scope='session')
def ngi_xyz_opk_radians_csv_file(
    ngi_xyz_opk_csv_file: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """
    Exterior parameters for NGI data in (x, y, z), (omega, phi, kappa) CSV format.

    Includes a header and .proj file.  Angles in radians.
    """
    filename = tmp_path_factory.mktemp('data').joinpath('ngi_xyz_opk_radians.csv')
    shutil.copy(ngi_xyz_opk_csv_file.with_suffix('.prj'), filename.with_suffix('.prj'))

    dialect = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    with open(ngi_xyz_opk_csv_file, 'r', newline=None) as rf, open(filename, 'w', newline='') as wf:
        reader = csv.reader(rf, **dialect)
        writer = csv.writer(wf, **dialect)
        writer.writerow(next(iter(reader)))  # write header
        for row in reader:
            row[4:7] = np.radians(np.float64(row[4:7])).tolist()
            writer.writerow(row)
    return filename


@pytest.fixture(scope='session')
def odm_lla_rpy_csv_file() -> Path:
    """
    Exterior parameters for ODM data in (latitude, longitude, altitude), (roll, pitch, yaw) CSV
    format.

    Includes a header.
    """
    return root_path.joinpath('tests/data/io/odm_lla_rpy.csv')


@pytest.fixture(scope='session')
def odm_xyz_opk_csv_file() -> Path:
    """
    Exterior parameters for ODM data in (x, y, z), (omega, phi, kappa) CSV format.

    Includes a header.
    """
    return root_path.joinpath('tests/data/io/odm_xyz_opk.csv')


@pytest.fixture(scope='session')
def rpc_image_file() -> Path:
    """Quickbird2 image file with RPC metadata."""
    return root_path.joinpath('tests/data/rpc/qb2_basic1b.tif')


@pytest.fixture(scope='session')
def rpc_param_file() -> Path:
    """Orthority RPC parameter file for the Quickbird2 image."""
    return root_path.joinpath('tests/data/rpc/rpc_param.yaml')


@pytest.fixture(scope='session')
def pinhole_int_param_dict(interior_args: dict) -> dict:
    """A pinhole camera interior parameter dictionary."""
    return {'pinhole test camera': dict(cam_type=CameraType.pinhole, **interior_args)}


@pytest.fixture(scope='session')
def opencv_int_param_dict(interior_args: dict, opencv_dist_param: dict) -> dict:
    """An opencv camera interior parameter dictionary."""
    return {
        'cv test camera': dict(cam_type=CameraType.opencv, **interior_args, **opencv_dist_param)
    }


@pytest.fixture(scope='session')
def brown_int_param_dict(interior_args: dict, brown_dist_param: dict) -> dict:
    """A brown camera interior parameter dictionary."""
    return {
        'brown test camera': dict(cam_type=CameraType.brown, **interior_args, **brown_dist_param)
    }


@pytest.fixture(scope='session')
def fisheye_int_param_dict(interior_args: dict, fisheye_dist_param: dict) -> dict:
    """A fisheye camera interior parameter dictionary."""
    return {
        'fisheye test camera': dict(
            cam_type=CameraType.opencv, **interior_args, **fisheye_dist_param
        )
    }


@pytest.fixture(scope='session')
def mult_int_param_dict(
    pinhole_int_param_dict: dict,
    brown_int_param_dict: dict,
    opencv_int_param_dict: dict,
    fisheye_int_param_dict: dict,
) -> dict:
    """An interior parameter dictionary consisting of multiple cameras."""
    return dict(
        **pinhole_int_param_dict,
        **brown_int_param_dict,
        **opencv_int_param_dict,
        **fisheye_int_param_dict,
    )


@pytest.fixture(scope='session')
def mult_ext_param_dict(xyz: tuple, opk: tuple, mult_int_param_dict: dict):
    """An exterior parameter dictionary referencing multiple cameras."""
    ext_param_dict = {}
    for i, cam_id in enumerate(mult_int_param_dict.keys()):
        ext_param_dict[f'src_image_{i}'] = dict(xyz=xyz, opk=opk, camera=cam_id)
    return ext_param_dict


@pytest.fixture(scope='session')
def odm_int_param_file(tmp_path_factory: pytest.TempPathFactory, mult_int_param_dict: dict) -> Path:
    """An interior parameter file in ODM cameras.json format."""
    filename = tmp_path_factory.mktemp('data').joinpath('odm_int_param_file.json')
    int_param = oty_to_osfm_int_param(mult_int_param_dict)
    with open(filename, 'w') as f:
        json.dump(int_param, f)
    return filename


@pytest.fixture(scope='session')
def osfm_int_param_file(
    tmp_path_factory: pytest.TempPathFactory, mult_int_param_dict: dict
) -> Path:
    """An interior parameter file in OpenSfM reconstruction.json format."""
    filename = tmp_path_factory.mktemp('data').joinpath('osfm_int_param_file.json')
    int_param = oty_to_osfm_int_param(mult_int_param_dict)
    int_param = [dict(cameras=int_param)]
    with open(filename, 'w') as f:
        json.dump(int_param, f)
    return filename


@pytest.fixture(scope='session')
def exif_image_file(odm_image_file: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An image file with EXIF tags including sensor size, and no XMP tags."""
    dst_filename = tmp_path_factory.mktemp('data').joinpath('exif.tif')
    with rio.open(odm_image_file, 'r') as src_im:
        dst_profile = src_im.profile.copy()
        with rio.open(dst_filename, 'w', **dst_profile) as dst_im:
            dst_tags = src_im.tags()
            dst_tags.update(
                EXIF_FocalPlaneResolutionUnit='4',
                EXIF_FocalPlaneXResolution=f'({dst_profile["width"] / 13.2:.4f})',
                EXIF_FocalPlaneYResolution=f'({dst_profile["height"] / 8.8:.4f})',
            )
            dst_im.update_tags(**dst_tags)
            dst_im.write(src_im.read())
    return dst_filename


@pytest.fixture(scope='session')
def exif_no_focal_image_file(
    exif_image_file: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """An image file with EXIF tags including sensor size, but without focal length and XMP tags."""
    dst_filename = tmp_path_factory.mktemp('data').joinpath('exif.tif')
    with rio.open(exif_image_file, 'r') as src_im:
        dst_profile = src_im.profile.copy()
        with rio.open(dst_filename, 'w', **dst_profile) as dst_im:
            dst_tags = src_im.tags()
            dst_tags.pop('EXIF_FocalLength')
            dst_im.update_tags(**dst_tags)
            dst_im.write(src_im.read())
    return dst_filename


@pytest.fixture(scope='session')
def xmp_no_dewarp_image_file(
    odm_image_file: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """An image file with EXIF & XMP tags, excluding EXIF sensor size and XMP DewarpData tags."""
    dst_filename = tmp_path_factory.mktemp('data').joinpath('exif.tif')
    with rio.open(odm_image_file, 'r') as src_im:
        dst_profile = src_im.profile.copy()
        with rio.open(dst_filename, 'w', **dst_profile) as dst_im:
            dst_im.update_tags(**src_im.tags())
            for namespace in src_im.tag_namespaces():
                # note there is an apparent rio/gdal bug with ':' in the 'xml:XMP' namspace/ tag
                # name, where 'xml:XMP=' gets prefixed to the value
                ns_dict = src_im.tags(ns=namespace)
                if namespace == 'xml:XMP':
                    ns_dict[namespace] = re.sub(
                        r'[ ]*?drone-dji:DewarpData(.*?)"\n', '', ns_dict[namespace]
                    )
                dst_im.update_tags(ns=namespace, **ns_dict)
            dst_im.write(src_im.read())
    return dst_filename
