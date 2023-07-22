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
import copy
from typing import Tuple, Dict
from pathlib import Path

import pytest
import numpy as np
import rasterio as rio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

from simple_ortho.camera import Camera, PinholeCamera, BrownCamera, OpenCVCamera, FisheyeCamera, create_camera
from simple_ortho.enums import CameraType, Interp
from simple_ortho.utils import distort_image, nan_equals
from simple_ortho.ortho import Ortho


def test_init(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_pinhole_camera: Camera, ortho_crs_no_vdatum: str
):
    """ Test Ortho initialisation with specified ortho CRS. """
    ortho = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, nadir_pinhole_camera, crs=ortho_crs_no_vdatum)
    with rio.open(nadir_dem_30m_float_no_vdatum, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._ortho_crs == rio.CRS.from_string(ortho_crs_no_vdatum)
    assert ortho._dem_crs == dem_crs
    assert ortho._crs_equal == (ortho._ortho_crs == dem_crs)
    assert ortho._dem_array is not None
    assert ortho._dem_transform is not None


def test_init_src_crs(src_file_rgb_byte_crs: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_pinhole_camera: Camera):
    """ Test Ortho initialisation with CRS from source file. """
    ortho = Ortho(src_file_rgb_byte_crs, nadir_dem_30m_float_no_vdatum, nadir_pinhole_camera, crs=None)
    with rio.open(src_file_rgb_byte_crs, 'r') as src_im:
        src_crs = src_im.crs
    with rio.open(nadir_dem_30m_float_no_vdatum, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._ortho_crs == src_crs
    assert ortho._dem_crs == dem_crs
    assert ortho._crs_equal == (ortho._ortho_crs == dem_crs)
    assert ortho._dem_array is not None
    assert ortho._dem_transform is not None


def test_init_nocrs_error(src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_pinhole_camera: Camera):
    """ Test Ortho initialisation without a CRS raises an error. """
    with pytest.raises(ValueError) as ex:
        _ = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, nadir_pinhole_camera, crs=None)
    assert 'crs' in str(ex)


def test_init_geogcrs_error(src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_pinhole_camera: Camera):
    """ Test Ortho initialisation with a geographic CRS raises an error. """
    with pytest.raises(ValueError) as ex:
        _ = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, nadir_pinhole_camera, crs='EPSG:4326')
    assert 'geographic' in str(ex)


def test_init_dem_coverage_error(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_camera_args: Tuple,
    ortho_crs_no_vdatum: str
):
    """ Test Ortho initialisation without DEM coverage raises an error. """
    # create a camera pointing away from DEM bounds
    camera_args = list(nadir_camera_args)
    camera_args[0] = (0, 0, 1000)
    camera = PinholeCamera(*camera_args)

    with pytest.raises(ValueError) as ex:
        _ = Ortho(
            src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, camera, crs=ortho_crs_no_vdatum
        )
    assert 'bounds' in str(ex)


@pytest.mark.parametrize(
    'interp, resolution', [*zip(Interp, [(10, 10)] * len(Interp)), *zip(Interp, [(50, 50)] * len(Interp))],
)
def test_reproject_dem(
    src_file_rgb_byte: Path, nadir_dem_30m_float_wgs84_wgs84_vdatum: Path, nadir_pinhole_camera: Camera,
    ortho_crs_no_vdatum: str, interp: Interp, resolution: Tuple
):
    """ Test DEM is reprojected when it's CRS / resolution is different to the ortho CRS / resolution. """
    ortho = Ortho(
        src_file_rgb_byte, nadir_dem_30m_float_wgs84_wgs84_vdatum, nadir_pinhole_camera, crs=ortho_crs_no_vdatum
    )
    # find bounds of initial dem
    with rio.open(nadir_dem_30m_float_wgs84_wgs84_vdatum, 'r') as dem_im:
        init_crs = dem_im.crs
    init_bounds = array_bounds(*ortho._dem_array.shape, ortho._dem_transform)
    init_bounds = np.array(transform_bounds(init_crs, ortho._ortho_crs, *init_bounds))

    # reproject
    array, transform = ortho._reproject_dem(interp, resolution)
    bounds = np.array(array_bounds(*array.shape, transform))

    # test validity
    assert transform != ortho._dem_transform
    assert array != ortho._dem_array
    assert np.all(np.abs((transform[0], transform[4])) == resolution)
    assert bounds == pytest.approx(init_bounds, abs=max(resolution))
    assert np.nanmean(array) == pytest.approx(np.nanmean(ortho._dem_array), abs=1)


def test_reproject_dem_crs_equal(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_pinhole_camera: Camera,
    ortho_crs_no_vdatum: str
):
    """ Test DEM is not reprojected when it's CRS & resolution are the same as the ortho CRS & resolution. """
    ortho = Ortho(
        src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, nadir_pinhole_camera, crs=ortho_crs_no_vdatum
    )
    with rio.open(nadir_dem_30m_float_no_vdatum, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic_spline, resolution)

    assert transform == ortho._dem_transform
    assert np.all(nan_equals(array, ortho._dem_array))


@pytest.mark.parametrize(
    'dem_file, crs', [
        ('nadir_dem_30m_float_wgs84_vdatum', 'ortho_crs_egm96_vdatum'),
        ('nadir_dem_30m_float_wgs84_vdatum', 'ortho_crs_egm2008_vdatum'),
        ('nadir_dem_30m_float_egm96_vdatum', 'ortho_crs_wgs84_vdatum'),
        ('nadir_dem_30m_float_egm96_vdatum', 'ortho_crs_egm2008_vdatum'),
    ],
)
def test_reproject_dem_vdatum(
    src_file_rgb_byte: Path, nadir_dem_30m_float_wgs84_vdatum: Path, nadir_pinhole_camera: Camera,
    dem_file: str, crs: str, request: pytest.FixtureRequest
):
    """ Test DEM reprojection altitude adjustment when DEM and ortho vertical datums are specified. """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)
    ortho = Ortho(src_file_rgb_byte, dem_file, nadir_pinhole_camera, crs=crs)
    with rio.open(dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic_spline, resolution)

    assert transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert array.shape == ortho._dem_array.shape

    mask = ~np.isnan(array) & ~np.isnan(ortho._dem_array)
    assert array[mask] != pytest.approx(ortho._dem_array[mask], abs=2)


@pytest.mark.parametrize(
    'camera, src_file, dem_file', [
        ('nadir_pinhole_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        ('oblique_pinhole_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
        ('nadir_opencv_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        ('oblique_opencv_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
        ('nadir_brown_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        ('oblique_brown_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
        ('nadir_fisheye_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        ('oblique_fisheye_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
    ],
)
def test_process(
    camera: str, src_file: str, dem_file: str, ortho_crs_no_vdatum, tmp_path: Path, request: pytest.FixtureRequest
):
    camera: Camera = request.getfixturevalue(camera)
    src_file: Path = request.getfixturevalue(src_file)
    dem_file: Path = request.getfixturevalue(dem_file)

    ortho = Ortho(src_file, dem_file, camera, ortho_crs_no_vdatum, dem_band=1)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, (2, 2), write_mask=True)

    assert ortho_file.exists()
