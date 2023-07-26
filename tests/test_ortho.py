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
import logging
from typing import Tuple, Dict
from pathlib import Path

import pytest
import numpy as np
import rasterio as rio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.features import shapes
from rasterio.windows import from_bounds

from simple_ortho.camera import Camera, PinholeCamera, BrownCamera, OpenCVCamera, FisheyeCamera, create_camera
from simple_ortho.enums import CameraType, Interp, Compress
from simple_ortho.utils import distort_image, nan_equals
from simple_ortho.ortho import Ortho

logging.basicConfig(level=logging.DEBUG)


def test_init(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_pinhole_camera: Camera, ortho_crs_no_vdatum: str
):
    """ Test Ortho initialisation with specified ortho CRS. """
    # TODO: create a default ortho fixture?
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
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_camera_args: Dict, ortho_crs_no_vdatum: str
):
    """ Test Ortho initialisation without DEM coverage raises an error. """
    # create a camera positioned away from DEM bounds
    camera = PinholeCamera(**nadir_camera_args)
    camera.update_extrinsic((0, 0, 0), (0, 0, 0))

    with pytest.raises(ValueError) as ex:
        _ = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, camera, crs=ortho_crs_no_vdatum)
    assert 'bounds' in str(ex)


def test_init_horizon_fov_error(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_camera_args: Dict, ortho_crs_no_vdatum: str
):
    """ Test Ortho initialisation with a horizontal FOV camera raises an error. """
    # create a camera pointing away from DEM bounds
    camera = PinholeCamera(**nadir_camera_args)
    camera.update_extrinsic((0, 0, 0), (np.pi/2, 0, 0))

    with pytest.raises(ValueError) as ex:
        _ = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, camera, crs=ortho_crs_no_vdatum)
    assert 'horizon' in str(ex)


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

    # find initial dem bounds
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
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_pinhole_camera: Camera, ortho_crs_no_vdatum: str
):
    """ Test DEM is not reprojected when it's CRS & resolution are the same as the ortho CRS & resolution. """
    ortho = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, nadir_pinhole_camera, crs=ortho_crs_no_vdatum)

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
        ('nadir_dem_30m_float_no_vdatum', 'ortho_crs_egm96_vdatum'),
        ('nadir_dem_30m_float_egm96_vdatum', 'ortho_crs_no_vdatum'),
    ],
)
def test_reproject_dem_vdatum_both(
    src_file_rgb_byte: Path, nadir_dem_30m_float_wgs84_vdatum: Path, nadir_pinhole_camera: Camera, dem_file: str,
    crs: str, request: pytest.FixtureRequest
):
    """ Test DEM reprojection altitude adjustment when both DEM and ortho vertical datums are specified. """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)

    ortho = Ortho(src_file_rgb_byte, dem_file, nadir_pinhole_camera, crs=crs)
    with rio.open(dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic_spline, resolution)

    assert not ortho._crs_equal
    assert transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert array.shape == ortho._dem_array.shape

    mask = ~np.isnan(array) & ~np.isnan(ortho._dem_array)
    assert array[mask] != pytest.approx(ortho._dem_array[mask], abs=2)


@pytest.mark.parametrize(
    'dem_file, crs', [
        ('nadir_dem_30m_float_no_vdatum', 'ortho_crs_egm96_vdatum'),
        ('nadir_dem_30m_float_egm96_vdatum', 'ortho_crs_no_vdatum'),
    ],
)
def test_reproject_dem_vdatum_one(
    src_file_rgb_byte: Path, nadir_dem_30m_float_wgs84_vdatum: Path, nadir_pinhole_camera: Camera, dem_file: str,
    crs: str, request: pytest.FixtureRequest
):
    """ Test DEM reprojection does no altitude adjustment when one of DEM and ortho vertical datums are specified. """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)

    ortho = Ortho(src_file_rgb_byte, dem_file, nadir_pinhole_camera, crs=crs)
    with rio.open(dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic_spline, resolution)

    assert not ortho._crs_equal
    assert transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert np.nanmean(array) == pytest.approx(np.nanmean(ortho._dem_array), abs=2)


@pytest.mark.parametrize(
    '_position, _rotation', [
        # varying rotations starting at `rotation` fixture value and keeping FOV below horizon
        ((2e4, 3e4, 1e3), (-3., 2., 10.)),
        ((2e4, 3e4, 1e3), (-18., 12., 10.)),
        ((2e4, 3e4, 1e3), (-33., 22., 10.)),
        ((2e4, 3e4, 1e3), (-48., 22., 10.)),
        # varying positions with partial dem coverage
        ((2e4, 3.05e4, 1e3), (-3., 2., 10.)),
        ((2e4, 3e4, 2e3), (-3., 2., 10.)),
        ((2e4, 3e4, 4e3), (-3., 2., 10.)),
    ],
)  # yapf: disable
def test_mask_dem(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_camera_args: Dict, ortho_crs_no_vdatum: str,
    _position: Tuple, _rotation: Tuple, tmp_path: Path
):
    """ Test the DEM (ortho boundary) mask contains the ortho valid data mask (without cropping). """
    # note that these tests should use the pinhole camera model to ensure no artefacts outside the ortho boundary, and
    #  DEM < camera height to ensure no ortho artefacts in DEM > camera height areas.
    camera: Camera = PinholeCamera(
        _position, np.radians(_rotation), nadir_camera_args['focal_len'], nadir_camera_args['im_size'],
        nadir_camera_args['sensor_size']
    )
    resolution = (5, 5)
    num_pts = 400

    # create an ortho image without DEM masking
    ortho = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, camera, crs=ortho_crs_no_vdatum)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic_spline, resolution)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    with rio.open(src_file_rgb_byte, 'r') as src_im:
        ortho_profile = ortho._create_ortho_profile(
            src_im, dem_array.shape, dem_transform, dtype='uint8', compress=Compress.deflate
        )
        with rio.open(ortho_file, 'w', **ortho_profile) as ortho_im:
            ortho._remap(src_im, ortho_im, dem_array, full_remap=True, interp=Interp.nearest, write_mask=False)

    # create the dem mask
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(), dem_transform, full_remap=True, crop=False, mask=True, num_pts=num_pts
    )
    dem_mask = ~np.isnan(dem_array_mask)

    # read the ortho nodata mask
    assert ortho_file.exists()
    with rio.open(ortho_file, 'r') as ortho_im:
        ortho_mask = ortho_im.dataset_mask().astype('bool')

    # test dem mask validity
    assert dem_transform_mask == dem_transform
    assert dem_mask.shape == ortho_mask.shape
    assert dem_mask[ortho_mask].sum() / ortho_mask.sum() > 0.99

    if False:
        # debug plotting code
        # %matplotlib
        from matplotlib import pyplot
        from rasterio.plot import show

        def plot_poly(mask: np.ndarray, transform=dem_transform_crop, ico='k'):
            """ Plot polygons from mask. """
            poly_list = [poly for poly, _ in shapes(mask.astype('uint8'), transform=transform)]

            for poly in poly_list[:-1]:
                coords = np.array(poly['coordinates'][0]).T
                pyplot.plot(coords[0], coords[1], ico)

        with rio.open(ortho_file, 'r') as ortho_im:
            ortho_array = ortho_im.read()

        for image in (ortho_array, dem_array):
            pyplot.figure()
            show(image, transform=dem_transform, cmap='gray')
            plot_poly(ortho_mask, transform=dem_transform, ico='y--')
            plot_poly(dem_mask, transform=dem_transform, ico='r:')


@pytest.mark.parametrize(
    '_position, _rotation', [
        # varying rotations starting at `rotation` fixture value and keeping FOV below horizon
        ((2e4, 3e4, 1e3), (-3., 2., 10.)),
        ((2e4, 3e4, 1e3), (-18., 12., 10.)),
        ((2e4, 3e4, 1e3), (-33., 22., 10.)),
        ((2e4, 3e4, 1e3), (-48., 22., 10.)),
        # varying positions with partial dem coverage
        ((2e4, 3.05e4, 1e3), (-3., 2., 10.)),
        ((2e4, 3e4, 2e3), (-3., 2., 10.)),
        ((2e4, 3e4, 4e3), (-3., 2., 10.)),
    ],
)  # yapf: disable
def test_mask_dem_crop(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_camera_args: Dict, ortho_crs_no_vdatum: str,
    _position: Tuple, _rotation: Tuple, tmp_path: Path
):
    """ Test the DEM mask is cropped to ortho boundaries. """
    camera: Camera = PinholeCamera(
        _position, np.radians(_rotation), nadir_camera_args['focal_len'], nadir_camera_args['im_size'],
        nadir_camera_args['sensor_size']
    )
    resolution = (5, 5)
    num_pts = 400

    # mask the dem without cropping
    ortho = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, camera, crs=ortho_crs_no_vdatum)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic_spline, resolution)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(), dem_transform, full_remap=True, crop=False, mask=True, num_pts=num_pts
    )
    mask = ~np.isnan(dem_array_mask)

    # crop & mask the dem
    dem_array_crop, dem_transform_crop = ortho._mask_dem(
        dem_array.copy(), dem_transform, full_remap=True, crop=True, mask=True, num_pts=num_pts
    )
    mask_crop = ~np.isnan(dem_array_crop)

    # find the window of mask_crop in mask
    bounds_crop = array_bounds(*dem_array_crop.shape, dem_transform_crop)
    win_crop = from_bounds(*bounds_crop, dem_transform_mask)

    # test windowed portion of mask is identical to mask_crop, and unwindowed portion contains no masked pixels
    assert np.all(mask_crop == mask[win_crop.toslices()])
    assert mask.sum() == mask[win_crop.toslices()].sum()


def test_mask_dem_coverage_error(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_camera_args: Dict, ortho_crs_no_vdatum: str
):
    """ Test _mask_dem() without DEM coverage raises an error. """
    camera: Camera = PinholeCamera(**nadir_camera_args)

    # init & reproject with coverage
    ortho = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, camera, crs=ortho_crs_no_vdatum)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic_spline, (30., 30.))

    # update camera for no coverage
    camera.update_extrinsic((0., 0., 1000.), (0., 0., 0.))

    # test
    with pytest.raises(ValueError) as ex:
        ortho._mask_dem(dem_array, dem_transform, full_remap=True)
    assert 'bounds' in str(ex)


def test_mask_dem_above_camera_error(
    src_file_rgb_byte: Path, nadir_dem_30m_float_no_vdatum: Path, nadir_camera_args: Dict, ortho_crs_no_vdatum: str
):
    """ Test _mask_dem() raises an error when the DEM is higher the camera. """
    camera: Camera = PinholeCamera(**nadir_camera_args)

    # move the camera below the DEM
    _position = list(nadir_camera_args['position'])
    _position[2] -= 1000
    camera.update_extrinsic(_position, nadir_camera_args['rotation'])

    # init & reproject
    ortho = Ortho(src_file_rgb_byte, nadir_dem_30m_float_no_vdatum, camera, crs=ortho_crs_no_vdatum)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic_spline, (30., 30.))

    # test
    with pytest.raises(ValueError) as ex:
        ortho._mask_dem(dem_array, dem_transform, full_remap=True)
    assert 'higher' in str(ex)


@pytest.mark.parametrize(
    'camera, src_file, dem_file', [
        ('nadir_pinhole_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        ('oblique_pinhole_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
        ('nadir_opencv_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        ('oblique_opencv_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
        ('nadir_brown_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        ('oblique_brown_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
        ('nadir_fisheye_camera', 'src_file_rgb_byte', 'nadir_dem_30m_float_no_vdatum'),
        # ('oblique_fisheye_camera', 'src_file_rgb_byte', 'oblique_dem_30m_float_no_vdatum'),
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

