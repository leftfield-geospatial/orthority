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
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pytest
import cv2
import rasterio as rio
from rasterio.enums import MaskFlags
from rasterio.features import shapes
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from simple_ortho.camera import Camera, PinholeCamera, create_camera
from simple_ortho.enums import Interp, Compress, CameraType
from simple_ortho.ortho import Ortho
from simple_ortho.utils import nan_equals, distort_image
from simple_ortho import errors
from tests.conftest import _dem_resolution, checkerboard

logging.basicConfig(level=logging.DEBUG)


def test_init(rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str):
    """ Test Ortho initialisation with specified ortho CRS. """
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs)
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._crs == rio.CRS.from_string(utm34n_crs)
    assert ortho._dem_crs == dem_crs
    assert ortho._crs_equal == (ortho._crs == dem_crs)
    assert ortho._dem_array is not None
    assert ortho._dem_transform is not None


def test_init_src_crs(rgb_byte_utm34n_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera):
    """ Test Ortho initialisation with CRS from source file. """
    ortho = Ortho(rgb_byte_utm34n_src_file, float_utm34n_dem_file, pinhole_camera, crs=None)
    with rio.open(rgb_byte_utm34n_src_file, 'r') as src_im:
        src_crs = src_im.crs
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._crs == src_crs
    assert ortho._dem_crs == dem_crs
    assert ortho._crs_equal == (ortho._crs == dem_crs)
    assert ortho._dem_array is not None
    assert ortho._dem_transform is not None


@pytest.mark.parametrize('dem_band', [1, 2])
def test_init_dem_band(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str, dem_band: int
):
    """ Test Ortho initialisation with `dem_band` reads the correct DEM band. """
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs, dem_band=dem_band)
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_bounds = array_bounds(*ortho._dem_array.shape, ortho._dem_transform)
        dem_win = dem_im.window(*dem_bounds)
        dem_array = dem_im.read(indexes=dem_band, window=dem_win).astype('float32')
    assert np.all(ortho._dem_array == dem_array)


def test_init_dem_band_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str
):
    """ Test Ortho initialisation with incorrect `dem_band` raises an error. """
    with pytest.raises(errors.DemBandError) as ex:
        Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs, dem_band=3)
    assert 'dem_band' in str(ex)


def test_init_nocrs_error(rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera):
    """ Test Ortho initialisation without a CRS raises an error. """
    with pytest.raises(errors.CrsMissingError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=None)
    assert 'crs' in str(ex)


def test_init_geogcrs_error(rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera):
    """ Test Ortho initialisation with a geographic CRS raises an error. """
    with pytest.raises(errors.CrsError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs='EPSG:4326')
    assert 'geographic' in str(ex)


def test_init_dem_coverage_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test Ortho initialisation without DEM coverage of ortho bounds raises an error. """
    # create a camera positioned away from dem bounds
    camera = PinholeCamera(**camera_args)
    camera.update((0, 0, 0), (0, 0, 0))

    with pytest.raises(ValueError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    assert 'bounds' in str(ex)


def test_init_horizon_fov_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test Ortho initialisation with a horizontal FOV camera raises an error. """
    # create a camera pointing away from dem bounds
    camera = PinholeCamera(**camera_args)
    camera.update((0, 0, 0), (np.pi / 2, 0, 0))

    with pytest.raises(ValueError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    assert 'horizon' in str(ex)


@pytest.mark.parametrize(
    'interp, resolution', [*zip(Interp, [(10, 10)] * len(Interp)), *zip(Interp, [(50, 50)] * len(Interp))],
)
def test_reproject_dem(
    rgb_byte_src_file: Path, float_wgs84_wgs84_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str, interp: Interp,
    resolution: Tuple
):
    """ Test DEM is reprojected when it's CRS / resolution is different to the ortho CRS / resolution. """
    ortho = Ortho(rgb_byte_src_file, float_wgs84_wgs84_dem_file, pinhole_camera, crs=utm34n_crs, dem_band=2)

    # find initial dem bounds
    init_bounds = array_bounds(*ortho._dem_array.shape, ortho._dem_transform)
    init_bounds = np.array(transform_bounds(ortho._dem_crs, ortho._crs, *init_bounds))

    # reproject
    array, transform = ortho._reproject_dem(interp, resolution)
    bounds = np.array(array_bounds(*array.shape, transform))

    # test validity
    assert transform != ortho._dem_transform
    assert array.shape != ortho._dem_array.shape
    assert np.all(np.abs((transform[0], transform[4])) == resolution)
    assert bounds == pytest.approx(init_bounds, abs=max(resolution))
    assert np.all(bounds[:2] <= init_bounds[:2]) and np.all(bounds[-2:] >= init_bounds[-2:])
    assert np.nanmean(array) == pytest.approx(np.nanmean(ortho._dem_array), abs=1e-3)


def test_reproject_dem_crs_equal(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str
):
    """ Test DEM is not reprojected when it's CRS & resolution are the same as the ortho CRS & resolution. """
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs)

    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic, resolution)

    assert transform == ortho._dem_transform
    assert np.all(nan_equals(array, ortho._dem_array))


# @formatter:off
@pytest.mark.parametrize(
    'dem_file, crs', [
        ('float_utm34n_wgs84_dem_file', 'utm34n_egm96_crs'),
        ('float_utm34n_wgs84_dem_file', 'utm34n_egm2008_crs'),
        ('float_utm34n_egm96_dem_file', 'utm34n_wgs84_crs'),
    ],
)  # yapf: disable  # @formatter:on
def test_reproject_dem_vdatum_both(
    rgb_byte_src_file: Path, dem_file: str, pinhole_camera: Camera, crs: str, request: pytest.FixtureRequest
):
    """ Test DEM reprojection altitude adjustment when both DEM and ortho vertical datums are specified. """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)

    ortho = Ortho(rgb_byte_src_file, dem_file, pinhole_camera, crs=crs, dem_band=2)
    array, transform = ortho._reproject_dem(Interp.cubic, _dem_resolution)

    # crop array & transform to correspond to ortho._dem_array & ortho._dem_transform (assumes ortho._dem_array lies
    # inside array, which it should)
    dem_win = rio.windows.from_bounds(
        *array_bounds(*ortho._dem_array.shape, ortho._dem_transform), transform=transform
    ).round_offsets().round_lengths()
    test_array = array[dem_win.toslices()]
    test_transform = rio.windows.transform(dem_win, transform)

    assert not ortho._crs_equal
    assert test_transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert test_array.shape == ortho._dem_array.shape

    mask = ~np.isnan(test_array) & ~np.isnan(ortho._dem_array)
    assert test_array[mask] != pytest.approx(ortho._dem_array[mask], abs=5)


# @formatter:off
@pytest.mark.parametrize(
    'dem_file, crs', [
        ('float_utm34n_dem_file', 'utm34n_egm96_crs'),
        ('float_utm34n_egm96_dem_file', 'utm34n_crs'),
    ],
)  # yapf: disable  # @formatter:on
def test_reproject_dem_vdatum_one(
    rgb_byte_src_file: Path, dem_file: str, pinhole_camera: Camera, crs: str, request: pytest.FixtureRequest
):
    """ Test DEM reprojection does no altitude adjustment when one of DEM and ortho vertical datums are specified. """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)

    ortho = Ortho(rgb_byte_src_file, dem_file, pinhole_camera, crs=crs, dem_band=2)
    with rio.open(dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic, resolution)

    # crop array & transform to correspond to ortho._dem_array & ortho._dem_transform (assumes ortho._dem_array lies
    # inside array, which it should)
    dem_win = rio.windows.from_bounds(
        *array_bounds(*ortho._dem_array.shape, ortho._dem_transform), transform=transform
    ).round_offsets().round_lengths()
    test_array = array[dem_win.toslices()]
    test_transform = rio.windows.transform(dem_win, transform)

    assert not ortho._crs_equal
    assert test_transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert test_array.shape == ortho._dem_array.shape

    mask = ~np.isnan(test_array) & ~np.isnan(ortho._dem_array)
    assert test_array[mask] == pytest.approx(ortho._dem_array[mask], abs=1e-3)


@pytest.mark.parametrize('num_pts', [40, 100, 400, 1000, 4000])
def test_src_boundary(rgb_pinhole_utm34n_ortho: Ortho, num_pts: int):
    """
    Test _get_src_boundary(full_remap=True) generates a boundary with the correct corners and length.
    test_camera.test_undistort_alpha() covers the full_remap=False case.
    """
    # reference coords to test against
    w, h = np.array(rgb_pinhole_utm34n_ortho._camera._im_size, dtype='float32') - 1
    ref_ji = {(0., 0.), (w, 0.), (w, h), (0., h)}

    # get the boundary and simplify
    ji = rgb_pinhole_utm34n_ortho._get_src_boundary(num_pts=num_pts).astype('float32')
    test_ji = cv2.approxPolyDP(ji.T, epsilon=1e-6, closed=True)
    test_ji = set([tuple(*pt) for pt in test_ji])

    assert ji.shape[1] == num_pts
    assert test_ji == ref_ji


# @formatter:off
@pytest.mark.parametrize(
    'xyz_offset, opk_offset', [
        # varying rotations starting at `rotation` fixture value and keeping FOV below horizon
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (-15, 10, 0)),
        ((0, 0, 0), (-30, 20, 0)),
        ((0, 0, 0), (-45, 20, 0)),
        # varying positions with partial dem coverage
        ((0, 5.5e2, 0), (0, 0, 0)),
        ((0, 0, 1.1e3), (0, 0, 0)),
        ((0, 0, 2.e3), (0, 0, 0)),
    ],
)  # yapf: disable  # @formatter:on
def test_mask_dem(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str, xyz_offset: Tuple,
    opk_offset: Tuple, tmp_path: Path
):
    """ Test the similarity of the DEM (ortho boundary) and ortho valid data mask (without cropping). """
    # Note that these tests should use the pinhole camera model to ensure no artefacts outside the ortho boundary, and
    #  DEM < camera height to ensure no ortho artefacts in DEM > camera height areas.  While the DEM mask excludes
    #  (boundary) occluded pixels, the ortho image mask does not i.e. to compare these masks, there should be no
    #  DEM - ortho occlusion.
    # TODO: add test with dem that includes occlusion
    _xyz = tuple(np.array(camera_args['xyz']) + xyz_offset)
    _opk = tuple(np.array(camera_args['opk']) + np.radians(opk_offset))
    camera: Camera = PinholeCamera(
        camera_args['im_size'], camera_args['focal_len'], sensor_size=camera_args['sensor_size'], xyz=_xyz, opk=_opk,
    )
    resolution = (3, 3)
    num_pts = 400
    dem_interp = Interp.cubic

    # create an ortho image without DEM masking
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs, dem_band=1)
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    with rio.open(rgb_byte_src_file, 'r') as src_im:
        ortho_profile, _ = ortho._create_ortho_profile(
            src_im, dem_array.shape, dem_transform, dtype='uint8', compress=Compress.deflate
        )
        with rio.open(ortho_file, 'w', **ortho_profile) as ortho_im:
            ortho._remap(src_im, ortho_im, dem_array, full_remap=True, write_mask=False)

    # create the dem mask
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(), dem_transform, dem_interp, full_remap=True, crop=False, mask=True, num_pts=num_pts
    )
    dem_mask = ~np.isnan(dem_array_mask)

    # read the ortho nodata mask
    assert ortho_file.exists()
    with rio.open(ortho_file, 'r') as ortho_im:
        ortho_mask = ortho_im.dataset_mask().astype('bool')

    # test dem mask contains, and is similar to the ortho mask
    assert dem_transform_mask == dem_transform
    assert dem_mask.shape == ortho_mask.shape
    assert dem_mask[ortho_mask].sum() / ortho_mask.sum() > 0.9
    cc = np.corrcoef(dem_mask.flatten(), ortho_mask.flatten())
    assert (np.all(dem_mask) and np.all(ortho_mask)) or (cc[0, 1] > 0.95)

    if False:
        # debug plotting code
        # %matplotlib
        from matplotlib import pyplot
        from rasterio.plot import show

        def plot_poly(mask: np.ndarray, transform=dem_transform, ico='k'):
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
            pyplot.plot(*ortho._camera._T[:2], 'cx')


def test_mask_dem_crop(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """ Test the DEM mask is cropped to mask boundaries. """
    ortho = rgb_pinhole_utm34n_ortho
    resolution = (5, 5)
    num_pts = 400
    dem_interp = Interp.cubic

    # mask the dem without cropping
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(), dem_transform, dem_interp, full_remap=True, crop=False, mask=True, num_pts=num_pts
    )
    mask = ~np.isnan(dem_array_mask)

    # crop & mask the dem
    dem_array_crop, dem_transform_crop = ortho._mask_dem(
        dem_array.copy(), dem_transform, dem_interp, full_remap=True, crop=True, mask=True, num_pts=num_pts
    )
    mask_crop = ~np.isnan(dem_array_crop)

    # find the window of mask_crop in mask
    bounds_crop = array_bounds(*dem_array_crop.shape, dem_transform_crop)
    win_crop = from_bounds(*bounds_crop, dem_transform_mask)

    # test windowed portion of mask is identical to mask_crop, and unwindowed portion contains no masked pixels
    assert np.all(mask_crop == mask[win_crop.toslices()])
    assert mask.sum() == mask[win_crop.toslices()].sum()

    # test mask_crop extends to the boundaries
    ij = np.where(mask_crop)
    assert np.min(ij, axis=1) == pytest.approx((0, 0), abs=1)
    assert np.max(ij, axis=1) == pytest.approx(np.array(mask_crop.shape) - 1, abs=1)


def test_mask_dem_partial(
    rgb_byte_src_file: Path, float_utm34n_partial_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test the DEM mask excludes DEM nodata and is cropped to mask boundaries. """
    camera: Camera = PinholeCamera(**camera_args)
    resolution = (5, 5)
    num_pts = 400
    dem_interp = Interp.cubic
    ortho = Ortho(rgb_byte_src_file, float_utm34n_partial_dem_file, camera, utm34n_crs)

    # mask the dem without cropping
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    valid_mask = ~np.isnan(dem_array)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(), dem_transform, dem_interp, full_remap=True, crop=False, mask=True, num_pts=num_pts
    )
    mask = ~np.isnan(dem_array_mask)

    # crop & mask the dem
    dem_array_crop, dem_transform_crop = ortho._mask_dem(
        dem_array.copy(), dem_transform, dem_interp, full_remap=True, crop=True, mask=True, num_pts=num_pts
    )
    mask_crop = ~np.isnan(dem_array_crop)

    # find the window of mask_crop in mask
    bounds_crop = array_bounds(*dem_array_crop.shape, dem_transform_crop)
    win_crop = from_bounds(*bounds_crop, dem_transform_mask)

    # test the dem contains nodata
    assert not np.all(valid_mask[win_crop.toslices()])

    # test mask_crop extends to the boundaries
    ij = np.where(mask_crop)
    assert np.min(ij, axis=1) == pytest.approx((0, 0), abs=1)
    assert np.max(ij, axis=1) == pytest.approx(np.array(mask_crop.shape) - 1, abs=1)

    # test mask_crop excludes dem nodata
    assert np.all(mask_crop == mask_crop & valid_mask[win_crop.toslices()])


def test_mask_dem_coverage_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test DEM masking without DEM coverage raises an error. """
    camera: Camera = PinholeCamera(**camera_args)

    # init & reproject with coverage
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic, (30., 30.))

    # update camera for no coverage
    camera.update((0., 0., 1000.), (0., 0., 0.))

    # test
    with pytest.raises(ValueError) as ex:
        ortho._mask_dem(dem_array, dem_transform, Interp.cubic, full_remap=True)
    assert 'boundary' in str(ex)


def test_mask_dem_above_camera_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test DEM masking raises an error when the DEM is higher the camera. """
    camera: Camera = PinholeCamera(**camera_args)

    # move the camera below the DEM
    _xyz = list(camera_args['xyz'])
    _xyz[2] -= 1000
    camera.update(_xyz, camera_args['opk'])

    # init & reproject
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic, (30., 30.))

    # test
    with pytest.raises(ValueError) as ex:
        ortho._mask_dem(dem_array, dem_transform, Interp.cubic, full_remap=True)
    assert 'higher' in str(ex)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera'], )
def test_undistort(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, utm34n_crs: str, camera: str, request: pytest.FixtureRequest
):
    """ Test _undistort method by comparing source & distorted-undistorted checkerboard images. """
    nodata = 0
    interp = Interp.cubic
    camera: Camera = request.getfixturevalue(camera)
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)

    # create checkerboard source image
    image = checkerboard(camera._im_size[::-1])

    # distort then undistort
    dist_image = distort_image(camera, image, nodata=nodata, interp=interp)
    undist_image = ortho._undistort(dist_image, nodata=nodata, interp=interp)

    # test similarity of source and distorted-undistorted images
    dist_mask = dist_image != nodata
    cc_dist = np.corrcoef(image[dist_mask], dist_image[dist_mask])
    undist_mask = undist_image != nodata
    cc = np.corrcoef(image[undist_mask], undist_image[undist_mask])
    assert cc[0, 1] > cc_dist[0, 1] or cc[0, 1] == 1
    assert cc[0, 1] > 0.95


# @formatter:off
@pytest.mark.parametrize(
    'src_file, compress', [
        ('rgb_byte_src_file', Compress.auto),
        ('rgb_byte_src_file', Compress.jpeg),
        ('rgb_byte_src_file', Compress.deflate),
        ('float_src_file', Compress.auto),
        ('float_src_file', Compress.deflate),
    ]
)  # yapf: disable  # @formatter:on
def test_process_compress(
    src_file: str, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str, compress: Compress,
    tmp_path: Path, request: pytest.FixtureRequest
):
    """ Test the ortho compression, interleaving and photometric interpretation are set correctly. """
    src_file: Path = request.getfixturevalue(src_file)
    ortho = Ortho(src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs, dem_band=1)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, (30, 30), compress=compress)

    assert ortho_file.exists()
    with rio.open(src_file, 'r') as src_im, rio.open(ortho_file, 'r') as ortho_im:
        if compress == Compress.auto:
            compress = Compress.jpeg if src_im.dtypes[0] == 'uint8' else compress.deflate
        interleave, photometric = (('pixel', 'ycbcr') if compress == Compress.jpeg and src_im.count == 3 else
                                   ('band', 'minisblack'))
        assert ortho_im.profile['compress'] == compress.name
        assert ortho_im.profile['interleave'] == interleave
        if 'photometric' in ortho_im.profile:
            assert ortho_im.profile['photometric'] == photometric


def test_process_compress_jpeg_error(
    float_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str, tmp_path: Path,
):
    """ Test that jpeg compresssion raises an error when the source image dtype is not uint8. """
    ortho = Ortho(float_src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs, dem_band=1)
    ortho_file = tmp_path.joinpath('test_ortho.tif')

    with pytest.raises(ValueError) as ex:
        ortho.process(ortho_file, (5, 5), compress=Compress.jpeg)
    assert 'uint8' in str(ex)


# @formatter:off
@pytest.mark.parametrize(
    'src_file, dtype', [
        ('rgb_byte_src_file', None),
        ('float_src_file', None),
        # all opencv supported dtypes
        ('rgb_byte_src_file', 'uint8'),
        ('rgb_byte_src_file', 'uint16'),
        ('rgb_byte_src_file', 'int16'),
        ('rgb_byte_src_file', 'float32'),
        ('rgb_byte_src_file', 'float64'),
    ]
)  # yapf: disable  # @formatter:on
def test_process_dtype(
    src_file: str, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str, dtype: str, tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """ Test the ortho `dtype` is set correctly. """
    src_file: Path = request.getfixturevalue(src_file)
    ortho = Ortho(src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs, dem_band=1)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, (30, 30), dtype=dtype)

    assert ortho_file.exists()
    with rio.open(src_file, 'r') as src_im, rio.open(ortho_file, 'r') as ortho_im:
        dtype = src_im.profile['dtype'] if dtype is None else dtype
        assert ortho_im.profile['dtype'] == dtype


@pytest.mark.parametrize('dtype', ['int8', 'uint32', 'int32', 'uint64', 'int64'])
def test_process_dtype_error(rgb_pinhole_utm34n_ortho: Ortho, dtype: str, tmp_path: Path):
    """ Test unsupported dtypes raise an error. """
    ortho_file = tmp_path.joinpath('test_ortho.tif')

    with pytest.raises(ValueError) as ex:
        rgb_pinhole_utm34n_ortho.process(ortho_file, (30, 30), dtype=dtype)
    assert dtype in str(ex)


@pytest.mark.parametrize('resolution', [(30., 30.), (60., 60.), (60., 30.)])
def test_process_resolution(rgb_pinhole_utm34n_ortho: Ortho, resolution: Tuple, tmp_path: Path):
    """ Test ortho `resolution` is set correctly. """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, resolution)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        assert ortho_im.res == resolution


@pytest.mark.parametrize(
    # varying rotations starting at `rotation` fixture value & keeping full DEM coverage
    'opk_offset', [(0, 0, 0), (-15, 10, 0)],
)
def test_process_auto_resolution(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str, opk_offset: Tuple,
    xyz: Tuple, tmp_path: Path
):
    """ Test that auto resolution generates approx as many ortho pixels as source pixels. """
    _opk = tuple(np.array(camera_args['opk']) + np.radians(opk_offset))
    camera: Camera = PinholeCamera(
        camera_args['im_size'], camera_args['focal_len'], sensor_size=camera_args['sensor_size'], xyz=xyz,
        opk=_opk
    )
    dem_interp = Interp.cubic

    # find the auto res and masked dem
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs, dem_band=2)
    resolution = ortho._get_auto_res()
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array, dem_transform, dem_interp, full_remap=True, crop=True, mask=True
    )
    mask = ~np.isnan(dem_array_mask)

    assert np.array(camera._im_size).prod() == pytest.approx(mask.sum(), rel=0.05)


# @formatter:off
@pytest.mark.parametrize(
    'src_file, write_mask, per_band', [
        ('rgb_byte_src_file', True, True),
        ('rgb_byte_src_file', False, True),
        ('rgb_byte_src_file', True, False),
        ('rgb_byte_src_file', False, False),
        ('float_src_file', True, False),
        ('float_src_file', False, False),
    ]
)  # yapf: disable  # @formatter:on
def test_process_write_mask_per_band(
    src_file: str, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str, write_mask: bool,
    per_band: bool, tmp_path: Path, request: pytest.FixtureRequest,
):
    """ Test ``write_mask=True`` writes an internal ortho mask irrespective of the value of `per_band`. """
    src_file: Path = request.getfixturevalue(src_file)
    ortho = Ortho(src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, (30, 30), write_mask=write_mask, per_band=per_band)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        mask_flag = MaskFlags.per_dataset if write_mask else MaskFlags.nodata
        assert all([mf[0] == mask_flag for mf in ortho_im.mask_flag_enums])


# @formatter:off
@pytest.mark.parametrize(
    'compress', [Compress.jpeg, Compress.deflate, Compress.auto]
)  # yapf: disable  # @formatter:on
def test_process_write_mask_compress(
    rgb_pinhole_utm34n_ortho: Ortho, compress: Compress, tmp_path: Path
):
    """ Test ``write_mask=None`` writes an internal ortho mask when jpeg compression is used. """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, (30, 30), write_mask=None, compress=compress)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        mask_flag = MaskFlags.per_dataset if ortho_im.profile['compress'] == 'jpeg' else MaskFlags.nodata
        assert all([mf[0] == mask_flag for mf in ortho_im.mask_flag_enums])


@pytest.mark.parametrize(
    # all opencv supported dtypes
    'dtype', ['uint8', 'uint16', 'int16', 'float32', 'float64'],
)
def test_process_nodata(rgb_pinhole_utm34n_ortho: Ortho, dtype: str, tmp_path: Path):
    """ Test the ortho `nodata` is set correctly. """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, (30, 30), dtype=dtype, compress=Compress.deflate)

    assert ortho_file.exists()
    with rio.open(ortho_file, 'r') as ortho_im:
        assert ortho_im.profile['dtype'] in Ortho._nodata_vals
        assert nan_equals(ortho_im.profile['nodata'], Ortho._nodata_vals[ortho_im.profile['dtype']])


@pytest.mark.parametrize('interp', [Interp.average, Interp.bilinear, Interp.cubic, Interp.lanczos], )
def test_process_interp(rgb_pinhole_utm34n_ortho: Ortho, interp: Interp, tmp_path: Path):
    """ Test the process `interp` setting by comparing with an ``interp='nearest'`` reference ortho. """
    resolution = (10, 10)

    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_ref_file, resolution, interp=Interp.nearest, compress=Compress.deflate)

    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_test_file, resolution, interp=interp, compress=Compress.deflate)

    assert ortho_ref_file.exists() and ortho_test_file.exists()
    with rio.open(ortho_ref_file, 'r') as ref_im, rio.open(ortho_test_file, 'r') as test_im:
        ref_array = ref_im.read(masked=True)
        test_array = test_im.read(masked=True)
        assert test_array.shape == ref_array.shape
        assert test_array.mask.sum() == pytest.approx(ref_array.mask.sum(), rel=0.05)
        assert len(np.unique(test_array.compressed())) > len(np.unique(ref_array.compressed()))
        assert test_array.mean() == pytest.approx(ref_array.mean(), rel=0.2)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera'])
def test_process_full_remap(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera: str, utm34n_crs, tmp_path: Path,
    request: pytest.FixtureRequest
):
    """ Test ortho equivalence for ``full_remap=True/False`` with ``alpha=1``. """
    camera: Camera = request.getfixturevalue(camera)
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, utm34n_crs)
    resolution = (3, 3)

    # create a ref (full_remap=True) and test (full_remap=False) ortho for this camera
    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    ortho.process(ortho_ref_file, resolution, full_remap=True, compress=Compress.deflate)
    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_test_file, resolution, full_remap=False, compress=Compress.deflate)

    # compare ref & test ortho extents, masks and pixels
    assert ortho_ref_file.exists() and ortho_test_file.exists()
    with rio.open(ortho_ref_file, 'r') as ref_im, rio.open(ortho_test_file, 'r') as test_im:
        ref_win = ref_im.window(*test_im.bounds)
        ref_array = ref_im.read(window=ref_win)
        ref_mask = ref_im.dataset_mask(window=ref_win).astype('bool')
        ref_bounds = np.array(ref_im.bounds)
        test_array = test_im.read()
        test_mask = test_im.dataset_mask().astype('bool')
        test_bounds = np.array(test_im.bounds)

        assert test_array.shape == ref_array.shape
        assert ref_bounds == pytest.approx(test_bounds, abs=resolution[0])
        cc = np.corrcoef(ref_mask.flatten(), test_mask.flatten())
        assert cc[0, 1] > 0.95
        assert cc[0, 1] == pytest.approx(1., abs=1e-3) if isinstance(camera, PinholeCamera) else cc[0, 1] < 1.

        mask = ref_mask & test_mask
        cc = np.corrcoef(ref_array[:, mask].flatten(), test_array[:, mask].flatten())
        assert cc[0, 1] > 0.95
        assert cc[0, 1] == pytest.approx(1., abs=1e-3) if isinstance(camera, PinholeCamera) else cc[0, 1] < 1.


@pytest.mark.parametrize(
    'cam_type, dist_param', [
        (CameraType.pinhole, {}),
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
    ],
)  # yapf: disable
def test_process_alpha(
    cam_type: CameraType, dist_param: str, camera_args: Dict, rgb_byte_src_file: Path, float_utm34n_dem_file: Path,
    utm34n_crs: str, tmp_path: Path, request: pytest.FixtureRequest
):
    """ Test ortho with ``alpha=1`` contains and is similar to ortho with ``alpha=0``. """
    dist_param: Dict = request.getfixturevalue(dist_param) if dist_param else {}
    camera_alpha1 = create_camera(cam_type, **camera_args, **dist_param, alpha=1.)
    camera_alpha0 = create_camera(cam_type, **camera_args, **dist_param, alpha=0.)
    resolution = (3, 3)

    # create a ref (alpha=1) and test (alpha=0) ortho for this camera
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera_alpha1, utm34n_crs, dem_band=1)
    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    ortho.process(ortho_ref_file, resolution, full_remap=False, compress=Compress.deflate)

    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera_alpha0, utm34n_crs, dem_band=1)
    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_test_file, resolution, full_remap=False,compress=Compress.deflate)

    # compare ref & test ortho extents, masks and pixels
    assert ortho_ref_file.exists() and ortho_test_file.exists()
    with rio.open(ortho_ref_file, 'r') as ref_im, rio.open(ortho_test_file, 'r') as test_im:
        ref_win = ref_im.window(*test_im.bounds)
        ref_array = ref_im.read()
        ref_mask = ref_im.dataset_mask().astype('bool')
        ref_bounds = np.array(ref_im.bounds)
        test_array = test_im.read()
        test_mask = test_im.dataset_mask().astype('bool')
        test_bounds = np.array(test_im.bounds)

        # test ref_mask contains test_mask
        assert test_mask.shape == (ref_win.height, ref_win.width)
        assert np.all(ref_bounds[:2] <= test_bounds[:2]) and np.all(ref_bounds[-2:] >= test_bounds[:2])
        if cam_type is CameraType.pinhole:
            assert ref_mask.sum() == test_mask.sum()
        else:
            assert ref_mask.sum() > test_mask.sum()
        ref_mask = ref_mask[ref_win.toslices()]
        assert (ref_mask[test_mask].sum() / test_mask.sum()) > 0.99

        # test similarity of ortho pixels in intersecting ref-test area
        mask = ref_mask & test_mask
        ref_array = ref_array[(slice(0, ref_im.count), *ref_win.toslices())]
        cc = np.corrcoef(ref_array[:, mask].flatten(), test_array[:, mask].flatten())
        assert cc[0, 1] > 0.95


def test_process_per_band(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """ Test ortho equivalence for ``per_band=True/False``. """
    resolution = (5, 5)

    # create a ref (per_band=True) and test (per_band=False) ortho
    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_ref_file, resolution, per_band=True, compress=Compress.deflate)

    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_test_file, resolution, per_band=False, compress=Compress.deflate)

    # compare ref and test orthos
    assert ortho_ref_file.exists() and ortho_test_file.exists()
    with rio.open(ortho_ref_file, 'r') as ref_im, rio.open(ortho_test_file, 'r') as test_im:
        ref_array = ref_im.read()
        test_array = test_im.read()

        assert test_array.shape == ref_array.shape
        assert np.all(test_array == ref_array)


def test_process_overwrite(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """ Test overwriting an existing file with ``overwrite=True``. """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho_file.touch()
    rgb_pinhole_utm34n_ortho.process(ortho_file, (30, 30), overwrite=True)
    assert ortho_file.exists()


def test_process_overwrite_error(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """ Test overwriting an existing file raises an error with ``overwrite=False``. """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho_file.touch()
    with pytest.raises(FileExistsError) as ex:
        rgb_pinhole_utm34n_ortho.process(ortho_file, (30, 30), overwrite=False)
    assert ortho_file.name in str(ex)


def test_process_overview(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """ Test the existence of overview(s) on a big enough ortho. """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, (0.25, 0.25))
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        assert min(ortho_im.shape) >= 512
        assert len(ortho_im.overviews(1)) > 0


@pytest.mark.parametrize('camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera'])
def test_process(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera: str, utm34n_crs, tmp_path: Path,
    request: pytest.FixtureRequest
):
    """ Test ortho image format and content for different cameras. """
    # create Ortho object and process, using the planar DEM band
    camera: Camera = request.getfixturevalue(camera)
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, utm34n_crs, dem_band=2)
    resolution = (5, 5)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, resolution, full_remap=True, compress=Compress.deflate, interp=Interp.nearest)
    dem_bounds = np.array(array_bounds(*ortho._dem_array.shape, ortho._dem_transform))
    assert ortho_file.exists()

    with rio.open(rgb_byte_src_file, 'r') as src_im, rio.open(ortho_file, 'r') as ortho_im:
        ortho_bounds = np.array(ortho_im.bounds)

        # test ortho format
        assert ortho_im.count == src_im.count
        assert ortho_im.dtypes == src_im.dtypes
        assert ortho_im.res == resolution
        assert np.all(ortho_bounds[:2] >= dem_bounds[:2]) and np.all(ortho_bounds[-2:] <= dem_bounds[-2:])
        assert ortho_im.profile['compress'] == 'deflate'
        assert ortho_im.profile['interleave'] == 'band'

        # test ortho content
        src_array = src_im.read()
        ortho_array = ortho_im.read()
        ortho_mask = ortho_im.dataset_mask().astype('bool')
        assert np.all(np.unique(src_array) == np.unique(ortho_array[:, ortho_mask]))
        assert src_array.mean() == pytest.approx(ortho_array[:, ortho_mask].mean(), abs=15)
        assert src_array.std() == pytest.approx(ortho_array[:, ortho_mask].std(), abs=15)


# TODO: dem reproject changes bounds with different v datum
# TODO: add tests for other CRSs, spec'd in proj4 string, with vertical datum & with ortho & DEM in different CRSs
##
