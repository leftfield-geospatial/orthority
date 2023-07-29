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
import rasterio as rio
from rasterio.enums import MaskFlags
from rasterio.features import shapes
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from simple_ortho.camera import Camera, PinholeCamera
from simple_ortho.enums import Interp, Compress
from simple_ortho.ortho import Ortho
from simple_ortho.utils import nan_equals

logging.basicConfig(level=logging.DEBUG)


def test_init(rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str):
    """ Test Ortho initialisation with specified ortho CRS. """
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs)
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._ortho_crs == rio.CRS.from_string(utm34n_crs)
    assert ortho._dem_crs == dem_crs
    assert ortho._crs_equal == (ortho._ortho_crs == dem_crs)
    assert ortho._dem_array is not None
    assert ortho._dem_transform is not None


def test_init_src_crs(rgb_byte_utm34n_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera):
    """ Test Ortho initialisation with CRS from source file. """
    ortho = Ortho(rgb_byte_utm34n_src_file, float_utm34n_dem_file, pinhole_camera, crs=None)
    with rio.open(rgb_byte_utm34n_src_file, 'r') as src_im:
        src_crs = src_im.crs
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._ortho_crs == src_crs
    assert ortho._dem_crs == dem_crs
    assert ortho._crs_equal == (ortho._ortho_crs == dem_crs)
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
    with pytest.raises(ValueError) as ex:
        Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs, dem_band=3)
    assert 'dem_band' in str(ex)


def test_init_nocrs_error(rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera):
    """ Test Ortho initialisation without a CRS raises an error. """
    with pytest.raises(ValueError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=None)
    assert 'crs' in str(ex)


def test_init_geogcrs_error(rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera):
    """ Test Ortho initialisation with a geographic CRS raises an error. """
    with pytest.raises(ValueError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs='EPSG:4326')
    assert 'geographic' in str(ex)


def test_init_dem_coverage_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test Ortho initialisation without DEM coverage raises an error. """
    # create a camera positioned away from dem bounds
    camera = PinholeCamera(**camera_args)
    camera.update_extrinsic((0, 0, 0), (0, 0, 0))

    with pytest.raises(ValueError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    assert 'bounds' in str(ex)


def test_init_horizon_fov_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test Ortho initialisation with a horizontal FOV camera raises an error. """
    # create a camera pointing away from dem bounds
    camera = PinholeCamera(**camera_args)
    camera.update_extrinsic((0, 0, 0), (np.pi / 2, 0, 0))

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
    with rio.open(float_wgs84_wgs84_dem_file, 'r') as dem_im:
        init_crs = dem_im.crs
    init_bounds = array_bounds(*ortho._dem_array.shape, ortho._dem_transform)
    init_bounds = np.array(transform_bounds(init_crs, ortho._ortho_crs, *init_bounds))

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
    array, transform = ortho._reproject_dem(Interp.cubic_spline, resolution)

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
    with rio.open(dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic_spline, resolution)

    assert not ortho._crs_equal
    assert transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert array.shape == ortho._dem_array.shape

    mask = ~np.isnan(array) & ~np.isnan(ortho._dem_array)
    assert array[mask] != pytest.approx(ortho._dem_array[mask], abs=5)


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
    array, transform = ortho._reproject_dem(Interp.cubic_spline, resolution)

    assert not ortho._crs_equal
    assert transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert array.shape == ortho._dem_array.shape

    mask = ~np.isnan(array) & ~np.isnan(ortho._dem_array)
    assert array[mask] == pytest.approx(ortho._dem_array[mask], abs=1e-3)


# @formatter:off
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
)  # yapf: disable  # @formatter:on
def test_mask_dem(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str, _position: Tuple,
    _rotation: Tuple, tmp_path: Path
):
    """ Test the DEM (ortho boundary) mask contains the ortho valid data mask (without cropping). """
    # note that these tests should use the pinhole camera model to ensure no artefacts outside the ortho boundary, and
    #  DEM < camera height to ensure no ortho artefacts in DEM > camera height areas.
    camera: Camera = PinholeCamera(
        _position, np.radians(_rotation), camera_args['focal_len'], camera_args['im_size'], camera_args['sensor_size']
    )
    resolution = (5, 5)
    num_pts = 400

    # create an ortho image without DEM masking
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic_spline, resolution)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    with rio.open(rgb_byte_src_file, 'r') as src_im:
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


# @formatter:off
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
)  # yapf: disable  # @formatter:on
def test_mask_dem_crop(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str, _position: Tuple,
    _rotation: Tuple, tmp_path: Path
):
    """ Test the DEM mask is cropped to ortho boundaries. """
    camera: Camera = PinholeCamera(
        _position, np.radians(_rotation), camera_args['focal_len'], camera_args['im_size'], camera_args['sensor_size']
    )
    resolution = (5, 5)
    num_pts = 400

    # mask the dem without cropping
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
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
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test DEM masking without DEM coverage raises an error. """
    camera: Camera = PinholeCamera(**camera_args)

    # init & reproject with coverage
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic_spline, (30., 30.))

    # update camera for no coverage
    camera.update_extrinsic((0., 0., 1000.), (0., 0., 0.))

    # test
    with pytest.raises(ValueError) as ex:
        ortho._mask_dem(dem_array, dem_transform, full_remap=True)
    assert 'bounds' in str(ex)


def test_mask_dem_above_camera_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera_args: Dict, utm34n_crs: str
):
    """ Test DEM masking raises an error when the DEM is higher the camera. """
    camera: Camera = PinholeCamera(**camera_args)

    # move the camera below the DEM
    _position = list(camera_args['position'])
    _position[2] -= 1000
    camera.update_extrinsic(_position, camera_args['rotation'])

    # init & reproject
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic_spline, (30., 30.))

    # test
    with pytest.raises(ValueError) as ex:
        ortho._mask_dem(dem_array, dem_transform, full_remap=True)
    assert 'higher' in str(ex)


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
def test_process_write_mask(
    src_file: str, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str, write_mask: bool,
    per_band: bool, tmp_path: Path, request: pytest.FixtureRequest,
):
    """ Test ``write_mask=True`` with ``per_band=True/False`` writes an internal mask to ortho file. """
    src_file: Path = request.getfixturevalue(src_file)
    ortho = Ortho(src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, (30, 30), write_mask=write_mask, per_band=per_band)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        mask_flag = MaskFlags.per_dataset if write_mask else MaskFlags.nodata
        assert all([mf[0] == mask_flag for mf in ortho_im.mask_flag_enums])


@pytest.mark.parametrize(
    # all opencv supported dtypes
    'dtype', ['uint8', 'uint16', 'int16', 'float32', 'float64'],
)
def test_process_nodata(rgb_pinhole_utm34n_ortho: Ortho, dtype: str, tmp_path: Path):
    """ Test the ortho `nodata` is set correctly. """
    nodata_vals = dict(uint8=0, uint16=0, int16=np.iinfo('int16').min, float32=float('nan'), float64=float('nan'))
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, (30, 30), dtype=dtype)

    assert ortho_file.exists()
    with rio.open(ortho_file, 'r') as ortho_im:
        assert ortho_im.profile['dtype'] in nodata_vals
        assert nan_equals(ortho_im.profile['nodata'], nodata_vals[ortho_im.profile['dtype']])


@pytest.mark.parametrize('interp', [Interp.average, Interp.bilinear, Interp.cubic, Interp.lanczos], )
def test_process_interp(rgb_pinhole_utm34n_ortho: Ortho, interp: Interp, tmp_path: Path):
    """ Test the process `interp` setting by comparing with an ``interp='nearest'`` reference ortho. """
    resolution = (30, 30)

    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_ref_file, resolution, interp=Interp.nearest, compress=Compress.deflate)

    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_test_file, resolution, interp=interp, compress=Compress.deflate)

    assert ortho_ref_file.exists() and ortho_test_file.exists()
    with rio.open(ortho_ref_file, 'r') as ref_im, rio.open(ortho_test_file, 'r') as test_im:
        ref_array = ref_im.read(masked=True)
        test_array = test_im.read(masked=True)
        assert test_array.shape == ref_array.shape
        assert (test_array.mean() > 1) and (test_array.mean() < 255)
        assert test_array.mean() != ref_array.mean()
        assert test_array.mean() == pytest.approx(ref_array.mean(), 10)


@pytest.mark.parametrize('camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'nadir_fisheye_camera'])
def test_process_full_remap(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, camera: str, utm34n_crs, tmp_path: Path,
    request: pytest.FixtureRequest
):
    """ Test ortho equivalence for ``full_remap=True/False``. """
    camera: Camera = request.getfixturevalue(camera)
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, utm34n_crs)
    resolution = (5, 5)

    # create a ref (full_remap=True) and test (full_remap=False) ortho for this camera
    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    ortho.process(ortho_ref_file, resolution, full_remap=True, compress=Compress.deflate)

    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_test_file, resolution, full_remap=False, compress=Compress.deflate)

    # compare valid portions of ref and test orthos
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
        assert np.all(ref_bounds[:2] <= test_bounds[:2]) and np.all(ref_bounds[-2:] >= test_bounds[-2:])

        mask = ref_mask & test_mask
        cc = np.corrcoef(ref_array[:, mask].flatten(), test_array[:, mask].flatten())
        assert cc[0, 1] > 0.99


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


@pytest.mark.parametrize('camera', ['pinhole_camera', 'brown_camera', 'opencv_camera', 'nadir_fisheye_camera'])
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
        assert src_array.mean() == pytest.approx(ortho_array[:, ortho_mask].mean(), abs=10)
        assert src_array.std() == pytest.approx(ortho_array[:, ortho_mask].std(), abs=10)


# TODO: dem reproject changes bounds with different v datum
# TODO: tests to ensure ortho contains full ortho bounds e.g. dem mask *contains* ortho bounds

##
