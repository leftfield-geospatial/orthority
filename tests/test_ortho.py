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

import logging
import tracemalloc
from itertools import product
from math import factorial
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import rasterio as rio
from rasterio.enums import MaskFlags
from rasterio.features import shapes
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from orthority import common, errors, param_io
from orthority.camera import Camera, PinholeCamera, create_camera
from orthority.enums import CameraType, Compress, Driver, Interp
from orthority.errors import OrthorityError
from orthority.ortho import Ortho
from tests.conftest import _dem_resolution

logger = logging.getLogger(__name__)


def _validate_ortho_files(
    files: Sequence[Path], cc_thresh: float = 0.75, num_ovl_thresh: int = None
):
    """Validate the similarity of overlapping areas in ortho files."""
    cc_array = np.full((len(files),) * 2, fill_value=np.nan)
    num_ovl_thresh = num_ovl_thresh or factorial(len(files) - 1)

    for i1, file1 in enumerate(files):
        for i2, file2 in enumerate(files[i1 + 1 :]):
            with rio.open(file1, 'r') as im1, rio.open(file2, 'r') as im2:
                # find windows for the overlap area in each image
                ovl_bl = np.array([im1.bounds[:2], im2.bounds[:2]]).max(axis=0)
                ovl_tr = np.array([im1.bounds[2:], im2.bounds[2:]]).min(axis=0)
                if np.any(ovl_bl >= ovl_tr):
                    continue  # no overlap
                ovl_bounds = [*ovl_bl, *ovl_tr]
                win1 = im1.window(*ovl_bounds)
                win2 = im2.window(*ovl_bounds)

                # read overlap area in each image & find common mask
                array1 = im1.read(1, window=win1, masked=True)
                array2 = im2.read(1, window=win2, masked=True)
                mask = ~(array1.mask | array2.mask)

                # test similarity if overlap area > 5%
                im1_mask = im1.dataset_mask().astype('bool', copy=False)
                if (mask.sum() / im1_mask.sum()) > 0.05:
                    cc = np.corrcoef(array1[mask], array2[mask])
                    log_str = (
                        f"Overlap similarity of '{file1.name}' and '{file2.name}': {cc[0, 1]:.4f}"
                    )
                    logger.info(log_str)
                    cc_array[i1, i2] = cc[0, 1]
                    assert cc[0, 1] > cc_thresh, log_str

    # rough test on number of overlapping files
    assert np.sum(~np.isnan(cc_array)) >= num_ovl_thresh


def test_init(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str
):
    """Test Ortho initialisation with specified world CRS."""
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs)
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._crs == rio.CRS.from_string(utm34n_crs)
    assert ortho._dem_crs == dem_crs
    assert ortho._dem_array is not None
    assert ortho._dem_transform is not None


def test_init_src_crs(
    rgb_byte_utm34n_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera
):
    """Test Ortho initialisation with CRS from source file."""
    ortho = Ortho(rgb_byte_utm34n_src_file, float_utm34n_dem_file, pinhole_camera, crs=None)
    with rio.open(rgb_byte_utm34n_src_file, 'r') as src_im:
        src_crs = src_im.crs
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_crs = dem_im.crs

    assert ortho._crs == src_crs
    assert ortho._dem_crs == dem_crs
    assert ortho._dem_array is not None
    assert ortho._dem_transform is not None


@pytest.mark.parametrize('dem_band', [1, 2])
def test_init_dem_band(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_crs: str,
    dem_band: int,
):
    """Test Ortho initialisation with ``dem_band`` reads the correct DEM band."""
    ortho = Ortho(
        rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs, dem_band=dem_band
    )
    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        dem_bounds = array_bounds(*ortho._dem_array.shape, ortho._dem_transform)
        dem_win = dem_im.window(*dem_bounds)
        dem_array = dem_im.read(indexes=dem_band, window=dem_win).astype('float32')
    assert np.all(ortho._dem_array == dem_array)


def test_init_dem_band_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str
):
    """Test Ortho initialisation with incorrect ``dem_band`` raises an error."""
    with pytest.raises(OrthorityError) as ex:
        Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs, dem_band=3)
    assert 'DEM band' in str(ex.value)


def test_init_nocrs_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera
):
    """Test Ortho initialisation without a CRS raises an error."""
    with pytest.raises(errors.CrsMissingError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=None)
    assert 'crs' in str(ex.value)


def test_init_dem_coverage_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, frame_args: dict, utm34n_crs: str
):
    """Test Ortho initialisation without DEM coverage of ortho bounds raises an error."""
    # create a camera positioned away from dem bounds
    camera = PinholeCamera(**frame_args)
    camera.update((0, 0, 0), (0, 0, 0))

    with pytest.raises(OrthorityError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    assert 'DEM' in str(ex.value)


def test_init_horizon_fov_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, frame_args: dict, utm34n_crs: str
):
    """Test Ortho initialisation with a horizontal FOV camera raises an error."""
    # create a camera pointing away from dem bounds
    camera = PinholeCamera(**frame_args)
    camera.update((0, 0, 0), (np.pi / 2, 0, 0))

    with pytest.raises(OrthorityError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    assert 'horizon' in str(ex.value)


def test_dem_above_camera_error(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, frame_args: dict, utm34n_crs: str
):
    """Test reading the DEM raises an error when it is higher than the (frame) camera."""
    camera = PinholeCamera(**frame_args)

    # move the camera below the DEM
    _xyz = (*frame_args['xyz'][:2], frame_args['xyz'][2] - 1000)
    camera.update(_xyz, frame_args['opk'])

    # init & reproject
    with pytest.raises(OrthorityError) as ex:
        _ = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    assert 'DEM' in str(ex.value)


@pytest.mark.parametrize(
    'dem_file, crs',
    [
        ('float_utm34n_dem_file', 'utm34n_crs'),
        ('float_utm34n_msl_dem_file', 'utm34n_egm2008_crs'),
        ('float_wgs84_wgs84_dem_file', 'utm34n_egm2008_crs'),
    ],
)
def test_get_init_dem(
    rgb_byte_src_file: Path,
    pinhole_camera: Camera,
    dem_file: str,
    crs: str,
    request: pytest.FixtureRequest,
):
    """Test the bounds of the initial DEM contain the world / ortho boundary at z=min(DEM),
    with different DEM and world / ortho CRSs, some including vertical CRSs.
    """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)
    dem_interp = Interp.cubic
    resolution = _dem_resolution

    ortho = Ortho(rgb_byte_src_file, dem_file, pinhole_camera, crs=crs)
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    test_bounds = array_bounds(*dem_array.shape, dem_transform)
    min_z = np.nanmin(dem_array)
    xyz = ortho.camera.world_boundary(min_z)
    ref_bounds = *xyz[:2].min(axis=1), *xyz[:2].max(axis=1)

    assert test_bounds[:2] <= ref_bounds[:2]
    assert test_bounds[2:] >= ref_bounds[2:]


def test_get_init_dem_vert_scale(
    rgb_byte_src_file: Path,
    float_utm34n_egm2008_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_egm2008_crs: str,
    rgb_pinhole_utm34n_ortho: Ortho,
):
    """Test that initial DEM bounds account for vertical scale / CRS by comparing bounds where the
    DEM and camera have the same relative geometry, but different vertical CRS.
    """
    ortho_egm2008 = Ortho(
        rgb_byte_src_file, float_utm34n_egm2008_dem_file, pinhole_camera, crs=utm34n_egm2008_crs
    )
    dem_bounds_egm2008 = array_bounds(*ortho_egm2008._dem_array.shape, ortho_egm2008._dem_transform)
    dem_bounds_novertcrs = array_bounds(
        *rgb_pinhole_utm34n_ortho._dem_array.shape, rgb_pinhole_utm34n_ortho._dem_transform
    )
    assert dem_bounds_egm2008 == pytest.approx(dem_bounds_novertcrs, abs=1e-1)


@pytest.mark.parametrize(
    # varying rotations starting at ``rotation`` fixture value & keeping full DEM coverage
    'opk_offset',
    [(0, 0, 0), (-15, 10, 0), (-30, 20, 0)],
)
def test_get_gsd_opk(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    frame_args: dict,
    utm34n_crs: str,
    opk_offset: tuple,
    tmp_path: Path,
):
    """Test the GSD as ortho resolution generates approx as many ortho pixels as source pixels
    with different camera orientations.
    """
    _opk = tuple(np.array(frame_args['opk']) + np.radians(opk_offset))
    camera = PinholeCamera(**frame_args)
    camera.update(xyz=frame_args['xyz'], opk=_opk)
    dem_interp = Interp.cubic

    # find the gsd resolution and masked dem
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs, dem_band=2)
    resolution = (ortho._get_gsd(),) * 2
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array, dem_transform, dem_interp, crop=False, mask=True
    )
    mask = ~np.isnan(dem_array_mask)

    assert np.array(camera.im_size).prod() == pytest.approx(mask.sum(), rel=0.05)


@pytest.mark.parametrize(
    'dem_file, crs',
    [
        ('float_utm34n_dem_file', 'utm34n_crs'),
        ('float_utm34n_msl_dem_file', 'utm34n_egm2008_crs'),
        ('float_wgs84_wgs84_dem_file', 'utm34n_egm2008_crs'),
    ],
)
def test_get_gsd_vert_crs(
    rgb_byte_src_file: Path,
    pinhole_camera: Camera,
    dem_file: str,
    crs: str,
    request: pytest.FixtureRequest,
):
    """Test the GSD as ortho resolution generates approx as many ortho pixels as source pixels
    with different DEM and world / ortho CRSs, some with vertical CRSs.
    """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)
    dem_interp = Interp.cubic

    ortho = Ortho(rgb_byte_src_file, dem_file, pinhole_camera, crs=crs, dem_band=2)

    # find the gsd resolution and masked dem
    resolution = (ortho._get_gsd(),) * 2
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array, dem_transform, dem_interp, crop=True, mask=True
    )
    mask = ~np.isnan(dem_array_mask)

    assert np.array(ortho.camera.im_size).prod() == pytest.approx(mask.sum(), rel=0.05)


@pytest.mark.parametrize(
    'interp, resolution',
    [*zip(Interp, [(10, 10)] * len(Interp)), *zip(Interp, [(50, 50)] * len(Interp))],
)
def test_reproject_dem(
    rgb_byte_src_file: Path,
    float_wgs84_wgs84_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_crs: str,
    interp: Interp,
    resolution: tuple[float, float],
):
    """Test DEM is reprojected when it's CRS & resolution is different to the world / ortho CRS &
    ortho resolution.
    """
    ortho = Ortho(
        rgb_byte_src_file, float_wgs84_wgs84_dem_file, pinhole_camera, crs=utm34n_crs, dem_band=2
    )

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
    assert bounds == pytest.approx(init_bounds, abs=2 * max(resolution))
    assert np.nanmean(array) == pytest.approx(np.nanmean(ortho._dem_array), abs=1e-3)


def test_reproject_dem_crs_equal(
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, pinhole_camera: Camera, utm34n_crs: str
):
    """Test DEM is not reprojected when it's CRS & resolution are the same as the world CRS & ortho
    resolution.
    """
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, pinhole_camera, crs=utm34n_crs)

    with rio.open(float_utm34n_dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic, resolution)

    assert transform == ortho._dem_transform
    assert np.all(common.nan_equals(array, ortho._dem_array))


@pytest.mark.parametrize(
    'dem_file, crs',
    [
        ('float_utm34n_egm2008_dem_file', 'utm34n_egm96_crs'),
        ('float_utm34n_egm96_dem_file', 'utm34n_egm2008_crs'),
        ('float_utm34n_egm96_dem_file', 'utm34n_wgs84_crs'),
    ],
)
def test_reproject_dem_vert_crs_both(
    rgb_byte_src_file: Path,
    dem_file: str,
    pinhole_camera: Camera,
    crs: str,
    request: pytest.FixtureRequest,
):
    """Test DEM reprojection altitude adjustment when both DEM and ortho vertical CRSs are
    specified.
    """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)

    ortho = Ortho(rgb_byte_src_file, dem_file, pinhole_camera, crs=crs, dem_band=2)
    array, transform = ortho._reproject_dem(Interp.cubic, _dem_resolution)

    # crop array & transform to correspond to ortho._dem_array & ortho._dem_transform (assumes
    # ortho._dem_array lies inside array, which it should)
    dem_win = (
        rio.windows.from_bounds(
            *array_bounds(*ortho._dem_array.shape, ortho._dem_transform), transform=transform
        )
        .round_offsets()
        .round_lengths()
    )
    test_array = array[dem_win.toslices()]
    test_transform = rio.windows.transform(dem_win, transform)

    assert ortho._dem_crs != ortho._crs
    assert test_transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert test_array.shape == ortho._dem_array.shape

    mask = ~np.isnan(test_array) & ~np.isnan(ortho._dem_array)
    assert test_array[mask] != pytest.approx(ortho._dem_array[mask], abs=0.1)
    assert test_array[mask] == pytest.approx(ortho._dem_array[mask], abs=abs(Ortho._egm_minmax[0]))


@pytest.mark.parametrize(
    'dem_file, crs',
    [('float_utm34n_dem_file', 'utm34n_egm96_crs'), ('float_utm34n_egm96_dem_file', 'utm34n_crs')],
)
@pytest.mark.skipif(rio.get_proj_version() < (9, 1, 1), reason="requires PROJ 9.1.1 or higher")
def test_reproject_dem_vert_crs_one(
    rgb_byte_src_file: Path,
    dem_file: str,
    pinhole_camera: Camera,
    crs: str,
    request: pytest.FixtureRequest,
):
    """Test DEM reprojection does no altitude adjustment when one of DEM and ortho vertical CRSs
    are specified.
    """
    dem_file: Path = request.getfixturevalue(dem_file)
    crs: str = request.getfixturevalue(crs)

    ortho = Ortho(rgb_byte_src_file, dem_file, pinhole_camera, crs=crs, dem_band=2)
    with rio.open(dem_file, 'r') as dem_im:
        resolution = dem_im.res
    array, transform = ortho._reproject_dem(Interp.cubic, resolution)

    # crop array & transform to correspond to ortho._dem_array & ortho._dem_transform (assumes
    # ortho._dem_array lies inside array, which it should)
    dem_win = (
        rio.windows.from_bounds(
            *array_bounds(*ortho._dem_array.shape, ortho._dem_transform), transform=transform
        )
        .round_offsets()
        .round_lengths()
    )
    test_array = array[dem_win.toslices()]
    test_transform = rio.windows.transform(dem_win, transform)

    assert ortho._dem_crs != ortho._crs
    assert test_transform.almost_equals(ortho._dem_transform, precision=1e-6)
    assert test_array.shape == ortho._dem_array.shape

    mask = ~np.isnan(test_array) & ~np.isnan(ortho._dem_array)
    # prior proj versions promote 2D->3D with ellipsoidal height
    assert test_array[mask] == pytest.approx(ortho._dem_array[mask], abs=1e-3)


def test_reproject_dem_vert_crs_scale(
    rgb_byte_src_file: Path,
    float_utm34n_msl_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_egm2008_crs: str,
):
    """Test DEM reprojection z scaling when DEM height is in feet."""
    ortho = Ortho(
        rgb_byte_src_file,
        float_utm34n_msl_dem_file,
        pinhole_camera,
        crs=utm34n_egm2008_crs,
        dem_band=2,
    )
    array, transform = ortho._reproject_dem(Interp.cubic, _dem_resolution)
    assert np.nanmean(array) == pytest.approx(np.nanmean(ortho._dem_array) / 3.28084, abs=1e-3)


def test_reproject_resolution_error(rgb_pinhole_utm34n_ortho: Ortho):
    """Test DEM reprojection raises an error when the resolution exceeds the ortho bounds."""
    with pytest.raises(OrthorityError) as ex:
        _, _ = rgb_pinhole_utm34n_ortho._reproject_dem(Interp.cubic, (1000, 1000))
    assert 'resolution' in str(ex.value)


def test_mask_dem(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """Test the similarity of the masked DEM (ortho boundary) and ortho valid data mask (without
    cropping).
    """
    # Notes:
    # - The DEM intersection algorithm is tested more rigorously in
    # test_camera.test_world_boundary_zsurf().
    # - This test should use the pinhole camera model to ensure no artefacts outside the ortho
    # boundary, and DEM < camera height to ensure no ortho artefacts in DEM > camera height
    # areas.  While the DEM mask excludes (boundary) occluded pixels, the ortho image mask does
    # not i.e. to compare these masks, there should be no DEM - ortho occlusion.
    resolution = (3, 3)
    num_pts = 400
    dem_interp = Interp.cubic
    ortho = rgb_pinhole_utm34n_ortho

    # create an ortho image & mask without DEM masking
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    j, i = np.meshgrid(range(0, dem_array.shape[1]), range(0, dem_array.shape[0]), indexing='xy')
    x, y = (dem_transform * rio.Affine.translation(0.5, 0.5)) * [j, i]
    im_array = np.ones((1, *ortho.camera.im_size[::-1]))
    ortho_array, ortho_mask = ortho.camera.remap(
        im_array, x, y, dem_array, nodata=float('nan'), interp=dem_interp
    )
    ortho_mask = ~ortho_mask

    # create the dem mask
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(),
        dem_transform,
        dem_interp,
        crop=False,
        mask=True,
        num_pts=num_pts,
    )
    dem_mask = ~np.isnan(dem_array_mask)

    # test dem mask contains, and is similar to the ortho mask
    assert dem_transform_mask == dem_transform
    assert dem_mask.shape == ortho_mask.shape
    assert dem_mask[ortho_mask].sum() / ortho_mask.sum() > 0.95
    cc = np.corrcoef(dem_mask.flatten(), ortho_mask.flatten())
    assert cc[0, 1] > 0.9

    if False:
        # debug plotting code
        # %matplotlib
        from matplotlib import pyplot
        from rasterio.plot import show

        def plot_poly(mask: np.ndarray, transform=dem_transform, ico='k'):
            """Plot polygons from mask."""
            poly_list = [poly for poly, _ in shapes(mask.astype('uint8'), transform=transform)]

            for poly in poly_list[:-1]:
                coords = np.array(poly['coordinates'][0]).T
                pyplot.plot(coords[0], coords[1], ico)

        for image in (ortho_array, dem_array):
            pyplot.figure()
            show(image, transform=dem_transform, cmap='gray')
            plot_poly(ortho_mask, transform=dem_transform, ico='y--')
            plot_poly(dem_mask, transform=dem_transform, ico='r:')
            pyplot.plot(*ortho._camera._T[:2], 'cx')


def test_mask_dem_crop(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """Test the DEM mask is cropped to mask boundaries."""
    ortho = rgb_pinhole_utm34n_ortho
    resolution = (5, 5)
    num_pts = 400
    dem_interp = Interp.cubic

    # mask the dem without cropping
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(),
        dem_transform,
        dem_interp,
        crop=False,
        mask=True,
        num_pts=num_pts,
    )
    mask = ~np.isnan(dem_array_mask)

    # crop & mask the dem
    dem_array_crop, dem_transform_crop = ortho._mask_dem(
        dem_array.copy(),
        dem_transform,
        dem_interp,
        crop=True,
        mask=True,
        num_pts=num_pts,
    )
    mask_crop = ~np.isnan(dem_array_crop)

    # find the window of mask_crop in mask
    bounds_crop = array_bounds(*dem_array_crop.shape, dem_transform_crop)
    win_crop = from_bounds(*bounds_crop, dem_transform_mask)

    # sanity testing
    assert dem_transform_mask == dem_transform
    assert np.any(mask_crop)

    # test windowed portion of mask is identical to mask_crop, and unwindowed portion contains no
    # masked pixels
    assert np.all(mask_crop == mask[win_crop.toslices()])
    assert mask.sum() == mask[win_crop.toslices()].sum()

    # test mask_crop extends to the boundaries
    ij = np.where(mask_crop)
    assert np.min(ij, axis=1) == pytest.approx((0, 0), abs=1)
    assert np.max(ij, axis=1) == pytest.approx(np.array(mask_crop.shape) - 1, abs=1)


def test_mask_dem_partial(
    rgb_byte_src_file: Path, float_utm34n_partial_dem_file: Path, frame_args: dict, utm34n_crs: str
):
    """Test the DEM mask excludes DEM nodata and is cropped to mask boundaries."""
    camera = PinholeCamera(**frame_args, distort=True)
    resolution = (5, 5)
    num_pts = 400
    dem_interp = Interp.cubic
    ortho = Ortho(rgb_byte_src_file, float_utm34n_partial_dem_file, camera, utm34n_crs)

    # mask the dem without cropping
    dem_array, dem_transform = ortho._reproject_dem(dem_interp, resolution)
    valid_mask = ~np.isnan(dem_array)
    dem_array_mask, dem_transform_mask = ortho._mask_dem(
        dem_array.copy(),
        dem_transform,
        dem_interp,
        crop=False,
        mask=True,
        num_pts=num_pts,
    )

    # crop & mask the dem
    dem_array_crop, dem_transform_crop = ortho._mask_dem(
        dem_array.copy(),
        dem_transform,
        dem_interp,
        crop=True,
        mask=True,
        num_pts=num_pts,
    )
    mask_crop = ~np.isnan(dem_array_crop)

    # find the window of dem_array_crop in dem_array_mask
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
    rgb_byte_src_file: Path, float_utm34n_dem_file: Path, frame_args: dict, utm34n_crs: str
):
    """Test DEM masking without DEM coverage raises an error."""
    camera = PinholeCamera(**frame_args)

    # init & reproject with coverage
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, crs=utm34n_crs)
    dem_array, dem_transform = ortho._reproject_dem(Interp.cubic, (30.0, 30.0))

    # update camera for no coverage
    camera.update((0.0, 0.0, 1000.0), (0.0, 0.0, 0.0))

    # test
    with pytest.raises(OrthorityError) as ex:
        ortho._mask_dem(dem_array, dem_transform, Interp.cubic)
    assert 'lies outside' in str(ex.value)


@pytest.mark.parametrize('resolution', [(30.0, 30.0), (60.0, 60.0), (60.0, 30.0)])
def test_process_resolution(rgb_pinhole_utm34n_ortho: Ortho, resolution: tuple, tmp_path: Path):
    """Test ortho ``resolution`` is set correctly."""
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, resolution)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        assert ortho_im.res == resolution


@pytest.mark.parametrize('interp', [Interp.average, Interp.bilinear, Interp.cubic, Interp.lanczos])
def test_process_interp(rgb_pinhole_utm34n_ortho: Ortho, interp: Interp, tmp_path: Path):
    """Test the process ``interp`` setting by comparing with an ``interp='nearest'`` reference
    ortho.
    """
    resolution = (10, 10)

    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(
        ortho_ref_file, resolution, interp=Interp.nearest, compress=Compress.deflate
    )

    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(
        ortho_test_file, resolution, interp=interp, compress=Compress.deflate
    )

    assert ortho_ref_file.exists() and ortho_test_file.exists()
    with rio.open(ortho_ref_file, 'r') as ref_im, rio.open(ortho_test_file, 'r') as test_im:
        ref_array = ref_im.read(masked=True)
        test_array = test_im.read(masked=True)
        assert test_array.shape == ref_array.shape
        assert test_array.mask.sum() == pytest.approx(ref_array.mask.sum(), rel=0.05)
        assert len(np.unique(test_array.compressed())) > len(np.unique(ref_array.compressed()))
        test_array.mask |= ref_array.mask
        ref_array.mask |= test_array.mask
        cc = np.corrcoef(test_array.flatten(), ref_array.flatten())
        assert cc[0, 1] > 0.9
        assert cc[0, 1] != 1.0


@pytest.mark.parametrize('dem_interp', [Interp.bilinear, Interp.cubic, Interp.lanczos])
def test_process_dem_interp(rgb_pinhole_utm34n_ortho: Ortho, dem_interp: Interp, tmp_path: Path):
    """Test the process ``dem_interp`` setting by comparing with an ``dem_interp='nearest'``
    reference ortho.
    """
    # note that Interp.average is skipped as it gives similar upsampling results to Interp.nearest
    resolution = (10, 10)
    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(
        ortho_ref_file, resolution, dem_interp=Interp.nearest, compress=Compress.deflate
    )

    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(
        ortho_test_file, resolution, dem_interp=dem_interp, compress=Compress.deflate
    )

    assert ortho_ref_file.exists() and ortho_test_file.exists()
    with rio.open(ortho_ref_file, 'r') as ref_im, rio.open(ortho_test_file, 'r') as test_im:
        ref_array = ref_im.read(masked=True)
        test_win = test_im.window(*ref_im.bounds)
        test_array = test_im.read(masked=True, window=test_win, boundless=True)
        assert test_array.mask.sum() == pytest.approx(ref_array.mask.sum(), rel=0.1)
        test_array.mask |= ref_array.mask
        ref_array.mask |= test_array.mask
        cc = np.corrcoef(test_array.flatten(), ref_array.flatten())
        assert cc[0, 1] > 0.9
        assert cc[0, 1] != 1.0


def test_process_per_band(
    ms_float_src_file: Path,
    float_utm34n_dem_file: Path,
    frame_args: dict,
    utm34n_crs: str,
    tmp_path: Path,
):
    """Test ortho equivalence for ``per_band=True/False`` and that ``per_band=True`` uses less
    memory than ``per_band=False``."""
    # Note: Allocated memory depends on thread timing and is noisy.  For per_band to make
    # measurable memory differences, the source image needs to be relatively large, have many bands
    # and/or a 'big' dtype.  Also, bear in mind that tracemalloc does not track GDAL allocations.

    # create a camera for ms_float_src_file
    cam_args = dict(**frame_args)
    with rio.open(ms_float_src_file) as src_im:
        cam_args.update(im_size=src_im.shape[::-1])
    camera = PinholeCamera(**cam_args)

    # create orthos with per_band=True/False, tracking memory usage
    resolution = (5, 5)
    ortho_files = [tmp_path.joinpath('ref_ortho.tif'), tmp_path.joinpath('test_ortho.tif')]
    per_bands = [True, False]
    peak_mems = []
    try:
        tracemalloc.start()
        for ortho_file, per_band in zip(ortho_files, per_bands):
            start_mem = tracemalloc.get_traced_memory()
            ortho = Ortho(ms_float_src_file, float_utm34n_dem_file, camera, utm34n_crs)
            ortho.process(ortho_file, resolution, per_band=per_band, compress=Compress.deflate)
            end_mem = tracemalloc.get_traced_memory()
            peak_mems.append(end_mem[1] - start_mem[0])
            tracemalloc.clear_traces()  # clears the peak
            del ortho
    finally:
        tracemalloc.stop()

    # compare memory usage
    assert peak_mems[1] > peak_mems[0]

    # compare pre_band=True/False orthos
    assert ortho_files[0].exists() and ortho_files[1].exists()
    with rio.open(ortho_files[0], 'r') as ref_im, rio.open(ortho_files[1], 'r') as test_im:
        ref_array = ref_im.read()
        test_array = test_im.read()

        assert test_array.shape == ref_array.shape
        assert np.all(common.nan_equals(test_array, ref_array))


@pytest.mark.parametrize(
    'camera, camera_und',
    [
        ('pinhole_camera', 'pinhole_camera_und'),
        ('brown_camera', 'brown_camera_und'),
        ('opencv_camera', 'opencv_camera_und'),
        ('fisheye_camera', 'fisheye_camera_und'),
    ],
)
def test_process_distort(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    camera: str,
    camera_und: str,
    utm34n_crs,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Test ortho similarity for frame cameras with ``distort=True/False`` and ``alpha=1``."""
    # note that these tests are mostly duplicated by test_camera.test_frame_remap_equiv

    camera: Camera = request.getfixturevalue(camera)
    camera_und: Camera = request.getfixturevalue(camera_und)
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, utm34n_crs)
    ortho_und = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera_und, utm34n_crs)
    resolution = (3, 3)

    # Create ref (distort=True) and test (distort=False) orthos. Note that
    # distort=False erodes the ortho mask to remove nodata blur so the reference is expected
    # to contain the test mask.
    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    ortho.process(ortho_ref_file, resolution, compress=Compress.deflate)
    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    ortho_und.process(ortho_test_file, resolution, compress=Compress.deflate)

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

        # test ref_mask contains and is similar to test_mask
        assert test_array.shape == ref_array.shape
        assert ref_bounds == pytest.approx(test_bounds, abs=resolution[0])
        assert ref_mask[test_mask].sum() / test_mask.sum() > 0.99
        cc = np.corrcoef(ref_mask.flatten(), test_mask.flatten())
        assert cc[0, 1] > 0.9
        assert (
            cc[0, 1] == pytest.approx(1.0, abs=1e-3)
            if type(camera) == PinholeCamera
            else cc[0, 1] < 1.0
        )

        # test similarity of ortho pixels in intersecting ref-test area
        mask = ref_mask & test_mask
        cc = np.corrcoef(ref_array[:, mask].flatten(), test_array[:, mask].flatten())
        assert cc[0, 1] > 0.95
        assert (
            cc[0, 1] == pytest.approx(1.0, abs=1e-3)
            if type(camera) == PinholeCamera
            else cc[0, 1] < 1.0
        )


@pytest.mark.parametrize(
    'cam_type, dist_param',
    [
        (CameraType.pinhole, {}),
        (CameraType.brown, 'brown_dist_param'),
        (CameraType.opencv, 'opencv_dist_param'),
        (CameraType.fisheye, 'fisheye_dist_param'),
    ],
)
def test_process_alpha(
    cam_type: CameraType,
    dist_param: str,
    frame_args: dict,
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    utm34n_crs: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Test ortho with ``alpha=1`` contains and is similar to ortho with ``alpha=0``."""
    # note that these tests are mostly duplicated by test_camera.test_frame_remap_alpha,
    # but without bounds testing
    dist_param: dict = request.getfixturevalue(dist_param) if dist_param else {}
    camera_alpha1 = create_camera(cam_type, **frame_args, **dist_param, alpha=1.0, distort=False)
    camera_alpha0 = create_camera(cam_type, **frame_args, **dist_param, alpha=0.0, distort=False)
    resolution = (3, 3)

    # create a ref (alpha=1) and test (alpha=0) orthos
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera_alpha1, utm34n_crs, dem_band=1)
    ortho_ref_file = tmp_path.joinpath('ref_ortho.tif')
    ortho.process(ortho_ref_file, resolution, compress=Compress.deflate)

    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera_alpha0, utm34n_crs, dem_band=1)
    ortho_test_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_test_file, resolution, compress=Compress.deflate)

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
        if cam_type is CameraType.pinhole:
            assert np.all(ref_bounds[:2] <= test_bounds[:2]) and np.all(
                ref_bounds[-2:] >= test_bounds[:2]
            )
            assert ref_mask.sum() == test_mask.sum()
        else:
            assert np.all(ref_bounds[:2] < test_bounds[:2]) and np.all(
                ref_bounds[-2:] > test_bounds[:2]
            )
            assert ref_mask.sum() > test_mask.sum()
        ref_mask = ref_mask[ref_win.toslices()]
        assert (ref_mask[test_mask].sum() / test_mask.sum()) > 0.99

        # test similarity of ortho pixels in intersecting ref-test area
        mask = ref_mask & test_mask
        ref_array = ref_array[(slice(0, ref_im.count), *ref_win.toslices())]
        cc = np.corrcoef(ref_array[:, mask].flatten(), test_array[:, mask].flatten())
        assert cc[0, 1] > 0.95


@pytest.mark.parametrize('driver', Driver)
def test_process_driver(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path, driver: Driver):
    """Test the ``Ortho.process()`` ``driver`` argument."""
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, _dem_resolution, driver=driver)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        assert ortho_im.driver.lower() == 'gtiff'
        im_struct = ortho_im.tags(ns='IMAGE_STRUCTURE')
        if driver is Driver.gtiff:
            assert 'LAYOUT' not in im_struct or im_struct['LAYOUT'].lower() != 'cog'
        else:
            assert im_struct['LAYOUT'].lower() == 'cog'


@pytest.mark.parametrize(
    'src_file, write_mask, per_band',
    [
        ('rgb_byte_src_file', True, True),
        ('rgb_byte_src_file', False, True),
        ('rgb_byte_src_file', True, False),
        ('rgb_byte_src_file', False, False),
        ('float_src_file', True, False),
        ('float_src_file', False, False),
    ],
)
def test_process_write_mask_per_band(
    src_file: str,
    float_utm34n_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_crs: str,
    write_mask: bool,
    per_band: bool,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Test ``write_mask=True`` writes an internal ortho mask irrespective of the value of
    ``per_band``.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ortho = Ortho(src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, _dem_resolution, write_mask=write_mask, per_band=per_band)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        mask_flag = MaskFlags.per_dataset if write_mask else MaskFlags.nodata
        assert all([mf[0] == mask_flag for mf in ortho_im.mask_flag_enums])


@pytest.mark.parametrize(
    'write_mask, compress',
    [(None, Compress.jpeg), (None, Compress.deflate), (False, None), (True, None)],
)
def test_process_write_mask(
    rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path, write_mask: bool, compress: Compress
):
    """Test the ``Ortho.process()`` ``write_mask`` argument and its default value interaction with
    compression.
    """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(
        ortho_file, _dem_resolution, compress=compress, write_mask=write_mask
    )
    assert ortho_file.exists()

    if write_mask is None:
        write_mask = True if compress is Compress.jpeg else False
    mask_flag = MaskFlags.per_dataset if write_mask else MaskFlags.nodata

    with rio.open(ortho_file, 'r') as ortho_im:
        assert all([mf[0] == mask_flag for mf in ortho_im.mask_flag_enums])
        assert (
            (ortho_im.nodata is None)
            if write_mask
            else common.nan_equals(ortho_im.nodata, common._nodata_vals[ortho_im.dtypes[0]])
        )


@pytest.mark.parametrize(
    'src_file, dtype',
    [
        ('rgb_byte_src_file', None),
        ('float_src_file', None),
        *[('rgb_byte_src_file', dt) for dt in common._nodata_vals.keys()],
    ],
)
def test_process_dtype(
    float_utm34n_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_crs: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    src_file: str,
    dtype: str,
):
    """Test the ``Ortho.process()`` ``dtype`` argument and its default value interaction with
    the source image dtype.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ortho = Ortho(src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs, dem_band=1)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, _dem_resolution, dtype=dtype)

    assert ortho_file.exists()
    with rio.open(src_file, 'r') as src_im, rio.open(ortho_file, 'r') as ortho_im:
        dtype = dtype or src_im.profile['dtype']
        assert ortho_im.profile['dtype'] == dtype


@pytest.mark.parametrize(
    'compress, dtype', [(None, 'uint8'), (None, 'float32'), *[(c, None) for c in Compress]]
)
def test_process_compress(
    rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path, compress: Compress, dtype: str
):
    """Test the ``Ortho.process()`` ``compress`` argument and its default value interaction with
    ``dtype``.
    """
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, _dem_resolution, dtype=dtype, compress=compress)

    if compress is None:
        compress = Compress.jpeg if dtype == 'uint8' else Compress.deflate

    with rio.open(ortho_file, 'r') as ortho_im:
        assert compress.value in ortho_im.profile['compress'].lower()


@pytest.mark.parametrize('build_ovw, driver', [*product([True, False], Driver)])
def test_process_overview(
    rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path, build_ovw: bool, driver: Driver
):
    """Test overview(s) are created according to the ``build_ovw`` value."""
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, (0.25, 0.25), driver=driver, build_ovw=build_ovw)
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        assert min(ortho_im.shape) >= 512
        if build_ovw:
            assert len(ortho_im.overviews(1)) > 0
        else:
            assert len(ortho_im.overviews(1)) == 0


@pytest.mark.parametrize('driver', Driver)
def test_process_colorinterp(
    ms_float_src_file: Path,
    float_utm34n_dem_file: Path,
    pinhole_camera: Camera,
    utm34n_crs: str,
    tmp_path: Path,
    driver: Driver,
):
    """Test ``Ortho.process()`` copies ``colorinterp`` from source to ortho."""
    ortho = Ortho(ms_float_src_file, float_utm34n_dem_file, pinhole_camera, utm34n_crs, dem_band=1)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, _dem_resolution, driver=driver, compress=Compress.deflate)

    assert ortho_file.exists()
    with rio.open(ms_float_src_file, 'r') as src_im, rio.open(ortho_file, 'r') as ortho_im:
        assert ortho_im.colorinterp == src_im.colorinterp


def test_process_creation_options(rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path):
    """Test ``Ortho.process()`` configures the ortho with ``creation_options``."""
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(
        ortho_file,
        _dem_resolution,
        creation_options=dict(tiled=True, compress='jpeg', jpeg_quality=50),
    )
    assert ortho_file.exists()

    with rio.open(ortho_file, 'r') as ortho_im:
        assert ortho_im.profile['tiled'] == True
        assert ortho_im.profile['compress'].lower() == 'jpeg'
        assert ortho_im.tags(ns='IMAGE_STRUCTURE')['JPEG_QUALITY'] == '50'


@pytest.mark.parametrize(
    'camera',
    ['pinhole_camera', 'brown_camera', 'opencv_camera', 'fisheye_camera', 'rpc_camera_proj'],
)
def test_process_camera(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    camera: str,
    utm34n_crs,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Test ortho image format and content for different cameras."""
    # create Ortho object and process, using the planar DEM band
    camera: Camera = request.getfixturevalue(camera)
    ortho = Ortho(rgb_byte_src_file, float_utm34n_dem_file, camera, utm34n_crs, dem_band=2)
    resolution = (5, 5)
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    ortho.process(ortho_file, resolution, compress=Compress.deflate, interp=Interp.nearest)
    dem_bounds = np.array(array_bounds(*ortho._dem_array.shape, ortho._dem_transform))
    assert ortho_file.exists()

    with rio.open(rgb_byte_src_file, 'r') as src_im, rio.open(ortho_file, 'r') as ortho_im:
        # test ortho bounds and content
        ortho_bounds = np.array(ortho_im.bounds)
        assert np.all(ortho_bounds[:2] >= dem_bounds[:2]) and np.all(
            ortho_bounds[-2:] <= dem_bounds[-2:]
        )

        src_array = src_im.read()
        ortho_array = ortho_im.read()
        ortho_mask = ortho_im.dataset_mask().astype('bool')
        assert np.all(np.unique(src_array) == np.unique(ortho_array[:, ortho_mask]))
        assert src_array.mean() == pytest.approx(ortho_array[:, ortho_mask].mean(), abs=15)
        assert src_array.std() == pytest.approx(ortho_array[:, ortho_mask].std(), abs=15)


def test_process_progress(
    rgb_pinhole_utm34n_ortho: Ortho, tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test ortho progress bar display."""
    # default bar
    ortho_file = tmp_path.joinpath('test_ortho.tif')
    rgb_pinhole_utm34n_ortho.process(ortho_file, _dem_resolution, progress=True)
    cap = capsys.readouterr()
    assert 'blocks' in cap.err and '100%' in cap.err

    # no bar
    rgb_pinhole_utm34n_ortho.process(ortho_file, _dem_resolution, overwrite=True, progress=False)
    cap = capsys.readouterr()
    assert 'blocks' not in cap.err and '100%' not in cap.err

    # custom bar
    desc = 'custom'
    rgb_pinhole_utm34n_ortho.process(
        ortho_file, _dem_resolution, overwrite=True, progress=dict(desc=desc)
    )
    cap = capsys.readouterr()
    assert desc in cap.err


def test_process_ngi(
    ngi_image_files: list[Path],
    ngi_dem_file: Path,
    ngi_legacy_config_file: Path,
    ngi_legacy_csv_file: Path,
    ngi_crs: str,
    tmp_path: Path,
):
    """Test integration and ortho overlap using NGI aerial images."""
    int_param_dict = param_io.read_oty_int_param(ngi_legacy_config_file)
    ext_param_dict = param_io.CsvReader(ngi_legacy_csv_file, crs=ngi_crs).read_ext_param()
    camera = create_camera(**next(iter(int_param_dict.values())))

    ortho_files = []
    for src_file in ngi_image_files:
        ortho_file = tmp_path.joinpath(src_file.stem + '_ORTHO.tif')
        ext_param = ext_param_dict[src_file.stem]
        camera.update(xyz=ext_param['xyz'], opk=ext_param['opk'])
        ortho = Ortho(src_file, ngi_dem_file, camera, crs=ngi_crs)
        ortho.process(ortho_file, resolution=(5, 5))

        assert ortho_file.exists()
        ortho_files.append(ortho_file)

    _validate_ortho_files(ortho_files)


def test_process_odm(
    odm_image_files: list[Path],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_crs: str,
    tmp_path: Path,
):
    """Test integration and ortho overlap using ODM drone images."""
    reader = param_io.OsfmReader(odm_reconstruction_file, crs=odm_crs)
    int_param_dict = reader.read_int_param()
    ext_param_dict = reader.read_ext_param()
    camera = create_camera(**next(iter(int_param_dict.values())))

    ortho_files = []
    for src_file in odm_image_files:
        ortho_file = tmp_path.joinpath(src_file.stem + '_ORTHO.tif')
        ext_param = ext_param_dict[src_file.stem]
        camera.update(xyz=ext_param['xyz'], opk=ext_param['opk'])
        ortho = Ortho(src_file, odm_dem_file, camera, crs=odm_crs)
        ortho.process(ortho_file, resolution=(0.25, 0.25))

        assert ortho_file.exists()
        ortho_files.append(ortho_file)

    _validate_ortho_files(ortho_files, num_ovl_thresh=5)
