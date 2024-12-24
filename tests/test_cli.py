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
import shutil
import tracemalloc
from pathlib import Path

import click
import fsspec
import numpy as np
import pytest
import rasterio as rio
import yaml
from click.testing import CliRunner

from orthority.cli import _ortho, cli, simple_ortho
from orthority.factory import FrameCameras

logger = logging.getLogger(__name__)


@pytest.fixture(scope='session')
def frame_legacy_ngi_cli_str(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_legacy_config_file: Path,
    ngi_legacy_csv_file: Path,
) -> str:
    """``oty frame`` CLI string to orthorectify an NGI image with legacy format interior and
    exterior params.
    """
    return (
        f'frame --dem {ngi_dem_file} --int-param {ngi_legacy_config_file} '
        f'--ext-param {ngi_legacy_csv_file} {ngi_image_file}'
    )


@pytest.fixture(scope='session')
def sharpen_cli_str(pan_file: Path, ms_file: Path) -> str:
    """``oty sharpen`` CLI string to pan-sharpen test images."""
    return f'sharpen --pan {pan_file} --multispectral {ms_file}'


def test_oty_verbosity(
    frame_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty -v frame`` generates debug logs."""
    cli_str = f'-v {frame_legacy_ngi_cli_str} --res 50 --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert 'DEBUG:' in result.stdout


def test_frame_help(runner: CliRunner):
    """Test ``oty frame --help``."""
    result = runner.invoke(cli, 'frame --help'.split())
    assert result.exit_code == 0, result.stdout
    assert len(result.stdout) > 0


def test_frame_src_file_wildcard(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with a SOURCE image wildcard (that is not expanded by the console)."""
    src_wildcard = ngi_image_file.parent.joinpath('*RGB.tif')
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --res 30 --out-dir {tmp_path}'
        f' {src_wildcard.as_posix()}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == len([*src_wildcard.parent.glob(src_wildcard.name)])


def test_frame_src_file_not_found_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with a non-existing SOURCE image raises an error ."""
    # use a source path whose file name is in the exterior params, but the path doesn't exist
    src_file = f'unknown/{ngi_image_file.name}'
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {src_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        'SOURCE' in result.stdout and 'No such file' in result.stdout and src_file in result.stdout
    )


def test_frame_dem_missing_error(
    ngi_image_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` without ``--dem`` raises an error."""
    cli_str = (
        f'frame --int-param {ngi_oty_int_param_file} --ext-param {ngi_oty_ext_param_file} '
        f'--out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dem' in result.stdout and 'missing' in result.stdout.lower()


def test_frame_dem_not_found_error(
    ngi_image_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with a non-existing ``--dem`` raises an error ."""
    dem_file = 'unknown.tif'
    cli_str = (
        f'frame --dem {dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--dem' in result.stdout and 'No such file' in result.stdout and dem_file in result.stdout
    )


def test_frame_int_param_missing_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` without ``--int-param`` raises an error ."""
    cli_str = (
        f'frame --dem {ngi_dem_file} --ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} '
        f'{ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--int-param' in result.stdout and 'missing' in result.stdout.lower()


def test_frame_int_param_not_found_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with a non-existing ``--int-param`` raises an error ."""
    int_param_file = 'unknown.yaml'
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {int_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'No such file' in result.stdout and int_param_file in result.stdout


def test_frame_int_param_ext_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with an unrecognised ``--int-param`` extension raises an error."""
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_oty_ext_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'interior parameter' in result.stdout and 'not supported' in result.stdout


def test_frame_int_param_invalid_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    rpc_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with an invalid ``--int-param`` file raises an error."""
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {rpc_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'Could not parse' in result.stdout and rpc_param_file.name in result.stdout


def test_frame_ext_param_missing_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` without ``--ext-param`` raises an error ."""
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} --out-dir {tmp_path} '
        f'{ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--ext-param' in result.stdout and 'missing' in result.stdout.lower()


def test_frame_ext_param_not_found_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with a non-existing ``--ext-param`` raises an error ."""
    ext_param_file = 'unknown.geojson'
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'No such file' in result.stdout and ext_param_file in result.stdout


def test_frame_ext_param_ext_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with an unrecognised ``--ext-param`` extension raises an error."""
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_oty_int_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'exterior parameter' in result.stdout and 'not supported' in result.stdout


def test_frame_ext_param_invalid_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    gcp_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with an invalid ``--ext-param`` file raises an error."""
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {gcp_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'Could not parse' in result.stdout and gcp_file.name in result.stdout


def test_frame_crs_src(
    frame_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame`` reads the world / ortho CRS from a projected source image."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ngi_image_file, 'r') as src_im, rio.open(ortho_files[0], 'r') as ortho_im:
        assert src_im.crs == ortho_im.crs


def test_frame_crs_auto(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_lla_rpy_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` auto-determines the CRS for LLA-RPY CSV format exterior parameters."""
    cli_str = (
        f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_lla_rpy_csv_file} --out-dir {tmp_path} --res 5 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_frame_crs_cli(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` uses a CRS specified with ``--crs``."""
    # use odm_xyz_opk_csv_file exterior params so there is no auto-crs
    cli_str = (
        f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_xyz_opk_csv_file} --out-dir {tmp_path} --crs {odm_crs} --res 5 '
        f'{odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_frame_crs_file(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` uses a text file CRS specified with ``--crs``."""
    # use odm_xyz_opk_csv_file exterior params so there is no auto-crs
    crs_file = tmp_path.joinpath('test_crs.txt')
    crs_file.write_text(odm_crs)
    # use file:// prefix to test differentiation with CRS string e.g. 'EPSG:...'
    cli_str = (
        f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_xyz_opk_csv_file} --out-dir {tmp_path} --crs {crs_file} --res 5 '
        f'{odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_frame_crs_prj(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` reads the CRS in a CSV exterior parameter .prj file."""
    # copy csv file to tmp_path and create .prj file
    csv_file = tmp_path.joinpath(odm_xyz_opk_csv_file.name)
    csv_file.write_text(odm_xyz_opk_csv_file.read_text())
    prj_file = csv_file.with_suffix('.prj')
    prj_file.write_text(odm_crs)

    # create ortho & test
    cli_str = (
        f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} --ext-param {csv_file} '
        f'--out-dir {tmp_path} --res 5 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_frame_crs_missing_error(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` raises an error when ``--crs`` is needed but not passed."""
    cli_str = (
        f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_xyz_opk_csv_file} --out-dir {tmp_path} {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'missing' in result.stdout.lower()


def test_frame_crs_geographic_error(
    frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame`` raises an error when ``--crs`` is geographic."""
    cli_str = frame_legacy_ngi_cli_str + ' --crs EPSG:4326'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'projected' in result.stdout


def test_frame_crs_invalid_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame`` raises an error when ``--crs`` is invalid."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --crs unknown'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_resolution_square(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --res`` with square resolution."""
    resolution = (96.0, 96.0)
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res {resolution[0]}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.res == resolution


def test_frame_resolution_non_square(
    frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame --res`` with non-square resolution."""
    resolution = (48.0, 96.0)
    cli_str = (
        frame_legacy_ngi_cli_str
        + f' --out-dir {tmp_path} --res {resolution[0]} --res {resolution[1]}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.res == resolution


def test_frame_resolution_auto(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame`` generates an ortho without the ``--res`` option."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test_frame_dem_band(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --dem-band`` with valid band."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --dem-band 1'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test_frame_dem_band_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --dem-band`` raises an error with an out of range band."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --dem-band 2'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dem-band' in result.stdout


def test_frame_interp(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --interp`` by comparing ``nearest`` to ``average`` interpolation orthos."""
    ortho_std = []
    for interp in ['average', 'nearest']:
        # create ortho
        out_dir = tmp_path.joinpath(interp)
        out_dir.mkdir()
        cli_str = (
            frame_legacy_ngi_cli_str
            + f' --out-dir {out_dir} --res 24 --compress deflate --interp {interp}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho std dev
        with rio.open(ortho_files[0], 'r') as im:
            array = im.read(masked=True)
        ortho_std.append(float(array.std()))

    # compare orthos
    assert ortho_std[0] != pytest.approx(ortho_std[1], abs=0.1)


def test_frame_interp_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --interp`` raises an error with an invalid interpolation value."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --interp other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--interp' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_dem_interp(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --dem-interp`` by comparing ``nearest`` to ``average`` DEM interpolation
    orthos.
    """
    ortho_std = []
    for dem_interp in ['average', 'nearest']:
        # create ortho
        out_dir = tmp_path.joinpath(dem_interp)
        out_dir.mkdir()
        cli_str = (
            frame_legacy_ngi_cli_str
            + f' --out-dir {out_dir} --res 30 --compress deflate --dem-interp {dem_interp}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho std dev
        with rio.open(ortho_files[0], 'r') as im:
            array = im.read(masked=True)
        ortho_std.append(float(array.std()))

    # compare orthos
    assert ortho_std[0] != pytest.approx(ortho_std[1], abs=0.01)


def test_frame_dem_interp_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --dem-interp`` raises an error with an invalid interpolation value."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --dem-interp other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dem-interp' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_per_band(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_legacy_config_file: Path,
    ngi_legacy_csv_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame --per-band`` by comparing memory usage between ``--per-band`` and ``--no-
    per-band``.
    """
    # make a temporary 4 band float64 source image from ngi_image_file (for --per-band to make
    # a measurable memory difference, the source image needs to be relatively large, have many
    # bands and/or a 'big' dtype.)
    src_file = tmp_path.joinpath(ngi_image_file.name)
    with rio.open(ngi_image_file, 'r') as ngi_im:
        array = ngi_im.read(out_dtype='float32')
        array = np.stack((*array, array[0]), axis=0)
        profile = ngi_im.profile
        profile.update(
            count=array.shape[0], dtype=array.dtype, compress='deflate', photometric='minisblack'
        )
        with rio.open(src_file, 'w', **profile) as src_im:
            src_im.write(array)

    # compare memory usage between --no-per-band and --per-band
    cli_str = (
        f'frame --dem {ngi_dem_file} --int-param {ngi_legacy_config_file} '
        f'--ext-param {ngi_legacy_csv_file} --res 30 --compress deflate {src_file}'
    )
    mem_peaks = []
    try:
        tracemalloc.start()
        for per_band in ['no-per-band', 'per-band']:
            out_dir = tmp_path.joinpath(per_band)
            out_dir.mkdir()
            cli_str = cli_str + f' --out-dir {out_dir} --{per_band}'

            # find peak memory used by the command
            mem_start = tracemalloc.get_traced_memory()
            result = runner.invoke(cli, cli_str.split())
            mem_end = tracemalloc.get_traced_memory()
            mem_peaks.append(mem_end[1] - mem_start[0])
            tracemalloc.clear_traces()  # clears the peak

            assert result.exit_code == 0, result.stdout
            ortho_files = [*out_dir.glob('*_ORTHO.tif')]
            assert len(ortho_files) == 1
    finally:
        tracemalloc.stop()

    assert mem_peaks[1] < mem_peaks[0]


def test_frame_full_remap(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame --full-remap`` by comparing ``--full-remap`` and ``--no-full-remap``
    orthos.
    """
    ortho_arrays = []
    for full_remap in ['full-remap', 'no-full-remap']:
        # create ortho
        out_dir = tmp_path.joinpath(full_remap)
        out_dir.mkdir()
        cli_str = (
            f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
            f'--ext-param {odm_reconstruction_file} --out-dir {out_dir} --res 1 '
            f'--compress deflate --{full_remap} {odm_image_file}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho
        with rio.open(ortho_files[0], 'r') as im:
            array = im.read(masked=True)
        ortho_arrays.append(array)

    # compare ortho masks
    assert np.any(ortho_arrays[1].mask != ortho_arrays[0].mask)


def test_frame_alpha(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame --alpha`` by comparing ``--alpha 0`` and ``--alpha 1`` orthos."""
    ortho_bounds = []
    for alpha in [0, 1]:
        # create ortho
        out_dir = tmp_path.joinpath(str(alpha))
        out_dir.mkdir()
        cli_str = (
            f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
            f'--ext-param {odm_reconstruction_file} --out-dir {out_dir} --res 1 '
            f'--compress deflate --no-full-remap --alpha {alpha} {odm_image_file}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho bounds
        with rio.open(ortho_files[0], 'r') as im:
            ortho_bounds.append(im.bounds)

    # compare ortho bounds
    assert ortho_bounds[0][:2] > ortho_bounds[1][:2] and ortho_bounds[0][-2:] < ortho_bounds[1][-2:]


def test_frame_alpha_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --alpha`` raises an error with an invalid alpha value."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --no-full-remap --alpha 2'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--alpha' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_lla_crs(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_lla_rpy_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame --lla-crs`` by comparing orthos created with different ``--lla-crs``
    values.
    """
    # TODO: this doesn't work with PyPI rasterio <1.3.9
    ortho_bounds = []
    res = 5
    for i, lla_crs in enumerate(['EPSG:4326+4326', 'EPSG:4326+3855']):
        # create ortho
        out_dir = tmp_path.joinpath(str(i))
        out_dir.mkdir()
        cli_str = (
            f'frame --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
            f'--ext-param {odm_lla_rpy_csv_file} --out-dir {out_dir} --res {res} '
            f'--crs {odm_crs}+3855 --lla-crs {lla_crs} {odm_image_file}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho bounds
        with rio.open(ortho_files[0], 'r') as im:
            ortho_bounds.append(im.bounds)

    # compare ortho bounds
    assert ortho_bounds[1] != pytest.approx(ortho_bounds[0], abs=res)


def test_frame_lla_crs_projected_error(
    frame_legacy_ngi_cli_str: str, ngi_crs: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame --lla-crs`` raises an error with a projected CRS value."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --lla-crs {ngi_crs}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--lla-crs' in result.stdout and 'geographic' in result.stdout


def test_frame_lla_crs_invalid_error(
    frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame`` raises an error when ``--lla_crs`` is invalid."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --lla-crs unknown'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--lla-crs' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_radians(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_xyz_opk_csv_file: Path,
    ngi_xyz_opk_radians_csv_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame --radians`` by comparing orthos created with exterior parameter orientation
    angles in radians and degrees.
    """
    ortho_arrays = []
    for radians, ext_param_file in zip(
        ['degrees', 'radians'], [ngi_xyz_opk_csv_file, ngi_xyz_opk_radians_csv_file]
    ):
        # create ortho
        out_dir = tmp_path.joinpath(radians)
        out_dir.mkdir()
        cli_str = (
            f'frame --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
            f'--ext-param {ext_param_file} --out-dir {out_dir} --res 24 '
            f'--{radians} {ngi_image_file}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho
        with rio.open(ortho_files[0], 'r') as im:
            ortho_arrays.append(im.read(1))

    # compare ortho similarity
    assert ortho_arrays[1] == pytest.approx(ortho_arrays[0], abs=1)


def test_frame_driver(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --driver`` creates an ortho with the correct driver."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --driver cog'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.driver.lower() == 'gtiff'
        assert im.tags(ns='IMAGE_STRUCTURE')['LAYOUT'].lower() == 'cog'


def test_frame_driver_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --diver`` with an invalid driver raises an error."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --driver other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--driver' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_write_mask(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --write-mask`` writes an internal mask to the ortho with
    ``compress=deflate``.
    """
    cli_str = (
        frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress deflate --write-mask --res 24'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.profile['compress'] == 'deflate'
        assert all([mf[0] == rio.enums.MaskFlags.per_dataset for mf in im.mask_flag_enums])


def test_frame_dtype(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --dtype`` creates an ortho with the correct dtype."""
    dtype = 'float32'
    cli_str = (
        frame_legacy_ngi_cli_str
        + f' --out-dir {tmp_path} --compress deflate --dtype {dtype} --res 24'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.profile['dtype'] == dtype


def test_frame_dtype_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --dtype`` with an invalid dtype raises an error."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress deflate --dtype int32'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dtype' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_compress(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --compress`` creates an ortho with the correct compression."""
    compress = 'lzw'
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress {compress} --res 24'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.profile['compress'] == compress


def test_frame_compress_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --compress`` with an invalid compression raises an error."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--compress' in result.stdout and 'invalid' in result.stdout.lower()


def test_frame_build_ovw(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --build-ovw`` builds overviews."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 5 --build-ovw'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert len(im.overviews(1)) > 0


def test_frame_no_build_ovw(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --no-build-ovw`` does not build overviews."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 5 --no-build-ovw'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert len(im.overviews(1)) == 0


def test_frame_creation_option(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame --creation-option`` creates an ortho with the correct creation options."""
    cli_str = frame_legacy_ngi_cli_str + (
        f' --out-dir {tmp_path} -co tiled=yes -co compress=jpeg -co jpeg_quality=50'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.profile['tiled'] == True
        assert im.profile['compress'].lower() == 'jpeg'
        assert im.tags(ns='IMAGE_STRUCTURE')['JPEG_QUALITY'] == '50'


def test_frame_creation_option_error(
    frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame --creation-option`` raises an error with invalid syntax."""
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} -co compress jpeg'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--creation-option' in result.stdout and 'invalid' in result.stdout.lower()


@pytest.mark.parametrize(
    'int_param_file, ext_param_file',
    [
        ('ngi_oty_int_param_file', 'ngi_oty_ext_param_file'),
        ('ngi_legacy_config_file', 'ngi_xyz_opk_csv_file'),
        ('odm_reconstruction_file', 'odm_reconstruction_file'),
    ],
)
def test_frame_export_params(
    int_param_file: str,
    ext_param_file: str,
    tmp_path: Path,
    runner: CliRunner,
    request: pytest.FixtureRequest,
):
    """Test ``oty frame --export-params`` exports interior & exterior parameters provided in
    different formats (with no ``--dem-file``).
    """
    # NOTE this doubles as a test of reading params in different formats
    int_param_file: Path = request.getfixturevalue(int_param_file)
    ext_param_file: Path = request.getfixturevalue(ext_param_file)
    cli_str = (
        f'frame --int-param {int_param_file} --ext-param {ext_param_file} --out-dir {tmp_path} '
        f'--export-params'
    )

    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    int_param_file = tmp_path.joinpath('int_param.yaml')
    ext_param_file = tmp_path.joinpath('ext_param.geojson')
    assert int_param_file.exists() and ext_param_file.exists()


def test_frame_out_dir_error(frame_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty frame`` raises an error when --out-dir does not exist."""
    out_dir = 'unknown'
    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {out_dir} --res 24'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--out-dir' in result.stdout and 'directory' in result.stdout and out_dir in result.stdout
    )


def test_frame_overwrite(
    frame_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame --overwrite`` overwrites an existing ortho."""
    ortho_file = tmp_path.joinpath(ngi_image_file.stem + '_ORTHO.tif')
    ortho_file.touch()

    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --overwrite'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1
    assert ortho_file == ortho_files[0]

    with rio.open(ortho_files[0], 'r'):
        pass


def test_frame_overwrite_error(
    frame_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty frame`` raises an error when the ortho file already exists."""
    ortho_file = tmp_path.joinpath(ngi_image_file.stem + '_ORTHO.tif')
    ortho_file.touch()

    cli_str = frame_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'exists' in result.stdout and ortho_file.name in result.stdout


def test_frame_urls(
    ngi_image_url: str,
    ngi_dem_url: str,
    ngi_oty_int_param_url: str,
    ngi_oty_ext_param_url: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty frame`` with source, DEM & parameter file URLs."""
    cli_str = (
        f'frame --dem {ngi_dem_url} --int-param {ngi_oty_int_param_url} '
        f'--ext-param {ngi_oty_ext_param_url} --out-dir {tmp_path} {ngi_image_url}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test_exif_source_error(
    ngi_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif`` raises an error with a non-EXIF source image."""
    cli_str = f'exif --dem {ngi_dem_file} --out-dir {tmp_path} {ngi_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        'SOURCE' in result.stdout
        and 'tags' in result.stdout
        and ngi_image_file.name in result.stdout
    )


def test_exif_source_not_found_error(
    ngi_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif`` raises an error with a non-existing source image."""
    src_file = 'unknown.tif'
    cli_str = f'exif --dem {ngi_dem_file} --out-dir {tmp_path} {src_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        'SOURCE' in result.stdout
        and 'No such file' in result.stdout
        and src_file in result.stdout.lower()
    )


def test_exif_crs(
    odm_image_file: Path, odm_dem_file: Path, odm_crs: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif --crs`` creates orthos with the correct CRS."""
    crs = odm_crs + '+3855'  # differentiate from auto CRS
    cli_str = f'exif --dem {odm_dem_file} --out-dir {tmp_path} --res 5 --crs {crs} {odm_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as ortho_im:
        assert ortho_im.crs == rio.CRS.from_string(crs)


def test_exif_crs_auto(
    odm_image_file: Path, odm_dem_file: Path, odm_crs: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif`` auto determines the CRS."""
    cli_str = f'exif --dem {odm_dem_file} --out-dir {tmp_path} --res 5 {odm_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as ortho_im:
        assert ortho_im.crs == rio.CRS.from_string(odm_crs)


def test_exif_crs_geographic_error(
    odm_image_file: Path, odm_dem_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif`` raises an error when ``--crs`` is geographic."""
    cli_str = f'exif --dem {odm_dem_file} --out-dir {tmp_path} --crs EPSG:4326 {odm_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'projected' in result.stdout


def test_exif_crs_invalid_error(
    odm_image_file: Path, odm_dem_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif`` raises an error when ``--crs`` is invalid."""
    cli_str = f'exif --dem {odm_dem_file} --out-dir {tmp_path} --crs unknown {odm_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'invalid' in result.stdout.lower()


def test_exif_option(
    odm_image_file: Path, odm_dem_file: Path, odm_crs: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif`` passes through a ``--res`` option."""
    res = 5
    cli_str = f'exif --dem {odm_dem_file} --out-dir {tmp_path} --res {res} {odm_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as ortho_im:
        assert ortho_im.res == (res, res)


def test_exif_lla_crs(
    odm_image_file: Path,
    odm_dem_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty exif --lla-crs`` by comparing orthos created with different ``--lla-crs``
    values.
    """
    ortho_bounds = []
    res = 5
    for i, lla_crs in enumerate(['EPSG:4979', 'EPSG:4326+3855']):
        # create ortho
        out_dir = tmp_path.joinpath(str(i))
        out_dir.mkdir()
        cli_str = (
            f'exif --dem {odm_dem_file} --out-dir {out_dir} --res 5 --crs {odm_crs}+3855 '
            f'--lla-crs {lla_crs} {odm_image_file}'
        )

        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho bounds
        with rio.open(ortho_files[0], 'r') as im:
            ortho_bounds.append(im.bounds)

    # compare ortho bounds
    assert ortho_bounds[1] != pytest.approx(ortho_bounds[0], abs=res / 2)


def test_odm_dataset_dir(
    odm_dataset_dir: Path, odm_image_files: list[Path], tmp_path: Path, runner: CliRunner
):
    """Test ``oty odm`` creates orthos in ``<--dataset-dir>/orthority`` sub-folder."""
    shutil.copytree(odm_dataset_dir, tmp_path, dirs_exist_ok=True)  # copy test data to tmp_path
    cli_str = f'odm --dataset-dir {tmp_path} --res 5 --overwrite'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.joinpath('orthority').glob('*_ORTHO.tif')]
    src_files = [*tmp_path.joinpath('images').glob('*.tif')]
    assert len(ortho_files) == len(src_files)


def test_odm_out_dir(
    odm_dataset_dir: Path, odm_image_files: list[Path], tmp_path: Path, runner: CliRunner
):
    """Test ``oty odm --out-dir`` creates orthos in the ``--out-dir`` folder."""
    cli_str = f'odm --dataset-dir {odm_dataset_dir} --res 5 --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    src_files = [*odm_dataset_dir.joinpath('images').glob('*.tif')]
    assert len(ortho_files) == len(src_files)


def test_odm_option(
    odm_dataset_dir: Path, odm_dem_file: Path, odm_crs: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty odm`` passes through a ``--res`` option."""
    res = 5
    cli_str = f'odm --dataset-dir {odm_dataset_dir} --res {res} --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]

    with rio.open(odm_dem_file, 'r') as dem_im:
        dem_crs = dem_im.crs
    for ortho_file in ortho_files:
        with rio.open(ortho_file, 'r') as ortho_im:
            assert ortho_im.crs == dem_crs
            assert ortho_im.res == (res, res)


def test_odm_dataset_dir_error(tmp_path: Path, runner: CliRunner):
    """Test ``oty odm`` raises an error with a non-existing --dataset-dir."""
    dataset_dir = 'unknown'
    cli_str = f'odm --dataset-dir {dataset_dir}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--dataset-dir' in result.stdout
        and 'directory' in result.stdout
        and dataset_dir in result.stdout
    )


def test_rpc_image(rpc_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty rpc`` with a source image that has RPC metadata."""
    cli_str = f'rpc --dem {ngi_dem_file} --out-dir {tmp_path} {rpc_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout

    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1
    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_epsg(4979)


def test_rpc_image_no_rpc_error(
    ngi_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty rpc`` raises an error with a source image that has no RPC metadata."""
    cli_str = f'rpc --dem {ngi_dem_file} --out-dir {tmp_path} {ngi_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        'SOURCE' in result.stdout
        and ngi_image_file.name in result.stdout
        and 'No RPC parameters' in result.stdout
    )


def test_rpc_image_not_found_error(ngi_dem_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty rpc`` raises an error with a non-existent source image."""
    src_file = 'unknown.tif'
    cli_str = f'rpc --dem {ngi_dem_file} --out-dir {tmp_path} {src_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        'SOURCE' in result.stdout and 'No such file' in result.stdout and src_file in result.stdout
    )


def test_rpc_param_file(
    rpc_image_file: Path,
    ngi_dem_file: Path,
    rpc_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty rpc`` with ``--rpc-param`` uses the RPC parameter file."""
    # create source image with no RPCs (and same name as rpc_image_file)
    src_file = tmp_path.joinpath(rpc_image_file.name)
    with rio.open(rpc_image_file, 'r') as rpc_im:
        profile = rpc_im.profile
        profile.pop('rpcs', None)  # make sure no rpcs
        with rio.open(src_file, 'w', **profile) as no_rpc_im:
            no_rpc_im.write(rpc_im.read())

    # orthorectify
    cli_str = (
        f'rpc --dem {ngi_dem_file} --rpc-param {rpc_param_file} --out-dir {tmp_path} {src_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout

    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1
    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_epsg(4979)


def test_rpc_param_file_invalid_error(
    rpc_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty rpc`` with ``--rpc-param`` raises an error with an invalid parameter file."""
    cli_str = (
        f'rpc --dem {ngi_dem_file} --rpc-param {ngi_oty_int_param_file} --out-dir {tmp_path}'
        f' {rpc_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--rpc-param' in result.stdout
        and 'Could not parse' in result.stdout
        and ngi_oty_int_param_file.name in result.stdout
    )


def test_rpc_param_file_not_found_error(
    rpc_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty rpc`` with ``--rpc-param`` raises an error with a non-existent parameter file."""
    rpc_param_file = 'unknown.yaml'
    cli_str = (
        f'rpc --dem {ngi_dem_file} --rpc-param {rpc_param_file} --out-dir {tmp_path}'
        f' {rpc_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--rpc-param' in result.stdout
        and 'No such file' in result.stdout
        and rpc_param_file in result.stdout
    )


def test_rpc_image_crs(rpc_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty rpc`` with ``--crs`` and a source image with RPC metadata, generates orthos with
    the correct CRS.
    """
    crs = 'EPSG:3857'
    cli_str = f'rpc --dem {ngi_dem_file} --crs {crs} --res 30 --out-dir {tmp_path} {rpc_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(crs)


def _test_rpc_param_file_crs(
    rpc_image_file: Path,
    ngi_dem_file: Path,
    rpc_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty rpc`` with ``--crs`` and a ``--rpc-param`` RPC parameter file, generates orthos
    with the correct CRS.
    """
    # TODO: When this is run directly after test_rpc_image_crs in a conda environment with
    #  rasterio 1.4.3 and libgdal 1.3.10, there is an intermittent seg fault.  Other than seeing
    #  that this happens in PROJ, I have not gotten far with debugging.
    crs = 'EPSG:3857'
    cli_str = (
        f'rpc --dem {ngi_dem_file} --rpc-param {rpc_param_file} --crs {crs} --res 30 '
        f'--out-dir {tmp_path} {rpc_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(crs)


def test_rpc_option(rpc_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty rpc`` passes through a ``--res`` option."""
    res = 0.0005  # degrees
    cli_str = f'rpc --dem {ngi_dem_file} --res {res} --out-dir {tmp_path} {rpc_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.res == (res, res)


def test_rpc_gcp_refine(
    rpc_image_file: Path, ngi_dem_file: Path, gcp_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty rpc`` with the ``--gcp-refine`` option by comparing orthos with and without
    refinement.
    """
    # create source image with no GCPs (and same name as rpc_image_file)
    no_gcp_file = tmp_path.joinpath(rpc_image_file.name)
    with rio.open(rpc_image_file, 'r') as gcp_im:
        profile = gcp_im.profile
        profile.update(rpcs=gcp_im.rpcs)
        profile.pop('gcps', None)  # make sure no GCPs
        with rio.open(no_gcp_file, 'w', **profile) as no_gcp_im:
            no_gcp_im.write(gcp_im.read())

    ortho_bounds = []
    for i, (gcp_refine_str, src_file) in enumerate(
        zip(
            ['', '--gcp-refine tags', f'--gcp-refine {gcp_file}'],
            [rpc_image_file, rpc_image_file, no_gcp_file],
        )
    ):
        # create ortho
        out_dir = tmp_path.joinpath(str(i))
        out_dir.mkdir()
        cli_str = f"rpc --dem {ngi_dem_file} {gcp_refine_str} --out-dir {out_dir} -o {src_file}"
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho bounds
        with rio.open(ortho_files[0], 'r') as im:
            ortho_bounds.append(im.bounds)

    # compare ortho bounds
    assert ortho_bounds[1] != pytest.approx(ortho_bounds[0], abs=1e-4)
    assert ortho_bounds[2] != pytest.approx(ortho_bounds[0], abs=1e-4)


def test_rpc_gcp_refine_file_invalid_error(
    rpc_image_file: Path,
    ngi_dem_file: Path,
    odm_reconstruction_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty rpc`` with ``--gcp-refine`` raises an error with an invalid GCP file."""
    cli_str = (
        f'rpc --dem {ngi_dem_file} --gcp-refine {odm_reconstruction_file} --out-dir {tmp_path}'
        f' {rpc_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--gcp-refine' in result.stdout
        and 'Could not parse' in result.stdout
        and odm_reconstruction_file.name in result.stdout
    )


def test_rpc_gcp_refine_file_not_found_error(
    rpc_image_file: Path,
    ngi_dem_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty rpc`` with ``--gcp-refine`` raises an error with a non-existent GCP file."""
    cli_str = f'rpc --dem {ngi_dem_file} --gcp-refine unknown --out-dir {tmp_path} {rpc_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--gcp-refine' in result.stdout
        and 'No such file' in result.stdout
        and 'unknown' in result.stdout
    )


def test_rpc_refine_method(
    rpc_image_file: Path, ngi_dem_file: Path, gcp_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty rpc`` with the ``--refine-method`` option by comparing orthos refined with
    different methods.
    """
    ortho_bounds = []
    for refine_method in ['shift', 'shift-drift']:
        # create ortho
        out_dir = tmp_path.joinpath(refine_method)
        out_dir.mkdir()
        cli_str = (
            f"rpc --dem {ngi_dem_file} --gcp-refine tags --refine-method {refine_method} "
            f"--out-dir {out_dir} -o {rpc_image_file}"
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        ortho_files = [*out_dir.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # read ortho bounds
        with rio.open(ortho_files[0], 'r') as im:
            ortho_bounds.append(im.bounds)

    # compare ortho bounds
    assert ortho_bounds[1] != pytest.approx(ortho_bounds[0], abs=1e-6)


def test_sharpen(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` generates an output image."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()
    with rio.open(out_file, 'r'):
        pass


def test_sharpen_pan_missing_error(ms_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` without --pan raises an error."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = f'sharpen --multispectral {ms_file} --out-file {out_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--pan' in result.stdout and 'missing' in result.stdout.lower()


def test_sharpen_pan_not_found_error(ms_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` raises an error with a non-existent --pan file."""
    pan_file = 'unknown.tif'
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = f'sharpen --pan {pan_file} --multispectral {ms_file} --out-file {out_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--pan' in result.stdout and 'No such file' in result.stdout and pan_file in result.stdout
    )


def test_sharpen_ms_missing_error(pan_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` without --multispectral raises an error."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = f'sharpen --pan {pan_file} --out-file {out_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--multispectral' in result.stdout and 'missing' in result.stdout.lower()


def test_sharpen_ms_not_found_error(pan_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` raises an error with a non-existent --multispectral file."""
    ms_file = 'unknown.tif'
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = f'sharpen --pan {pan_file} --multispectral {ms_file} --out-file {out_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert (
        '--multispectral' in result.stdout
        and 'No such file' in result.stdout
        and ms_file in result.stdout
    )


def test_sharpen_out_file_missing_error(pan_file: Path, ms_file: Path, runner: CliRunner):
    """Test ``oty sharpen`` without --out-file raises an error."""
    cli_str = f'sharpen --pan {pan_file} --multispectral {ms_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--out-file' in result.stdout and 'missing' in result.stdout.lower()


def test_sharpen_pan_index_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` raises an error when --pan-index is out of range."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --pan-index 2'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--pan-index' in result.stdout and 'out of range' in result.stdout


def test_sharpen_ms_index(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --ms-index defines MS bands for pan-sharpening."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --ms-index 1 --ms-index 2'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()
    with rio.open(out_file, 'r') as out_im:
        assert out_im.count == 2


def test_sharpen_ms_index_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` raises an error when --ms-index is out of range."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --ms-index 1 --ms-index 0'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--ms-index' in result.stdout and 'out of range' in result.stdout


def test_sharpen_weight(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --weight sets MS to pan weights by comparing output images sharpened
    with different weights.
    """
    out_filenames = ['pan_sharp1.tif', 'pan_sharp2.tif']
    weight_strs = [
        '--ms-index 1 --ms-index 2 --weight 1 --weight 1',
        '--ms-index 1 --ms-index 2 --weight 1 --weight 2',
    ]
    out_arrays = []
    for out_filename, weight_str in zip(out_filenames, weight_strs):
        out_file = tmp_path.joinpath(out_filename)
        cli_str = sharpen_cli_str + f' --out-file {out_file} {weight_str}'
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        assert out_file.exists()

        with rio.open(out_file, 'r') as im:
            out_arrays.append(im.read())

    assert np.any(out_arrays[0] != out_arrays[1])


def test_sharpen_weight_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` raises an error when the --weight's are invalid."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --weight 1 --weight 1'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'weights' in result.stdout and 'same number' in result.stdout

    cli_str = sharpen_cli_str + f' --out-file {out_file} --weight 1 --weight -1'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'Weight values' in result.stdout


def test_sharpen_interp(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --interp by comparing output files sharpened with ``nearest`` and
    ``bilinear`` interpolation.
    """
    out_arrays = []
    for interp in ['bilinear', 'nearest']:
        out_file = tmp_path.joinpath(f'{interp}.tif')
        cli_str = sharpen_cli_str + f' --out-file {out_file} --interp {interp}'
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0, result.stdout
        assert out_file.exists()

        with rio.open(out_file, 'r') as im:
            out_arrays.append(im.read())

    assert np.any(out_arrays[0] != out_arrays[1])


def test_sharpen_interp_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --interp raises an error with an invalid interpolation value."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --interp other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--interp' in result.stdout and 'invalid' in result.stdout.lower()


def test_sharpen_driver(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --driver creates a pan-sharpened file with the correct driver."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --driver cog'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert out_im.driver.lower() == 'gtiff'
        assert out_im.tags(ns='IMAGE_STRUCTURE')['LAYOUT'].lower() == 'cog'


def test_sharpen_driver_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --driver with an invalid diver raises an error."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --driver other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--driver' in result.stdout and 'invalid' in result.stdout.lower()


def test_sharpen_write_mask(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --write-mask writes an internal mask to the pan-sharpened file with
    deflate compression.
    """
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --compress deflate --write-mask'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert out_im.profile['compress'] == 'deflate'
        assert all([mf[0] == rio.enums.MaskFlags.per_dataset for mf in out_im.mask_flag_enums])


def test_sharpen_dtype(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --dtype creates a pan-sharpened file with the correct dtype."""
    dtype = 'float32'
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --compress deflate --dtype {dtype}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert out_im.profile['dtype'] == dtype


def test_sharpen_dtype_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --dtype with an invalid dtype raises an error."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --compress deflate --dtype int32'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dtype' in result.stdout and 'invalid' in result.stdout.lower()


def test_sharpen_compress(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --compress creates a pan-sharpened file with the correct compression."""
    compress = 'lzw'
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --compress {compress}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert out_im.profile['compress'] == compress


def test_sharpen_compress_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --compress with an invalid compression raises an error."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --compress other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--compress' in result.stdout and 'invalid' in result.stdout.lower()


def test_sharpen_build_ovw(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --build-ovw builds overviews."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --build-ovw'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert len(out_im.overviews(1)) > 0


def test_sharpen_no_build_ovw(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --no-build-ovw does not build overviews."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + f' --out-file {out_file} --no-build-ovw'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert len(out_im.overviews(1)) == 0


def test_sharpen_creation_option(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --creation-option creates a pan-sharpened file with the correct
    creation options.
    """
    out_file = tmp_path.joinpath('pan_sharp.tif')
    cli_str = sharpen_cli_str + (
        f' --out-file {out_file} -co tiled=yes -co compress=jpeg -co jpeg_quality=50'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r') as out_im:
        assert out_im.profile['tiled'] == True
        assert out_im.profile['compress'].lower() == 'jpeg'
        assert out_im.tags(ns='IMAGE_STRUCTURE')['JPEG_QUALITY'] == '50'


def test_sharpen_overwrite(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` --overwrite overwrites an existing pan-sharpened file."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    out_file.touch()
    cli_str = sharpen_cli_str + f' --out-file {out_file} --overwrite'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()

    with rio.open(out_file, 'r'):
        pass


def test_sharpen_overwrite_error(sharpen_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty sharpen`` raises an error when the pan-sharpened file already exists."""
    out_file = tmp_path.joinpath('pan_sharp.tif')
    out_file.touch()
    cli_str = sharpen_cli_str + f' --out-file {out_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'exists' in result.stdout and out_file.name in result.stdout


def test_simple_ortho(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_legacy_config_file: Path,
    ngi_legacy_csv_file: Path,
    tmp_path: Path,
):
    """Test legacy ``simple-ortho`` CLI with NGI data."""
    cli_str = (
        f'-od {tmp_path} -rc {ngi_legacy_config_file} {ngi_image_file} {ngi_dem_file} '
        f'{ngi_legacy_csv_file}'
    )
    simple_ortho(cli_str.split())
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test_simple_ortho_write_conf(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_legacy_config_file: Path,
    ngi_legacy_csv_file: Path,
    tmp_path: Path,
):
    """Test legacy ``simple-ortho -wc`` writes a config file."""
    conf_file = tmp_path.joinpath('test_config.yaml')
    cli_str = (
        f'-od {tmp_path} -rc {ngi_legacy_config_file} -wc {conf_file} {ngi_image_file} '
        f'{ngi_dem_file} {ngi_legacy_csv_file}'
    )
    simple_ortho(cli_str.split())
    assert conf_file.exists()
    assert yaml.safe_load(conf_file.read_text()) == yaml.safe_load(
        ngi_legacy_config_file.read_text()
    )


def test__frame_mult_camera(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    mult_int_param_dict: dict,
    mult_ext_param_dict: dict,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend with multiple camera ID interior & exterior parameters."""
    # note that this test, and those below, basically duplicate similarly named tests in
    # test_factory
    src_files = []
    for src_file_key in mult_ext_param_dict.keys():
        src_file = tmp_path.joinpath(src_file_key).with_suffix(rgb_byte_src_file.suffix)
        shutil.copy(rgb_byte_src_file, src_file)
        src_files.append(str(src_file))

    cameras = FrameCameras(mult_int_param_dict, mult_ext_param_dict)

    _ortho(
        src_files=fsspec.open_files(src_files, 'rb'),
        dem_file=fsspec.open(str(float_utm34n_dem_file), 'rb'),
        cameras=cameras,
        crs=rio.CRS.from_string(utm34n_crs),
        dem_band=1,
        export_params=False,
        out_dir=fsspec.open(str(tmp_path), 'wb'),
        overwrite=False,
    )
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == len(src_files)


def test__frame_single_camera(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    pinhole_int_param_dict: dict,
    xyz: tuple,
    opk: tuple,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend with an exterior parameter camera ID of ``None`` and a single set of
    interior parameters.
    """
    ext_param_dict = {rgb_byte_src_file.name: dict(xyz=xyz, opk=opk, camera=None)}
    cameras = FrameCameras(pinhole_int_param_dict, ext_param_dict)

    _ortho(
        src_files=(fsspec.open(str(rgb_byte_src_file), 'rb'),),
        dem_file=fsspec.open(str(float_utm34n_dem_file), 'rb'),
        cameras=cameras,
        crs=rio.CRS.from_string(utm34n_crs),
        dem_band=1,
        export_params=False,
        out_dir=fsspec.open(str(tmp_path), 'wb'),
        overwrite=False,
    )
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test__frame_filename_not_found_error(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    pinhole_int_param_dict: dict,
    xyz: tuple,
    opk: tuple,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend raises an error when there are no exterior parameters for the
    source image.
    """
    ext_param_dict = {'unknown.tif': dict(xyz=xyz, opk=opk, camera='pinhole test camera')}
    cameras = FrameCameras(pinhole_int_param_dict, ext_param_dict)

    with pytest.raises(click.UsageError) as ex:
        _ortho(
            src_files=(fsspec.open(str(rgb_byte_src_file), 'rb'),),
            dem_file=fsspec.open(str(float_utm34n_dem_file), 'rb'),
            cameras=cameras,
            crs=rio.CRS.from_string(utm34n_crs),
            dem_band=1,
            export_params=False,
            out_dir=fsspec.open(str(tmp_path)),
            overwrite=False,
        )
    assert rgb_byte_src_file.name in str(ex.value) and 'exterior parameters' in str(ex.value)


def test__frame_camera_not_found_error(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    mult_int_param_dict: dict,
    xyz: tuple,
    opk: tuple,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend raises an error when there are no interior parameters for the
    exterior parameter camera ID.
    """
    camera = 'other'
    ext_param_dict = {rgb_byte_src_file.name: dict(xyz=xyz, opk=opk, camera=camera)}
    cameras = FrameCameras(mult_int_param_dict, ext_param_dict)

    with pytest.raises(click.UsageError) as ex:
        _ortho(
            src_files=(fsspec.open(str(rgb_byte_src_file), 'rb'),),
            dem_file=fsspec.open(str(float_utm34n_dem_file), 'rb'),
            cameras=cameras,
            crs=rio.CRS.from_string(utm34n_crs),
            dem_band=1,
            alpha=1.0,
            export_params=False,
            out_dir=fsspec.open(str(tmp_path)),
            overwrite=False,
            full_remap=True,
        )
    assert camera in str(ex.value) and 'interior parameters' in str(ex.value)


def test__frame_mult_camera_no_camera_error(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    mult_int_param_dict: dict,
    xyz: tuple,
    opk: tuple,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend raises an error when the exterior parameter camera ID is ``None`` and
    there are multiple sets of interior parameters.
    """
    ext_param_dict = {rgb_byte_src_file.name: dict(xyz=xyz, opk=opk, camera=None)}
    cameras = FrameCameras(mult_int_param_dict, ext_param_dict)

    with pytest.raises(click.UsageError) as ex:
        _ortho(
            src_files=(fsspec.open(str(rgb_byte_src_file), 'rb'),),
            dem_file=fsspec.open(str(float_utm34n_dem_file), 'rb'),
            cameras=cameras,
            crs=rio.CRS.from_string(utm34n_crs),
            dem_band=1,
            alpha=1.0,
            export_params=False,
            out_dir=fsspec.open(str(tmp_path)),
            overwrite=False,
            full_remap=True,
        )
    assert (
        rgb_byte_src_file.name in str(ex.value) and 'exterior parameters' in str(ex.value).lower()
    )
