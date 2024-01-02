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

import copy
import logging
import shutil
import tracemalloc
from pathlib import Path

import click
import numpy as np
import pytest
import rasterio as rio
import yaml
from click.testing import CliRunner

from orthority.cli import _ortho, cli, simple_ortho

logger = logging.getLogger(__name__)


@pytest.fixture(scope='session')
def ortho_legacy_ngi_cli_str(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_legacy_config_file: Path,
    ngi_legacy_csv_file: Path,
) -> str:
    """``oty ortho`` CLI string to orthorectify an NGI image with legacy format interior and
    exterior params.
    """
    return (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_legacy_config_file} '
        f'--ext-param {ngi_legacy_csv_file} {ngi_image_file}'
    )


def test_oty_verbosity(
    ortho_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty -v ortho`` generates debug logs."""
    cli_str = f'-v {ortho_legacy_ngi_cli_str} --res 50 --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    assert 'DEBUG:' in result.stdout


def test_ortho_help(runner: CliRunner):
    """Test ``oty ortho --help``."""
    result = runner.invoke(cli, 'ortho --help'.split())
    assert result.exit_code == 0, result.stdout
    assert len(result.stdout) > 0


@pytest.mark.parametrize(
    'src_file',
    ['unknown.tif', 'https://un.known/unknown.tif', 'https://github.com/unknown/unknown.tif'],
)
def test_ortho_src_file_not_found_error(
    src_file: str,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` with a non-existing SOURCE image raises an error ."""
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {src_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'SOURCE' in result.stdout


def test_ortho_dem_missing_error(
    ngi_image_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` without ``--dem`` raises an error ."""
    cli_str = (
        f'ortho --int-param {ngi_oty_int_param_file} --ext-param {ngi_oty_ext_param_file} '
        f'--out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dem' in result.stdout and 'missing' in result.stdout.lower()


@pytest.mark.parametrize(
    'dem_file',
    ['unknown.tif', 'https://un.known/unknown.tif', 'https://github.com/unknown/unknown.tif'],
)
def test_ortho_dem_not_found_error(
    ngi_image_file: Path,
    dem_file: str,
    ngi_oty_int_param_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` with a non-existing ``--dem`` raises an error ."""
    cli_str = (
        f'ortho --dem {dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dem' in result.stdout


def test_ortho_int_param_missing_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` without ``--int-param`` raises an error ."""
    cli_str = (
        f'ortho --dem {ngi_dem_file} --ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} '
        f'{ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--int-param' in result.stdout and 'missing' in result.stdout.lower()


@pytest.mark.parametrize(
    'int_param_file',
    ['unknown.yaml', 'https://un.known/unknown.yaml', 'https://github.com/unknown/unknown.yaml'],
)
def test_ortho_int_param_not_found_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    int_param_file: str,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` with a non-existing ``--int-param`` raises an error ."""
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {int_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--int-param' in result.stdout


def test_ortho_int_param_ext_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_ext_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` with an unrecognised ``--int-param`` extension raises an error."""
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_oty_ext_param_file} '
        f'--ext-param {ngi_oty_ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--int-param' in result.stdout and 'not supported' in result.stdout.lower()


def test_ortho_ext_param_missing_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` without ``--ext-param`` raises an error ."""
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} --out-dir {tmp_path} '
        f'{ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--ext-param' in result.stdout and 'missing' in result.stdout.lower()


@pytest.mark.parametrize(
    'ext_param_file',
    [
        'unknown.geojson',
        'https://un.known/unknown.geojson',
        'https://github.com/unknown/unknown.geojson',
    ],
)
def test_ortho_ext_param_not_found_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    ext_param_file: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` with a non-existing ``--ext-param`` raises an error ."""
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ext_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--ext-param' in result.stdout


def test_ortho_ext_param_ext_error(
    ngi_image_file: Path,
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` with an unrecognised ``--ext-param`` extension raises an error."""
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_oty_int_param_file} --out-dir {tmp_path} {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--ext-param' in result.stdout and 'not supported' in result.stdout.lower()


def test_ortho_crs_src(
    ortho_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho`` reads the world / ortho CRS from a projected source image."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ngi_image_file, 'r') as src_im, rio.open(ortho_files[0], 'r') as ortho_im:
        assert src_im.crs == ortho_im.crs


def test_ortho_crs_auto(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_lla_rpy_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` auto-determines the CRS for LLA-RPY CSV format exterior parameters."""
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_lla_rpy_csv_file} --out-dir {tmp_path} --res 5 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_ortho_crs_cli(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` uses a CRS specified with ``--crs``."""
    # use odm_xyz_opk_csv_file exterior params so there is no auto-crs
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_xyz_opk_csv_file} --out-dir {tmp_path} --crs {odm_crs} --res 5 '
        f'{odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_ortho_crs_file(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` uses a text file CRS specified with ``--crs``."""
    # use odm_xyz_opk_csv_file exterior params so there is no auto-crs
    crs_file = tmp_path.joinpath('test_crs.txt')
    crs_file.write_text(odm_crs)
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_xyz_opk_csv_file} --out-dir {tmp_path} --crs {crs_file} --res 5 '
        f'{odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_ortho_crs_prj(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` reads the CRS in a CSV exterior parameter .prj file."""
    # copy csv file to tmp_path and create .prj file
    csv_file = tmp_path.joinpath(odm_xyz_opk_csv_file.name)
    csv_file.write_text(odm_xyz_opk_csv_file.read_text())
    prj_file = csv_file.with_suffix('.prj')
    prj_file.write_text(odm_crs)

    # create ortho & test
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} --ext-param {csv_file} '
        f'--out-dir {tmp_path} --res 5 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.crs == rio.CRS.from_string(odm_crs)


def test_ortho_crs_missing_error(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_xyz_opk_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` raises an error when ``--crs`` is needed but not passed."""
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_xyz_opk_csv_file} --out-dir {tmp_path} {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'missing' in result.stdout.lower()


def test_ortho_crs_geographic_error(
    ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho`` raises an error when ``--crs`` is geographic."""
    cli_str = ortho_legacy_ngi_cli_str + ' --crs EPSG:4326'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'projected' in result.stdout


def test_ortho_crs_invalid_error(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho`` raises an error when ``--crs`` is invalid."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --crs unknown'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--crs' in result.stdout and 'invalid' in result.stdout.lower()


def test_ortho_resolution_square(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --res`` with square resolution."""
    resolution = (96.0, 96.0)
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res {resolution[0]}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.res == resolution


def test_ortho_resolution_non_square(
    ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho --res`` with non-square resolution."""
    resolution = (48.0, 96.0)
    cli_str = (
        ortho_legacy_ngi_cli_str
        + f' --out-dir {tmp_path} --res {resolution[0]} --res {resolution[1]}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.res == resolution


def test_ortho_resolution_auto(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho`` generates an ortho without the ``--res`` option."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test_ortho_dem_band(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --dem-band`` with valid band."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --dem-band 1'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test_ortho_dem_band_error(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --dem-band`` raises an error with an out of range band."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --dem-band 2'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dem-band' in result.stdout


def test_ortho_interp(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --interp`` by comparing 'nearest' to 'average' interpolation orthos."""
    # create average interpolation dem
    out_dir_average = tmp_path.joinpath('average')
    out_dir_average.mkdir()
    cli_str = (
        ortho_legacy_ngi_cli_str
        + f' --out-dir {out_dir_average} --res 24 --compress deflate --interp average '
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ref_ortho_files = [*out_dir_average.glob('*_ORTHO.tif')]
    assert len(ref_ortho_files) == 1

    # create nearest interpolation dem
    out_dir_nearest = tmp_path.joinpath('nearest')
    out_dir_nearest.mkdir()
    cli_str = (
        ortho_legacy_ngi_cli_str
        + f' --out-dir {out_dir_nearest} --res 24 --compress deflate --interp nearest'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    test_ortho_files = [*out_dir_nearest.glob('*_ORTHO.tif')]
    assert len(test_ortho_files) == 1

    # compare nearest and average interpolation orthos
    with rio.open(ref_ortho_files[0], 'r') as ref_im, rio.open(test_ortho_files[0], 'r') as test_im:
        ref_array = ref_im.read(masked=True)
        test_array = test_im.read(masked=True)
        assert test_array.std() > ref_array.std()


def test_ortho_interp_error(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --interp`` raises an error with an invalid interpolation value."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --interp other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--interp' in result.stdout and 'invalid' in result.stdout.lower()


def test_ortho_dem_interp(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --dem-interp`` by comparing 'nearest' to 'average' DEM interpolation
    orthos.
    """
    # create average dem interpolation ortho
    out_dir_average = tmp_path.joinpath('average')
    out_dir_average.mkdir()
    cli_str = ortho_legacy_ngi_cli_str + (
        f' --out-dir {out_dir_average} --res 30 --compress deflate --dem-interp average'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ref_ortho_files = [*out_dir_average.glob('*_ORTHO.tif')]
    assert len(ref_ortho_files) == 1

    # create nearest dem interpolation ortho
    out_dir_nearest = tmp_path.joinpath('nearest')
    out_dir_nearest.mkdir()
    cli_str = ortho_legacy_ngi_cli_str + (
        f' --out-dir {out_dir_nearest} --res 30 --compress deflate --dem-interp nearest'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    test_ortho_files = [*out_dir_nearest.glob('*_ORTHO.tif')]
    assert len(test_ortho_files) == 1

    # compare nearest and average dem interpolation orthos
    with rio.open(ref_ortho_files[0], 'r') as ref_im, rio.open(test_ortho_files[0], 'r') as test_im:
        ref_array = ref_im.read(masked=True)
        test_array = test_im.read(masked=True)
        assert test_im.bounds == pytest.approx(ref_im.bounds, abs=ref_im.res[0])
        assert test_array.std() != ref_array.std()


def test_ortho_dem_interp_error(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --dem-interp`` raises an error with an invalid interpolation value."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --dem-interp other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dem-interp' in result.stdout and 'invalid' in result.stdout.lower()


def test_ortho_per_band(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --per-band`` by comparing memory usage between ``--per-band`` and ``--no-
    per-band``.
    """
    tracemalloc.start()
    try:
        # create --per-band ortho
        cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --per-band'
        result = runner.invoke(cli, cli_str.split())
        _, per_band_peak = tracemalloc.get_traced_memory()
        tracemalloc.clear_traces()
        assert result.exit_code == 0, result.stdout
        ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        # create --no-per-band ortho
        cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --no-per-band -o'
        result = runner.invoke(cli, cli_str.split())
        _, no_per_band_peak = tracemalloc.get_traced_memory()
        assert result.exit_code == 0, result.stdout
        ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
        assert len(ortho_files) == 1

        assert no_per_band_peak > per_band_peak
    finally:
        tracemalloc.stop()


def test_ortho_full_remap(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho --full-remap`` by comparing ``--full-remap`` and ``--no-full-remap``
    orthos.
    """
    # create --full-remap ortho
    out_dir_full_remap = tmp_path.joinpath('full_remap')
    out_dir_full_remap.mkdir()
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_reconstruction_file} --out-dir {out_dir_full_remap} --res 1 '
        f'--compress deflate --full-remap {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    fr_ortho_files = [*out_dir_full_remap.glob('*_ORTHO.tif')]
    assert len(fr_ortho_files) == 1

    # create --no-full-remap ortho
    out_dir_no_full_remap = tmp_path.joinpath('no_full_remap')
    out_dir_no_full_remap.mkdir()
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_reconstruction_file} --out-dir {out_dir_no_full_remap} --res 1 '
        f'--compress deflate --no-full-remap {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    nfr_ortho_files = [*out_dir_no_full_remap.glob('*_ORTHO.tif')]
    assert len(nfr_ortho_files) == 1

    # compare --full-remap and --no-full-remap orthos
    with rio.open(fr_ortho_files[0], 'r') as fr_im, rio.open(nfr_ortho_files[0], 'r') as nfr_im:
        fr_array = fr_im.read(masked=True)
        nfr_array = nfr_im.read(masked=True)
        assert np.any(nfr_array.mask != fr_array.mask)

        mask = nfr_array.mask | fr_array.mask
        nfr_array.mask = mask
        fr_array.mask = mask
        cc = np.corrcoef(nfr_array.compressed(), fr_array.compressed())
        assert 0.95 < cc[0, 1] < 1


def test_ortho_alpha(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho --alpha`` by comparing ``--alpha 0`` and ``--alpha 1`` orthos."""
    # create --alpha 1 ortho
    out_dir_alpha_1 = tmp_path.joinpath('alpha_1')
    out_dir_alpha_1.mkdir()
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_reconstruction_file} --out-dir {out_dir_alpha_1} --res 1 '
        f'--compress deflate --no-full-remap --alpha 1 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    a1_ortho_files = [*out_dir_alpha_1.glob('*_ORTHO.tif')]
    assert len(a1_ortho_files) == 1

    # create --alpha 0 ortho
    out_dir_alpha_0 = tmp_path.joinpath('alpha_0')
    out_dir_alpha_0.mkdir()
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_reconstruction_file} --out-dir {out_dir_alpha_0} --res 1 '
        f'--compress deflate --no-full-remap --alpha 0 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    a0_ortho_files = [*out_dir_alpha_0.glob('*_ORTHO.tif')]
    assert len(a0_ortho_files) == 1

    # compare --alpha 1 and --alpha 0 orthos
    with rio.open(a1_ortho_files[0], 'r') as a1_im, rio.open(a0_ortho_files[0], 'r') as a0_im:
        a1_win = a1_im.window(*a0_im.bounds)
        a1_array = a1_im.read(masked=True, window=a1_win)
        a0_array = a0_im.read(masked=True)
        assert a0_im.bounds[:2] > a1_im.bounds[:2] and a0_im.bounds[-2:] < a1_im.bounds[-2:]

        mask = a0_array.mask | a1_array.mask
        a0_array.mask = mask
        a1_array.mask = mask
        cc = np.corrcoef(a0_array.compressed(), a1_array.compressed())
        assert cc[0, 1] > 0.99


def test_ortho_alpha_error(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --alpha`` raises an error with an invalid alpha value."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --no-full-remap --alpha 2'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--alpha' in result.stdout and 'invalid' in result.stdout.lower()


def test_ortho_lla_crs(
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_reconstruction_file: Path,
    odm_lla_rpy_csv_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho --lla-crs`` by comparing orthos created with different ``--lla-crs``
    values.
    """
    # create an ortho with ellipsoidal height --lla-crs
    out_dir_ellps = tmp_path.joinpath('lla_crs_ellps')
    out_dir_ellps.mkdir()
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_lla_rpy_csv_file} --out-dir {out_dir_ellps} --res 5 '
        f'--crs {odm_crs}+4326 --lla-crs EPSG:4326+4326 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ellps_ortho_files = [*out_dir_ellps.glob('*_ORTHO.tif')]
    assert len(ellps_ortho_files) == 1

    # create an ortho with geoidal height --lla-crs
    out_dir_geoid = tmp_path.joinpath('lla_crs_geoid')
    out_dir_geoid.mkdir()
    cli_str = (
        f'ortho --dem {odm_dem_file} --int-param {odm_reconstruction_file} '
        f'--ext-param {odm_lla_rpy_csv_file} --out-dir {out_dir_geoid} --res 5 '
        f'--crs {odm_crs}+4326 --lla-crs EPSG:4326+3855 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    geoid_ortho_files = [*out_dir_geoid.glob('*_ORTHO.tif')]
    assert len(geoid_ortho_files) == 1

    # compare
    with rio.open(ellps_ortho_files[0], 'r') as ellps_im, rio.open(
        geoid_ortho_files[0], 'r'
    ) as geoid_im:
        assert ellps_im.bounds != pytest.approx(geoid_im.bounds, abs=geoid_im.res[0])


def test_ortho_lla_crs_projected_error(
    ortho_legacy_ngi_cli_str: str, ngi_crs: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho --lla-crs`` raises an error with a projected CRS value."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --lla-crs {ngi_crs}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--lla-crs' in result.stdout and 'geographic' in result.stdout


def test_ortho_lla_crs_invalid_error(
    ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho`` raises an error when ``--lla_crs`` is invalid."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --lla-crs unknown'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--lla-crs' in result.stdout and 'invalid' in result.stdout.lower()


def test_ortho_radians(
    ngi_image_file: tuple[Path, ...],
    ngi_dem_file: Path,
    ngi_oty_int_param_file: Path,
    ngi_xyz_opk_csv_file: Path,
    ngi_xyz_opk_radians_csv_file: Path,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho --radians`` by comparing orthos created exterior parameter orientation
    angles in radians and degrees.
    """
    # create an ortho with orientation angles in degrees
    out_dir_degrees = tmp_path.joinpath('degrees')
    out_dir_degrees.mkdir()
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_xyz_opk_csv_file} --out-dir {out_dir_degrees} --res 24 '
        f'--degrees {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files_degrees = [*out_dir_degrees.glob('*_ORTHO.tif')]
    assert len(ortho_files_degrees) == 1

    # create an ortho with orientation angles in radians
    out_dir_radians = tmp_path.joinpath('radians')
    out_dir_radians.mkdir()
    cli_str = (
        f'ortho --dem {ngi_dem_file} --int-param {ngi_oty_int_param_file} '
        f'--ext-param {ngi_xyz_opk_radians_csv_file} --out-dir {out_dir_radians} --res 24 '
        f'--radians {ngi_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files_radians = [*out_dir_radians.glob('*_ORTHO.tif')]
    assert len(ortho_files_radians) == 1

    # compare
    with rio.open(ortho_files_degrees[0], 'r') as degrees_im, rio.open(
        ortho_files_radians[0], 'r'
    ) as radians_im:
        degrees_array = degrees_im.read(1)
        radians_array = radians_im.read(1)
        assert radians_array == pytest.approx(degrees_array, abs=1)


def test_ortho_write_mask(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --write-mask`` writes an internal mask to the ortho with
    ``compress=deflate``.
    """
    cli_str = (
        ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress deflate --write-mask --res 24'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.profile['compress'] == 'deflate'
        assert all([mf[0] == rio.enums.MaskFlags.per_dataset for mf in im.mask_flag_enums])


def test_ortho_dtype(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --dtype`` creates an ortho with the correct dtype."""
    dtype = 'float32'
    cli_str = (
        ortho_legacy_ngi_cli_str
        + f' --out-dir {tmp_path} --compress deflate --dtype {dtype} --res 24'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.profile['dtype'] == dtype


def test_ortho_dtype_error(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --dtype`` with an invalid dtype raises an error."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress deflate --dtype int32'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--dtype' in result.stdout and 'invalid' in result.stdout.lower()


@pytest.mark.parametrize('compress', ['jpeg', 'deflate'])
def test_ortho_compress(
    ortho_legacy_ngi_cli_str: str, compress: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho --compress`` creates an ortho with the correct compression."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress {compress} --res 24'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert im.profile['compress'] == compress


def test_ortho_compress_error(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --dtype`` with an invalid dtype raises an error."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --compress other'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert '--compress' in result.stdout and 'invalid' in result.stdout.lower()


def test_ortho_build_ovw(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --build-ovw`` builds overviews."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 5 --build-ovw'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert len(im.overviews(1)) > 0


def test_ortho_no_build_ovw(ortho_legacy_ngi_cli_str: str, tmp_path: Path, runner: CliRunner):
    """Test ``oty ortho --no-build-ovw`` does not build overviews."""
    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 5 --no-build-ovw'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1

    with rio.open(ortho_files[0], 'r') as im:
        assert len(im.overviews(1)) == 0


@pytest.mark.parametrize(
    'int_param_file, ext_param_file',
    [
        ('ngi_oty_int_param_file', 'ngi_oty_ext_param_file'),
        ('ngi_legacy_config_file', 'ngi_xyz_opk_csv_file'),
        ('odm_reconstruction_file', 'odm_reconstruction_file'),
    ],
)
def test_ortho_export_params(
    int_param_file: str,
    ext_param_file: str,
    tmp_path: Path,
    runner: CliRunner,
    request: pytest.FixtureRequest,
):
    """Test ``oty ortho --export-params`` exports interior & exterior parameters provided in
    different formats (with no ``--dem-file``).
    """
    # note this doubles as a test of reading params in different formats
    int_param_file: Path = request.getfixturevalue(int_param_file)
    ext_param_file: Path = request.getfixturevalue(ext_param_file)
    cli_str = (
        f'ortho --int-param {int_param_file} --ext-param {ext_param_file} --out-dir {tmp_path} '
        f'--export-params'
    )

    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    int_param_file = tmp_path.joinpath('int_param.yaml')
    ext_param_file = tmp_path.joinpath('ext_param.geojson')
    assert int_param_file.exists() and ext_param_file.exists()


def test_ortho_overwrite(
    ortho_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho --overwrite`` overwrites and existing ortho."""
    ortho_file = tmp_path.joinpath(ngi_image_file.stem + '_ORTHO.tif')
    ortho_file.touch()

    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24 --overwrite'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1
    assert ortho_file == ortho_files[0]


def test_ortho_overwrite_error(
    ortho_legacy_ngi_cli_str: str, ngi_image_file: Path, tmp_path: Path, runner: CliRunner
):
    """Test ``oty ortho`` raises an error when the ortho file already exists."""
    ortho_file = tmp_path.joinpath(ngi_image_file.stem + '_ORTHO.tif')
    ortho_file.touch()

    cli_str = ortho_legacy_ngi_cli_str + f' --out-dir {tmp_path} --res 24'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert result.exception is not None and 'exists' in str(result.exception)


def test_ortho_urls(
    ngi_image_url: str,
    ngi_dem_url: str,
    ngi_oty_int_param_url: str,
    ngi_oty_ext_param_url: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty ortho`` with source, DEM & parameter file URLs."""
    cli_str = (
        f'ortho --dem {ngi_dem_url} --int-param {ngi_oty_int_param_url} '
        f'--ext-param {ngi_oty_ext_param_url} --out-dir {tmp_path} {ngi_image_url}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test_exif_crs(
    odm_image_file: Path, odm_dem_file: Path, odm_crs: str, tmp_path: Path, runner: CliRunner
):
    """Test ``oty exif --crs`` creates orthos with the correct CRS."""
    crs = odm_crs + '+4326'  # differentiate from auto CRS
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
    odm_image_file: tuple[Path, ...],
    odm_dem_file: Path,
    odm_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test ``oty exif --lla-crs`` by comparing orthos created with different ``--lla-crs``
    values.
    """
    # create an ortho with ellipsoidal height --lla-crs
    out_dir_ellps = tmp_path.joinpath('lla_crs_ellps')
    out_dir_ellps.mkdir()
    cli_str = (
        f'exif --dem {odm_dem_file} --out-dir {out_dir_ellps} --res 5 --crs {odm_crs}+4326 '
        f'--lla-crs EPSG:4326+4326 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ellps_ortho_files = [*out_dir_ellps.glob('*_ORTHO.tif')]
    assert len(ellps_ortho_files) == 1

    # create an ortho with geoidal height --lla-crs
    out_dir_geoid = tmp_path.joinpath('lla_crs_geoid')
    out_dir_geoid.mkdir()
    cli_str = (
        f'exif --dem {odm_dem_file} --out-dir {out_dir_geoid} --res 5 --crs {odm_crs}+4326 '
        f'--lla-crs EPSG:4326+3855 {odm_image_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    geoid_ortho_files = [*out_dir_geoid.glob('*_ORTHO.tif')]
    assert len(geoid_ortho_files) == 1

    # compare
    with rio.open(ellps_ortho_files[0], 'r') as ellps_im, rio.open(
        geoid_ortho_files[0], 'r'
    ) as geoid_im:
        assert ellps_im.bounds != pytest.approx(geoid_im.bounds, abs=geoid_im.res[0])


def test_exif_error(ngi_image_file: Path, ngi_dem_file: Path, tmp_path: Path, runner: CliRunner):
    """Test ``oty exif`` raises an error with a non-EXIF source image."""
    cli_str = f'exif --dem {ngi_dem_file} --out-dir {tmp_path} {ngi_image_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0, result.stdout
    assert 'SOURCE' in result.stdout and 'tags' in result.stdout.lower()


def test_odm_dataset_dir(
    odm_dataset_dir: Path, odm_image_files: tuple[Path, ...], tmp_path: Path, runner: CliRunner
):
    """Test ``oty odm`` creates orthos in '<--dataset-dir>/orthority' sub-folder."""
    shutil.copytree(odm_dataset_dir, tmp_path, dirs_exist_ok=True)  # copy test data to tmp_path
    cli_str = f'odm --dataset-dir {tmp_path} --res 5 --overwrite'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0, result.stdout
    ortho_files = [*tmp_path.joinpath('orthority').glob('*_ORTHO.tif')]
    src_files = [*tmp_path.joinpath('images').glob('*.tif')]
    assert len(ortho_files) == len(src_files)


def test_odm_out_dir(
    odm_dataset_dir: Path, odm_image_files: tuple[Path, ...], tmp_path: Path, runner: CliRunner
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


def test__ortho_mult_camera(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    mult_int_param_dict: dict,
    mult_ext_param_dict: dict,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend with multiple camera ID interior & exterior parameters."""
    src_files = []
    for src_file_key in mult_ext_param_dict.keys():
        src_file = tmp_path.joinpath(src_file_key).with_suffix(rgb_byte_src_file.suffix)
        shutil.copy(rgb_byte_src_file, src_file)
        src_files.append(src_file)

    _ortho(
        src_files=src_files,
        dem_file=float_utm34n_dem_file,
        int_param_dict=mult_int_param_dict,
        ext_param_dict=copy.deepcopy(mult_ext_param_dict),
        crs=utm34n_crs,
        dem_band=1,
        alpha=1.0,
        export_params=False,
        out_dir=tmp_path,
        overwrite=False,
    )
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == len(src_files)


def test__ortho_single_no_camera(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    pinhole_int_param_dict: dict,
    xyz: tuple,
    opk: tuple,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend with an exterior parameter camera ID of None and a single set of
    interior parameters.
    """
    ext_param_dict = {rgb_byte_src_file.name: dict(xyz=xyz, opk=opk, camera=None)}
    _ortho(
        src_files=(rgb_byte_src_file,),
        dem_file=float_utm34n_dem_file,
        int_param_dict=pinhole_int_param_dict,
        ext_param_dict=ext_param_dict,
        crs=utm34n_crs,
        dem_band=1,
        alpha=1.0,
        export_params=False,
        out_dir=tmp_path,
        overwrite=False,
    )
    ortho_files = [*tmp_path.glob('*_ORTHO.tif')]
    assert len(ortho_files) == 1


def test__ortho_mult_camera_unknown_camera_error(
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
    with pytest.raises(click.BadParameter) as ex:
        _ortho(
            src_files=(rgb_byte_src_file,),
            dem_file=float_utm34n_dem_file,
            int_param_dict=mult_int_param_dict,
            ext_param_dict=ext_param_dict,
            crs=utm34n_crs,
            dem_band=1,
            alpha=1.0,
            export_params=False,
            out_dir=tmp_path,
            overwrite=False,
        )
    assert camera in str(ex) and '--int-param' in ex.value.param_hint


def test__ortho_mult_camera_no_camera_error(
    rgb_byte_src_file: Path,
    float_utm34n_dem_file: Path,
    mult_int_param_dict: dict,
    xyz: tuple,
    opk: tuple,
    utm34n_crs: str,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the _ortho backend raises an error when the exterior parameter camera ID is None and
    there are multiple sets of interior parameters.
    """
    ext_param_dict = {rgb_byte_src_file.name: dict(xyz=xyz, opk=opk, camera=None)}
    with pytest.raises(click.BadParameter) as ex:
        _ortho(
            src_files=(rgb_byte_src_file,),
            dem_file=float_utm34n_dem_file,
            int_param_dict=mult_int_param_dict,
            ext_param_dict=ext_param_dict,
            crs=utm34n_crs,
            dem_band=1,
            alpha=1.0,
            export_params=False,
            out_dir=tmp_path,
            overwrite=False,
        )
    assert rgb_byte_src_file.name in str(ex) and '--ext-param' in ex.value.param_hint


def test__ortho_ext_param_not_found_error(
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
    with pytest.raises(click.BadParameter) as ex:
        _ortho(
            src_files=(rgb_byte_src_file,),
            dem_file=float_utm34n_dem_file,
            int_param_dict=pinhole_int_param_dict,
            ext_param_dict=ext_param_dict,
            crs=utm34n_crs,
            dem_band=1,
            alpha=1.0,
            export_params=False,
            out_dir=tmp_path,
            overwrite=False,
        )
    assert rgb_byte_src_file.name in str(ex) and '--ext-param' in ex.value.param_hint
