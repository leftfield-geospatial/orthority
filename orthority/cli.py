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

import argparse
import csv
import logging
import re
import warnings
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import click
import fsspec
import numpy as np
import rasterio as rio
import yaml
from fsspec.core import OpenFile
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from orthority import root_path, common
from orthority.camera import create_camera, FrameCamera
from orthority.enums import CameraType, Compress, Interp, RpcRefine, Driver
from orthority.errors import CrsError, CrsMissingError, OrthorityError
from orthority.factory import Cameras, FrameCameras, RpcCameras
from orthority.fit import _default_rpc_refine_method
from orthority.ortho import Ortho
from orthority.pan_sharp import PanSharpen
from orthority.version import __version__

logger = logging.getLogger(__name__)

# TODO: use universal_pathlib if/when it matures for all filename options instead of OpenFile,
#  and the common.get_filename and common.join_ofile work arounds.


class RstCommand(click.Command):
    """click.Command subclass for formatting help with RST markup."""

    def get_help(self, ctx: click.Context):
        """
        Strip some RST markup from the help text for CLI display.

        Doesn't work with grid tables.
        """

        # Note that this can't easily be done in __init__, as each sub-command's __init__ gets
        # called, which ends up re-assigning self.wrap_text to reformat_text
        if not hasattr(self, 'click_wrap_text'):
            self.click_wrap_text = click.formatting.wrap_text

        sub_strings = {
            # convert from RST friendly to click literal (unwrapped) block marker
            '\b\n': '\n\b',
            # strip RST literal (unwrapped) marker in e.g. tables and bullet lists
            r'\| ': '',
            # strip RST ref directive '\n.. _<name>:\n'
            r'\n\.\. _.*:\n': '',
            # convert from RST '::' to ':'
            '::': ':',
            # convert from RST '``literal``' to 'literal'
            '``(.*?)``': r'\g<1>',
            # convert ':option:`--name <group-command --name>`' to '--name'
            r':option:`(.*?)(\s+<.*?>)?`': r'\g<1>',
            # convert ':option:`--name`' to '--name'
            ':option:`(.*?)`': r'\g<1>',
            # convert ':file:`file/na.me`' to "'file/na.me'"
            ':file:`(.*?)`': r"'\g<1>'",
            # strip '----...'
            # '-{4,}': r'',
            # convert from RST cross-ref '`<name> <link>`__' to 'name'
            r'`(.*?)(\s+<.*?>)?`_+': r'\g<1>',
            # # convert from RST cross-ref '`<name> <link>`__' to 'link'
            # r'`(.*?)<(.*?)>`_+': r'\g<2>',
        }

        def reformat_text(text, width, **kwargs):
            for sub_key, sub_value in sub_strings.items():
                text = re.sub(sub_key, sub_value, text, flags=re.DOTALL)
            wr_text = self.click_wrap_text(text, width, **kwargs)
            # change double newline to single newline separated list
            return re.sub('\n\n(\s*?)- ', '\n- ', wr_text, flags=re.DOTALL)

        click.formatting.wrap_text = reformat_text
        return click.Command.get_help(self, ctx)


def _configure_logging(verbosity: int):
    """Configure python logging level."""

    def showwarning(message, category, filename, lineno, file=None, line=None):
        """Redirect orthority warnings to the source module's logger, otherwise show warning as
        usual.
        """
        # adapted from https://discuss.python.org/t/some-easy-and-pythonic-way-to-bind-warnings-to-loggers/14009/2
        package_root = Path(__file__).parents[1]
        file_path = Path(filename)
        try:
            module_path = file_path.relative_to(package_root)
            is_relative_to = True
        except ValueError:
            is_relative_to = False

        if file is not None or not is_relative_to:
            orig_show_warning(message, category, filename, lineno, file, line)
        else:
            module_name = module_path.with_suffix('').as_posix().replace('/', '.')
            logger = logging.getLogger(module_name)
            logger.warning(str(message))

    # redirect orthority warnings to module logger
    orig_show_warning = warnings.showwarning
    warnings.showwarning = showwarning

    # Configure the package logger, leaving logs from dependencies on their defaults (e.g. for
    # rasterio, they stay hidden).
    # Adapted from rasterio: https://github.com/rasterio/rasterio/blob/main/rasterio/rio/main.py.
    log_level = max(10, 20 - 10 * verbosity)
    pkg_logger = logging.getLogger(__package__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)


def _read_crs(crs: str):
    """Read a CRS from a string, text file, or image file."""
    crs_path = Path(crs)
    if crs_path.suffix.lower() in ['.tif', '.tiff']:
        # read CRS from geotiff path / URL
        with common.suppress_no_georef(), common.OpenRaster(crs, 'r') as im:
            crs = im.crs
    else:
        if crs_path.exists() or urlparse(crs).scheme in fsspec.available_protocols():
            # read string from text file path / valid URI
            with common.Open(crs, 'rt') as f:
                crs = f.read()
        # read CRS from string
        crs = rio.CRS.from_string(crs)
    return crs


def _crs_cb(ctx: click.Context, param: click.Parameter, crs: str):
    """Click callback to validate and parse the CRS."""
    if crs is not None:
        try:
            crs = _read_crs(crs)
        except Exception as ex:
            raise click.BadParameter(f'{str(ex)}', param=param)
    return crs


def _src_files_cb(
    ctx: click.Context, param: click.Parameter, src_files: list[str]
) -> list[OpenFile]:
    """Click callback to form a list of source image OpenFile instances. ``src_files`` can be a
    list of file paths / URIs, or a list of path / URI glob patterns.
    """
    if not src_files or len(src_files) == 0:
        return []
    try:
        src_files = fsspec.open_files(src_files, 'rb')
    except Exception as ex:
        raise click.BadParameter(str(ex), param=param)
    if len(src_files) == 0:
        raise click.BadParameter('No source image(s) found.', param=param)
    return src_files


def _file_cb(
    ctx: click.Context, param: click.Parameter, path_uri: str, mode: str = 'rb'
) -> OpenFile:
    """Click callback to convert a file path / URI to an OpenFile instance."""
    ofile = None
    if path_uri:
        try:
            ofile = fsspec.open(path_uri, mode)
        except Exception as ex:
            raise click.BadParameter(str(ex), param=param)
    return ofile


def _lla_crs_cb(ctx: click.Context, param: click.Parameter, lla_crs: str) -> str:
    """Click callback to validate and parse the LLA CRS."""
    if lla_crs is not None:
        try:
            lla_crs = _read_crs(lla_crs)
        except Exception as ex:
            raise click.BadParameter(f'{str(ex)}', param=param)
        if not lla_crs.is_geographic:
            raise click.BadParameter(f"CRS should be a geographic system.", param=param)
    return lla_crs


def _resolution_cb(
    ctx: click.Context, param: click.Parameter, resolution: tuple
) -> tuple[float, float]:
    """Click callback to validate and parse the resolution."""
    if len(resolution) == 1:
        resolution *= 2
    elif len(resolution) > 2:
        raise click.BadParameter(f'At most two resolution values should be specified.', param=param)
    return resolution


def _dir_cb(ctx: click.Context, param: click.Parameter, uri_path: str) -> OpenFile:
    """Click callback to convert a directory to an fsspec OpenFile, and validate."""
    try:
        # some dirs (e.g. gcs buckets) don't work with a trailing slash on the path, so strip it
        ofile = fsspec.open(uri_path.rstrip('/'))
    except Exception as ex:
        raise click.BadParameter(str(ex), param=param)

    if not ofile.fs.isdir(ofile.path):
        raise click.BadParameter(
            f"'{uri_path}' is not a directory or cannot be accessed.", param=param
        )
    return ofile


def _creation_option_cb(ctx: click.Context, param: click.Option, value):
    """click callback to validate and parse creation options (e.g. ``-co KEY1=VAL1 -co
    KEY2=VAL2``). Note: ``==VAL`` breaks this as ``str.split('=', 1)`` is used.
    """
    # adapted from rasterio https://github.com/rasterio/rasterio
    if not value:
        return {}
    else:
        out = {}
        for pair in value:
            if '=' not in pair:
                raise click.BadParameter(f'Invalid syntax for KEY=VAL: {pair}')
            else:
                k, v = pair.split('=', 1)
                k = k.lower().strip()
                v = v.lower().strip()
                out[k] = None if v in ['none', 'null', 'nil'] else yaml.safe_load(v)
        return out


def _odm_out_dir_cb(ctx: click.Context, param: click.Parameter, out_dir: str) -> OpenFile:
    """Click callback to create the default, or validate a user-supplied, ODM output directory."""
    if not out_dir:
        ofile = common.join_ofile(ctx.params['dataset_dir'], 'orthority')
        try:
            # Note this does not raise an exception for (read-only) html protocol.  Not ideal, but
            # this code won't be reached for html as _dir_cb does raise an exception for
            # --dataset-dir.
            ofile.fs.mkdirs(ofile.path, exist_ok=True)
        except Exception as ex:
            raise click.BadParameter(f"Cannot create / access '{out_dir}'. {str(ex)}", param=param)
    else:
        ofile = _dir_cb(ctx, param, out_dir)
    return ofile


def _gcp_refine_cb(
    ctx: click.Context, param: click.Parameter, path_uri: str | bool
) -> OpenFile | bool:
    """Click callback to parse the ``--gcp-refine`` option."""
    return (
        _file_cb(ctx, param, path_uri, mode='rt') if str(path_uri).lower() != 'tags' else path_uri
    )


def _weights_cb(ctx: click.Context, param: click.Parameter, weights: list[float]):
    """Click callback to validate the ``--weight`` option."""
    if weights is not None:
        if np.any(np.array(weights) < 0):
            raise click.BadParameter(
                'Weight values should greater than or equal to 0.', param=param
            )
    return weights


def _ortho(
    src_files: list[OpenFile],
    dem_file: OpenFile,
    cameras: Cameras,
    crs: rio.CRS,
    dem_band: int,
    export_params: bool,
    out_dir: OpenFile,
    overwrite: bool,
    **kwargs,
):
    """
    Orthorectify images given a DEM filename and camera factory.

    Backend function for orthorectification sub-commands.
    """
    # TODO: investigate simplifying progress bars with rich

    if export_params:
        # convert interior / exterior params to oty format files & exit
        logger.info('Writing parameter files...')
        try:
            cameras.write_param(out_dir, overwrite=overwrite)
        except FileExistsError as ex:
            raise click.UsageError(str(ex))
        except CrsMissingError:
            raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')
        return
    elif not dem_file:
        raise click.MissingParameter(param_hint="'-d' / '--dem'", param_type='option')

    # open & validate dem_file (open it once here so it is not opened repeatedly in
    # orthorectification below)
    try:
        dem_ctx = common.OpenRaster(dem_file, 'r')
    except FileNotFoundError as ex:
        raise click.BadParameter(str(ex), param_hint="'-d' / '--dem'")

    with dem_ctx as dem_im:
        # validate dem_band
        if dem_band <= 0 or dem_band > dem_im.count:
            dem_name = common.get_filename(dem_im)
            raise click.BadParameter(
                f"DEM band {dem_band} is invalid for '{dem_name}' with {dem_im.count} band(s).",
                param_hint="'-db' / '--dem-band'",
            )

        for src_file in tqdm(src_files, **common.get_tqdm_kwargs(desc='Total', unit='files')):
            # create camera for src_file
            try:
                camera = cameras.get(src_file)
            except OrthorityError as ex:
                raise click.UsageError(str(ex))

            # open & validate src_file (open it once here so it is not opened repeatedly)
            try:
                src_ctx = common.OpenRaster(src_file, 'r')
            except FileNotFoundError as ex:
                raise click.BadParameter(str(ex), param_hint="'SOURCE...'")

            with src_ctx as src_im:
                # finalise and validate world / ortho crs
                crs = crs or src_im.crs
                if not crs:
                    raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')

                # orthorectify
                src_file_path = Path(src_file.path)
                tqdm_kwargs = common.get_tqdm_kwargs(
                    desc=src_file_path.name, unit='blocks', leave=False
                )
                ortho_ofile = common.join_ofile(
                    out_dir, f'{src_file_path.stem}_ORTHO.tif', mode='wb'
                )
                try:
                    ortho = Ortho(src_im, dem_im, camera, crs, dem_band=dem_band)
                    ortho.process(ortho_ofile, overwrite=overwrite, progress=tqdm_kwargs, **kwargs)
                except (FileExistsError, OrthorityError) as ex:
                    raise click.UsageError(str(ex))


# Define click options that are common to more than one command
src_files_arg = click.argument(
    'src_files',
    nargs=-1,
    metavar='SOURCE...',
    type=click.Path(dir_okay=False),
    callback=_src_files_cb,
)
dem_file_option = click.option(
    '-d',
    '--dem',
    'dem_file',
    type=click.Path(dir_okay=False),
    default=None,
    callback=partial(_file_cb, mode='rb'),
    help='Path / URI of a DEM file covering the source image(s).',
)
int_param_file_option = click.option(
    '-ip',
    '--int-param',
    'int_param_file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    callback=partial(_file_cb, mode='rt'),
    help='Path / URI of an interior parameter file.',
)
ext_param_file_option = click.option(
    '-ep',
    '--ext-param',
    'ext_param_file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    callback=partial(_file_cb, mode='rt'),
    help='Path / URI of an exterior parameter file.',
)
crs_option = click.option(
    '-c',
    '--crs',
    type=click.STRING,
    default=None,
    show_default='auto',
    callback=_crs_cb,
    help='CRS of ortho image(s) and world coordinates as an EPSG, proj4, or WKT string; path / URI '
    'of a text file containing string; or path / URI of an image with metadata CRS.',
)
lla_crs_option = click.option(
    '-lc',
    '--lla-crs',
    type=click.STRING,
    default='EPSG:4979',
    show_default=True,
    callback=_lla_crs_cb,
    help='CRS of any geographic coordinate exterior parameters as an EPSG, proj4, or WKT string; '
    'path / URI of a text file containing string; or path / URI of an image with metadata CRS.',
)
radians_option = click.option(
    '-rd/-dg',
    '--radians/--degrees',
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Orientation angle units. Only used for ``--ext-param`` in CSV format.',
)
resolution_option = click.option(
    '-r',
    '--res',
    'resolution',
    type=click.FLOAT,
    default=None,
    show_default='ground sampling distance',
    multiple=True,
    callback=_resolution_cb,
    help='Ortho image resolution in units of the ``--crs``.  Can be used twice for non-square '
    'pixels: ``--res PIXEL_WIDTH --res PIXEL_HEIGHT``.',
)
dem_band_option = click.option(
    '-db',
    '--dem-band',
    type=click.INT,
    nargs=1,
    default=Ortho._default_alg_config['dem_band'],
    show_default=True,
    help='Index of the DEM band to use (1 based).',
)
interp_option = click.option(
    '-i',
    '--interp',
    type=click.Choice(Interp, case_sensitive=True),
    default=Ortho._default_alg_config['interp'],
    show_default=True,
    help=f'Interpolation method for remapping source to ortho image.',
)
dem_interp_option = click.option(
    '-di',
    '--dem-interp',
    type=click.Choice(Interp, case_sensitive=True),
    default=Ortho._default_alg_config['dem_interp'],
    show_default=True,
    help=f'Interpolation method for DEM reprojection.',
)
per_band_option = click.option(
    '-pb/-npb',
    '--per-band/--no-per-band',
    type=click.BOOL,
    default=Ortho._default_alg_config['per_band'],
    show_default=True,
    help='Orthorectify band-by-band (``--per-band``) or all bands at once (``--no-per-band``). '
    '``--no-per-band`` is faster but uses more memory.',
)
full_remap_option = click.option(
    '-fr/-nfr',
    '--full-remap/--no-full-remap',
    type=click.BOOL,
    default=FrameCamera._default_distort,
    show_default=True,
    help='Orthorectify the source image with full camera model (``--full-remap``), '
    'or an undistorted source image with pinhole camera model (``--no-full-remap``).  '
    '``--no-full-remap`` is faster but can reduce ortho image quality.',
)
alpha_option = click.option(
    '-a',
    '--alpha',
    type=click.FloatRange(0, 1),
    nargs=1,
    default=1,
    show_default=True,
    help='Scaling of the ``--no-full-remap`` undistorted image: ``0`` includes the largest '
    'source image portion that allows all undistorted pixels to be valid.  ``1`` includes all '
    'source pixels in the undistorted image.',
)
driver_option = click.option(
    '-dv',
    '--driver',
    type=click.Choice(Driver, case_sensitive=True),
    default=common._default_out_config['driver'],
    show_default=True,
    help=f'Ortho image driver.',
)
write_mask_option = click.option(
    '-wm/-nwm',
    '--write-mask/--no-write-mask',
    type=click.BOOL,
    default=common._default_out_config['write_mask'],
    show_default='true for jpeg compression.',
    help='Mask valid pixels with an internal mask (``--write-mask``), or with a nodata value '
    'based on ``--dtype`` (``--no-write-mask``). An internal mask helps remove nodata noise '
    'caused by lossy compression.',
)
dtype_option = click.option(
    '-dt',
    '--dtype',
    type=click.Choice(list(common._nodata_vals.keys()), case_sensitive=True),
    default=common._default_out_config['dtype'],
    show_default='source image data type.',
    help=f'Ortho image data type.',
)
compress_option = click.option(
    '-cm',
    '--compress',
    type=click.Choice(Compress, case_sensitive=True),
    default=common._default_out_config['compress'],
    show_default="jpeg for uint8 --dtype, deflate otherwise",
    help=f'Ortho image compression.',
)
build_ovw_option = click.option(
    '-bo/-nbo',
    '--build-ovw/--no-build-ovw',
    type=click.BOOL,
    default=True,
    show_default=True,
    help='Build overviews for the ortho image(s).',
)
creation_option = click.option(
    '-co',
    '--creation-option',
    'creation_options',
    metavar='NAME=VALUE',
    multiple=True,
    default=(),
    show_default='auto',
    callback=_creation_option_cb,
    help='Creation option(s) for the ortho image(s).  If supplied, ``--compress`` is '
    'ignored.  See the `GDAL docs <https://gdal.org/en/latest/drivers/raster/index.html>`_  '
    'for details.',
)
export_params_option = click.option(
    '-e',
    '--export-params',
    is_flag=True,
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Export camera parameters to orthority format file(s), and exit.',
)
out_dir_option = click.option(
    '-od',
    '--out-dir',
    type=click.Path(file_okay=False),
    default=str(Path.cwd()),
    show_default='current working',
    callback=_dir_cb,
    help='Path / URI of the output file directory.',
)
overwrite_option = click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Overwrite existing output(s).',
)


@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity.')
@click.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=__version__, message='%(version)s')
@click.pass_context
def cli(ctx: click.Context, verbose, quiet) -> None:
    """Orthorectification toolkit."""
    # configure logging
    verbosity = verbose - quiet
    _configure_logging(verbosity)

    # enter context managers for sub-command raster operations
    env = rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False, GDAL_TIFF_INTERNAL_MASK=True)
    ctx.with_resource(common.suppress_no_georef())
    ctx.with_resource(env)

    # redirect logs through tqdm.write, so they do not interfere with progress bars
    ctx.with_resource(logging_redirect_tqdm([logging.getLogger(__package__)], tqdm_class=tqdm))


@cli.command(
    cls=RstCommand,
    short_help='Orthorectify with interior and exterior parameter files.',
    epilog='See https://orthority.readthedocs.io/ for more detail on file formats and usage.',
)
@src_files_arg
@dem_file_option
@int_param_file_option
@ext_param_file_option
@crs_option
@resolution_option
@dem_band_option
@interp_option
@dem_interp_option
@per_band_option
@full_remap_option
@alpha_option
@lla_crs_option
@radians_option
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@creation_option
@driver_option
@export_params_option
@out_dir_option
@overwrite_option
def frame(
    int_param_file: OpenFile,
    ext_param_file: OpenFile,
    crs: rio.CRS,
    full_remap: bool,
    alpha: float,
    lla_crs: rio.CRS,
    radians: bool,
    **kwargs,
) -> None:
    """
    Orthorectify SOURCE images with frame camera model(s) defined by interior and exterior
    parameter files.

    SOURCE images can be specified with paths, URIs or path / URI wildcard patterns.

    Interior parameters are supported in Orthority :file:`.yaml`, OpenDroneMap
    :file:`cameras.json`, and OpenSfM :file:`reconstruction.json` formats.  Exterior parameters
    are supported in Orthority :file:`.geojson`, CSV, and OpenSfM :file:`reconstruction.json`
    formats.  Parameter file extensions are used to distinguish their format.

    The :option:`--int-param <oty-frame --int-param>` and :option:`--ext-param <oty-frame
    --ext-param>` options are required.  The :option:`--dem <oty-frame --dem>` option is
    required, except when exporting camera parameters with :option:`--export-params <oty-frame
    --export-params>`.  Depending on the input file formats, :option:`--crs <oty-frame --crs>`
    may also be required::

        oty frame --dem dem.tif --int-param int_param.yaml --ext-param ext_param.csv --crs EPSG:32651 source*.tif

    Camera parameters can be exported to Orthority format files with :option:`--export-params
    <oty-frame --export-params>`::

        oty frame --int-param reconstruction.json --ext-param reconstruction.json --export-params

    Ortho images and exported files are placed in the current working directory by default. This
    can be changed with :option:`--out-dir <oty-frame --out-dir>`.
    """
    # create camera factory
    try:
        cameras = FrameCameras(
            int_param_file,
            ext_param_file,
            io_kwargs=dict(crs=crs, lla_crs=lla_crs, radians=radians),
            cam_kwargs=dict(distort=full_remap, alpha=alpha),
        )
    except CrsMissingError:
        raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')
    except CrsError as ex:
        raise click.BadParameter(str(ex), param_hint="'-c' / '--crs'")
    except (FileNotFoundError, OrthorityError) as ex:
        raise click.UsageError(str(ex))

    # get factory CRS if not set already
    crs = crs or cameras.crs

    # orthorectify
    _ortho(cameras=cameras, crs=crs, **kwargs)


@cli.command(
    cls=RstCommand,
    short_help='Orthorectify with image EXIF / XMP tags.',
    epilog='See https://orthority.readthedocs.io/ for more detail.',
)
@src_files_arg
@dem_file_option
@crs_option
@resolution_option
@dem_band_option
@interp_option
@dem_interp_option
@per_band_option
@full_remap_option
@alpha_option
@lla_crs_option
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@creation_option
@driver_option
@export_params_option
@out_dir_option
@overwrite_option
def exif(
    src_files: list[OpenFile],
    crs: rio.CRS,
    full_remap: bool,
    alpha: float,
    lla_crs: rio.CRS,
    **kwargs,
) -> None:
    """
    Orthorectify SOURCE images with frame camera model(s) defined by image EXIF / XMP tags.

    SOURCE images can be specified with paths, URIs or path / URI wildcard patterns.

    SOURCE image tags should include DewarpData, focal length & sensor size or 35mm equivalent
    focal length; camera position and camera roll, pitch & yaw.  DewarpData is converted to a
    Brown model if it is present, otherwise a pinhole model is used.  Pinhole approximation and
    tag value accuracy affect ortho image accuracy.

    The :option:`--dem <oty-exif --dem>` option is required, except when exporting camera
    parameters with :option:`--export-params <oty-exif --export-params>`.  If :option:`--crs
    <oty-exif --crs>` is not supplied, a UTM world / ortho CRS is auto-determined from the camera
    positions::

        oty exif --dem dem.tif source*.tif

    Camera parameters can be exported to Orthority format files with :option:`--export-params
    <oty-exif --export-params>`::

        oty exif ---export-params source*.tif

    Ortho images and exported files are placed in the current working directory by default. This
    can be changed with :option:`--out-dir <oty-exif --out-dir>`.
    """
    # set up progress bar args
    tqdm_kwargs = common.get_tqdm_kwargs(desc='Reading parameters', unit='files', leave=False)

    # create camera factory
    try:
        cameras = FrameCameras.from_images(
            src_files,
            io_kwargs=dict(crs=crs, lla_crs=lla_crs, progress=tqdm_kwargs),
            cam_kwargs=dict(distort=full_remap, alpha=alpha),
        )
    except CrsError as ex:
        raise click.BadParameter(str(ex), param_hint="'-c' / '--crs'")
    except (FileNotFoundError, OrthorityError) as ex:
        raise click.BadParameter(str(ex), param_hint="'SOURCE...'")

    # get auto UTM CRS, if CRS not set already
    crs = crs or cameras.crs

    # orthorectify
    _ortho(src_files=src_files, cameras=cameras, crs=crs, **kwargs)


@cli.command(
    cls=RstCommand,
    short_help='Orthorectify with OpenDroneMap outputs.',
    epilog='See https://orthority.readthedocs.io/ for more detail.',
)
@click.option(
    '-dd',
    '--dataset-dir',
    type=click.Path(file_okay=False),
    required=True,
    default=None,
    callback=_dir_cb,
    help='Path / URI of the ODM dataset to process.',
)
@crs_option
@resolution_option
@interp_option
@dem_interp_option
@per_band_option
@full_remap_option
@alpha_option
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@creation_option
@driver_option
@export_params_option
@click.option(
    '-od',
    '--out-dir',
    type=click.Path(file_okay=False),
    default=None,
    show_default='<dataset-dir>/orthority',
    callback=_odm_out_dir_cb,
    help='Path / URI of the output file directory.',
)
@overwrite_option
def odm(
    dataset_dir: OpenFile,
    crs: rio.CRS,
    resolution: tuple[float, float],
    full_remap: bool,
    alpha: float,
    out_dir: OpenFile,
    **kwargs,
) -> None:
    """
    Orthorectify images in a processed OpenDroneMap dataset that includes a DSM.

    The images, DSM and camera models are read from the dataset. If :option:`--crs <oty-odm
    --crs>` is not supplied (recommended), the world / ortho CRS is also read from the dataset.
    :option:`--dataset-dir <oty-odm --dataset-dir>` is the only required option::

        oty odm --dataset-dir dataset

    Camera parameters can be exported to Orthority format files with :option:`--export-params
    <oty-odm --export-params>`::

        oty odm --dataset-dir dataset --export-params

    Ortho images and exported files are placed in the :file:`{dataset}/orthority` subdirectory
    by default.  This can be changed with :option:`--out-dir <oty-odm --out-dir>`.
    """
    # find source images
    src_exts = ['.jpg', '.jpeg', '.tif', '.tiff']
    images_ofile = common.join_ofile(dataset_dir, 'images/')
    src_files = images_ofile.fs.find(images_ofile.path)  # use existing fs rather than open_files
    src_files = [sf for sf in src_files if sf[sf.rfind('.') :].lower() in src_exts]
    src_files = [OpenFile(images_ofile.fs, sf, 'rb') for sf in src_files]
    if len(src_files) == 0:
        raise click.BadParameter(
            f"No images found in '<dataset-dir>/images/'.", param_hint="'-dd' / '--dataset-dir'"
        )

    # read CRS from DSM
    dem_ofile = common.join_ofile(dataset_dir, 'odm_dem/dsm.tif', mode='rb')
    try:
        dem_ctx = common.OpenRaster(dem_ofile, 'r')
    except FileNotFoundError as ex:
        raise click.BadParameter(
            f"No DSM found in '<dataset-dir>/odm_dem/'. {str(ex)}",
            param_hint="'-dd' / '--dataset-dir'",
        )
    with dem_ctx as dem_im:
        crs = crs or dem_im.crs

    # create camera factory
    rec_ofile = common.join_ofile(dataset_dir, 'opensfm/reconstruction.json', mode='rt')
    try:
        cameras = FrameCameras(
            int_param=rec_ofile,
            ext_param=rec_ofile,
            io_kwargs=dict(crs=crs),
            cam_kwargs=dict(distort=full_remap, alpha=alpha),
        )
    except FileNotFoundError as ex:
        raise click.BadParameter(
            f"No 'reconstruction.json' file found in '<dataset-dir>/opensfm/'. {str(ex)}",
            param_hint="'-dd' / '--dataset-dir'",
        )
    except OrthorityError as ex:
        raise click.UsageError(str(ex))

    # orthorectify
    _ortho(
        src_files=src_files,
        dem_file=dem_ofile,
        cameras=cameras,
        crs=crs,
        resolution=resolution,
        dem_band=1,
        out_dir=out_dir,
        **kwargs,
    )


@cli.command(
    # make these short help messages consistent between frame / rpc models
    cls=RstCommand,
    short_help='Orthorectify with RPC camera model(s).',
    epilog='See https://orthority.readthedocs.io/ for more detail.',
)
@src_files_arg
@dem_file_option
@click.option(
    '-rp',
    '--rpc-param',
    'rpc_param_file',
    type=click.Path(dir_okay=False),
    default=None,
    callback=partial(_file_cb, mode='rt'),
    help='Path / URI of an Orthority RPC parameter file.',
)
@click.option(
    '-gr',
    '--gcp-refine',
    type=click.Path(dir_okay=False),
    default=None,
    callback=_gcp_refine_cb,
    help="Refine camera model(s) with GCP(s).  Can be supplied with the path / URI value of an "
    "Orthority format GCP file, or with 'tags' to read GCPs from source image tags.",
)
@click.option(
    '-rm',
    '--refine-method',
    type=click.Choice(RpcRefine, case_sensitive=True),
    default=_default_rpc_refine_method,
    show_default=True,
    help='Refinement method to use with ``--gcp-refine``.',
)
@crs_option
@resolution_option
@dem_band_option
@interp_option
@dem_interp_option
@per_band_option
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@creation_option
@driver_option
@export_params_option
@out_dir_option
@overwrite_option
def rpc(
    src_files: list[OpenFile],
    rpc_param_file: OpenFile,
    gcp_refine: OpenFile | bool,
    refine_method: RpcRefine,
    crs: rio.CRS,
    **kwargs,
) -> None:
    """
    Orthorectify SOURCE images with RPC camera model(s).

    SOURCE images can be specified with paths, URIs or path / URI wildcard patterns.

    Camera parameters are read from SOURCE image tags / sidecar files, or from
    :option:`--rpc-param <oty-rpc --rpc-param>` if provided.

    The :option:`--dem <oty-rpc --dem>` option is required, except when exporting camera
    parameters with :option:`--export-params <oty-rpc --export-params>`.  If :option:`--crs
    <oty-rpc --crs>` is not supplied, a 3D WGS84 geographic world / ortho CRS is used::

        oty rpc --dem dem.tif source*.tif

    Camera parameters can be refined with GCPs using :option:`--gcp-refine <oty-rpc
    --gcp-refine>`::

        oty rpc --dem dem.tif --gcp-refine tags source*.tif

    Camera parameters can be exported to Orthority format file(s) with :option:`--export-params
    <oty-rpc --export-params>`.  If :option:`--export-params <oty-rpc --export-params>` is
    supplied with :option:`--gcp-refine <oty-rpc --gcp-refine>`, the refined parameters are
    exported as well as the GCPs::

        oty rpc ---export-params --gcp-refine tags source*.tif

    Ortho images and exported files are placed in the current working directory by default. This
    can be changed with :option:`--out-dir <oty-rpc --out-dir>`.
    """
    # set CRS to the RPC camera default (WGS84) if no CRS supplied, otherwise pass user CRS in
    # cam_kwargs
    if not crs:
        crs = rio.CRS.from_epsg(4979)
        cam_kwargs = {}
    else:
        cam_kwargs = dict(crs=crs)

    if rpc_param_file:
        # create camera factory from parameter file
        try:
            cameras = RpcCameras(rpc_param_file, cam_kwargs=cam_kwargs)
        except (FileNotFoundError, OrthorityError) as ex:
            raise click.BadParameter(str(ex), param_hint="'-rp' / '--rpc-param'")
    else:
        # set up progress bar args
        tqdm_kwargs = common.get_tqdm_kwargs(desc='Reading parameters', unit='files', leave=False)

        # create camera factory from image tags / sidecar file(s)
        try:
            cameras = RpcCameras.from_images(
                src_files, io_kwargs=dict(progress=tqdm_kwargs), cam_kwargs=cam_kwargs
            )
        except (FileNotFoundError, OrthorityError) as ex:
            raise click.BadParameter(str(ex), param_hint="'SOURCE...'")

    if gcp_refine is not None:
        # refine model(s) with GCPs
        fit_kwargs = dict(method=refine_method)
        try:
            if str(gcp_refine).lower() == 'tags':
                # set up progress bar args
                tqdm_kwargs = common.get_tqdm_kwargs(desc='Reading GCPs', unit='files', leave=False)
                cameras.refine(
                    src_files, io_kwargs=dict(progress=tqdm_kwargs), fit_kwargs=fit_kwargs
                )
            else:
                cameras.refine(gcp_refine, fit_kwargs=fit_kwargs)
        except (FileNotFoundError, OrthorityError) as ex:
            raise click.BadParameter(str(ex), param_hint="'-gr / --gcp-refine'")

    # orthorectify
    _ortho(src_files=src_files, cameras=cameras, crs=crs, **kwargs)


@cli.command(
    cls=RstCommand,
    short_help='Pan-sharpen.',
    epilog='See https://orthority.readthedocs.io/ for more detail.',
)
@click.option(
    '-p',
    '--pan',
    'pan_file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    callback=partial(_file_cb, mode='rb'),
    help='Path / URI of the panchromatic image.',
)
@click.option(
    '-ms',
    '--multispectral',
    'ms_file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    callback=partial(_file_cb, mode='rb'),
    help='Path / URI of the multispectral image.',
)
@click.option(
    '-of',
    '--out-file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    callback=partial(_file_cb, mode='wb'),
    help='Path / URI of the pan-sharpened image.',
)
@click.option(
    '-pi',
    '--pan-index',
    type=click.INT,
    default=PanSharpen._default_alg_config['pan_index'],
    show_default=True,
    help='Index of the panchromatic band to use (1 based).',
)
@click.option(
    '-mi',
    '--ms-index',
    'ms_indexes',
    type=click.INT,
    multiple=True,
    default=PanSharpen._default_alg_config['ms_indexes'],
    show_default='all non-alpha bands',
    help='Indexes of the multispectral bands to use (1 based).',
)
@click.option(
    '-w',
    '--weight',
    'weights',
    type=click.FLOAT,
    multiple=True,
    default=PanSharpen._default_alg_config['weights'],
    show_default='auto',
    callback=_weights_cb,
    help='Multispectral to panchromatic weights (≥0).',
)
@click.option(
    '-i',
    '--interp',
    type=click.Choice(Interp, case_sensitive=True),
    default=PanSharpen._default_alg_config['interp'],
    show_default=True,
    help=f'Interpolation method for upsampling the multispectral image.',
)
@write_mask_option
@click.option(
    '-dt',
    '--dtype',
    type=click.Choice(list(common._nodata_vals.keys()), case_sensitive=True),
    default=common._default_out_config['dtype'],
    show_default='source image data type.',
    help=f'Pan-sharpened image data type.',
)
@click.option(
    '-cm',
    '--compress',
    type=click.Choice(Compress, case_sensitive=True),
    default=common._default_out_config['compress'],
    show_default="jpeg for uint8 --dtype, deflate otherwise",
    help=f'Pan-sharpened image compression.',
)
@click.option(
    '-bo/-nbo',
    '--build-ovw/--no-build-ovw',
    type=click.BOOL,
    default=True,
    show_default=True,
    help='Build overviews for the pan-sharpened image.',
)
@click.option(
    '-co',
    '--creation-option',
    'creation_options',
    metavar='NAME=VALUE',
    multiple=True,
    default=(),
    show_default='auto',
    callback=_creation_option_cb,
    help='Creation option(s) for the pan-sharpened image.  If supplied, ``--compress`` is '
    'ignored.  See the `GDAL docs <https://gdal.org/en/latest/drivers/raster/index.html>`_  '
    'for details.',
)
@click.option(
    '-dv',
    '--driver',
    type=click.Choice(Driver, case_sensitive=True),
    default=common._default_out_config['driver'],
    show_default=True,
    help=f'Pan-sharpened image driver.',
)
@overwrite_option
def sharpen(
    pan_file: OpenFile,
    ms_file: OpenFile,
    pan_index: int,
    ms_indexes: list[int],
    weights: list[float],
    **kwargs,
) -> None:
    """
    Increases the resolution of a multispectral image to that of a panchromatic image using the
    Gram-Schmidt pan-sharpening method.

    Panchromatic and multispectral image bounds should overlap if they are georeferenced. If one
    or both of the images are not georeferenced, they are assumed to having matching bounds.

    Panchromatic and multispectral images are specified with :option:`--pan <oty-sharpen --pan>`
    and :option:`--multispectral <oty-sharpen --multispectral>`, and the pan-sharpened output
    with :option:`--out-file <oty-sharpen --out-file>`::

        oty sharpen --pan pan.tif --multispectral ms.tif --out-file pan_sharp.tif

    A subset of multispectral bands for sharpening can be specified with :option:`--ms-index
    <oty-sharpen --ms-index>`::

        oty sharpen -mi 3 -mi 2 -mi 1 --pan pan.tif --multispectral ms.tif --out-file pan_sharp.tif

    Multispectral to panchromatic weights are estimated from the images by default.  User values
    can be provided with :option:`--weight <oty-sharpen --weight>`.  There should be as many
    weights as multispectral bands::

        oty sharpen -mi 3 -mi 2 -mi 1 -w 0.4 -w 0.5 -w 0.3 --pan pan.tif --multispectral ms.tif --out-file pan_sharp.tif
    """
    # open pan & ms files
    try:
        pan_ctx = common.OpenRaster(pan_file, 'r')
    except FileNotFoundError as ex:
        raise click.BadParameter(str(ex), param_hint="'-p' / '--pan'")

    try:
        ms_ctx = common.OpenRaster(ms_file, 'r')
    except FileNotFoundError as ex:
        raise click.BadParameter(str(ex), param_hint="'-ms' / '--multispectral'")

    with pan_ctx as pan_im, ms_ctx as ms_im:
        # validate indexes
        if pan_index <= 0 or pan_index > pan_im.count:
            pan_name = common.get_filename(pan_im)
            raise click.BadParameter(
                f"Pan index {pan_index} out of range for '{pan_name}' with {pan_im.count} band(s).",
                param_hint="'-pi' / '--pan-index'",
            )

        ms_err_indexes = np.array(ms_indexes)
        ms_err_indexes = ms_err_indexes[(ms_err_indexes <= 0) | (ms_err_indexes > ms_im.count)]
        if len(ms_err_indexes) > 0:
            ms_name = common.get_filename(ms_im)
            raise click.BadParameter(
                f"Multispectral indexes {tuple(ms_err_indexes.tolist())} are out of range for "
                f"'{ms_name}' with {ms_im.count} band(s).",
                param_hint="'-mi' / '--ms-index'",
            )

        # pan-sharpen
        try:
            pan_sharp = PanSharpen(pan_im, ms_im)
            pan_sharp.process(
                pan_index=pan_index, ms_indexes=ms_indexes, weights=weights, progress=True, **kwargs
            )
        except (FileExistsError, OrthorityError) as ex:
            raise click.UsageError(str(ex))


def _simple_ortho(
    src_im_file,
    dem_file,
    pos_ori_file,
    ortho_dir=None,
    read_conf=None,
    write_conf=None,
    verbosity=2,
):
    """
    Legacy orthorectification (deprecated).

    Parameters
    ----------
    src_im_file: str, pathlib.Path
        Source image file(s).
    dem_file: str, pathlib.Path
        DEM file covering source image file(s).
    pos_ori_file: str, pathlib.Path
        Position and orientation file for source image file(s).
    ortho_dir: str, pathlib.Path, optional
        Output directory.
    read_conf: str, pathlib.Path, optional
        Read configuration from this file.
    write_conf: str, pathlib.Path, optional
        Write configuration to this file and exit.
    verbosity: int
        Logging verbosity 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)
    """

    def check_args(src_im_file, dem_file, pos_ori_file, ortho_dir=None):
        """Check arguments for errors."""
        # check files exist
        for src_im_file_spec in src_im_file:
            src_im_file_path = Path(src_im_file_spec)
            if len(list(src_im_file_path.parent.glob(src_im_file_path.name))) == 0:
                raise Exception(f'Could not find any source image(s) matching {src_im_file_spec}')

        if not Path(dem_file).exists():
            raise Exception(f'DEM file {dem_file} does not exist')

        if not Path(pos_ori_file).exists():
            raise Exception(f'Camera position and orientation file {pos_ori_file} does not exist')

        # check and create ortho_dir if necessary
        if ortho_dir is not None:
            ortho_dir = Path(ortho_dir)
            if not ortho_dir.exists():
                logger.warning(f'Creating ortho directory {ortho_dir}')
                ortho_dir.mkdir()

    try:
        # set logging level
        if verbosity is not None:
            _configure_logging(2 - verbosity)

        logger.warning(
            "This command is deprecated and will be removed in a future version.  Please switch to"
            " 'oty' and its sub-commands.\n"
        )

        # read configuration
        if read_conf is None:
            config_filename = root_path.joinpath('config.yaml')
        else:
            config_filename = Path(read_conf)

        if not config_filename.exists():
            raise Exception(f'Config file {config_filename} does not exist')

        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)

        # write configuration if requested and exit
        if write_conf is not None:
            out_config_filename = Path(write_conf)
            with open(out_config_filename, 'w') as f:
                yaml.dump(config, stream=f)
            logger.info(f'Wrote config to {out_config_filename}')
            return

        # prepare ortho config
        ortho_config = config.get('ortho', {})
        crs = ortho_config.pop('crs', None)
        dem_band = ortho_config.pop('dem_band', Ortho._default_alg_config['dem_band'])
        for key in ['driver', 'tile_size', 'nodata', 'interleave', 'photometric']:
            if key in ortho_config:
                ortho_config.pop(key)
                logger.warning(f"The '{key}' option is deprecated.")
        for key in ['interp', 'dem_interp']:
            if ortho_config.get(key, None) == 'cubic_spline':
                logger.warning(
                    f"'cubic_spline' interpolation is deprecated, using '{key}'='cubic'."
                )
                ortho_config[key] = 'cubic'
        distort = ortho_config.pop('full_remap', True)

        # prepare camera config
        camera = None
        camera_config = config['camera']
        camera_type = CameraType(camera_config.get('type', 'pinhole'))
        camera_config = {k: v for k, v in camera_config.items() if k not in ['name', 'type']}

        # checks paths etc
        check_args(src_im_file, dem_file, pos_ori_file, ortho_dir=ortho_dir)

        # read camera position and rotation
        with open(pos_ori_file, 'r', newline='') as f:
            reader = csv.DictReader(
                f,
                delimiter=' ',
                fieldnames=['file', 'x', 'y', 'z', 'omega', 'phi', 'kappa'],
            )
            cam_pos_orid = {
                row['file']: {k: float(row[k]) for k in reader.fieldnames[1:]} for row in reader
            }

        # loop through image file(s) or wildcard(s), or combinations thereof
        for src_im_file_spec in src_im_file:
            src_im_file_path = Path(src_im_file_spec)
            for src_filename in src_im_file_path.parent.glob(src_im_file_path.name):
                if src_filename.stem not in cam_pos_orid:
                    raise Exception(f'Could not find {src_filename.stem} in {pos_ori_file}')

                im_pos_ori = cam_pos_orid[src_filename.stem]
                opk = np.radians((im_pos_ori['omega'], im_pos_ori['phi'], im_pos_ori['kappa']))
                xyz = np.array((im_pos_ori['x'], im_pos_ori['y'], im_pos_ori['z']))

                # set ortho filename
                ortho_dir = src_filename.parent if not ortho_dir else ortho_dir
                ortho_filename = Path(ortho_dir).joinpath(src_filename.stem + '_ORTHO.tif')

                # Get src size
                with common.suppress_no_georef(), rio.open(src_filename) as src_im:
                    im_size = np.float64([src_im.width, src_im.height])

                if not camera:
                    # create a new camera
                    camera = create_camera(
                        camera_type, **camera_config, xyz=xyz, opk=opk, distort=distort
                    )
                else:
                    # update existing camera
                    camera.update(xyz, opk)

                if np.any(im_size != camera._im_size):
                    logger.warning(
                        f'{src_filename.name} size ({im_size}) does not match configuration.'
                    )

                # create Ortho and orthorectify
                logger.info(f'Orthorectifying {src_filename.name}:')
                ortho_im = Ortho(src_filename, dem_file, camera, crs=crs, dem_band=dem_band)
                ortho_im.process(ortho_filename, **ortho_config)

    except Exception as ex:
        logger.error('Exception: ' + str(ex))
        raise ex


def _get_simple_ortho_parser():
    """Return an argparse parser for the legacy ``simple-ortho`` CLI."""
    parser = argparse.ArgumentParser(
        description='Orthorectify an image with known DEM and camera model.'
    )
    parser.add_argument(
        "src_im_file",
        help="path(s) and or wildcard(s) specifying the source image file(s)",
        type=str,
        metavar='src_im_file',
        nargs='+',
    )
    parser.add_argument("dem_file", help="path to the DEM file", type=str)
    parser.add_argument(
        "pos_ori_file", help="path to the camera position and orientation file", type=str
    )
    parser.add_argument(
        "-od",
        "--ortho-dir",
        help="write ortho image(s) to this directory (default: write ortho image(s) to source directory)",
        type=str,
    )
    parser.add_argument(
        "-rc",
        "--read-conf",
        help="read custom config from this path (default: use config.yaml in simple-ortho root)",
        type=str,
    )
    parser.add_argument(
        "-wc", "--write-conf", help="write default config to this path and exit", type=str
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=[1, 2, 3, 4],
        help="logging level: 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)",
        type=int,
    )
    return parser


def simple_ortho(argv=None):
    """Entry point to legacy ``simple-ortho`` CLI."""
    args = _get_simple_ortho_parser().parse_args(argv)
    _simple_ortho(**vars(args))


if __name__ == '__main__':
    cli()

##
