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
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

import click
import numpy as np
import rasterio as rio
import yaml
from rasterio.errors import RasterioIOError

from orthority import param_io, root_path
from orthority.camera import create_camera
from orthority.enums import CameraType, Compress, Interp
from orthority.errors import CrsMissingError, DemBandError, ParamFileError
from orthority.ortho import Ortho
from orthority.utils import open_text, suppress_no_georef
from orthority.version import __version__

logger = logging.getLogger(__name__)


class PlainInfoFormatter(logging.Formatter):
    """
    Logging formatter to format INFO logs without the module name etc.

    prefix.
    """

    def format(self, record: logging.LogRecord):
        if record.levelno == logging.INFO:
            self._style._fmt = '%(message)s'
        else:
            self._style._fmt = '%(levelname)s:%(name)s: %(message)s'
        return super().format(record)


class RstCommand(click.Command):
    """click.Command subclass for formatting help with RST markup."""

    # TODO: can we lose this?
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
            # # convert from RST cross-ref '`<name> <link>`__' to 'name'
            # r'`(.*?)(\s+<.*?>)?`_+': r'\g<1>',
            # convert from RST cross-ref '`<name> <link>`__' to 'link'
            r'`(.*?)<(.*?)>`_+': r'\g<2>',
        }

        def reformat_text(text, width, **kwargs):
            for sub_key, sub_value in sub_strings.items():
                text = re.sub(sub_key, sub_value, text, flags=re.DOTALL)
            wr_text = self.click_wrap_text(text, width, **kwargs)
            # change double newline to single newline separated list
            return re.sub('\n\n(\s*?)- ', '\n- ', wr_text, flags=re.DOTALL)

        click.formatting.wrap_text = reformat_text
        return click.Command.get_help(self, ctx)


class CondReqOption(click.Option):
    """
    click.Option subclass whose ``required`` attribute is turned off when any the ``not_required``
    options is present.
    """

    # adapted from https://stackoverflow.com/questions/44247099/click-command-line-interfaces-make-options-required-if-other-optional-option-is
    def __init__(self, *args, not_required: list[str] | None = None, **kwargs):
        self.not_required = not_required or []
        click.Option.__init__(self, *args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        self.required = not any([opt in self.not_required for opt in opts])
        return click.Option.handle_parse_result(self, ctx, opts, args)


def _configure_logging(verbosity: int):
    """Configure python logging level."""
    # adapted from rasterio: https://github.com/rasterio/rasterio
    log_level = max(10, 20 - 10 * verbosity)

    # apply config to package logger, rather than root logger
    pkg_logger = logging.getLogger('orthority')
    formatter = PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)
    logging.captureWarnings(True)


def _read_crs(crs: str):
    """Read a CRS from a string, text file, or image file."""
    crs_file = Path(crs)
    if crs_file.suffix.lower() in ['.tif', '.tiff']:
        # read CRS from geotiff path / URL
        with suppress_no_georef(), rio.open(crs, 'r') as im:
            crs = im.crs
    else:
        if crs_file.exists() or urlparse(crs).scheme in ['https', 'http', 'ftp']:
            # read string from text file path / URL
            with open_text(crs) as f:
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
        if not crs.is_projected:
            raise click.BadParameter(f"CRS should be a projected system.", param=param)
    return crs


def _lla_crs_cb(ctx: click.Context, param: click.Parameter, lla_crs: str):
    """Click callback to validate and parse the LLA CRS."""
    if lla_crs is not None:
        try:
            lla_crs = _read_crs(lla_crs)
        except Exception as ex:
            raise click.BadParameter(f'{str(ex)}', param=param)
        if not lla_crs.is_geographic:
            raise click.BadParameter(f"CRS should be a geographic system.", param=param)
    return lla_crs


def _resolution_cb(ctx: click.Context, param: click.Parameter, resolution: tuple):
    """Click callback to validate and parse the resolution."""
    if len(resolution) == 1:
        resolution *= 2
    elif len(resolution) > 2:
        raise click.BadParameter(f'At most two resolution values should be specified.', param=param)
    return resolution


def _dataset_dir_cb(ctx: click.Context, param: click.Parameter, dataset_dir: Path):
    """Click callback to validate the ODM dataset directory."""
    req_paths = ['opensfm/reconstruction.json', 'odm_dem/dsm.tif', 'images']
    for req_path in req_paths:
        req_path = dataset_dir.joinpath(req_path)
        if not req_path.exists():
            raise click.BadParameter(f"Could not find '{req_path}'.", param=param)
    return dataset_dir


def _ortho(
    src_files: tuple[str, ...],
    dem_file: str,
    int_param_dict: dict[str, dict],
    ext_param_dict: dict[str, dict],
    crs: rio.CRS,
    dem_band: int,
    alpha: float,
    export_params: bool,
    out_dir: Path,
    overwrite: bool,
    **kwargs,
):
    """
    Orthorectify images given a DEM filename, and interior & exterior parameter dictionaries.

    Backend function for orthorectification sub-commands.
    """
    if export_params:
        # convert interior / exterior params to oty format files
        logger.info('Writing parameter files...')
        int_param_file = out_dir.joinpath('int_param.yaml')
        ext_param_file = out_dir.joinpath('ext_param.geojson')
        param_io.write_int_param(int_param_file, int_param_dict, overwrite)
        param_io.write_ext_param(ext_param_file, ext_param_dict, crs, overwrite)
        return
    elif not dem_file:
        raise click.MissingParameter(param_hint="'-d' / '--dem'", param_type='option')

    # open & validate dem_file path / URL (open it once here so it is not opened repeatedly in
    # orthorectification below)
    try:
        dem_im = rio.open(dem_file, 'r')
    except RasterioIOError as ex:
        raise click.BadParameter(str(ex), param_hint="'-d' / '--dem'")

    cameras = {}
    with dem_im:
        for src_i, src_file in enumerate(src_files):
            # get exterior params for src_file
            src_file_path = Path(src_file)
            ext_param = ext_param_dict.get(
                src_file_path.name, ext_param_dict.get(src_file_path.stem, None)
            )
            if not ext_param:
                raise click.BadParameter(
                    f"Could not find parameters for '{src_file_path.name}'.",
                    param_hint="'-ep' / '--ext-param'",
                )

            # get interior params for ext_param
            cam_id = ext_param.pop('camera')
            if cam_id:
                if cam_id not in int_param_dict:
                    raise click.BadParameter(
                        f"Could not find parameters for camera '{cam_id}'.",
                        param_hint="'-ip' / '--int-param'",
                    )
                int_param = int_param_dict[cam_id]
            elif len(int_param_dict) == 1:
                int_param = list(int_param_dict.values())[0]
            else:
                raise click.BadParameter(
                    f"'camera' ID for '{src_file_path.name}' should be specified.",
                    param_hint="'-ep' / '--ext-param'",
                )

            # create camera on first use and update exterior parameters
            if cam_id not in cameras:
                cameras[cam_id] = create_camera(**int_param, alpha=alpha)
            cameras[cam_id].update(**ext_param)

            # open & validate src_file path / URL (open it once here so it is not opened repeatedly
            # in orthorectification below)
            try:
                src_im = rio.open(src_file, 'r')
            except RasterioIOError as ex:
                raise click.BadParameter(str(ex), param_hint='SOURCE...')

            with src_im:
                # create ortho object & filename
                try:
                    ortho = Ortho(src_im, dem_im, cameras[cam_id], crs, dem_band=dem_band)
                except DemBandError as ex:
                    raise click.BadParameter(str(ex), param_hint="'-db' / '--dem-band'")
                except CrsMissingError:
                    raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')

                ortho_file = out_dir.joinpath(f'{src_file_path.stem}_ORTHO.tif')

                # orthorectify
                logger.info(
                    f"Orthorectifying '{src_file_path.name}' ({src_i + 1} of {len(src_files)}):"
                )
                ortho.process(ortho_file, overwrite=overwrite, **kwargs)


# Define click options that are common to more than one command
src_files_arg = click.argument(
    'src_files',
    nargs=-1,
    metavar='SOURCE...',
    type=click.Path(dir_okay=False),
    # help='Path / URL of source image(s) to be orthorectified..'
)
dem_file_option = click.option(
    '-d',
    '--dem',
    'dem_file',
    type=click.Path(dir_okay=False, readable=True),
    required=True,
    default=None,
    help='Path / URL of a DEM image covering the source image(s).',
    cls=CondReqOption,
    not_required=['export_params'],
)
int_param_file_option = click.option(
    '-ip',
    '--int-param',
    'int_param_file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    help='Path / URL of an interior parameter file.',
)
ext_param_file_option = click.option(
    '-ep',
    '--ext-param',
    'ext_param_file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    help='Path / URL of an exterior parameter file.',
)
crs_option = click.option(
    '-c',
    '--crs',
    type=click.STRING,
    default=None,
    show_default='auto',
    callback=_crs_cb,
    help='CRS of ortho image(s) and any projected coordinate exterior parameters as an EPSG, '
    'proj4, or WKT string; or path of a text file containing string.',
)
lla_crs_option = click.option(
    '-lc',
    '--lla-crs',
    type=click.STRING,
    default='EPSG:4979',
    show_default=True,
    callback=_lla_crs_cb,
    help='CRS of any geographic coordinate exterior parameters as an EPSG, proj4, or WKT string; '
    'or path of a text file containing string',
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
    help='Ortho image resolution in units of the ``--crs`` (usually meters).  Can be used '
    'twice for non-square pixels: ``--res PIXEL_WIDTH --res PIXEL_HEIGHT``.',
)
dem_band_option = click.option(
    '-db',
    '--dem-band',
    type=click.INT,
    nargs=1,
    default=Ortho._default_config['dem_band'],
    show_default=True,
    help='Index of the DEM band to use (1 based).',
)
interp_option = click.option(
    '-i',
    '--interp',
    type=click.Choice(Interp, case_sensitive=False),
    default=Ortho._default_config['interp'],
    show_default=True,
    help=f'Interpolation method for remapping source to ortho image.',
)
dem_interp_option = click.option(
    '-di',
    '--dem-interp',
    type=click.Choice(Interp, case_sensitive=False),
    default=Ortho._default_config['dem_interp'],
    show_default=True,
    help=f'Interpolation method for DEM reprojection.',
)
per_band_option = click.option(
    '-pb/-npb',
    '--per-band/--no-per-band',
    type=click.BOOL,
    default=Ortho._default_config['per_band'],
    show_default=True,
    help='Orthorectify band-by-band (``--per-band``) or all bands at once (``--no-per-band``). '
    '``--no-per-band`` is faster but uses more memory.',
)
full_remap_option = click.option(
    '-fr/-nfr',
    '--full-remap/--no-full-remap',
    type=click.BOOL,
    default=Ortho._default_config['full_remap'],
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
    help='Scaling of the ``--no-full-remap`` undistorted image: 0 includes the largest '
    'source image portion that allows all undistorted pixels to be valid.  1 includes all '
    'source pixels in the undistorted image.',
)
write_mask_option = click.option(
    '-wm/-nwm',
    '--write-mask/--no-write-mask',
    type=click.BOOL,
    default=Ortho._default_config['write_mask'],
    show_default='true for jpeg compression.',
    help='Mask valid pixels with an internal mask (``--write-mask``), or with a nodata value '
    'based on ``--dtype`` (``--no-write-mask``). An internal mask helps remove nodata noise '
    'caused by lossy compression.',
)
dtype_option = click.option(
    '-dt',
    '--dtype',
    type=click.Choice(list(Ortho._nodata_vals.keys()), case_sensitive=False),
    default=Ortho._default_config['dtype'],
    show_default='source image data type.',
    help=f'Ortho image data type.',
)
compress_option = click.option(
    '-c',
    '--compress',
    type=click.Choice(Compress, case_sensitive=False),
    default=Ortho._default_config['compress'],
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
export_params_option = click.option(
    '-ep',
    '--export-params',
    is_flag=True,
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Export interior & exterior parameters to orthority format files, and exit.',
)
out_dir_option = click.option(
    '-od',
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    default=Path.cwd(),
    show_default='current working',
    help='Directory in which to place output file(s).',
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
def cli(ctx: click.Context, verbose, quiet):
    """Orthorectification toolkit."""
    verbosity = verbose - quiet
    _configure_logging(verbosity)

    # enter context managers for sub-command raster operations
    env = rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False, GDAL_TIFF_INTERNAL_MASK=True)
    ctx.with_resource(suppress_no_georef())
    ctx.with_resource(env)


@cli.command(
    cls=RstCommand,
    short_help='Orthorectify with camera parameter files.',
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
@export_params_option
@out_dir_option
@overwrite_option
def ortho(
    src_files: tuple[str, ...],
    int_param_file: str,
    ext_param_file: str,
    crs: rio.CRS,
    lla_crs: rio.CRS,
    radians: bool,
    **kwargs,
):
    """
    Orthorectify SOURCE images with camera model(s) defined by interior and exterior parameter
    files.

    Interior parameters are supported in orthority (.yaml), OpenDroneMap :file:`cameras.json`,
    and OpenSfM :file:`reconstruction.json` formats.  Exterior parameters are supported in
    Orthority (.geojson), CSV, and OpenSfM :file:`reconstruction.json` formats.  Note that
    parameter file extensions are used to distinguish their format.

    The :option:`--dem <oty-ortho --dem>`, :option:`--int-param <oty-ortho --int-param>` and
    :option:`--ext-param <oty-ortho --ext-param>` options are required.  Depending on the input
    file formats, :option:`--crs <oty-ortho --crs>` may also be required::

        oty ortho --dem dem.tif --int-param int_param.yaml --ext-param ext_param.csv --crs EPSG:32651 source*.tif

    Camera parameters can be converted into Orthority format files with :option:`--export-params
    <oty-ortho --export-params>`.  With this option, :option:`--dem <oty-ortho --dem>` is not
    required::

        oty ortho --int-param reconstruction.json --ext-param reconstruction.json --export-params

    Ortho images and parameter files are placed in the current working directory by default.
    This can be overridden with :option:`--out-dir <oty-odm --out-dir>`.
    """
    # read interior params
    try:
        int_param_suffix = Path(int_param_file).suffix.lower()
        if int_param_suffix in ['.yaml', '.yml']:
            int_param_dict = param_io.read_oty_int_param(int_param_file)
        elif int_param_suffix == '.json':
            int_param_dict = param_io.read_osfm_int_param(int_param_file)
        else:
            raise ParamFileError(f"'{int_param_suffix}' file type not supported.")
    except (FileNotFoundError, URLError, HTTPError, ParamFileError) as ex:
        raise click.BadParameter(str(ex), param_hint="'-ip' / '--int-param'")

    # read exterior params
    try:
        ext_param_suffix = Path(ext_param_file).suffix.lower()
        if ext_param_suffix in ['.csv', '.txt']:
            reader = param_io.CsvReader(ext_param_file, crs=crs, lla_crs=lla_crs, radians=radians)
        elif ext_param_suffix == '.json':
            reader = param_io.OsfmReader(ext_param_file, crs=crs, lla_crs=lla_crs)
        elif ext_param_suffix == '.geojson':
            reader = param_io.OtyReader(ext_param_file)
        else:
            raise ParamFileError(f"'{ext_param_suffix}' file type not supported.")
    except (FileNotFoundError, URLError, HTTPError, ParamFileError) as ex:
        raise click.BadParameter(str(ex), param_hint="'-ep' / '--ext-param'")
    except CrsMissingError:
        raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')

    ext_param_dict = reader.read_ext_param()

    # get any parameter CRS, if no user CRS is supplied
    crs = crs or reader.crs

    # orthorectify
    _ortho(
        src_files=src_files,
        int_param_dict=int_param_dict,
        ext_param_dict=ext_param_dict,
        crs=crs,
        **kwargs,
    )


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
@export_params_option
@out_dir_option
@overwrite_option
def exif(src_files: tuple[str, ...], crs: rio.CRS, lla_crs: rio.CRS, **kwargs):
    """
    Orthorectify SOURCE images with camera model(s) defined by image EXIF / XMP tags.

    SOURCE image tags should include DewarpData, focal length & sensor size or 35mm equivalent
    focal length; camera position and camera roll, pitch & yaw.  DewarpData is converted to a
    Brown model if it is present, otherwise a pinhole model is used.  Pinhole approximation and
    tag value accuracy affect ortho image accuracy.

    The :option:`--dem <oty-exif --dem>` option is required.  If :option:`--crs <oty-exif --crs>`
    is not supplied, a UTM world / ortho CRS is auto-determined from the camera positions::

        oty exif --dem dem.tif source*.tif

    Camera parameters can be converted into Orthority format files with :option:`--export-params
    <oty-exif --export-params>`.  With this option, :option:`--dem <oty-exif --dem>` is not
    required::

        oty exif ---export-params

    Ortho images and parameter files are placed in the current working directory by
    default.  This can be overridden with :option:`--out-dir <oty-odm --out-dir>`.
    """
    # read interior & exterior params
    try:
        logger.info('Reading camera parameters:')
        reader = param_io.ExifReader(src_files, crs=crs, lla_crs=lla_crs)
        int_param_dict = reader.read_int_param()
        ext_param_dict = reader.read_ext_param()
    except (RasterioIOError, ParamFileError) as ex:
        raise click.BadParameter(str(ex), param_hint='SOURCE...')

    # get auto UTM CRS, if CRS not set already
    crs = crs or reader.crs

    # orthorectify
    _ortho(
        src_files=src_files,
        int_param_dict=int_param_dict,
        ext_param_dict=ext_param_dict,
        crs=crs,
        **kwargs,
    )


@cli.command(
    cls=RstCommand,
    short_help='Orthorectify with OpenDroneMap outputs.',
    epilog='See https://orthority.readthedocs.io/ for more detail.',
)
@click.option(
    '-dd',
    '--dataset-dir',
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    required=True,
    default=None,
    callback=_dataset_dir_cb,
    help='Path of the ODM dataset to process.',
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
@export_params_option
@click.option(
    '-od',
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    default=None,
    show_default='<dataset-dir>/orthority',
    help='Directory in which to place output file(s).',
)
@overwrite_option
def odm(dataset_dir: Path, crs: rio.CRS, resolution: tuple[float, float], out_dir: Path, **kwargs):
    """
    Orthorectify images in a processed OpenDroneMap dataset that includes a DSM.

    The images, DSM and camera models are read from the dataset. If :option:`--crs <oty-odm
    --crs>` is not supplied, the world / ortho CRS is also read from the dataset.
    :option:`--dataset-dir <oty-odm --dataset-dir>` is the only required option::

        oty odm --dataset-dir dataset

    Camera parameters can be converted into Orthority format files with :option:`--export-params
    <oty-odm --export-params>`::

        oty odm --dataset-dir dataset --export-params

    Ortho images and parameter files are placed in the :file:`{dataset}/orthority` subdirectory
    by default.  This can be overridden with :option:`--out-dir <oty-odm --out-dir>`.
    """
    # find source images
    src_exts = ['.jpg', '.jpeg', '.tif', '.tiff']
    src_files = tuple(
        [str(p) for p in dataset_dir.joinpath('images').glob('*.*') if p.suffix.lower() in src_exts]
    )
    if len(src_files) == 0:
        raise click.BadParameter(
            f"No images found in '{dataset_dir.joinpath('images')}'.",
            param_hint="'-dd' / '--dataset-dir'",
        )

    # set CRS from DSM
    dem_file = dataset_dir.joinpath('odm_dem/dsm.tif')
    with rio.open(dem_file, 'r') as dem_im:
        crs = crs or dem_im.crs

    # set and create output dir
    out_dir = out_dir or dataset_dir.joinpath('orthority')
    out_dir.mkdir(exist_ok=True)

    # read interior and exterior params from OpenSfM reconstruction file
    rec_file = dataset_dir.joinpath('opensfm', 'reconstruction.json')
    reader = param_io.OsfmReader(rec_file, crs=crs)
    int_param_dict = reader.read_int_param()
    ext_param_dict = reader.read_ext_param()

    # orthorectify
    _ortho(
        src_files=src_files,
        dem_file=str(dem_file),
        int_param_dict=int_param_dict,
        ext_param_dict=ext_param_dict,
        crs=crs,
        resolution=resolution,
        dem_band=1,
        out_dir=out_dir,
        **kwargs,
    )


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
        dem_band = ortho_config.pop('dem_band', Ortho._default_config['dem_band'])
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
                with suppress_no_georef(), rio.open(src_filename) as src_im:
                    im_size = np.float64([src_im.width, src_im.height])

                if not camera:
                    # create a new camera
                    camera = create_camera(camera_type, **camera_config, xyz=xyz, opk=opk)
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

# TODO: test CLI exceptions are meaningful
# TODO: a lot of args are spec'd as Tuples, is this correct or would List, or Iterable be more appropriate?

##
