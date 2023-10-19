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
import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import numpy as np
import rasterio as rio
import yaml

from orthority import io, root_path
from orthority.camera import create_camera
from orthority.enums import CameraType, Compress, Interp
from orthority.errors import CrsMissingError, DemBandError, ParamFileError
from orthority.ortho import Ortho
from orthority.utils import suppress_no_georef
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
            ':option:`(.*?)( <.*?>)?`': r'\g<1>',
            # convert ':option:`--name`' to '--name'
            ':option:`(.*?)`': r'\g<1>',
            # strip '----...'
            '-{4,}': r'',
            # convert from RST cross-ref '`<name> <<link>>`_' to 'name'
            '`([^<]*) <([^>]*)>`_': r'\g<1>',
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
    # adapted from rasterio https://github.com/rasterio/rasterio
    log_level = max(10, 20 - 10 * verbosity)

    # apply config to package logger, rather than root logger
    pkg_logger = logging.getLogger('orthority')
    formatter = PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)
    logging.captureWarnings(True)


def _read_src_crs(filename: Path) -> Optional[rio.CRS]:
    """Read CRS from source image file."""
    with suppress_no_georef(), rio.open(filename, 'r') as im:
        if not im.crs:
            logger.debug(f"No CRS found for source image: '{filename.name}'")
        else:
            logger.debug(f"Found source image '{filename.name}' CRS: '{im.crs.to_proj4()}'")
        return im.crs


def _read_crs(crs: str):
    """Read a CRS from a string, text file, or image file."""
    crs_file = Path(crs)
    if crs_file.is_file():
        # read CRS from geotiff / txt file
        if crs_file.suffix.lower() in ['.tif', '.tiff']:
            with suppress_no_georef(), rio.open(crs_file, 'r') as im:
                crs = im.crs
        else:
            crs_str = crs_file.read_text()
            crs = rio.CRS.from_string(crs_str)
    else:
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
        if crs.is_geographic:
            raise click.BadParameter(
                f"CRS should be a projected, not geographic system.", param=param
            )
    return crs


def _lla_crs_cb(ctx: click.Context, param: click.Parameter, lla_crs: str):
    """Click callback to validate and parse the LLA CRS."""
    if lla_crs is not None:
        try:
            lla_crs = _read_crs(lla_crs)
        except Exception as ex:
            raise click.BadParameter(f'{str(ex)}', param=param)
        if not lla_crs.is_geographic:
            raise click.BadParameter(
                f"CRS should be a geographic, not projected system.", param=param
            )
    return lla_crs


def _resolution_cb(ctx: click.Context, param: click.Parameter, resolution: Tuple):
    """Click callback to validate and parse the resolution."""
    if len(resolution) == 1:
        resolution *= 2
    elif len(resolution) > 2:
        raise click.BadParameter(f'At most two resolution values should be specified.', param=param)
    return resolution


def _odm_proj_dir_cb(ctx: click.Context, param: click.Parameter, proj_dir: Path):
    """Click callback to validate the ODM project directory."""
    req_paths = [
        Path('opensfm').joinpath('reconstruction.json'),
        Path('odm_dem').joinpath('dsm.tif'),
        Path('images'),
    ]
    for req_path in req_paths:
        req_path = proj_dir.joinpath(req_path)
        if not req_path.exists():
            raise click.BadParameter(f"Could not find '{req_path}'.", param=param)
    return proj_dir


def _ortho(
    src_files: Tuple[Path, ...],
    dem_file: Path,
    int_param_dict: Dict[str, Dict],
    ext_param_dict: Dict[str, Dict],
    crs: rio.CRS,
    dem_band: int,
    alpha: float,
    export_params: bool,
    out_dir: Path,
    overwrite: bool,
    **kwargs,
):
    """"""
    if export_params:
        # convert interior / exterior params to oty format files
        logger.info('Writing parameter files...')
        int_param_file = out_dir.joinpath('int_param.yaml')
        ext_param_file = out_dir.joinpath('ext_param.geojson')
        io.write_int_param(int_param_file, int_param_dict, overwrite)
        io.write_ext_param(ext_param_file, ext_param_dict, crs, overwrite)
        return
    elif not dem_file:
        raise click.MissingParameter(param_hint="'-d' / '--dem'", param_type='option')

    cameras = {}
    for src_i, src_file in enumerate(src_files):
        # get exterior params for src_file
        ext_param = ext_param_dict.get(src_file.name, ext_param_dict.get(src_file.stem, None))
        if not ext_param:
            raise click.BadParameter(
                f"Could not find parameters for '{src_file.name}'.",
                param_hint="'-ep' / '--ext-param'",
            )

        # get interior params for ext_param
        if ext_param['camera']:
            cam_id = ext_param['camera']
            if cam_id not in int_param_dict:
                raise click.BadParameter(
                    f"Could not find parameters for camera '{cam_id}'.",
                    param_hint="'-ip' / '--int-param'",
                )
            int_param = int_param_dict[cam_id]
        elif len(int_param_dict) == 1:
            cam_id = None
            int_param = list(int_param_dict.values())[0]
        else:
            raise click.BadParameter(
                f"'camera' ID for '{src_file.name}' should be specified.",
                param_hint="'-ep' / '--ext-param'",
            )

        # get/create camera and update exterior parameters
        camera = cameras[cam_id] if cam_id in cameras else create_camera(**int_param, alpha=alpha)
        camera.update(xyz=ext_param['xyz'], opk=ext_param['opk'])
        cameras[cam_id] = camera

        # create ortho object & filename
        try:
            ortho = Ortho(src_file, dem_file, camera, crs, dem_band=dem_band)
        except DemBandError as ex:
            raise click.BadParameter(str(ex), param_hint="'-db' / '--dem-band'")
        ortho_file = out_dir.joinpath(f'{src_file.stem}_ORTHO.tif')

        # orthorectify
        logger.info(f"Orthorectifying '{src_file.name}' ({src_i + 1} of {len(src_files)}):")
        ortho.process(ortho_file, overwrite=overwrite, **kwargs)


# Define click options that are common to more than one command
src_files_arg = click.argument(
    'src_files',
    nargs=-1,
    metavar='SOURCE...',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    # help='Path/URL of source image(s) to be orthorectified..'
)
dem_file_option = click.option(
    '-d',
    '--dem',
    'dem_file',
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None,
    help='Path/URL of a DEM image covering the source image(s). [required]',
)
int_param_file_option = click.option(
    '-ip',
    '--int-param',
    'int_param_file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    default=None,
    help='Path of an interior parameter file.',
)
ext_param_file_option = click.option(
    '-ep',
    '--ext-param',
    'ext_param_file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    default=None,
    help='Path of an exterior parameter file.',
)
crs_option = click.option(
    '-c',
    '--crs',
    type=click.STRING,
    default=None,
    show_default='auto',
    callback=_crs_cb,
    help='CRS of the ortho image and any projected coordinate exterior parameters as an EPSG, '
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
resolution_option = click.option(
    '-r',
    '--res',
    'resolution',
    type=click.FLOAT,
    default=None,
    show_default='auto',
    multiple=True,
    callback=_resolution_cb,
    help='Ortho image pixel size in units of the :option:`--crs` (usually meters).  Can be used '
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
    type=click.Choice(Interp.cv_list(), case_sensitive=False),
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
    '``--no-full-remap`` is faster but can reduce the ortho image quality.',
)
alpha_option = click.option(
    '-a',
    '--alpha',
    type=click.FloatRange(0, 1),
    nargs=1,
    default=1,
    show_default=True,
    help='Scaling of the ``--no-full-remap`` undistorted image: 0 results in an undistorted image '
    'with all valid pixels, 1 results in an undistorted image with all source pixels.',
)
write_mask_option = click.option(
    '-wm/-nwm',
    '--write-mask/--no-write-mask',
    type=click.BOOL,
    default=Ortho._default_config['write_mask'],
    show_default='true for jpeg compression.',
    help='Write an internal mask for the ortho image. Helps remove nodata noise caused by lossy '
    'compression.',
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
    show_default=True,
    help=f'Ortho image compression. `auto` uses `jpeg` compression for `uint8` :option:`--dtype`, '
    f'and `deflate` otherwise.',
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
    help='Export interior & exterior parameters to orthority format file(s), and exit.',
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
def cli(verbose, quiet):
    """Orthorectification toolkit."""
    verbosity = verbose - quiet
    _configure_logging(verbosity)


@cli.command(cls=RstCommand)
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
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@export_params_option
@out_dir_option
@overwrite_option
def ortho(
    src_files: Tuple[Path, ...],
    int_param_file: Path,
    ext_param_file: Path,
    crs: rio.CRS,
    lla_crs: rio.CRS,
    **kwargs,
):
    """
    Orthorectify using interior and exterior parameter files.

    Orthorectify images with camera models specified in interior and exterior parameter files.
    Interior parameters are supported in orthority (.yaml), ODM cameras (.json), and OpenSfM
    reconstruction (.json) formats.  Exterior parameters are supported in orthority (.geojson),
    custom CSV (.csv), and ODM / OpenSfM reconstruction (.json) formats.  Note that parameter
    file extensions are used to distinguish their format.

    If possible, a world / ortho CRS will be read from other sources, or auto-determined when
    :option:`--crs <oty-ortho --crs>` is not passed.

    See the `online docs <?>`_ for more detail on file formats and CRS.
    \b

    Examples
    ========

    Orthorectify `source.tif` using DEM `dem.tif` and interior and exterior parameters from
    `reconstruction.json`.  An ortho resolution and UTM CRS are auto-determined::

        oty ortho --dem dem.tif --int-param reconstruction.json --ext-param reconstruction.json source.tif

    Convert 'reconstruction.json' interior and exterior parameters to orthority format files in
    the current working directory::

        oty ortho --int-param reconstruction.json --ext-param reconstruction.json --export-params

    Orthorectify images matching '*rgb.tif' using DEM 'dem.tif', and 'int_param.yaml' interior &
    'ext_param.csv' exterior parameter files.  Specify a 1m ortho resolution and 'EPSG:32651'
    CRS.  Write ortho files to the 'data' directory using 'deflate' compression and a 'uint16'
    data type::

        oty ortho --dem dem.tif --int-param int_param.yaml --ext-param ext_param.csv --res 1 --crs EPSG:32651 --out-dir data --compress deflate --dtype uint16 *rgb.tif

    SOURCE... Path/URL(s) of source image(s) to orthorectify.
    """
    if not crs and len(src_files) > 0:
        # read crs from the first source image, if it has one
        crs = _read_src_crs(src_files[0])

    # read interior params
    try:
        if int_param_file.suffix.lower() in ['.yaml', '.yml']:
            int_param_dict = io.read_oty_int_param(int_param_file)
        elif int_param_file.suffix.lower() == '.json':
            int_param_dict = io.read_osfm_int_param(int_param_file)
        else:
            raise click.BadParameter(
                f"'{int_param_file.suffix}' file type not supported.",
                param_hint="'-ip' / '--int-param'",
            )
    except ParamFileError as ex:
        raise click.BadParameter(str(ex), param_hint="'-ip' / '--int-param'")

    # read exterior params
    try:
        if ext_param_file.suffix.lower() in ['.csv', '.txt']:
            reader = io.CsvReader(ext_param_file, crs=crs, lla_crs=lla_crs)
        elif ext_param_file.suffix.lower() == '.json':
            reader = io.OsfmReader(ext_param_file, crs=crs, lla_crs=lla_crs)
        elif ext_param_file.suffix.lower() == '.geojson':
            reader = io.OtyReader(ext_param_file, crs=crs, lla_crs=lla_crs)
        else:
            raise click.BadParameter(
                f"'{ext_param_file.suffix}' file type not supported.",
                param_hint="'-ep' / '--ext-param'",
            )
    except ParamFileError as ex:
        raise click.BadParameter(str(ex), param_hint="'-ep' / '--ext-param'")
    except CrsMissingError:
        raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')

    ext_param_dict = reader.read_ext_param()

    # finalise the crs
    crs = crs or reader.crs
    if not crs:
        raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')

    # orthorectify
    _ortho(
        src_files=src_files,
        int_param_dict=int_param_dict,
        ext_param_dict=ext_param_dict,
        crs=crs,
        **kwargs,
    )


@cli.command(cls=RstCommand)
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
def exif(src_files: Tuple[Path, ...], crs: rio.CRS, lla_crs: rio.CRS, **kwargs):
    """
    Orthorectify images with EXIF / XMP tags.

    Orthorectify images with pinhole camera models derived from EXIF / XMP metadata.  Image tags
    should include focal length & sensor size, or 35mm equivalent focal length; camera position;
    and camera / gimbal roll, pitch & yaw.

    See the `online docs <?>`_ for more detail.

    \b

    Examples
    ========

    Orthorectify images matching '*rgb.tif' with DEM 'dem.tif'::

        oty exif --dem dem.tif *rgb.tif

    Export interior and exterior parameters for images matching '*rgb.tif' to orthority format
    files::

        oty exif --export-params *rgb.tif

    SOURCE... Path/URL(s) of source image(s) to orthorectify.
    """
    if not crs and len(src_files) > 0:
        # get crs from the first source image, if it has one
        crs = _read_src_crs(src_files[0])

    # read interior & exterior params
    try:
        logger.info('Reading camera parameters:')
        reader = io.ExifReader(src_files, crs=crs, lla_crs=lla_crs)
        int_param_dict = reader.read_int_param()
        ext_param_dict = reader.read_ext_param()
    except ParamFileError as ex:
        raise click.BadParameter(str(ex), param_hint='SOURCE...')

    # get auto UTM crs, if crs not set already
    crs = crs or reader.crs

    # orthorectify
    _ortho(
        src_files=src_files,
        int_param_dict=int_param_dict,
        ext_param_dict=ext_param_dict,
        crs=crs,
        **kwargs,
    )


@cli.command(cls=RstCommand)
@click.option(
    '-pd',
    '--proj-dir',
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    required=True,
    default=None,
    callback=_odm_proj_dir_cb,
    help='Path of the ODM project folder to process.',
)
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
    show_default='<proj-dir>/orthority',
    help='Directory in which to place output file(s).',
)
@overwrite_option
def odm(proj_dir: Path, resolution: Tuple[float, float], out_dir: Path, **kwargs):
    """
    Orthorectify existing OpenDroneMap outputs.

    Orthorectify individual images using OpenDroneMap (ODM) generated camera models and DSM.  The
    DSM is required, and can be generated by running ODM with the `--dsm
    <https://docs.opendronemap.org/arguments/dsm/#dsm>`_ option.

    By default, the ortho resolution is read from the ODM orthophoto.  If that does not exist,
    it is read from the DSM. Ortho images & parameter files are placed in the '<odm
    project>/orthority' directory.

    The ``ortho`` sub-command  can be used for more control over options.
    \b

    Examples
    ========

    Orthorectify images in '<odm project>/images' directory using the '<odm project>' camera
    models and DSM::

        oty odm --proj-dir <odm project>

    Export '<odm project>' interior and exterior parameters to orthority format files::

        oty odm --proj-dir <odm project> --export-params

    Orthorectify images in '<odm project>/images' directory using the '<odm project>' camera
    models and DSM.  Use an ortho resolution of 0.1m and 'lanczos' interpolation to remap source
    to ortho.

        oty odm --proj-dir <odm project> --res 0.1 --interp lanczos
    """
    # find source images
    src_exts = ['.jpg', '.jpeg', '.tif', '.tiff']
    src_files = tuple(
        [p for p in proj_dir.joinpath('images').glob('*.*') if p.suffix.lower() in src_exts]
    )
    if len(src_files) == 0:
        raise click.BadParameter(
            f"No images found in '{proj_dir.joinpath('images')}'.",
            param_hint="'-pd' / '--proj-dir'",
        )

    # set crs from ODM orthophoto or DSM
    orthophoto_file = proj_dir.joinpath('odm_orthophoto', 'odm_orthophoto.tif')
    dem_file = proj_dir.joinpath('odm_dem', 'dsm.tif')
    crs_file = orthophoto_file if orthophoto_file.exists() else dem_file
    with rio.open(crs_file, 'r') as im:
        crs = im.crs

    # set and create output dir
    out_dir = out_dir or proj_dir.joinpath('orthority')
    out_dir.mkdir(exist_ok=True)

    # read interior and exterior params from OpenSfM reconstruction file
    rec_file = proj_dir.joinpath('opensfm', 'reconstruction.json')
    reader = io.OsfmReader(rec_file, crs=crs)
    int_param_dict = reader.read_int_param()
    ext_param_dict = reader.read_ext_param()

    # orthorectify
    _ortho(
        src_files=src_files,
        dem_file=dem_file,
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

    # TODO: add deprecation warning with link to docs
    # logger.warning(
    #     "This command is deprecated and will be removed in version 0.4.0.  Please switch to 'oty'
    #     and its " "sub-commands."
    # )

    try:
        # set logging level
        if verbosity is not None:
            pkg_logger = logging.getLogger('orthority')
            formatter = PlainInfoFormatter()
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(formatter)
            pkg_logger.addHandler(handler)
            pkg_logger.setLevel(10 * verbosity)
            logging.captureWarnings(True)

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
                fieldnames=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'],
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
                xyz = np.array(
                    (im_pos_ori['easting'], im_pos_ori['northing'], im_pos_ori['altitude'])
                )

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


def simple_ortho(argv=None):
    """Entry point to legacy 'simple-ortho' CLI."""

    def parse_args(argv=None):
        """Parse arguments."""
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
        return parser.parse_args(argv)

    args = parse_args(argv)
    _simple_ortho(**vars(args))


if __name__ == '__main__':
    cli()

# TODO: test CLI exceptions are meaningful
# TODO: add radians option for CSV files
# TODO: consider typing with PathLike, ArrayLike and Iterable, rather than Union[] etc
# TODO: a lot of args are spec'd as Tuples, is this correct or would List, or Iterable be more appropriate?
