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
import sys
from pathlib import Path
import re
import click
import yaml
import csv
from typing import Tuple, List, Dict, Union

from tqdm.auto import tqdm
import numpy as np
import rasterio as rio
from rasterio.errors import CRSError
from simple_ortho.ortho import Ortho
from simple_ortho.camera import create_camera
from simple_ortho.enums import Compress, Interp, CameraType
from simple_ortho.utils import suppress_no_georef
from simple_ortho.errors import  CrsError, ParamFileError, DemBandError, CrsMissingError
from simple_ortho import io
from simple_ortho.version import __version__


logger = logging.getLogger(__name__)


class PlainInfoFormatter(logging.Formatter):
    """ logging formatter to format INFO logs without the module name etc. prefix. """
    # TODO: do we need this ?
    def format(self, record: logging.LogRecord):
        if record.levelno == logging.INFO:
            self._style._fmt = '%(message)s'
        else:
            self._style._fmt = '%(levelname)s:%(name)s: %(message)s'
        return super().format(record)


class RstCommand(click.Command):
    """ click.Command subclass for formatting help with RST markup. """

    def get_help(self, ctx: click.Context):
        """ Strip some RST markup from the help text for CLI display.  Will not work with grid tables. """

        # Note that this can't easily be done in __init__, as each sub-command's __init__ gets called,
        # which ends up re-assigning self.wrap_text to reformat_text
        if not hasattr(self, 'click_wrap_text'):
            self.click_wrap_text = click.formatting.wrap_text

        sub_strings = {
            '\b\n': '\n\b',                 # convert from RST friendly to click literal (unwrapped) block marker
            r'\| ': '',                     # strip RST literal (unwrapped) marker in e.g. tables and bullet lists
            '\n\.\. _.*:\n': '',            # strip RST ref directive '\n.. _<name>:\n'
            '`(.*?) <(.*?)>`_': r'\g<1>',   # convert from RST cross-ref '`<name> <<link>>`_' to 'name'
            '::': ':',                      # convert from RST '::' to ':'
            '``(.*?)``': r'\g<1>',          # convert from RST '``literal``' to 'literal'
            ':option:`(.*?)( <.*?>)?`': r'\g<1>',   # convert ':option:`--name <group-command --name>`' to '--name'
            ':option:`(.*?)`': r'\g<1>',    # convert ':option:`--name`' to '--name'
            '-{4,}': r'',                   # strip '----...'
        }  # yapf: disable

        def reformat_text(text: str, width: int, **kwargs):
            for sub_key, sub_value in sub_strings.items():
                text = re.sub(sub_key, sub_value, text, flags=re.DOTALL)
            wr_text = self.click_wrap_text(text, width, **kwargs)
            # change double newline to single newline separated list
            return re.sub('\n\n(\s*?)- ', '\n- ', wr_text, flags=re.DOTALL)

        click.formatting.wrap_text = reformat_text
        return click.Command.get_help(self, ctx)


def _configure_logging(verbosity: int):
    """ Configure python logging level."""
    # adapted from rasterio https://github.com/rasterio/rasterio
    log_level = max(10, 20 - 10 * verbosity)

    # apply config to package logger, rather than root logger
    pkg_logger = logging.getLogger(__package__)
    formatter = PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)
    logging.captureWarnings(True)


def _crs_cb(ctx: click.Context, param: click.Parameter, crs: str):
    """ click callback to validate and parse the CRS. """
    if crs is not None:
        try:
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
        except Exception as ex:
            raise click.BadParameter(f'{crs}.  {str(ex)}', param=param)
        if crs.is_geographic:
            raise click.BadParameter(f"CRS should be a projected, not geographic system.", param=param)
    return crs


def _resolution_cb(ctx: click.Context, param: click.Parameter, resolution: Tuple):
    """ click callback to validate and parse the resolution. """
    if len(resolution) == 1:
        resolution *= 2
    elif len(resolution) > 2:
        raise click.BadParameter(f'At most two resolution values should be specified.', param=param)
    return resolution


def _odm_root_cb(ctx: click.Context, param: click.Parameter, odm_root: Path):
    """ click callback to validate the ODM output directory. """
    req_paths = [Path('opensfm').joinpath('reconstruction.json'), Path('odm_dem').joinpath('dsm.tif'), Path('images')]
    for req_path in req_paths:
        req_path = odm_root.joinpath(req_path)
        if not (req_path).exists():
            raise click.BadParameter(f'Could not find {req_path}.', param=param)
    return odm_root


def _read_src_crs(filename: Path) -> Union[None, rio.CRS]:
    with suppress_no_georef(), rio.open(filename, 'r') as im:
        if not im.crs:
            logger.debug(f'No CRS found for source image: {filename.name}')
        return im.crs


def _ortho(
    src_files: Tuple[Path, ...], dem_file: Path, int_param_dict: Dict[str, Dict], ext_param_dict: Dict[str, Dict],
    crs: rio.CRS, resolution: Tuple[float, float], dem_band: int, write_params: bool, out_dir: Path, overwrite: bool,
    **kwargs
):
    """ """
    # TODO: multiple file progress as a master bar, or 'x of N' prefix to individual bars
    if write_params:
        # convert interior / exterior params to oty format files
        logger.info('Writing parameter files...')
        if not out_dir:
            raise click.MissingParameter(
                '--write-params requires --out-dir to be specified.', param_hint='--out-dir', param_type='option'
            )
        int_param_file = out_dir.joinpath('int_param.yaml')
        ext_param_file = out_dir.joinpath('ext_param.geojson')
        io.write_int_param(int_param_file, int_param_dict, overwrite)
        io.write_ext_param(ext_param_file, ext_param_dict, crs, overwrite)
        return

    for src_i, src_file in enumerate(src_files):
        # get exterior params for src_file
        ext_param = ext_param_dict.get(src_file.name, ext_param_dict.get(src_file.stem, None))
        if not ext_param:
            raise click.BadParameter(
                f"Could not find parameters for '{src_file.name}'.", param_hint='--ext-param'
            )

        # get interior params for ext_param
        if ext_param['camera']:
            cam_id = ext_param['camera']
            if cam_id not in int_param_dict:
                raise click.BadParameter(f"Could not find parameters for camera '{cam_id}'.", param_hint='--int-param')
            int_param = int_param_dict[cam_id]
        elif len(int_param_dict) == 1:
            int_param = list(int_param_dict.values())[0]
        else:
            raise click.BadParameter(f"'camera' ID for {src_file.name} should be specified.", param_hint='--ext-param')

        # get image size
        with suppress_no_georef(), rio.open(src_file, 'r') as src_im:
            im_size = (src_im.width, src_im.height)

        # create camera and ortho objects
        # TODO: generalise exterior params / camera so that the cli can just expand the dict and not need to know
        #  about the internals
        camera = create_camera(position=ext_param['xyz'], rotation=ext_param['opk'], im_size=im_size, **int_param)
        try:
            ortho = Ortho(src_file, dem_file, camera, crs, dem_band=dem_band)
        except DemBandError as ex:
            raise click.BadParameter(str(ex), param_hint='--dem_band')
        out_root = out_dir or src_file.parent
        ortho_file = out_root.joinpath(f'{src_file.stem}_ORTHO.tif')

        # orthorectify
        logger.info(f'Orthorectifying {src_file.name} ({src_i +1 } of {len(src_files)}):')
        ortho.process(ortho_file, resolution, **kwargs)


# TODO: add mosaic, and write param options
# Define click options that are common to more than one command
src_files_arg = click.argument(
    'src_files', nargs=-1, metavar='SOURCE...', type=click.Path(exists=True, dir_okay=False, path_type=Path),
    # help='Path/URL of source image(s) to be orthorectified..'
)
dem_file_option = click.option(
    '-d', '--dem', 'dem_file', type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True, default=None, help='Path/URL of a DEM image covering the source image(s).'
)
int_param_file_option = click.option(
    '-ip', '--int-param', 'int_param_file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, default=None,
    help='Path of a file specifying camera internal parameters.'
)
ext_param_file_option = click.option(
    '-ep', '--ext-param', 'ext_param_file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, default=None,
    help='Path of a file specifying camera external parameters.'
)
crs_option = click.option(
    '-c', '--crs', type=click.STRING, default=None, show_default='source image CRS.', callback=_crs_cb,
    help='CRS of the ortho image as an EPSG, proj4, or WKT string, or text file containing string.  Should be a '
         'projected, and not geographic CRS.  Can be omitted if source image(s) are projected in this CRS.'
)
dem_band_option = click.option(
    '-db', '--dem-band', type=click.INT, nargs=1, default=1, show_default=True,
    help='Index of the DEM band to use (1 based).'
)
resolution_option = click.option(
    '-r', '--res', 'resolution', type=click.FLOAT, required=True, default=None, multiple=True, callback=_resolution_cb,
    help='Ortho image pixel size in units of the :option:`--crs` (usually meters).  Can be used twice for non-square '
         'pixels: ``--res PIXEL_WIDTH --res PIXEL_HEIGHT``'
)
interp_option = click.option(
    '-i', '--interp', type=click.Choice(Interp.cv_list(), case_sensitive=False),
    default=Ortho._default_config['interp'], show_default=True,
    help=f'Interpolation method for remapping source to ortho image.'
)
dem_interp_option = click.option(
    '-di', '--dem-interp', type=click.Choice(Interp, case_sensitive=False),
    default=Ortho._default_config['dem_interp'], show_default=True,
    help=f'Interpolation method for reprojecting the DEM.'
)
per_band_option = click.option(
    '-pb/-npb', '--per-band/--no-per-band', type=click.BOOL, default=Ortho._default_config['per_band'],
    show_default=True, help='Orthorectify band-by-band (``--per-band``) or all bands at once (``--no-per-band``). '
                            '``--no-per-band`` is faster but uses more memory.'
)
# TODO: change name to something friendlier, or exclude entirely
full_remap_option = click.option(
    '-fr/-nfr', '--full-remap/--no-full-remap', type=click.BOOL, default=Ortho._default_config['full_remap'],
    show_default=True, help='Orthorectify the source image with full camera model (``--full-remap``), or an '
                            'undistorted source image with a pinhole camera model (``--no-full-remap``).  '
                            '``--no-full-remap`` is faster but can reduce the extents and quality of the ortho image.'
)
# TODO: "internal mask"?
write_mask_option = click.option(
    '-wm/-nwm', '--write-mask/--no-write-mask', type=click.BOOL, default=Ortho._default_config['write_mask'],
    show_default='true for jpeg compression.',
    help='Write an internal mask for the ortho image. Helps remove nodata noise caused by lossy compression.'
)
dtype_option = click.option(
    '-dt', '--dtype', type=click.Choice(list(Ortho._nodata_vals.keys()), case_sensitive=False),
    default=Ortho._default_config['dtype'], show_default='source image data type.', help=f'Ortho image data type.'
)
compress_option = click.option(
    '-c', '--compress', type=click.Choice(Compress, case_sensitive=False), default=Ortho._default_config['compress'],
    show_default=True, help=f'Ortho image compression. `auto` uses `jpeg` compression for `uint8` :option:`--dtype`, '
                            f'and `deflate` otherwise.'
)
build_ovw_option = click.option(
    '-bo/-nbo', '--build-ovw/--no-build-ovw', type=click.BOOL, default=True, show_default=True,
    help='Build overviews for the ortho image(s).'
)
write_params_option = click.option(
    '-wp', '--write-params', is_flag=True, type=click.BOOL, default=False, show_default=True,
    help='Write interior / exterior parameters to orthority format file(s), and exit.'
)
out_dir_option = click.option(
    '-od', '--out-dir', type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path), default=None,
    show_default='source file directory.', help='Directory in which to place output file(s).'
)
overwrite_option = click.option(
    '-o', '--overwrite', is_flag=True, type=click.BOOL, default=False, show_default=True,
    help='Overwrite existing ortho image(s).'
)


@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity.')
@click.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=__version__, message='%(version)s')
def cli(verbose, quiet):
    """ Orthorectify remotely sensed imagery. """
    # TODO: how can you pass options, but not chain commands afer wildcard arguments
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
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@write_params_option
@out_dir_option
@overwrite_option
def ortho(
    src_files: Tuple[Path, ...], int_param_file: Path, ext_param_file: Path, crs: rio.CRS, **kwargs
):
    """ Orthorectify using interior and exterior parameter files. """
    if not crs and len(src_files) > 0:
        # read crs from the first source image, if it has one
        # TODO: what if this crs is used, but changes in other source images?
        crs = _read_src_crs(src_files[0])

    # read interior params
    # TODO: create meaningful CLI exceptions
    try:
        if int_param_file.suffix.lower() in ['.yaml', '.yml']:
            int_param_dict = io.read_oty_int_param(int_param_file)
        elif int_param_file.suffix.lower() == '.json':
            int_param_dict = io.read_osfm_int_param(int_param_file)
        else:
            raise click.BadParameter(f"'{int_param_file.suffix}' file type not supported.", param_hint='--int-param')
    except ParamFileError as ex:
        raise click.BadParameter(str(ex), param_hint='--int-param')

    # read exterior params
    try:
        if ext_param_file.suffix.lower() in ['.csv', '.txt']:
            reader = io.CsvReader(ext_param_file, crs=crs)
        elif ext_param_file.suffix.lower() == '.json':
            reader = io.OsfmReader(ext_param_file, crs=crs)
        elif ext_param_file.suffix.lower() == '.geojson':
            reader = io.OtyReader(ext_param_file, crs=crs)
        else:
            raise click.BadParameter(f"'{ext_param_file.suffix}' file type not supported.", param_hint='--ext-param')

    except ParamFileError as ex:
        raise click.BadParameter(str(ex), param_hint='--ext-param')

    except CrsMissingError:
        raise click.MissingParameter(param_hint='--crs', param_type='option')

    ext_param_dict = reader.read_ext_param()

    # finalise the crs
    crs = crs or reader.crs
    if not crs:
        raise click.MissingParameter(param_hint='--crs', param_type='option')

    # orthorectify
    _ortho(src_files=src_files, int_param_dict=int_param_dict, ext_param_dict=ext_param_dict, crs=crs, **kwargs)


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
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@write_params_option
@out_dir_option
@overwrite_option
def exif(src_files: Tuple[Path, ...], crs: rio.CRS, **kwargs):
    """ Orthorectify using EXIF / XMP metadata. """
    if not crs and len(src_files) > 0:
        # get crs from the first source image, if it has one
        crs = _read_src_crs(src_files[0])

    # read interior & exterior params
    try:
        logger.info('Reading camera parameters:')
        reader = io.ExifReader(src_files, crs=crs)
        int_param_dict = reader.read_int_param()
        ext_param_dict = reader.read_ext_param()
    except ParamFileError as ex:
        raise click.BadParameter(str(ex), param_hint='SOURCE...')  # TODO: match param hint to the help, or don't catch

    # get auto UTM crs, if crs not set already
    crs = crs or reader.crs

    # orthorectify
    _ortho(src_files=src_files, int_param_dict=int_param_dict, ext_param_dict=ext_param_dict, crs=crs, **kwargs)


@cli.command(cls=RstCommand)
@click.option(
    '-or', '--odm-root', type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path), required=True,
    default=None,  callback=_odm_root_cb, help='Root path of OpenDroneMap outputs.'
)
@interp_option
@dem_interp_option
@per_band_option
@full_remap_option
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@write_params_option
@out_dir_option
@overwrite_option
def odm(odm_root: Path, out_dir: Path, **kwargs):
    """ Orthorectify using OpenDroneMap outputs. """
    # find source images
    src_files = None
    src_wildcards = ['*.jpg', '*.jpeg', '*.tif', '*.tiff']
    src_wildcards = src_wildcards + [src_wildcard.upper() for src_wildcard in src_wildcards]
    for src_wildcard in src_wildcards:
        src_files = (*odm_root.joinpath('images').glob(src_wildcard),)
        if len(src_files) > 0:
            break
    if not src_files:
        raise click.BadParameter(f'No images found in {odm_root.joinpath("images")}.', param_hint='--odm-root')

    # set crs and resolution from ODM orthophoto or DSM
    orthophoto_file = odm_root.joinpath('odm_orthophoto', 'odm_orthophoto.tif')
    dem_file = odm_root.joinpath('odm_dem', 'dsm.tif')
    if orthophoto_file.exists():
        with rio.open(orthophoto_file, 'r') as ortho_im:
            crs = ortho_im.crs
            resolution = ortho_im.res
    else:
        with rio.open(dem_file, 'r') as dem_im:
            crs = dem_im.crs
            resolution = dem_im.res

    # create and set output dir
    if not out_dir:
        out_dir = odm_root.joinpath('orthority')
        out_dir.mkdir(exist_ok=True)

    # read internal and external params from OpenSfM reconstruction file
    rec_file = odm_root.joinpath('opensfm', 'reconstruction.json')
    reader = io.OsfmReader(rec_file, crs=crs)
    int_param_dict = reader.read_int_param()
    ext_param_dict = reader.read_ext_param()

    # orthorectify
    _ortho(
        src_files=src_files, dem_file=dem_file, int_param_dict=int_param_dict, ext_param_dict=ext_param_dict, crs=crs,
        resolution=resolution, dem_band=1, out_dir=out_dir, **kwargs
    )