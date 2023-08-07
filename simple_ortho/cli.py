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

import numpy as np
import rasterio as rio
from rasterio.errors import CRSError
from simple_ortho.ortho import Ortho
from simple_ortho.camera import create_camera
from simple_ortho.enums import Compress, Interp, CameraType
from simple_ortho.utils import suppress_no_georef
from simple_ortho.io import read_int_param
from simple_ortho.version import __version__


logger = logging.getLogger(__name__)


class PlainInfoFormatter(logging.Formatter):
    """ logging formatter to format INFO logs without the module name etc. prefix. """

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


def crs_cb(ctx: click.Context, param: click.Parameter, crs: str):
    """ click callback to validate and parse the CRS. """
    if crs is not None:
        try:
            crs_file = Path(crs)
            if crs_file.exists():  # read string from file, if it exists
                with open(crs_file, 'r') as f:
                    crs = f.read()

            crs = rio.CRS.from_string(crs)
        except CRSError as ex:
            raise click.BadParameter(f'{crs}.\n {str(ex)}', param=param)
    return crs


def resolution_cb(ctx: click.Context, param: click.Parameter, resolution: Tuple):
    """ click callback to validate and parse the resolution. """
    if len(resolution) == 1:
        resolution *= 2
    elif len(resolution) > 2:
        raise click.BadParameter(f'at most two resolution values should be specified.', param=param)
    return resolution


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
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path), required=True, default=None,
    help='Path of a file specifying camera internal parameters.'
)
ext_param_file_option = click.option(
    '-ep', '--ext-param', 'ext_param_file',
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path), required=True, default=None,
    help='Path of a file specifying camera external parameters.'
)
crs_option = click.option(
    '-c', '--crs', type=click.STRING, default=None, show_default='source image CRS.', callback=crs_cb,
    help='CRS of the ortho image as an EPSG, proj4, or WKT string, or text file containing string.  Should be a '
         'projected, and not geographic CRS.  Can be omitted if source image(s) are projected in this CRS.'
)
dem_band_option = click.option(
    '-db', '--dem-band', type=click.INT, nargs=1, default=1, show_default=True,
    help='Index of the DEM band to use (1 based).'
)
# TODO: allow single number for square pixel
resolution_option = click.option(
    '-r', '--res', 'resolution', type=click.FLOAT, required=True, default=None, multiple=True, callback=resolution_cb,
    help='Ortho image pixel size in units of the :option:`--crs` (usually meters).  Can be used twice for non-square '
         'pixels: ``--res PIXEL_WIDTH --res PIXEL_HEIGHT``'
)
# TODO: change interp -> src_interp
src_interp_option = click.option(
    '-si', '--src-interp', 'interp', type=click.Choice(Interp.cv_list(), case_sensitive=False),
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
ortho_dir_option = click.option(
    '-od', '--ortho-dir', type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path), default=None,
    show_default='source image directory.', help='Directory in which to place ortho image(s).'
)
overwrite_option = click.option(
    '-o', '--overwrite', is_flag=True, type=click.BOOL, default=False, show_default=True,
    help='Overwrite existing ortho image(s).'
)


@click.group()
@click.option('--verbose', '-v', count=True, help="Increase verbosity.")
@click.option('--quiet', '-q', count=True, help="Decrease verbosity.")
@click.version_option(version=__version__, message='%(version)s')
def cli(verbose, quiet):
    """ Orthorectify remotely sensed imagery. """
    verbosity = verbose - quiet
    _configure_logging(verbosity)


@click.command(cls=RstCommand)
@src_files_arg
@dem_file_option
@int_param_file_option
@ext_param_file_option
@crs_option
@resolution_option
@dem_band_option
@src_interp_option
@dem_interp_option
@per_band_option
@full_remap_option
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@ortho_dir_option
@overwrite_option
def opk(
    src_files: Tuple[Path, ...], dem_file: Path, int_param_file: Path, ext_param_file: Path, crs: rio.CRS,
    resolution: Tuple[float, float], dem_band: int, ortho_dir: Path, **kwargs
):
    """
    Orthorectify image(s) with known camera model and DEM.

    Orthorectify images with camera (easting, northing, altitude) position & (omega, phi, kappa) rotation specified in
    a text file, and camera intrinsic parameters in a configuration file.
    """
    # read camera config
    try:
        int_param_dict = read_int_param(int_param_file)
    except ValueError as ex:
        raise click.BadParameter(str(ex), param_hint='--int-param')

    # read position & orientation file
    with open(ext_param_file, 'r', newline='') as f:
        reader = csv.DictReader(
            f, delimiter=' ', fieldnames=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'],
        )
        ext_param_dict = {
            row['file']: {k: float(row[k]) for k in reader.fieldnames[1:]} for row in reader
        }  # yapf: disable
    # TODO: check format of pos_ori_file

    for src_file in src_files:
        # extract position and orientation for src_file from pos_ori_file
        if src_file.stem not in ext_param_dict:
            raise click.BadParameter(
                f'{ext_param_file.name} does not contain an entry for {src_file.name}', param_hint='--ext-param'
            )

        src_ext_param_dict = ext_param_dict[src_file.stem]
        position = (src_ext_param_dict['easting'], src_ext_param_dict['northing'], src_ext_param_dict['altitude'])
        rotation = np.radians((src_ext_param_dict['omega'], src_ext_param_dict['phi'], src_ext_param_dict['kappa']))

        # get src_file image size
        with suppress_no_georef(), rio.open(src_file, 'r') as src_im:
            im_size = (src_im.width, src_im.height)

        # create the camera, ortho object, and ortho filename
        # TODO: call it orientation or rotation?
        int_param_key = next(iter(int_param_dict))
        camera = create_camera(position=position, rotation=rotation, im_size=im_size, **int_param_dict[int_param_key])
        ortho = Ortho(src_file, dem_file, camera, crs, dem_band=dem_band)
        ortho_root = ortho_dir or src_file.parent
        ortho_file = ortho_root.joinpath(f'{src_file.stem}_ORTHO.tif')

        # orthorectify
        logger.info(f'Orthorectifying {src_file.name}:')
        ortho.process(ortho_file, resolution, **kwargs)


cli.add_command(opk)
