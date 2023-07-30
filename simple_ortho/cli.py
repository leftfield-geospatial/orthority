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


def _crs_cb(ctx, param, crs):
    """ click callback to validate and parse the CRS. """
    if crs is not None:
        try:
            crs_file = Path(crs)
            if crs_file.exists():  # read string from file, if it exists
                with open(crs_file, 'r') as f:
                    crs = f.read()

            crs = rio.CRS.from_string(crs)
        except CRSError as ex:
            raise click.BadParameter(f'Invalid CRS: {crs}.\n {str(ex)}', param=param)
    return crs


# Define click options that are common to more than one command
src_files_arg = click.argument(
    'src_files', nargs=-1, metavar='SOURCE...', type=click.Path(exists=True, dir_okay=False, path_type=Path),
    # help='Path/URL of source image(s) to be orthorectified..'
)
dem_file_option = click.option(
    '-d', '--dem', 'dem_file', type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True, default=None, help='Path/URL to a DEM image covering the source image(s).'
)
pos_ori_file_option = click.option(
    '-po', '--pos-ori', 'pos_ori_file', type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True, default=None, help='Path of a camera position & orientation file.'
)
cam_conf_file_option = click.option(
    '-cc', '--cam-conf', 'cam_conf_file', type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True, default=None, help='Path of a camera configuration file.'
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
# TODO: allow single number for square pixel
resolution_option = click.option(
    '-r', '--res', 'resolution', type=click.FLOAT, nargs=2, required=True, default=None, metavar='X Y',
    help='Ortho image (x, y) pixel size in units of the :option:`--crs` (usually meters).'
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
@pos_ori_file_option
@cam_conf_file_option
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
    src_files: Tuple[Path, ...], dem_file: Path, pos_ori_file: Path, cam_conf_file: Path, crs: rio.CRS,
    resolution: Tuple[float, float], dem_band: int, ortho_dir: Path, **kwargs
):
    """
    Orthorectify image(s) with known camera model and DEM.

    Orthorectify images with camera (easting, northing, altitude) position & (omega, phi, kappa) rotation specified in
    a text file, and camera intrinsic parameters in a configuration file.
    """
    # read camera config
    with open(cam_conf_file, 'r') as f:
        cam_conf = yaml.safe_load(f)
    cam_conf = cam_conf.get('camera', cam_conf)     # support old format camera section
    req_keys = ('type', 'focal_len')
    if not all([k in cam_conf for k in req_keys]):
        raise click.BadParameter(
            f'{cam_conf_file.name} does not contain all the keys: {req_keys}.', param_hint='--cam-conf'
        )
    cam_conf.pop('name', None)
    cam_type = cam_conf.pop('type')
    # TODO: check the other params are valid for the camera type

    # read position & orientation file
    with open(pos_ori_file, 'r', newline='') as f:
        reader = csv.DictReader(
            f, delimiter=' ', fieldnames=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'],
        )
        pos_ori_dict = {
            row['file']: {k: float(row[k]) for k in reader.fieldnames[1:]} for row in reader
        }  # yapf: disable
    # TODO: check format of pos_ori_file

    for src_file in src_files:
        # extract position and orientation for src_file from pos_ori_file
        if src_file.stem not in pos_ori_dict:
            raise click.BadParameter(
                f'{pos_ori_file.name} does not contain an entry for {src_file.name}', param_hint='--pos-ori'
            )

        im_pos_ori = pos_ori_dict[src_file.stem]
        position = (im_pos_ori['easting'], im_pos_ori['northing'], im_pos_ori['altitude'])
        rotation = np.radians((im_pos_ori['omega'], im_pos_ori['phi'], im_pos_ori['kappa']))

        # get src_file image size
        with suppress_no_georef(), rio.open(src_file, 'r') as src_im:
            cam_conf['im_size'] = (src_im.width, src_im.height)

        # create the camera, ortho object, and ortho filename
        camera = create_camera(cam_type, position, rotation, **cam_conf)
        ortho = Ortho(src_file, dem_file, camera, crs, dem_band=dem_band)
        ortho_root = ortho_dir or src_file.parent
        ortho_file = ortho_root.joinpath(f'{src_file.stem}_ORTHO.tif')

        # orthorectify
        logger.info(f'Orthorectifying {src_file.name}:')
        ortho.process(ortho_file, resolution, **kwargs)


cli.add_command(opk)
