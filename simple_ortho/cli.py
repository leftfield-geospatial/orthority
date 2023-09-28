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
from typing import Tuple, List, Dict, Union, Optional

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
        """ Strip some RST markup from the help text for CLI display.  Doesn't work with grid tables. """

        # Note that this can't easily be done in __init__, as each sub-command's __init__ gets called,
        # which ends up re-assigning self.wrap_text to reformat_text
        if not hasattr(self, 'click_wrap_text'):
            self.click_wrap_text = click.formatting.wrap_text

        sub_strings = {
            '\b\n': '\n\b',                 # convert from RST friendly to click literal (unwrapped) block marker
            r'\| ': '',                     # strip RST literal (unwrapped) marker in e.g. tables and bullet lists
            r'\n\.\. _.*:\n': '',            # strip RST ref directive '\n.. _<name>:\n'
            '::': ':',                      # convert from RST '::' to ':'
            '``(.*?)``': r'\g<1>',          # convert from RST '``literal``' to 'literal'
            ':option:`(.*?)( <.*?>)?`': r'\g<1>',   # convert ':option:`--name <group-command --name>`' to '--name'
            ':option:`(.*?)`': r'\g<1>',    # convert ':option:`--name`' to '--name'
            '-{4,}': r'',                   # strip '----...'
            '`([^<]*) <([^>]*)>`_': r'\g<1>',  # convert from RST cross-ref '`<name> <<link>>`_' to 'name'
        }  # yapf: disable

        def reformat_text(text, width, **kwargs):
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
    pkg_logger = logging.getLogger('simple_ortho')
    formatter = PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)
    logging.captureWarnings(True)


def _read_src_crs(filename: Path) -> Optional[rio.CRS]:
    """ Read CRS from source image file. """
    with suppress_no_georef(), rio.open(filename, 'r') as im:
        if not im.crs:
            logger.debug(f'No CRS found for source image: {filename.name}')
        return im.crs


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


def _odm_proj_dir_cb(ctx: click.Context, param: click.Parameter, proj_dir: Path):
    """ click callback to validate the ODM project directory. """
    req_paths = [Path('opensfm').joinpath('reconstruction.json'), Path('odm_dem').joinpath('dsm.tif'), Path('images')]
    for req_path in req_paths:
        req_path = proj_dir.joinpath(req_path)
        if not req_path.exists():
            raise click.BadParameter(f'Could not find {req_path}.', param=param)
    return proj_dir


def _ortho(
    src_files: Tuple[Path, ...], dem_file: Path, int_param_dict: Dict[str, Dict], ext_param_dict: Dict[str, Dict],
    crs: rio.CRS, dem_band: int, alpha: float, export_params: bool, out_dir: Path, overwrite: bool, **kwargs
):
    """ """
    # TODO: multiple file progress as a master bar, or 'x of N' prefix to individual bars
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
                f"Could not find parameters for '{src_file.name}'.", param_hint='--ext-param'
            )

        # get interior params for ext_param
        if ext_param['camera']:
            cam_id = ext_param['camera']
            if cam_id not in int_param_dict:
                raise click.BadParameter(f"Could not find parameters for camera '{cam_id}'.", param_hint='--int-param')
            int_param = int_param_dict[cam_id]
        elif len(int_param_dict) == 1:
            cam_id = None
            int_param = list(int_param_dict.values())[0]
        else:
            raise click.BadParameter(f"'camera' ID for {src_file.name} should be specified.", param_hint='--ext-param')

        # get camera if it exists, otherwise create camera, then update with exterior parameters and store
        camera = cameras[cam_id] if cam_id in cameras else create_camera(**int_param, alpha=alpha)
        camera.update(xyz=ext_param['xyz'], opk=ext_param['opk'])
        cameras[cam_id] = camera

        # create camera and ortho objects
        # TODO: generalise exterior params / camera so that the cli can just expand the dict and not need to know
        #  about the internals
        try:
            ortho = Ortho(src_file, dem_file, camera, crs, dem_band=dem_band)
        except DemBandError as ex:
            raise click.BadParameter(str(ex), param_hint='--dem_band')
        ortho_file = out_dir.joinpath(f'{src_file.stem}_ORTHO.tif')

        # orthorectify
        logger.info(f'Orthorectifying {src_file.name} ({src_i +1 } of {len(src_files)}):')
        ortho.process(ortho_file, overwrite=overwrite, **kwargs)


# TODO: add mosaic, and write param options
# Define click options that are common to more than one command
src_files_arg = click.argument(
    'src_files', nargs=-1, metavar='SOURCE...', type=click.Path(exists=True, dir_okay=False, path_type=Path),
    # help='Path/URL of source image(s) to be orthorectified..'
)
dem_file_option = click.option(
    '-d', '--dem', 'dem_file', type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None, help='Path/URL of a DEM image covering the source image(s). [required]'
)
int_param_file_option = click.option(
    '-ip', '--int-param', 'int_param_file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, default=None,
    help='Path of an internal parameter file.'
)
ext_param_file_option = click.option(
    '-ep', '--ext-param', 'ext_param_file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, default=None,
    help='Path of an external parameter file.'
)
crs_option = click.option(
    '-c', '--crs', type=click.STRING, default=None, show_default='auto', callback=_crs_cb,
    help='CRS of the exterior parameters and ortho image as an EPSG, proj4, or WKT string; or path of a text file '
         'containing string.'
)
resolution_option = click.option(
    '-r', '--res', 'resolution', type=click.FLOAT, default=None, show_default='auto', multiple=True,
    callback=_resolution_cb,
    help='Ortho image pixel size in units of the :option:`--crs` (usually meters).  Can be used twice for non-square '
         'pixels: ``--res PIXEL_WIDTH --res PIXEL_HEIGHT``.'
)
dem_band_option = click.option(
    '-db', '--dem-band', type=click.INT, nargs=1, default=Ortho._default_config['dem_band'], show_default=True,
    help='Index of the DEM band to use (1 based).'
)
interp_option = click.option(
    '-i', '--interp', type=click.Choice(Interp.cv_list(), case_sensitive=False),
    default=Ortho._default_config['interp'], show_default=True,
    help=f'Interpolation method for remapping source to ortho image.'
)
dem_interp_option = click.option(
    '-di', '--dem-interp', type=click.Choice(Interp, case_sensitive=False),
    default=Ortho._default_config['dem_interp'], show_default=True,
    help=f'Interpolation method for DEM reprojection.'
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
                            'undistorted source image with pinhole camera model (``--no-full-remap``).  '
                            '``--no-full-remap`` is faster but can reduce the ortho image quality.'
)
alpha_option = click.option(
    '-a', '--alpha', type=click.FloatRange(0, 1), nargs=1, default=1, show_default=True,
    help='Scaling of the ``--no-full-remap`` undistorted image: 0 results in an undistorted image with all valid '
         'pixels, 1 results in an undistorted image with all source pixels.'
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
export_params_option = click.option(
    '-ep', '--export-params', is_flag=True, type=click.BOOL, default=False, show_default=True,
    help='Export interior & exterior parameters to orthority format file(s), and exit.'
)
out_dir_option = click.option(
    '-od', '--out-dir', type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    default=Path.cwd(), show_default='current working', help='Directory in which to place output file(s).'
)
overwrite_option = click.option(
    '-o', '--overwrite', is_flag=True, type=click.BOOL, default=False, show_default=True,
    help='Overwrite existing output(s).'
)


@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity.')
@click.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=__version__, message='%(version)s')
def cli(verbose, quiet):
    """ Orthorectification toolkit. """
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
@alpha_option
@export_params_option
@out_dir_option
@overwrite_option
def ortho(
    src_files: Tuple[Path, ...], int_param_file: Path, ext_param_file: Path, crs: rio.CRS, **kwargs
):
    # yapf:disable  @formatter:off
    """
    Orthorectify using interior and exterior parameter files.

    Orthorectify images with camera models specified in interior and exterior parameter files. Interior parameters are
    supported in orthority (.yaml), ODM cameras (.json), and OpenSfM reconstruction (.json) formats.  Exterior
    parameters are supported in orthority (.geojson), custom CSV (.csv), and ODM / OpenSfM reconstruction (.json)
    formats.  Note that parameter file extensions are used to distinguish their format.

    If possible, an ortho CRS will be read from other sources, or auto-determined when :option:`--crs <oty-ortho
    --crs>` is not passed.

    See the `online docs <?>`_ for more detail on file formats and CRS.
    \b

    Examples
    ========

    Orthorectify `source.tif` using DEM `dem.tif` and interior and exterior parameters from `reconstruction.json`.  An
    ortho resolution and UTM CRS are auto-determined::

        oty ortho --dem dem.tif --int-param reconstruction.json --ext-param reconstruction.json source.tif

    Convert `reconstruction.json` internal and external parameters to orthority format files in the current working
    directory::

        oty ortho --int-param reconstruction.json --ext-param reconstruction.json --write-params

    Orthorectify images matching `*rgb.tif` using DEM `dem.tif`, and `int_param.yaml` interior &
    `ext_param.csv` exterior parameter files.  Specify a 1m ortho resolution and `EPSG:32651` CRS.  Write ortho files to
    the `data` directory using `deflate` compression and a `uint16` data type::

        oty ortho --dem dem.tif --int-param int_param.yaml --ext-param ext_param.csv --res 1 --crs EPSG:32651 --out-dir data --compress deflate --dtype uint16 *rgb.tif

    SOURCE... Path/URL(s) of source image(s) to orthorectify.
    """
    # yapf:enable @formatter:on
    if not crs and len(src_files) > 0:
        # read crs from the first source image, if it has one
        # TODO: what if this crs is used, but changes in other source images?
        crs = _read_src_crs(src_files[0])

    # read interior params
    # TODO: create meaningful CLI exceptions
    # TODO: if --write-params is passed it requires both --int-param & --ext-param but it would be useful to only
    #  require one
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
        raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')

    ext_param_dict = reader.read_ext_param()

    # finalise the crs
    crs = crs or reader.crs
    if not crs:
        raise click.MissingParameter(param_hint="'-c' / '--crs'", param_type='option')

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
@alpha_option
@export_params_option
@out_dir_option
@overwrite_option
def exif(src_files: Tuple[Path, ...], crs: rio.CRS, **kwargs):
    """
    Orthorectify images with EXIF / XMP tags.

    Orthorectify images with pinhole camera models derived from EXIF / XMP metadata.  Image tags should include focal
    length & sensor size, or 35mm equivalent focal length; camera position; and camera / gimbal roll, pitch & yaw.

    See the `online docs <?>`_ for more detail.

    \b

    Examples
    ========

    Orthorectify images matching `*rgb.tif` with DEM `dem.tif`::

        oty exif --dem dem.tif *rgb.tif

    Write internal and external parameters for images matching `*rgb.tif` to orthority format files, and exit::

        oty exif --write-params *rgb.tif

    SOURCE... Path/URL(s) of source image(s) to orthorectify.
    """
    # TODO: allow partial exif spec with --int-param and --ext-param overrides (?)
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
# TODO: root, project or dataset
@click.option(
    '-pd', '--proj-dir', type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path), required=True,
    default=None,  callback=_odm_proj_dir_cb, help='Path of the ODM project folder to process.'
)
@resolution_option
@interp_option
@dem_interp_option
@per_band_option
@full_remap_option
@write_mask_option
@dtype_option
@compress_option
@build_ovw_option
@alpha_option
@export_params_option
@overwrite_option
def odm(proj_dir: Path, resolution: Tuple[float, float], **kwargs):
    # yapf:disable @formatter:off
    """
    Orthorectify existing OpenDroneMap outputs.

    Orthorectify individual images using OpenDroneMap (ODM) generated camera models and DSM.  The DSM is required,
    and can be generated by running ODM with the `--dsm <https://docs.opendronemap.org/arguments/dsm/#dsm>`_ option.

    By default, the ortho resolution is read from the ODM orthophoto.  If that does not exist, it is read from the DSM.
    Ortho images & parameter files are placed in the `<odm project>/orthority` directory.

    The ``ortho`` sub-command  can be used for more control over options.
    \b

    Examples
    ========

    Orthorectify images in `<odm project>/images` directory using the `<odm project>` camera models and DSM::

        oty odm --proj-dir <odm project>

    Write the `<odm project>` interior and exterior parameters to orthority format files, and exit::

        oty odm --proj-dir <odm project> --write-params

    Orthorectify images in `<odm project>/images` directory using the `<odm project>` camera models and DSM.  Use an
    ortho resolution of 0.1m and `lanczos` interpolation to remap source to ortho.

        oty odm --proj-dir <odm project> --res 0.1 --interp lanczos
    """
    # yapf:enable @formatter:on
    # find source images
    src_exts = ['.jpg', '.jpeg', '.tif', '.tiff']
    src_files = tuple([p for p in proj_dir.joinpath('images').glob('*.*') if p.suffix.lower() in src_exts])
    if len(src_files) == 0:
        raise click.BadParameter(f'No images found in {proj_dir.joinpath("images")}.', param_hint='--odm-root')

    # set crs and resolution from ODM orthophoto or DSM
    orthophoto_file = proj_dir.joinpath('odm_orthophoto', 'odm_orthophoto.tif')
    dem_file = proj_dir.joinpath('odm_dem', 'dsm.tif')
    if orthophoto_file.exists():
        with rio.open(orthophoto_file, 'r') as ortho_im:
            crs = ortho_im.crs
            resolution = resolution or ortho_im.res
    else:
        with rio.open(dem_file, 'r') as dem_im:
            crs = dem_im.crs
            resolution = resolution or dem_im.res

    # create and set output dir
    out_dir = proj_dir.joinpath('orthority')
    out_dir.mkdir(exist_ok=True)

    # read internal and external params from OpenSfM reconstruction file
    rec_file = proj_dir.joinpath('opensfm', 'reconstruction.json')
    reader = io.OsfmReader(rec_file, crs=crs)
    int_param_dict = reader.read_int_param()
    ext_param_dict = reader.read_ext_param()

    # orthorectify
    _ortho(
        src_files=src_files, dem_file=dem_file, int_param_dict=int_param_dict, ext_param_dict=ext_param_dict, crs=crs,
        resolution=resolution, dem_band=1, out_dir=out_dir, **kwargs
    )


if __name__ == '__main__':
    cli()
