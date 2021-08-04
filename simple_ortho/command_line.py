"""
   Copyright 2021 Dugal Harris - dugalh@gmail.com

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
import datetime
import os
import pathlib

import numpy as np
import pandas as pd
import rasterio as rio
import yaml
from simple_ortho import get_logger
from simple_ortho import root_path
from simple_ortho import simple_ortho

# print formatting
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
logger = get_logger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Orthorectify an image with known DEM and camera model.')
    parser.add_argument("src_im_file", help="path(s) or wildcard(s) specifying the source image file(s)", type=str,
                        metavar='src_im_file', nargs='+')
    parser.add_argument("dem_file", help="path to the DEM file", type=str)
    parser.add_argument("pos_ori_file", help="path to the camera position and orientation file", type=str)
    parser.add_argument("-od", "--ortho-dir",
                        help="write ortho image(s) to this directory (default: write ortho image(s) to source directory)",
                        type=str)
    parser.add_argument("-rc", "--readconf",
                        help="read custom config from this path (default: use config.yaml in simple_ortho root)",
                        type=str)
    parser.add_argument("-wc", "--writeconf", help="write default config to this path and exit", type=str)
    parser.add_argument("-v", "--verbosity", choices=[1, 2, 3, 4],
                        help="logging level: 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)", type=int)
    return parser.parse_args()


def _process_args(args):
    # set logging level
    if args.verbosity is not None:
        logger.setLevel(10 * args.verbosity)
        simple_ortho.logger.setLevel(10 * args.verbosity)

    # read configuration
    if args.readconf is None:
        config_filename = root_path.joinpath('config.yaml')
    else:
        config_filename = pathlib.Path(args.readconf)

    if not config_filename.exists():
        raise Exception(f'Config file {config_filename} does not exist')

    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)

    # write configuration if requested and exit
    if args.writeconf is not None:
        out_config_filename = pathlib.Path(args.writeconf)
        with open(out_config_filename, 'w') as f:
            yaml.dump(config, stream=f)
        logger.info(f'Wrote config to {out_config_filename}')
        exit(0)

    # check files exist
    for src_im_file_spec in args.src_im_file:
        src_im_file_path = pathlib.Path(src_im_file_spec)
        if len(list(src_im_file_path.parent.glob(src_im_file_path.name))) == 0:
            raise Exception(f'Could not find any source image(s) matching {src_im_file_spec}')

    if not pathlib.Path(args.dem_file).exists():
        raise Exception(f'DEM file {args.dem_file} does not exist')

    if not pathlib.Path(args.pos_ori_file).exists():
        raise Exception(f'Camera position and orientation file {args.pos_ori_file} does not exist')

    if args.ortho_dir is not None:
        ortho_dir = pathlib.Path(args.ortho_dir)
        if not ortho_dir.is_dir():
            raise Exception(f'Ortho directory {args.ortho_dir} is not a valid directory')
        if not ortho_dir.exists():
            logger.warning(f'Creating ortho directory {args.ortho_dir}')
            os.mkdir(str(ortho_dir))

    return config


def main(args):
    """
    Orthorectify an image

    Parameters
    ----------
    args :  ArgumentParser.parse_args()
            Run `python ortho_im.py -h` to see help on arguments
    cam_pos_orid :  pandas.DataFrame
                   A pandas dataframe containing the camera position and orientation for each image
    config : dict
             Configuration dictionary - see config.yaml or the readme for details
    """

    try:
        # check args and get config
        config = _process_args(args)

        # read camera position and orientation and find row for src_im_file
        cam_pos_orid = pd.read_csv(args.pos_ori_file, header=None, sep=' ', index_col=0,
                                   names=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])

        # loop through image file(s) or wildcard(s), or combinations thereof
        for src_im_file_spec in args.src_im_file:
            src_im_file_path = pathlib.Path(src_im_file_spec)
            for src_im_filename in src_im_file_path.parent.glob(src_im_file_path.name):
                try:
                    if src_im_filename.stem not in cam_pos_orid.index:
                        raise Exception(f'Could not find {src_im_filename.stem} in {args.pos_ori_file}')

                    im_pos_ori = cam_pos_orid.loc[src_im_filename.stem]
                    orientation = np.array(np.pi * im_pos_ori[['omega', 'phi', 'kappa']] / 180.)
                    position = np.array([im_pos_ori['easting'], im_pos_ori['northing'], im_pos_ori['altitude']])

                    # set ortho filename
                    if args.ortho_dir is not None:
                        ortho_im_filename = pathlib.Path(args.ortho_dir).joinpath(src_im_filename.stem + '_ORTHO.tif')
                    else:
                        ortho_im_filename = None

                    # Get src geotransform
                    with rio.open(src_im_filename) as src_im:
                        geo_transform = src_im.transform
                        im_size = np.float64([src_im.width, src_im.height])

                    # create Camera
                    camera_config = config['camera']
                    camera = simple_ortho.Camera(camera_config['focal_len'], camera_config['sensor_size'], im_size,
                                                 geo_transform, position, orientation, dtype=np.float32)

                    # create OrthoIm  and orthorectify
                    logger.info(f'Orthorectifying {src_im_filename.name}')
                    start_ttl = datetime.datetime.now()
                    ortho_im = simple_ortho.OrthoIm(src_im_filename, args.dem_file, camera, config=config['ortho'],
                                                    ortho_im_filename=ortho_im_filename)
                    ortho_im.orthorectify()
                    ttl_time = (datetime.datetime.now() - start_ttl)
                    logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

                    if config['ortho']['build_ovw']:
                        start_ttl = datetime.datetime.now()
                        logger.info(f'Building overviews for {src_im_filename.name}')
                        ortho_im.build_ortho_overviews()
                        ttl_time = (datetime.datetime.now() - start_ttl)
                        logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

                except Exception as ex:
                    # catch exceptions so that problem image(s) don't prevent processing of a batch
                    logger.error('Exception: ' + str(ex))

    except Exception as ex:
        logger.error('Exception: ' + str(ex))
        raise ex

def main_entry():
    args = parse_arguments()
    main(args)