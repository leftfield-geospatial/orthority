"""
    Copyright (c) 2021 Dugal Harris - dugalh@gmail.com
    This project is licensed under the terms of the MIT license.
"""

import numpy as np
import pandas as pd
import pathlib
from simple_ortho import get_logger
from simple_ortho import simple_ortho
from simple_ortho import root_path
import yaml
import argparse
import glob
from scripts import ortho_im

# print formatting
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
logger = get_logger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_im_wildcard", help="source image wildcard pattern or directory (e.g. '.' or '*_CMP.TIF')", type=str)
    parser.add_argument("dem_file", help="path to the DEM file", type=str)
    parser.add_argument("pos_ori_file", help="path to the camera position and orientaion file", type=str)
    parser.add_argument("-rc", "--readconf", help="read custom config from this path (default: use config.yaml in simple_ortho root)", type=str)
    parser.add_argument("-v", "--verbosity", choices=[1, 2, 3, 4], help="logging level: 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)", type=int)
    return parser.parse_args()

def process_args(args):
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

    # check files exist
    if len(glob.glob(args.src_im_wildcard)) == 0:
        if len(glob.glob(args.src_im_wildcard + '*.tif')) == 0:
            raise Exception(f'Could not find any files matching {args.src_im_wildcard}')
        else:
            args.src_im_wildcard = args.src_im_wildcard + '*.tif'

    if not pathlib.Path(args.dem_file).exists():
        raise Exception(f'DEM file {args.dem_file} does not exist')

    if not pathlib.Path(args.pos_ori_file).exists():
        raise Exception(f'Camera position and orientaion file {args.pos_ori_file} does not exist')

    return config

def main(args):

    try:
        # parse the command line
        config = process_args(args)

        # read camera position and orientation and find row for src_im_file
        cam_pos_orid = pd.read_csv(args.pos_ori_file, header=None, sep=' ', index_col=0,
                           names=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])
        src_im_list = glob.glob(args.src_im_wildcard)
        logger.info(f'Batch orthorectifying {len(src_im_list)} file(s) matching {args.src_im_wildcard}')
        for src_i, src_im_filename in enumerate(src_im_list):
            try:
                src_im_filename = pathlib.Path(src_im_filename)
                args.src_im_file = str(src_im_filename)
                args.ortho = None
                logger.info(f'Processing {src_im_filename.stem} - file {src_i + 1} of {len(src_im_list)}:')
                ortho_im.main(args, cam_pos_orid=cam_pos_orid, config=config)

            except:
                pass    # logged in ortho_im

    except Exception as ex:
        logger.error('Exception: ' + str(ex))
        raise ex

if __name__ == "__main__":
    args = parse_arguments()
    main(args)