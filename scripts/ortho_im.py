"""
    Copyright (c) 2021 Dugal Harris - dugalh@gmail.com
    This project is licensed under the terms of the MIT license.
"""

import rasterio as rio
import numpy as np
import pandas as pd
import pathlib
from simple_ortho import get_logger
from simple_ortho import simple_ortho
from simple_ortho import root_path
import yaml
import argparse

# print formatting
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
logger = get_logger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_im_file", help="path to the source image file", type=str)
    parser.add_argument("dem_file", help="path to the DEM file", type=str)
    parser.add_argument("pos_ori_file", help="path to the camera position and orientaion file", type=str)
    parser.add_argument("-o", "--ortho", help="write ortho image to this path (default: append '_ORTHO' to src_im_file)", type=str)
    parser.add_argument("-rc", "--readconf", help="read custom config from this path (default: use config.yaml in simple_ortho root)", type=str)
    parser.add_argument("-wc", "--writeconf", help="write default config to this path and exit", type=str) # TODO make this so it doesn't require positionsal args
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

    # write configuration if requested and exit
    if args.writeconf is not None:
        out_config_filename = pathlib.Path(args.writeconf)
        with open(out_config_filename, 'w') as f:
            yaml.dump(config, stream=f)
        logger.info(f'Wrote config to {out_config_filename}')
        exit(0)

    # check files exist
    if not pathlib.Path(args.src_im_file).exists():
        raise Exception(f'Source image file {args.src_im_file} does not exist')

    if not pathlib.Path(args.dem_file).exists():
        raise Exception(f'DEM file {args.pos_ori_file} does not exist')

    if not pathlib.Path(args.pos_ori_file).exists():
        raise Exception(f'Camera position and orientaion file {args.pos_ori_file} does not exist')

    return config

def main(args, cam_pos_orid=None, config=None):
    """
    Orthorectify an image

    Parameters
    ----------
    args :  ArgumentParser.parse_args() object containing requisite parameters

    """

    try:
        # check args and get config
        if config is None:
            config = process_args(args)

        # read camera position and orientation and find row for src_im_file
        if cam_pos_orid is None:
            cam_pos_orid = pd.read_csv(args.pos_ori_file, header=None, sep=' ', index_col=0,
                               names=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])

        src_im_file_stem = pathlib.Path(args.src_im_file).stem[:-4] # TODO remove -4 i.e. make a new cam ori file(s)
        if not src_im_file_stem in cam_pos_orid.index:
            raise Exception(f'Could not find {src_im_file_stem} in {args.pos_ori_file}')

        im_pos_ori = cam_pos_orid.loc[src_im_file_stem]
        orientation = np.array(np.pi * im_pos_ori[['omega', 'phi', 'kappa']] / 180.)
        position = np.array([im_pos_ori['easting'], im_pos_ori['northing'], im_pos_ori['altitude']])

        # set ortho filename
        if args.ortho is not None:
            ortho_im_filename = pathlib.Path(args.ortho)
        else:
            ortho_im_filename = None

        # Get src geotransform
        with rio.open(args.src_im_file) as src_im:
            geo_transform = src_im.transform
            im_size = np.float64([src_im.width, src_im.height])

        # create Camera
        camera_config = config['camera']
        camera = simple_ortho.Camera(camera_config['focal_len'], camera_config['sensor_size'], camera_config['im_size'],
                                     geo_transform, position, orientation, dtype='float32')

        # create OrthoIm  and orthorectify
        ortho_im = simple_ortho.OrthoIm(args.src_im_file, args.dem_file, camera, config=config['ortho'],
                                        ortho_im_filename=ortho_im_filename)
        ortho_im.orthorectify()

    except Exception as ex:
        logger.error(ex)
        raise ex

if __name__ == "__main__":
    args = parse_arguments()
    main(args)