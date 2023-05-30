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
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def angle_axis_to_opk(angle_exis: Tuple[float]) -> Tuple[float]:
    """
    Convert given ODM angle/axis vector to omega, phi, kappa in PATB convention.
    """
    # convert ODM angle/axis to rotation matrix (see https://github.com/mapillary/OpenSfM/issues/121)
    R = cv2.Rodrigues(np.array(angle_exis))[0].T

    # ODM uses a camera co-ordinate system with x->right, y->down, and z->forward (looking through the camera at the
    # scene), while simple-ortho uses PATB, which is x->right, y->up, and z->backward.  Here we rotate from ODM
    # convention to PATB.
    R = np.dot(R, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

    # extract OPK from R (see https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf)
    omega = np.arctan2(-R[1, 2], R[2, 2])
    phi = np.arcsin(R[0, 2])
    kappa = np.arctan2(- R[0, 1], R[0, 0])
    return (omega, phi, kappa)


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Convert ODM `odm_report/shots.geojson` to simple-ortho compatible camera position and orientation.'
    )
    parser.add_argument('odm_file_path', help='path to input ODM `shots.geojson` file', type=str)
    parser.add_argument('out_file_path', help='path of output camera position/orientation file to create', type=str)
    parser.add_argument(
        '-o', '--overwrite', help='overwrite camera position/orientation file if it exists', action='store_true'
    )
    return parser.parse_args()


def main():
    """
    Convert ODM `odm_report/shots.geojson` to simple-ortho compatible camera position and orientation.
    """
    args = parse_arguments()

    # check paths
    odm_file_path = Path(args.odm_file_path)
    if not odm_file_path.exists():
        raise FileNotFoundError(f'ODM file {odm_file_path} does not exist.')

    out_file_path = Path(args.out_file_path)
    if out_file_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f'Output file {out_file_path} exists, and won\'t be overwritten without the `overwrite` option.'
            )

    # read ODM shots.geojson file into dict
    shots_dict = json.load(open(odm_file_path))

    # write simple-ortho compatible camera position/orientation file
    with open(out_file_path, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        for feature in shots_dict['features']:
            filename = Path(feature['properties']['filename'])
            pos = feature['properties']['translation']
            ori = feature['properties']['rotation']

            # convert ODM angle/axis to simple-ortho OPK
            opk = np.degrees(angle_axis_to_opk(ori))
            writer.writerow([filename.stem, *pos, *opk])


if __name__ == "__main__":
    main()
