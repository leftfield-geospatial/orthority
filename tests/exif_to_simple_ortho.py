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
from typing import Tuple, Union, List

import numpy as np
from rasterio import CRS
from rasterio.warp import transform
from simple_ortho.exif import Exif
# TODO: standardise OPK/rotation naming & units here and in camera.py.  also standardise docs here and in
#  odm_to_simple_ortho.py


def rpy_to_opk(rpy: Tuple[float], lla: Tuple[float], crs: CRS, cbb: Union[None, List[List]] = None) -> Tuple[float]:
    """
    Convert (roll, pitch, yaw) to (omega, phi, kappa) angles.

    (roll, pitch, yaw) are angles to rotate from body to navigation systems, where the body system is centered on and
    aligned with the gimbal/camera with (x->front, y->right, z->down).  The navigation system shares its center with
    the body system, but its xy-plane is perpendicular to the local plumbline (x->N, y->E, z->down).

    (omega, phi, kappa) are angles to rotate from world to camera coordinate systems. World coordinates are a
    projected system like UTM (origin fixed near earth surface, and usually some distance from camera), and camera
    coordinates are centered on and aligned with the camera (in PATB convention: x->right, y->up, z->backwards looking
    through the camera at the scene).

    Based on https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.

    Parameters
    ----------
    rpy: tuple of float
        (roll, pitch, yaw) camera angles in radians to rotate from body to navigation coordinate system.
    lla: tuple of float
        (latitude, longitude, altitude) navigation system co-ordinates of the body.
    crs: rasterio.CRS
        World coordinate reference system as a rasterio CRS object (the same CRS used for the ortho image).
    cbb: list of list of float, optional
        Optional camera to body rotation matrix.  Defaults to reference values: [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        which describe typical drone geometry where the camera top points in the flying direction & the camera is
        looking down.

    Returns
    -------
    tuple of float
        (omega, phi, kappa) angles in radians, to rotate from camera to world coordinate systems.
    """
    # Adapted from the OpenSFM exif module https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py which follows
    # https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.
    # To keep with the naming convention in the rest of simple-ortho, what the refernce calls Object (E) co-ordinates, I
    # call world coordinates, and what the reference calls Image (B) coordinates, I have called camera coordinates (
    # which helps differentiate it from pixel coordinates).

    lla = np.array(lla)
    roll, pitch, yaw = rpy
    nav_crs = CRS.from_epsg(4326)
    world_crs = CRS.from_string(crs) if isinstance(crs, str) else crs

    # find rotation matrix cnb, to rotate from body to navigation coordinates.
    rx = np.array(
        [[1, 0, 0],
         [0, np.cos(roll), -np.sin(roll)],
         [0, np.sin(roll), np.cos(roll)]]
    )

    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)],
         [0, 1, 0],
         [-np.sin(pitch), 0, np.cos(pitch)]]
    )

    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0],
         [np.sin(yaw), np.cos(yaw), 0],
         [0, 0, 1]]
    )

    cnb = rz.dot(ry).dot(rx)

    # find rotation matrix cen, to rotate from navigation to world coordinates (world is called object (E) in the
    # reference)
    delta = 1e-7
    lla1 = lla + (delta, 0, 0)
    lla2 = lla - (delta, 0, 0)

    # p1 & p2 must be in the world/ortho CRS, not ECEF as might be understood from the reference
    p1 = np.array(
        transform(
            nav_crs, world_crs, [lla1[1]], [lla1[0]], [lla1[2]]
        )
    ).squeeze()
    p2 = np.array(
        transform(
            nav_crs, world_crs, [lla2[1]], [lla2[0]], [lla2[2]]
        )
    ).squeeze()

    # approximate the relative alignment of world and navigation systems
    xnp = p1 - p2
    m = np.linalg.norm(xnp)
    xnp /= m                # unit vector in navigation system N direction
    znp = np.array([0, 0, -1]).T
    ynp = np.cross(znp, xnp)
    cen = np.array([xnp, ynp, znp]).T

    # cbb is the rotation from camera to body coordinates (camera is called image (B) in the reference).
    cbb = np.array(cbb) if cbb is not None else np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    # combine cen, cnb, cbb to find rotation from camera (B) to world (E) coordinates.
    ceb = cen.dot(cnb).dot(cbb)

    # extract OPK angles from ceb
    omega = np.arctan2(-ceb[1][2], ceb[2][2])
    phi = np.arcsin(ceb[0][2])
    kappa = np.arctan2(-ceb[0][1], ceb[0][0])
    return omega, phi, kappa


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='''
        Convert EXIF GPS position and (roll, pitch, yaw) angles to simple-ortho compatible camera position and 
        orientation.
        '''
    )
    parser.add_argument(
        "src_im_wildcard", help="wildcard pattern for source images containing EXIF data (e.g. 'DJI*.JPG')", type=str
    )
    parser.add_argument('out_file_path', help='path of output camera position/orientation file to create', type=str)
    parser.add_argument('ortho_crs', help='world / ortho image CRS as an EPSG, proj4 or WKT string', type=str)
    parser.add_argument(
        '-s', '--shots', help='instead of EXIF GPS positions, use camera positions from this ODM `shots.geojson` file',
        type=str
    )
    parser.add_argument(
        '-o', '--overwrite', help='overwrite camera position/orientation file if it exists', action='store_true'
    )
    return parser.parse_args()


def main():
    """
    Convert EXIF GPS position and (roll, pitch, yaw) angles to simple-ortho compatible camera position and orientation.
    """
    args = parse_arguments()

    # check & prepare arguments
    src_im_dir = Path(args.src_im_wildcard).parent
    src_im_wildcard = Path(args.src_im_wildcard).name
    if len([*src_im_dir.glob(src_im_wildcard)]) == 0:
        raise FileNotFoundError(f'Could not find any files matching {args.src_im_wildcard}')

    out_file_path = Path(args.out_file_path)
    if out_file_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f'Output file {out_file_path} exists, and won\'t be overwritten without the `overwrite` option.'
            )

    nav_crs = CRS.from_epsg(4326)
    world_crs = CRS.from_string(args.ortho_crs)
    # TODO: allow a UTM world_crs to be automatically determined
    # TODO: perhaps also, reproject camera positions to auto-determined UTM crs if the config.yaml crs is specified as
    #  EPSG:4326

    file_pos_dict = None
    if args.shots:
        # load positions the ODM shots.geojson file into a dict
        shots_file_path = Path(args.shots)
        if not shots_file_path.exists():
            raise FileNotFoundError(f'ODM shots file {shots_file_path} does not exist.')
        shots_dict = json.load(open(shots_file_path))
        file_pos_dict = {}
        for feature in shots_dict['features']:
            filename = feature['properties']['filename']
            pos = feature['properties']['translation']
            file_pos_dict[filename] = pos

    # write camera position & OPK values for each source image to output file
    with open(out_file_path, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        for filename in src_im_dir.glob(src_im_wildcard):
            exif = Exif(filename)
            if not exif.rpy:
                raise ValueError(f'File contains no (roll, pitch yaw) values: {filename.name}')
            if not exif.lla:
                raise ValueError(f'File contains no (latitude, longitude, altitude) values: {filename.name}')
            if file_pos_dict:
                if not str(filename.name) in file_pos_dict:
                    raise ValueError(f'ODM shots.geoson file has no entry for: {filename.name}')
                pos = file_pos_dict[str(filename.name)]
            else:
                # transform EXIF geo position to world coordinates
                pos = np.array(
                    transform(nav_crs, world_crs, [exif.lla[1]], [exif.lla[0]], [exif.lla[2]])
                ).squeeze()

            opk = rpy_to_opk(np.radians(exif.rpy), exif.lla, world_crs)
            writer.writerow([filename.stem, *pos, *np.degrees(opk)])
            print(exif, end='\n\n')


if __name__ == "__main__":
    main()
