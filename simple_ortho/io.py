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
import csv
import json
import yaml
import logging
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Callable
from enum import Enum
from csv import DictReader, Sniffer, Dialect

import numpy as np
import rasterio as rio
from rasterio.warp import transform
from rasterio.crs import CRS
import cv2

from simple_ortho.enums import CameraType
from simple_ortho.exif import Exif
from simple_ortho.utils import utm_crs_from_latlon

logger = logging.getLogger(__name__)


def read_yaml_int_param(filename: Path) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more cameras from a yaml file.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the yaml file to read.

    Returns
    -------
    dict
        A dictionary of camera id - camera interior parameters, key - value pairs.
    """
    with open(filename, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    def parse_cam_dict(cam_dict: Dict, cam_id: str) -> Dict:
        for req_key in ['type', 'focal_len', 'sensor_size']:
            if req_key not in cam_dict:
                raise ValueError(f'`{req_key}` is not defined for camera `{cam_id}` in {filename.name}.')

        cam_type = cam_dict.pop('type').lower()
        try:
            cam_dict['cam_type'] = CameraType(cam_type)
        except ValueError:
            raise ValueError(f'Unsupported camera type `{cam_type}` for camera `{cam_id}` in {filename.name}.')

        cam_dict.pop('name', None)
        cam_dict.pop('im_size', None)
        return cam_dict

    # flatten if in original simple-ortho format
    yaml_dict = yaml_dict.get('camera', yaml_dict)

    # convert to nested dict if in flat format
    first_value = next(iter(yaml_dict.values()))
    if not isinstance(first_value, dict):
        cam_id = yaml_dict.get('name', None) or 'default'
        yaml_dict = {cam_id: yaml_dict}

    # parse each set of camera parameters
    cams_dict = {}
    for cam_id, cam_dict in yaml_dict.items():
        cams_dict[cam_id] = parse_cam_dict(cam_dict, cam_id)
    return cams_dict


def read_json_int_param(filename: Path) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more cameras, from an ODM cameras.json or OpenSFM camera_models.json file.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the ODM / OpenSFM json file to read.

    Returns
    -------
    dict
        A dictionary of camera id - camera interior parameters, key - value pairs.
    """
    with open(filename, 'r') as f:
        json_dict = json.load(f)

    def parse_json_cam_dict(json_cam_dict: Dict, cam_id: str) -> Dict:
        """ Return a dict of interior camera parameters given an ODM / OpenSFM json dict for single camera. """
        cam_dict = {}
        for req_key in ['projection_type', 'width', 'height']:
            if req_key not in json_cam_dict:
                raise ValueError(f'`{req_key}` is not defined for camera `{cam_id}` in {filename.name}.')

        # read focal length(s) (json values are normalised by sensor width)
        if 'focal' in json_cam_dict:
            cam_dict['focal_len'] = json_cam_dict.pop('focal')
        elif 'focal_x' in json_cam_dict and 'focal_y' in json_cam_dict:
            cam_dict['focal_len'] = (json_cam_dict.pop('focal_x'), json_cam_dict.pop('focal_y'))
        else:
            raise ValueError(
                f'`focal`, or `focal_x` and `focal_y` are not defined for camera `{cam_id}` in {filename.name}.'
            )

        proj_type = json_cam_dict.pop('projection_type').lower()
        try:
            cam_dict['cam_type'] = CameraType.from_odm(proj_type)
        except ValueError:
            raise ValueError(f'Unsupported camera type `{proj_type}` for camera `{cam_id}` in {filename.name}.')

        image_size = (json_cam_dict.pop('width'), json_cam_dict.pop('height'))

        # TODO: normalised by width or max(width, height) ?
        # find a normalised sensor size in same units as focal_len, assuming square pixels (ODM / OpenSFM json files do
        # not define sensor size)
        cam_dict['sensor_size'] = (1, image_size[1] / image_size[0])

        # rename c_x->cx & c_y->cy
        for from_key, to_key in zip(['c_x', 'c_y'], ['cx', 'cy']):
            if from_key in json_cam_dict:
                cam_dict[to_key] = json_cam_dict.pop(from_key)

        # update param_dict with any remaining distortion coefficient parameters and return
        cam_dict.update(**json_cam_dict)
        return cam_dict

    cams_dict = {}
    for cam_id, json_cam_dict in json_dict.items():
        cams_dict[cam_id] = parse_json_cam_dict(json_cam_dict, cam_id)

    return cams_dict


def read_exif_int_param(exif: Exif) -> Dict[str, Dict]:
    """
    Read interior parameters for a camera from EXIF tags.

    Parameters
    ----------
    exif: simple_ortho.exif.Exif
        :class:`~simple_ortho.exif.Exif` instance to read.

    Returns
    -------
    dict
        A dictionary of a camera id - camera interior parameters, key - value pair.
    """
    cam_dict = dict(cam_type=CameraType.pinhole, name=exif.camera_name, im_size=exif.image_size)

    if exif.focal_len and exif.sensor_size:
        # prefer using focal length and sensor size directly
        cam_dict['focal_len'] = exif.focal_len
        cam_dict['sensor_size'] = exif.sensor_size
    elif exif.focal_len_35:
        logger.warning(f'Approximating the focal length for {exif.filename.name} from the 35mm equivalent.')
        if exif.sensor_size:
            # scale 35mm focal length to actual focal length in mm, assuming "35mm" = 36mm sensor width
            cam_dict['focal_len'] = exif.sensor_size[0] * exif.focal_len_35 / 36.
            cam_dict['sensor_size'] = exif.sensor_size
        else:
            # normalise 35mm focal length assuming "35mm" = 36 mm sensor width, and find a normalised sensor size
            # in same units, assuming square pixels
            cam_dict['focal_len'] = exif.focal_len_35 / 36.
            cam_dict['sensor_size'] = (1, exif.image_size[1] / exif.image_size[0])
    else:
        raise ValueError(
            f'EXIF tags in {exif.filename} should define a focal length and sensor size, or 35mm equivalent focal '
            f'length.'
        )

    cam_id = f'{exif.camera_name or "unknown"} pinhole {exif.focal_len:.4f}'
    return {cam_id: cam_dict}


def read_int_param(filename: Union[str, Path]) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more cameras, yaml file, ODM / OpenSFM json file, or JPEG / TIFF file with EXIF
    tags.

    Parameters
    ----------
    filename: str, Path
        Path of file to read.

    Returns
    -------
    dict
        A dictionary of camera id - camera interior parameters, key - value pairs.
    """
    filename = Path(filename)
    if filename.suffix.lower() in ['.yaml', '.yml']:
        return read_yaml_int_param(filename)
    elif filename.suffix.lower() == '.json':
        return read_json_int_param(filename)
    elif filename.suffix.lower() in ['.jpg', '.jpeg', '.tif', '.tiff']:
        return read_exif_int_param(Exif(filename))
    else:
        raise ValueError(f'Unrecognised file type: {filename.name}')


def aa_to_opk(angle_axis: Tuple[float]) -> Tuple[float, float, float]:
    """
    Convert ODM convention angle axis vector to PATB convention omega, phi, kappa angles.

    Parameters
    ----------
    angle_axis: tuple of float
        Angle axis vector in ODM convention (x->right, y->down, and z->forward looking through the camera at the scene).

    Returns
    -------
    tuple of float
        Omega, phi, kappa angles (radians) in PATB convention (x->right, y->up, and z->backward looking through the
        camera at the scene).
    """
    # convert ODM angle/axis to rotation matrix (see https://github.com/mapillary/OpenSfM/issues/121)
    R = cv2.Rodrigues(np.array(angle_axis))[0].T

    # ODM uses a camera co-ordinate system with x->right, y->down, and z->forward (looking through the camera at the
    # scene), while simple-ortho uses PATB, which is x->right, y->up, and z->backward.  Here we rotate from ODM
    # convention to PATB.
    R = np.dot(R, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

    # extract OPK from R (see https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf)
    omega = np.arctan2(-R[1, 2], R[2, 2])
    phi = np.arcsin(R[0, 2])
    kappa = np.arctan2(-R[0, 1], R[0, 0])
    return omega, phi, kappa


def rpy_to_opk(
    rpy: Tuple[float, float, float], lla: Tuple[float, float, float], crs: CRS, cbb: Union[None, List[List]] = None,
    lla_crs: CRS = CRS.from_epsg(4979),
) -> Tuple[float, float, float]:
    """
    Convert (roll, pitch, yaw) to (omega, phi, kappa) angles.

    (roll, pitch, yaw) are angles to rotate from body to navigation systems, where the body system is centered on and
    aligned with the gimbal/camera with (x->front, y->right, z->down).  The navigation system shares its center with
    the body system, but its xy-plane is perpendicular to the local plumbline (x->N, y->E, z->down).

    (omega, phi, kappa) are angles to rotate from world to camera coordinate systems. World coordinates are a
    projected system like UTM (origin fixed near earth surface, and usually some distance from camera), and camera
    coordinates are centered on and aligned with the camera (in PATB convention: x->right, y->up, z->backwards looking
    through the camera at the scene).

    Parameters
    ----------
    rpy: tuple of float
        (roll, pitch, yaw) camera angles in radians to rotate from body to navigation coordinate system.
    lla: tuple of float
        (latitude, longitude, altitude) navigation system coordinates of the body.
    crs: rasterio.crs.CRS
        World coordinate reference system as a rasterio CRS object (the same CRS used for the ortho image and camera
        positions).
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

    # To keep with the naming convention in the rest of simple-ortho, what the reference calls Object (E)
    # coordinates, I call world coordinates, and what the reference calls Image (B) coordinates, I have called camera
    # coordinates (to help differentiate from pixel coordinates).

    # TODO: incorporate vertical datum into lla_crs and test with destination crs with geoid vertical datum
    # TODO: can this be generalised to convert rotations between any CRS's?
    # TODO: call lla_crs->lla_crs and include in docstring, also world_crs -> ortho_crs ?
    # TODO: consider changing lla ordering to x,y,z everywhere
    lla = np.array(lla)
    # breakpoint()
    roll, pitch, yaw = rpy
    # lla_crs = CRS.from_epsg(4326)
    world_crs = CRS.from_string(crs) if isinstance(crs, str) else crs

    # find rotation matrix cnb, to rotate from body to navigation coordinates.
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    cnb = rz.dot(ry).dot(rx)

    # find rotation matrix cen, to rotate from navigation to world coordinates (world is called object (E) in the
    # reference)
    delta = 1e-7
    lla1 = lla + (delta, 0, 0)
    lla2 = lla - (delta, 0, 0)

    # p1 & p2 must be in the world/ortho CRS, not ECEF as might be understood from the reference
    p1 = np.array(transform(lla_crs, world_crs, [lla1[1]], [lla1[0]], [lla1[2]])).squeeze()
    p2 = np.array(transform(lla_crs, world_crs, [lla2[1]], [lla2[0]], [lla2[2]])).squeeze()

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


class ExtReader(object):
    def __init__(
        self, filename: Union[str, Path], crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = rio.CRS.from_epsg(4979)
    ):
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f'File not found: {filename}.')
        self._filename = filename
        self._crs = rio.CRS.from_string(crs) if isinstance(crs, str) else crs
        self._lla_crs = rio.CRS.from_string(lla_crs) if isinstance(lla_crs, str) else lla_crs

    def read(self) -> Dict[str, Dict]:
        return {}


class CsvFormat(Enum):
    xyz_opk = 1
    lla_opk = 2
    xyz_rpy = 3
    lla_rpy = 4

    @property
    def is_opk(self) -> bool:
        return self is CsvFormat.xyz_opk or self is CsvFormat.lla_opk

    @property
    def is_xyz(self) -> bool:
        return self is CsvFormat.xyz_opk or self is CsvFormat.xyz_rpy


class CsvExtReader(ExtReader):
    _type_schema = dict(
        filename=str, easting=float, northing=float, latitude=float, longitude=float, altitude=float, omega=float,
        phi=float, kappa=float, roll=float, pitch=float, yaw=float, camera=str,
    )
    _std_fieldnames = ['filename', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa']

    def __init__(
        self, filename: Union[str, Path], crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = rio.CRS.from_epsg(4979), fieldnames: List[str] = None,
        dialect: Dialect = None, radians: bool = False,
    ):
        # TODO: naming of crs: ortho_crs or xyz_crs or ena_crs or world_crs
        # TODO: make this class a context manager so file can be opened once (?)
        # TODO: we haven't thought of if/how to allow other coordinate conventions for opk / rpy
        # TODO: test vertical adjustment when crs and lla_crs have vertical datums
        # TODO: allow the CSV field naming spec to overlap with GDAL/QGIS spec for reading actual points
        # TODO: use super() and implement using a best practice abstract class pattern (?)

        filename = Path(filename)
        ExtReader.__init__(self, filename, crs, lla_crs=lla_crs)

        self._radians = radians
        self._fieldnames, self._dialect, self._has_header, self._format = self._parse(
            filename, fieldnames=fieldnames, dialect=dialect
        )

        if not self._crs:
            if self._format is CsvFormat.xyz_opk or self._format is CsvFormat.xyz_rpy:
                self._crs = self._read_prj_file_crs(filename)
            elif self._format is CsvFormat.lla_rpy:
                self._crs = self._find_lla_rpy_crs()
            else:   # lla_opk
                raise ValueError(f'A `crs` should be specified.')

    @property
    def crs(self) -> rio.CRS:
        return self._crs

    @staticmethod
    def _read_prj_file_crs(filename: Path) -> rio.CRS:
        prj_filename = filename.parent.joinpath(filename.stem, '.prj')
        # TODO: check exception messages make sense if used from CLI without --crs
        if not prj_filename.exists():
            raise ValueError(f'No {prj_filename.name} projection file found, a `crs` should be specified.')
        try:
            with open(prj_filename, newline='') as f:
                crs = rio.CRS.from_string(f.read())
        except Exception as ex:
            raise ValueError(f'Could not interpret CRS in {prj_filename.name}: str{ex}')
        return crs

    def _find_lla_rpy_crs(self) -> rio.CRS:
        latlons = []
        with open(self._filename, 'r', newline=None) as f:
            reader = DictReader(f, fieldnames=self._fieldnames, dialect=self._dialect)
            if self._has_header:
                _ = next(iter(reader))
            for row in reader:
                row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
                latlon = (row['latitude'], row['longitude'])
                latlons.append(latlon if self._radians else np.radians(latlon))

        mean_latlon = np.array(latlons).mean(axis=0)
        return utm_crs_from_latlon(*mean_latlon)

    @staticmethod
    def _parse_fieldnames(fieldnames: List[str], err_prefix: str = '`fieldnames`') -> CsvFormat:
        if 'filename' not in fieldnames:
            raise ValueError(f'{err_prefix} should include `filename`.')
        has_xyz = {'easting', 'northing', 'altitude'}.issubset(fieldnames)
        has_lla = {'latitude', 'longitude', 'altitude'}.issubset(fieldnames)
        has_opk = {'omega', 'phi', 'kappa'}.issubset(fieldnames)
        has_rpy = {'roll', 'pitch', 'yaw'}.issubset(fieldnames)
        if not (has_xyz or has_lla):
            raise ValueError(
                f'{err_prefix} should include `easting`, `northing` & `altitude`, or `latitude`, `longitude` & '
                f'`altitude`.'
            )
        if not (has_opk or has_rpy):
            raise ValueError(
                f'{err_prefix} should include `omega`, `phi` & `kappa`, or `roll`, `pitch` & `yaw`.'
            )
        # dictionary with key = (has_xyz, has_opk) and value = format
        format_dict = {
            (True, True): CsvFormat.xyz_opk,
            (True, False): CsvFormat.xyz_rpy,
            (False, True): CsvFormat.lla_opk,
            (False, False): CsvFormat.lla_rpy,
        }  # yapf: disable
        return format_dict[(has_xyz, has_opk)]

    @staticmethod
    def _parse(
        filename: Path, fieldnames: List[str] = None, dialect: Dialect = None
    ) -> Tuple[List[str], Dialect, bool, CsvFormat]:
        # logic as follows:
        # - if dialect is provided it is used as is and not checked against auto dialect
        # - if fieldnames provided, it is assumed there is no header, and they are used as is
        # - if fieldnames not provided and there is no header detected, it is assumed in std simple-ortho format

        def strip_lower_strlist(str_list: List[str]) -> List[str]:
            """ Strip, and lower the case, of a string list. """
            return [str_item.strip().lower() for str_item in str_list]

        # read a sample of the csv file (newline=None works around some delimiter detection problems with newline='')
        with open(filename, newline=None) as f:
            sample = f.read(10000)

        # auto-detect dialect if not provided
        sniffer = Sniffer()
        if not dialect:
            dialect = sniffer.sniff(sample, delimiters=',;: \t')
            dialect.skipinitialspace = True

        # find fieldnames and parse them to determine conversion method
        has_header = False
        if not fieldnames:
            has_header = sniffer.has_header(sample)
            if has_header:
                fieldnames = next(iter(csv.reader(sample.splitlines(), dialect=dialect)))
            else:
                fieldnames = CsvExtReader._std_fieldnames  # assume simple-ortho std format
            err_prefix = f'The {filename.name} fields '
        else:
            err_prefix = f'`fieldnames` '
        fieldnames = strip_lower_strlist(fieldnames)
        csv_fmt = CsvExtReader._parse_fieldnames(fieldnames, err_prefix=err_prefix)

        return fieldnames, dialect, has_header, csv_fmt

    def _convert(self, row: Dict[str, float], radians=False) -> Tuple[Tuple, Tuple]:
        xyz = lla = opk = rpy = None
        if self._format.is_xyz:
            xyz = (row['easting'], row['northing'], row['altitude'])
        else:
            lla = (row['latitude'], row['longitude'], row['altitude'])
            xyz = transform(self._lla_crs, self._crs, [lla[1]], [lla[0]], [lla[2]])
            xyz = tuple([coord[0] for coord in xyz])

        if self._format.is_opk:
            opk = (row['omega'], row['phi'], row['kappa'])
            opk = opk if radians else tuple(np.radians(opk))
        else:
            rpy = (row['roll'], row['pitch'], row['yaw'])
            rpy = rpy if radians else tuple(np.radians(rpy))
            if self._format.is_xyz:
                lla = transform(self._crs, self._lla_crs, [xyz[0]], [xyz[1]], [xyz[2]])
                lla = (lla[1][0], lla[0][0], lla[2][0])  # x, y order -> lat, lon order
            opk = rpy_to_opk(rpy, lla, self._crs, lla_crs=self._lla_crs)

        return xyz, opk

    def read(self) -> Dict[str, Dict]:
        ext_param_dict = {}
        with open(self._filename, 'r', newline=None) as f:
            reader = DictReader(f, fieldnames=self._fieldnames, dialect=self._dialect)
            if self._has_header:
                _ = next(iter(reader))
            for row in reader:
                row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
                xyz, opk = self._convert(row, radians=self._radians)
                ext_param_dict[row['filename']] = dict(xyz=xyz, opk=opk, int_id=row.get('camera', None))
        return ext_param_dict
