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
from tqdm.auto import tqdm

import numpy as np
import rasterio as rio
from rasterio.warp import transform
from rasterio.crs import CRS
from rasterio.errors import CRSError as RioCrsError
import cv2

from simple_ortho.enums import CameraType
from simple_ortho.exif import Exif
from simple_ortho.utils import utm_crs_from_latlon, validate_collection
from simple_ortho.errors import ParamFileError, CrsError, CrsMissingError

logger = logging.getLogger(__name__)


def read_yaml_int_param(filename: Union[str, Path]) -> Dict[str, Dict]:
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
    # TODO: allow partial spec that can be merged with other int params?
    filename = Path(filename)
    with open(filename, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    def parse_yaml_param(yaml_param: Dict, cam_id: str) -> Dict:
        # TODO: check all keys are valid for camera type
        int_param = {}
        for req_key in ['type', 'focal_len', 'sensor_size']:
            if req_key not in yaml_param:
                raise ParamFileError(f"'{req_key}' is missing for camera '{cam_id}'.")

        cam_type = yaml_param.pop('type').lower()
        try:
            int_param['cam_type'] = CameraType(cam_type)
        except ValueError:
            raise ParamFileError(f"Unsupported camera type '{cam_type}'.")

        yaml_param.pop('name', None)
        yaml_param.pop('im_size', None)

        int_param.update(**{k: tuple(v) if isinstance(v, list) else v for k, v in yaml_param.items()})
        return int_param

    # flatten if in original simple-ortho format
    yaml_dict = yaml_dict.get('camera', yaml_dict)

    # convert to nested dict if in flat format
    first_value = next(iter(yaml_dict.values()))
    if not isinstance(first_value, dict):
        cam_id = yaml_dict.get('name', None) or 'default'
        yaml_dict = {cam_id: yaml_dict}

    # parse each set of camera parameters
    int_param_dict = {}
    for cam_id, yaml_param in yaml_dict.items():
        int_param_dict[cam_id] = parse_yaml_param(yaml_param, cam_id)
    return int_param_dict


def _read_json_int_param(json_dict: Dict) -> Dict[str, Dict]:
    def parse_json_param(json_param: Dict, cam_id: str) -> Dict:
        """ Return a dict of interior camera parameters given an ODM / OpenSFM json dict for single camera. """
        # TODO: check all keys are valid for camera type
        int_param = {}
        for req_key in ['projection_type', 'width', 'height']:
            if req_key not in json_param:
                raise ParamFileError(f"'{req_key}' is missing for camera '{cam_id}'.")

        # set 'cam_type' from projection_type'
        proj_type = json_param.pop('projection_type').lower()
        try:
            int_param['cam_type'] = CameraType.from_odm(proj_type)
        except ValueError:
            raise ParamFileError(f"Unsupported camera type '{proj_type}'.")

        # read focal length(s) (json values are normalised by sensor width)
        if 'focal' in json_param:
            int_param['focal_len'] = json_param.pop('focal')
        elif 'focal_x' in json_param and 'focal_y' in json_param:
            int_param['focal_len'] = (json_param.pop('focal_x'), json_param.pop('focal_y'))
        else:
            raise ParamFileError(
                f"'focal', or 'focal_x' and 'focal_y' are missing for camera '{cam_id}'."
            )

        image_size = (json_param.pop('width'), json_param.pop('height'))

        # TODO: normalised by width or max(width, height) ?
        # find a normalised sensor size in same units as focal_len, assuming square pixels (ODM / OpenSFM json files do
        # not define sensor size)
        int_param['sensor_size'] = (1, image_size[1] / image_size[0])

        # rename c_x->cx & c_y->cy
        for from_key, to_key in zip(['c_x', 'c_y'], ['cx', 'cy']):
            if from_key in json_param:
                int_param[to_key] = json_param.pop(from_key)

        # update param_dict with any remaining distortion coefficient parameters and return
        int_param.update(**json_param)
        return int_param

    # extract cameras section if json_dict is from an OpenSFM reconstruction.json file
    if isinstance(json_dict, list) and len(json_dict) == 1 and 'cameras' in json_dict[0]:
        json_dict = json_dict[0]['cameras']

    int_param_dict = {}
    for cam_id, json_param in json_dict.items():
        cam_id = cam_id[3:] if cam_id.startswith('v2 ') else cam_id
        int_param_dict[cam_id] = parse_json_param(json_param, cam_id)

    return int_param_dict


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

    return _read_json_int_param(json_dict)


def _create_exif_cam_id(exif: Exif) -> str:
    focal_str = ''
    if exif.focal_len:
        focal_str = f'{exif.focal_len:.4f}'
    elif exif.focal_len_35:
        focal_str = f'{exif.focal_len_35:.4f} (35)'
    return f'{exif.make or "unknown"} {exif.model or "unknown"} pinhole {focal_str}'


def _read_exif_int_param(exif: Exif) -> Dict[str, Dict]:
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
    cam_dict = dict(cam_type=CameraType.pinhole)

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
        raise ParamFileError(
            f'No focal length & sensor size, or 35mm focal length tags in {exif.filename.name}.'
        )

    return {_create_exif_cam_id(exif): cam_dict}


def read_exif_int_param(filename: Union[str, Path]) -> Dict[str, Dict]:
    return _read_exif_int_param(Exif(filename))


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


class Reader(object):
    def __init__(
        self, crs: Union[str, rio.CRS] = None, lla_crs: Union[str, rio.CRS] = rio.CRS.from_epsg(4979)
    ):
        self._crs, self._lla_crs = self._parse_crss(crs, lla_crs)

    @staticmethod
    def _parse_crss(crs: Union[str, rio.CRS], lla_crs: Union[str, rio.CRS]) -> Tuple:
        if crs:
            try:
                crs = rio.CRS.from_string(crs) if isinstance(crs, str) else crs
            except RioCrsError as ex:
                raise CrsError(f"Could not interpret 'crs': {crs}. {str(ex)}")
            if crs.is_geographic:
                raise CrsError(f"'crs' should be a projected, not geographic system.")

        if lla_crs:
            try:
                lla_crs = rio.CRS.from_string(lla_crs) if isinstance(lla_crs, str) else lla_crs
            except RioCrsError as ex:
                raise CrsError(f"Could not interpret 'lla_crs': {lla_crs}. {str(ex)}")
            if not lla_crs.is_geographic:
                raise CrsError(f"'lla_crs' should be a geographic, not projected system.")
        return crs, lla_crs

    @property
    def crs(self) -> rio.CRS:
        return self._crs

    def read_int(self) -> Dict[str, Dict]:
        raise NotImplementedError()

    def read_ext(self) -> Dict[str, Dict]:
        raise NotImplementedError()


class CsvFormat(Enum):
    xyz_opk = 1
    """ Projected (easting, northing, altitude) position and (omega, phi, kappa) orientation. """
    lla_opk = 2
    """ Geographic (latitude, longitude, altitude) position and (omega, phi, kappa) orientation. """
    xyz_rpy = 3
    """ Projected (easting, northing, altitude) position and (roll, pitch, yaw) orientation. """
    lla_rpy = 4
    """ Geographic (latitude, longitude, altitude) position and (roll, pitch, yaw) orientation. """

    @property
    def is_opk(self) -> bool:
        return self is CsvFormat.xyz_opk or self is CsvFormat.lla_opk

    @property
    def is_xyz(self) -> bool:
        return self is CsvFormat.xyz_opk or self is CsvFormat.xyz_rpy


class CsvReader(Reader):
    _type_schema = dict(
        filename=str, easting=float, northing=float, latitude=float, longitude=float, altitude=float, omega=float,
        phi=float, kappa=float, roll=float, pitch=float, yaw=float, camera=lambda x: str(x) if x else x,
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
        # TODO: use super() and implement using a best practice abstract class pattern (?)
        # TODO: allow angles in CSV to have trailing r to specify radians (?)
        # TODO: consider moving CRS args to read, where they are actually used & possibly a separate method for
        #  reading a prj CRS
        Reader.__init__(self, crs, lla_crs=lla_crs)
        self._filename = Path(filename)
        if not self._filename.exists():
            raise FileNotFoundError(f'File not found: {self._filename}.')

        self._radians = radians
        self._fieldnames, self._dialect, self._has_header, self._format = self._parse(
            filename, fieldnames=fieldnames, dialect=dialect
        )
        self._crs = self._crs or self._get_crs()


    @staticmethod
    def _parse_fieldnames(fieldnames: List[str]) -> CsvFormat:
        if 'filename' not in fieldnames:
            raise ParamFileError(f'Fields should include `filename`.')
        has_xyz = {'easting', 'northing', 'altitude'}.issubset(fieldnames)
        has_lla = {'latitude', 'longitude', 'altitude'}.issubset(fieldnames)
        has_opk = {'omega', 'phi', 'kappa'}.issubset(fieldnames)
        has_rpy = {'roll', 'pitch', 'yaw'}.issubset(fieldnames)
        if not (has_xyz or has_lla):
            raise ParamFileError(
                f"Fields should include 'easting', 'northing' & 'altitude', or 'latitude', 'longitude' & "
                f"'altitude'."
            )
        if not (has_opk or has_rpy):
            raise ParamFileError(
                f"Fields should include 'omega', 'phi' & 'kappa', or 'roll', 'pitch' & 'yaw'."
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
                fieldnames = CsvReader._std_fieldnames  # assume simple-ortho std format
        fieldnames = strip_lower_strlist(fieldnames)
        csv_fmt = CsvReader._parse_fieldnames(fieldnames)

        return fieldnames, dialect, has_header, csv_fmt

    def _find_lla_rpy_crs(self) -> rio.CRS:
        latlons = []
        with open(self._filename, 'r', newline=None) as f:
            reader = DictReader(f, fieldnames=self._fieldnames, dialect=self._dialect)
            if self._has_header:
                _ = next(iter(reader))
            for row in reader:
                row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
                latlon = (row['latitude'], row['longitude'])
                latlons.append(latlon)

        mean_latlon = np.array(latlons).mean(axis=0)
        return utm_crs_from_latlon(*mean_latlon)

    def _get_crs(self) -> rio.CRS:
        crs = None
        if self._format is CsvFormat.xyz_opk or self._format is CsvFormat.xyz_rpy:
            # read CRS of projected xyz positions / opk orientations from .prj file, if it exists
            prj_filename = self._filename.with_suffix('.prj')
            # TODO: check exception messages make sense if used from CLI without --crs
            if prj_filename.exists():
                try:
                    crs = rio.CRS.from_string(prj_filename.read_text())
                except RioCrsError as ex:
                    raise CrsError(f'Could not interpret CRS in {prj_filename.name}: {str(ex)}')
                logger.debug(f"Using .prj file CRS: '{crs.to_proj4()}'")
            elif self._format is CsvFormat.xyz_rpy:
                raise CrsMissingError(f"'crs' should be specified for positions in {self._filename.name}.")

        elif self._format is CsvFormat.lla_rpy:
            # find a UTM CRS to transform the lla positions & rpy orientations into
            crs = self._find_lla_rpy_crs()
            logger.debug(f"Using auto UTM CRS: '{crs.to_proj4()}'")

        elif self._format is CsvFormat.lla_opk:
            # a user-supplied opk CRS is required to project lla into
            raise CrsMissingError(f"'crs' should be specified for orientations in {self._filename.name}.")

        return crs

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

    def read_ext(self) -> Dict[str, Dict]:
        ext_param_dict = {}
        with open(self._filename, 'r', newline=None) as f:
            reader = DictReader(f, fieldnames=self._fieldnames, dialect=self._dialect)
            if self._has_header:
                _ = next(iter(reader))
            for row in reader:
                row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
                xyz, opk = self._convert(row, radians=self._radians)
                ext_param_dict[row['filename']] = dict(xyz=xyz, opk=opk, camera=row.get('camera', None))
        return ext_param_dict


class OsfmReader(Reader):
    # TODO: allow to read interior params from camera section (like ExifReader?)
    # TODO: OSFM reconstruction is in a topocentric system AFAICT, so the direct transfer of 3D cartesian positions &
    #  rotations into a 2D+1D UTM CRS is an approximation, with similar issues to the RPY->OPK conversion.  Ideally the
    #  orthorectification should also happen in this topocentric system, with the DEM being transformed into it. Then
    #  the orthorectified image can be reprojected to UTM.
    # TODO: test the above theory be e.g. making crs=orthographic centered on reference_lla, then compare ortho ims.
    def __init__(
        self, filename: Union[str, Path], crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = rio.CRS.from_epsg(4979)
    ):
        Reader.__init__(self, crs=crs, lla_crs=lla_crs)
        self._filename = Path(filename)
        if not self._filename.exists():
            raise FileNotFoundError(f'File not found: {self._filename}.')

        self._json_dict = self._read_json_dict(Path(filename))
        if not self._crs:
            self._crs = self._find_utm_crs()
            logger.debug(f'Using auto UTM CRS: {self._crs.to_proj4()}')

    @staticmethod
    def _read_json_dict(filename: Path) -> Dict[str, Dict]:
        with open(filename, 'r') as f:
            json_data = json.load(f)

        template = [dict(cameras=dict, shots=dict, reference_lla=dict(latitude=float, longitude=float, altitude=float))]
        if not validate_collection(template, json_data):
            raise ParamFileError(f'{filename.name} is not a valid OpenSfM reconstruction file.')

        # keep root template keys and delete the rest
        json_dict = {k: json_data[0][k] for k in template[0].keys()}
        del json_data
        return json_dict

    def _find_utm_crs(self) -> rio.CRS:
        ref_lla = self._json_dict['reference_lla']
        ref_lla = (ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        return utm_crs_from_latlon(*ref_lla[:2])

    def read_int(self) -> Dict[str, Dict]:
        return _read_json_int_param(self._json_dict['cameras'])

    def read_ext(self) -> Dict[str, Dict]:
        ref_lla = self._json_dict['reference_lla']
        ref_lla = (ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        ref_xyz = transform(self._lla_crs, self._crs, [ref_lla[1]], [ref_lla[0]], [ref_lla[2]],)
        ref_xyz = tuple([coord[0] for coord in ref_xyz])

        ext_param_dict = {}
        for filename, shot_dict in self._json_dict['shots'].items():
            # adapted from https://github.com/OpenDroneMap/ODM/blob/master/opendm/shots.py
            rotation = cv2.Rodrigues(np.array(shot_dict['rotation']))[0]
            delta_xyz = -rotation.T.dot(shot_dict['translation'])
            xyz = tuple(ref_xyz + delta_xyz)
            opk = aa_to_opk(shot_dict['rotation'])
            cam_id = shot_dict['camera']
            cam_id = cam_id[3:] if cam_id.startswith('v2 ') else cam_id
            ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=cam_id)

        return ext_param_dict


def _read_exif_ext_param(
    exif: Exif, crs: Union[str, rio.CRS], lla_crs: Union[str, rio.CRS] = rio.CRS.from_epsg(4979)
) -> Dict:
    if not exif.lla:
        raise ParamFileError(
            f'No latitude, longitude & altitude tags in {exif.filename.name}.'
        )
    if not exif.rpy:
        raise ParamFileError(
            f'No camera / gimbal roll, pitch & yaw tags in {exif.filename.name}.'
        )
    rpy = tuple(np.radians(exif.rpy))
    opk = rpy_to_opk(rpy, exif.lla, crs, lla_crs=lla_crs)
    xyz = transform(lla_crs, crs, [exif.lla[1]], [exif.lla[0]], [exif.lla[2]])
    xyz = tuple([coord[0] for coord in xyz])
    return dict(xyz=xyz, opk=opk, camera=_create_exif_cam_id(exif))


class ExifReader(Reader):
    # TODO: test with empty filenames list
    # TODO: read crs from image file
    def __init__(
        self, filenames: Tuple[Union[str, Path], ...], crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = rio.CRS.from_epsg(4979)
    ):
        Reader.__init__(self, crs, lla_crs)
        filenames = [filenames] if isinstance(filenames, (str, Path)) else filenames
        self._exif_dict = self._read_exif(filenames)

        if not self._crs and len(self._exif_dict) > 0:
            self._crs = self._find_utm_crs()
            logger.debug(f'Using auto UTM CRS: {self._crs.to_proj4()}')

    @staticmethod
    def _read_exif(filenames: List[Union[str, Path]]) -> Dict[str, Exif]:
        exif_dict = {}
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} files [{elapsed}<{remaining}]'
        with rio.Env(GDAL_NUM_TRHREADS='ALL_CPUS'):
            for filename in tqdm(filenames, bar_format=bar_format, dynamic_ncols=True):
                filename = Path(filename)
                exif_dict[filename.name] = Exif(filename)
        return exif_dict

    def _find_utm_crs(self) -> Union[None, rio.CRS]:
        llas = []
        for e in self._exif_dict.values():
            if not e.lla:
                raise ParamFileError(f'No latitude, longitude & altitude tags in {e.filename.name}.')
            llas.append(e.lla)

        mean_latlon = np.array(llas)[:, :2].mean(axis=0)
        return utm_crs_from_latlon(*mean_latlon)

    def read_int(self) -> Dict[str, Dict]:
        int_param_dict = {}
        for filename, exif in self._exif_dict.items():
            int_param = _read_exif_int_param(exif)
            int_param_dict.update(**int_param)
        return int_param_dict

    def read_ext(self) -> Dict[str, Dict]:
        ext_param_dict = {}
        for filename, exif in self._exif_dict.items():
            ext_param_dict[filename] = _read_exif_ext_param(exif, crs=self._crs, lla_crs=self._lla_crs)
        return ext_param_dict


class OtyReader(Reader):
    def __init__(
        self, filename: Union[str, Path], crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = rio.CRS.from_epsg(4979)
    ):
        Reader.__init__(self, crs=crs, lla_crs=lla_crs)
        self._filename = Path(filename)
        if not self._filename.exists():
            raise FileNotFoundError(f'File not found: {self._filename}.')

        self._crs, self._json_dict = self._read_json_dict(Path(filename), self._crs)

    @staticmethod
    def _read_json_dict(filename: Path, crs: rio.CRS) -> Tuple[rio.CRS, Dict]:
        with open(filename, 'r') as f:
            json_dict = json.load(f)

        template = dict(
            type='FeatureCollection', xyz_opk_crs=str, features=[dict(
                type='Feature', properties=dict(filename=str, camera=str, xyz=list, opk=list),
                geometry=dict(type='Point', coordinates=list)
            )]
        )  # yapf: disable
        if not validate_collection(template, json_dict):
            raise ParamFileError(f'{filename.name} is not a valid GeoJSON exterior parameter file.')

        if not crs:
            try:
                crs = rio.CRS.from_string(json_dict['xyz_opk_crs'])
            except RioCrsError as ex:
                raise ParamFileError(f'Could not interpret CRS in {filename.name}: {str(ex)}')

        return crs, json_dict

    def read_ext(self) -> Dict[str, Dict]:
        ext_param_dict = {}
        for feat_dict in self._json_dict['features']:
            prop_dict = feat_dict['properties']
            filename = prop_dict['filename']
            ext_parm = dict(xyz=tuple(prop_dict['xyz']), opk=tuple(prop_dict['opk']), camera=prop_dict['camera'])
            ext_param_dict[filename] = ext_parm
        return ext_param_dict


def write_int_param(
    filename: Union[str, Path], int_param_dict: Dict[str, Dict], overwrite: bool = False
):
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            raise FileExistsError(f'Interior parameter file exists: {filename}.')
        filename.unlink()

    # convert 'cam_type' to 'type', with 'type' being the first item
    yaml_dict = {}
    for cam_id, int_param in int_param_dict.items():
        yaml_param = {}
        if 'cam_type' in int_param:
            yaml_param['type'] = int_param['cam_type'].value
        yaml_param.update(**{k: v for k, v in int_param.items() if k != 'cam_type'})
        yaml_dict[cam_id] = yaml_param

    with open(filename, 'w') as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, indent=4, default_flow_style=None)


def write_ext_param(
    filename: Union[str, Path], ext_param_dict: Dict[str, Dict], crs: Union[str, rio.CRS], overwrite: bool = False
):
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            raise FileExistsError(f'Exterior parameter file exists: {filename}.')
        filename.unlink()

    feat_list = []
    lla_crs = rio.CRS.from_epsg(4979)
    for src_file, ext_param in ext_param_dict.items():
        xyz = ext_param['xyz']
        lla = transform(crs, lla_crs, [xyz[0]], [xyz[1]], [xyz[2]])
        lla = [lla[0][0], lla[1][0], lla[2][0]]  # (lon, lat) order for geojson point
        props_dict = dict(filename=src_file, camera=ext_param['camera'], xyz=xyz, opk=ext_param['opk'])
        geom_dict = dict(type='Point', coordinates=list(lla))
        feat_dict = dict(type='Feature', properties=props_dict, geometry=geom_dict)
        feat_list.append(feat_dict)

    json_dict = dict(type='FeatureCollection', xyz_opk_crs=crs.to_string(), features=feat_list)
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=4)

