# Copyright The Orthority Contributors.
#
# This file is part of Orthority.
#
# Orthority is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Orthority is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Orthority.  If not, see <https://www.gnu.org/licenses/>.

import csv
import json
import logging
from csv import Dialect, DictReader, Sniffer
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio as rio
import yaml
from rasterio.crs import CRS
from rasterio.errors import CRSError as RioCrsError
from rasterio.warp import transform
from tqdm.auto import tqdm

from orthority.enums import CameraType, CsvFormat
from orthority.errors import CrsError, CrsMissingError, ParamFileError
from orthority.exif import Exif
from orthority.utils import utm_crs_from_latlon, validate_collection

logger = logging.getLogger(__name__)

_optional_schema = {
    CameraType.pinhole: ['cx', 'cy'],
    # fmt: off
    CameraType.opencv: [
        'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4', 't1', 't2', 'cx',
        'cy',
    ],
    # fmt: on
    CameraType.brown: ['k1', 'k2', 'p1', 'p2', 'k3', 'cx', 'cy'],
    CameraType.fisheye: ['k1', 'k2', 'k3', 'k4', 'cx', 'cy'],
}
"""Schema of valid optional parameters for each camera type."""

_default_lla_crs = CRS.from_epsg(4979)
"""Default CRS for geographic camera coordinates."""


def _read_osfm_int_param(json_dict: Dict) -> Dict[str, Dict]:
    """Read camera interior parameters from an ODM / OpenSfM json dictionary."""

    def parse_json_param(json_param: Dict, cam_id: str) -> Dict:
        """Validate & convert the given json dictionary for a single camera."""
        int_param = {}
        for req_key in ['projection_type', 'width', 'height']:
            if req_key not in json_param:
                raise ParamFileError(f"'{req_key}' is missing for camera '{cam_id}'.")

        # set 'cam_type' from 'projection_type'
        proj_type = json_param.pop('projection_type').lower()
        try:
            int_param['cam_type'] = CameraType.from_odm(proj_type)
        except ValueError:
            raise ParamFileError(f"Unsupported projection type '{proj_type}'.")

        im_size = (json_param.pop('width'), json_param.pop('height'))
        int_param['im_size'] = im_size

        # read focal length(s) (json values are normalised by max of sensor width & height)
        if 'focal' in json_param:
            int_param['focal_len'] = json_param.pop('focal')
        elif 'focal_x' in json_param and 'focal_y' in json_param:
            focal_x, focal_y = json_param.pop('focal_x'), json_param.pop('focal_y')
            int_param['focal_len'] = focal_x if focal_x == focal_y else (focal_x, focal_y)
        else:
            raise ParamFileError(
                f"'focal', or 'focal_x' and 'focal_y' are missing for camera '{cam_id}'."
            )

        # find a normalised sensor size in same units as focal_len, assuming square pixels (ODM /
        # OpenSFM json files do not define sensor size)
        int_param['sensor_size'] = tuple((np.array(im_size) / max(im_size)).tolist())

        # rename c_x->cx & c_y->cy
        for from_key, to_key in zip(['c_x', 'c_y'], ['cx', 'cy']):
            if from_key in json_param:
                int_param[to_key] = json_param.pop(from_key)

        # validate any remaining optional params, update param_dict & return
        err_keys = set(json_param.keys()).difference(_optional_schema[int_param['cam_type']])
        if len(err_keys) > 0:
            raise ParamFileError(f"Unsupported parameter(s) {err_keys} for camera '{cam_id}'")
        int_param.update(**json_param)
        return int_param

    # extract cameras section if json_dict is from an OpenSfM reconstruction.json file
    if isinstance(json_dict, list) and len(json_dict) == 1 and 'cameras' in json_dict[0]:
        json_dict = json_dict[0]['cameras']

    # parse each set of interior parameters
    int_param_dict = {}
    for cam_id, json_param in json_dict.items():
        cam_id = cam_id[3:] if cam_id.startswith('v2 ') else cam_id
        int_param_dict[cam_id] = parse_json_param(json_param, cam_id)

    return int_param_dict


def _create_exif_cam_id(exif: Exif) -> str:
    """Return a camera ID string for the given Exif object."""
    cam_id = ' '.join([exif.make, exif.model, exif.serial])
    if len(cam_id) == 0:
        cam_id = 'unknown'
    return cam_id


def _read_exif_int_param(exif: Exif) -> Dict[str, Dict]:
    """Read camera interior parameters from an Exif object."""
    if exif.dewarp:
        if len(exif.dewarp) != 9 or not any(exif.dewarp) or not exif.tag_im_size:
            logger.warning(f"Cannot interpret dewarp data for '{exif.filename.name}'.")
        else:
            # construct brown camera parameters from dewarp data and return
            cam_dict = dict(
                cam_type=CameraType.brown,
                im_size=exif.im_size,
                focal_len=tuple(exif.dewarp[:2]),
                sensor_size=(float(exif.tag_im_size[0]), float(exif.tag_im_size[1])),
            )
            cam_dict['cx'], cam_dict['cy'] = tuple(
                (np.array(exif.dewarp[2:4]) / max(exif.tag_im_size)).tolist()
            )
            dist_params = dict(zip(['k1', 'k2', 'p1', 'p2', 'k3'], exif.dewarp[-5:]))
            cam_dict.update(**dist_params)
            return {_create_exif_cam_id(exif): cam_dict}

    # construct pinhole camera parameters
    cam_dict = dict(cam_type=CameraType.pinhole, im_size=exif.im_size)
    if exif.focal_len and exif.sensor_size:
        # prefer using focal length and sensor size directly
        cam_dict['focal_len'] = exif.focal_len
        cam_dict['sensor_size'] = exif.sensor_size
    elif exif.focal_len_35:
        logger.warning(
            f"Approximating the focal length for '{exif.filename.name}' from the 35mm equivalent."
        )
        if exif.sensor_size:
            # scale 35mm focal length to actual focal length in mm, assuming "35mm" = 36mm max
            # sensor dimension
            cam_dict['focal_len'] = max(exif.sensor_size) * exif.focal_len_35 / 36.0
            cam_dict['sensor_size'] = exif.sensor_size
        else:
            # normalise 35mm focal length assuming "35mm" = 36 mm max sensor dimension, and find
            # a normalised sensor size in same units, assuming square pixels
            cam_dict['focal_len'] = exif.focal_len_35 / 36.0
            cam_dict['sensor_size'] = tuple((np.array(exif.im_size) / max(exif.im_size)).tolist())
    else:
        raise ParamFileError(
            f"No focal length & sensor size, or 35mm focal length tags in '{exif.filename.name}'."
        )

    return {_create_exif_cam_id(exif): cam_dict}


def _read_exif_ext_param(
    exif: Exif, crs: Union[str, rio.CRS], lla_crs: Union[str, rio.CRS]
) -> Dict:
    """Read camera exterior parameters from an Exif object."""
    if not exif.lla:
        raise ParamFileError(f"No latitude, longitude & altitude tags in '{exif.filename.name}'.")
    if not exif.rpy:
        raise ParamFileError(
            f"No camera / gimbal roll, pitch & yaw tags in '{exif.filename.name}'."
        )
    rpy = tuple(np.radians(exif.rpy).tolist())
    opk = _rpy_to_opk(rpy, exif.lla, crs, lla_crs=lla_crs)
    xyz = transform(lla_crs, crs, [exif.lla[1]], [exif.lla[0]], [exif.lla[2]])
    xyz = tuple([coord[0] for coord in xyz])
    return dict(xyz=xyz, opk=opk, camera=_create_exif_cam_id(exif))


def read_oty_int_param(filename: Union[str, Path]) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more cameras from an orthority format yaml file.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the yaml file to read.

    Returns
    -------
    dict
        A dictionary of camera interior parameters.
    """
    filename = Path(filename)
    with open(filename, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    def parse_yaml_param(yaml_param: Dict, cam_id: str = None) -> Dict:
        """Validate & convert the given yaml dictionary for a single camera."""
        # test required keys for all cameras
        for req_key in ['type', 'im_size', 'focal_len', 'sensor_size']:
            if req_key not in yaml_param:
                raise ParamFileError(f"'{req_key}' is missing for camera '{cam_id}'.")

        # convert type -> cam_type
        cam_type = yaml_param.pop('type').lower()
        try:
            int_param = dict(cam_type=CameraType(cam_type))
        except ValueError:
            raise ParamFileError(f"Unsupported camera type '{cam_type}'.")

        int_param['im_size'] = tuple(yaml_param.pop('im_size'))

        # pop known legacy keys not supported by Camera.__init__
        yaml_param.pop('name', None)

        # set focal_len & sensor_size
        focal_len = yaml_param.pop('focal_len')
        int_param['focal_len'] = tuple(focal_len) if isinstance(focal_len, list) else focal_len
        int_param['sensor_size'] = tuple(yaml_param.pop('sensor_size'))

        # validate any remaining distortion params, update param_dict & return
        err_keys = set(yaml_param.keys()).difference(_optional_schema[int_param['cam_type']])
        if len(err_keys) > 0:
            raise ParamFileError(f"Unsupported parameter(s) {err_keys} for camera '{cam_id}'")
        int_param.update(**yaml_param)
        return int_param

    # flatten if in original simple-ortho format
    yaml_dict = yaml_dict.get('camera', yaml_dict)

    # convert to nested dict if in flat format
    first_value = next(iter(yaml_dict.values()))
    if not isinstance(first_value, dict):
        cam_id = yaml_dict['name'] if 'name' in yaml_dict else 'unknown'
        yaml_dict = {cam_id: yaml_dict}

    # parse each set of interior parameters
    int_param_dict = {}
    for cam_id, yaml_param in yaml_dict.items():
        int_param_dict[cam_id] = parse_yaml_param(yaml_param, cam_id)
    return int_param_dict


def read_osfm_int_param(filename: Path) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more camera, from an ODM cameras.json or OpenSfM
    reconstruction.json file.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the ODM / OpenSFM json file to read.

    Returns
    -------
    dict
        A dictionary of camera interior parameters.
    """
    with open(filename, 'r') as f:
        json_dict = json.load(f)

    return _read_osfm_int_param(json_dict)


def read_exif_int_param(filename: Union[str, Path]) -> Dict[str, Dict]:
    """
    Read interior parameters for a camera from an image file with EXIF / XMP tags.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the image file to read.

    Returns
    -------
    dict
        A dictionary of camera interior parameters.
    """
    return _read_exif_int_param(Exif(filename))


def _rpy_to_rotation(rpy: Tuple[float, float, float]) -> np.ndarray:
    """Return the rotation matrix for the given (roll, pitch, yaw) orientations in radians."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    roll, pitch, yaw = rpy
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R_z.dot(R_y).dot(R_x)


def _opk_to_rotation(opk: Tuple[float, float, float]) -> np.ndarray:
    """Return the rotation matrix for the given (omega, phi, kappa) orientations in radians."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    omega, phi, kappa = opk
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
    )
    R_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    R_z = np.array(
        [[np.cos(kappa), -np.sin(kappa), 0], [np.sin(kappa), np.cos(kappa), 0], [0, 0, 1]]
    )
    return R_x.dot(R_y).dot(R_z)


def _rotation_to_opk(R: np.ndarray) -> Tuple[float, float, float]:
    """Return the (omega, phi, kappa) orientations in radians for the given rotation matrix."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    omega = np.arctan2(-R[1, 2], R[2, 2])
    phi = np.arcsin(R[0, 2])
    kappa = np.arctan2(-R[0, 1], R[0, 0])
    return omega, phi, kappa


def _aa_to_opk(aa: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert angle axis vector to (omega, phi, kappa) angles.

    Parameters
    ----------
    aa: tuple of float
        Angle axis vector to rotate from camera (OpenSfM / OpenCV convention) to world coordinates.

    Returns
    -------
    tuple of float
        (omega, phi, kappa) angles in radians to rotate from camera (Pix4D / PATB convention) to
        world coordinates.
    """
    # convert ODM angle/axis to rotation matrix (see
    # https://github.com/mapillary/OpenSfM/issues/121)
    R = cv2.Rodrigues(np.array(aa))[0].T

    # rotate from ODM to PATB convention
    R = np.dot(R, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

    # extract OPK from R
    omega, phi, kappa = _rotation_to_opk(R)
    return omega, phi, kappa


def _rpy_to_opk(
    rpy: Tuple[float, float, float],
    lla: Tuple[float, float, float],
    crs: CRS,
    lla_crs: CRS,
    C_bB: Union[Optional[List[List]], np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Convert (roll, pitch, yaw) to (omega, phi, kappa) angles for a given CRS.

    Coordinate conventions are as defined in: https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.

    Note this conversion does not correct for earth curvature & assumes ``crs`` is orthogonal and
    uniform scale (cartesian) in the vicinity of ``lla``.  ``crs`` should be conformal, and have
    its central meridian(s) close to ``lla`` for this assumption to be accurate.

    Parameters
    ----------
    rpy: tuple of float
        (roll, pitch, yaw) orientation to rotate from body to navigation coordinate system
        (radians).
    lla: tuple of float
        (latitude, longitude, altitude) geographic coordinates of the camera (degrees).
    crs: rasterio.crs.CRS
        CRS of the world / ortho coordinate system.
    lla_crs: rasterio.crs.CRS
        CRS of the ``lla`` geographic coordinates.
    C_bB: list of list of float, ndarray, optional
        Optional camera to body rotation matrix.  Defaults to :
        ``[[0, 1, 0], [1, 0, 0], [0, 0, -1]]``, which describes typical drone geometry.

    Returns
    -------
    tuple of float
        omega, phi, kappa angles in radians, to rotate from Pix4D / PATB convention camera to world
        coordinate systems.
    """
    # Adapted from the OpenSfM exif module
    # https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py and Pix4D doc
    # https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    # /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.

    # Note that what the Pix4D reference calls Object (E), and Image (B) coordinates, orthority
    # calls world and camera coordinates respectively.

    # RPY rotates from body (b) to navigation (n) coordinates, and OPK from camera ('Image' B) to
    # world ('Object' E) coordinates.  Body (b) coordinates are defined as x->front, y->right,
    # z->down, and navigation (n) coordinates as x->N, y->E, z->down.  Navigation is a tangent
    # plane type system and shares its origin with body (b) at ``lla``. Camera (B) coordinates
    # are in the Pix4D / PATB convention i.e. x->right, y->top, z->back (looking through the
    # camera at the scene), and are aligned with & centered on the body (b) system.  World (E)
    # coordinates are ENU i.e. x->E, y->N, z->up.
    lla = np.array(lla)
    crs = CRS.from_string(crs) if isinstance(crs, str) else crs

    # find rotation matrix C_nb, to rotate from body (b) to navigation (n) coordinates.
    C_nb = _rpy_to_rotation(rpy)

    # find rotation matrix C_En, to rotate from navigation (n) to world (object E) coordinates
    delta = 1e-7
    lla1 = lla + (delta, 0, 0)
    lla2 = lla - (delta, 0, 0)

    # p1 & p2 must be in the world CRS, not ECEF as might be understood from the references
    p1 = np.array(transform(lla_crs, crs, [lla1[1]], [lla1[0]], [lla1[2]])).squeeze()
    p2 = np.array(transform(lla_crs, crs, [lla2[1]], [lla2[0]], [lla2[2]])).squeeze()

    # approximate the relative alignment of world and navigation systems in the vicinity of lla
    x_np = p1 - p2
    m = np.linalg.norm(x_np)
    x_np /= m  # unit vector in navigation system N direction
    z_np = np.array([0, 0, -1]).T
    y_np = np.cross(z_np, x_np)
    C_En = np.array([x_np, y_np, z_np]).T

    # C_bB is the rotation from camera (image B) to body (b) coordinates
    C_bB = np.array(C_bB) if C_bB is not None else np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    # combine C_En, C_nb, C_bB to find rotation from camera (B) to world (E) coordinates.
    C_EB = C_En.dot(C_nb).dot(C_bB)

    # extract OPK angles from C_EB and return
    omega, phi, kappa = _rotation_to_opk(C_EB)
    return omega, phi, kappa


class Reader:
    def __init__(
        self, crs: Union[str, rio.CRS] = None, lla_crs: Union[str, rio.CRS] = _default_lla_crs
    ):
        """
        Abstract class for reading interior and exterior camera parameters.

        Parameters
        ----------
        crs: rio.crs.CRS
            CRS of the world coordinate system.
        lla_crs: rio.crs.CRS
            CRS of input geographic camera coordinates (if any).
        """
        self._crs, self._lla_crs = self._parse_crss(crs, lla_crs)

    @staticmethod
    def _parse_crss(crs: Union[str, rio.CRS], lla_crs: Union[str, rio.CRS]) -> Tuple:
        """Validate and convert CRSs."""
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
                raise CrsError(f"Could not interpret 'lla_crs': '{lla_crs}'. {str(ex)}")
            if not lla_crs.is_geographic:
                raise CrsError(f"'lla_crs' should be a geographic, not projected system.")
        return crs, lla_crs

    @property
    def crs(self) -> rio.CRS:
        """CRS of the world coordinate system."""
        return self._crs

    def read_int_param(self) -> Dict[str, Dict]:
        """
        Read interior camera parameters.

        Returns
        -------
        dict
            A dictionary of camera interior parameters.
        """
        raise NotImplementedError()

    def read_ext_param(self) -> Dict[str, Dict]:
        """
        Read exterior camera parameters.

        Returns
        -------
        dict
            A dictionary of camera exterior parameters.
        """
        raise NotImplementedError()


class CsvReader(Reader):
    _type_schema = dict(
        filename=str,
        easting=float,
        northing=float,
        latitude=float,
        longitude=float,
        altitude=float,
        omega=float,
        phi=float,
        kappa=float,
        roll=float,
        pitch=float,
        yaw=float,
        camera=lambda x: str(x) if x else x,
    )
    _legacy_fieldnames = ['filename', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa']

    def __init__(
        self,
        filename: Union[str, Path],
        crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = _default_lla_crs,
        fieldnames: List[str] = None,
        dialect: Dialect = None,
        radians: bool = False,
    ):
        """
        Class for reading camera exterior parameters from a CSV file.

        Reads tabular data from a text CSV file with a row per source image and column fields for
        the image file name, camera position coordinates and orientation angles.  Positions and
        orientations are converted into :class:`~orthority.camera.Camera` compatible values.

        Fields can be identified with a first line file header or by passing the ``fieldnames``
        argument. Recognised names and corresponding descriptions for the header and
        ``fieldnames`` fields are::

            =================================== ====================================================
            Field name(s)                       Description
            =================================== ====================================================
            'filename'                          Image file name, excluding parent path, with or
                                                without an extension.
            'easting', 'northing', 'altitude'   Camera position in ``crs`` coordinates and units.
            'latitude', 'longitude', 'altitude' Camera position in ``lla_crs`` coordinates and
                                                units.
            'omega', 'phi', 'kappa'             Camera orientation with units specified by
                                                ``radians``.
            'roll', 'pitch', 'yaw'              Camera orientation with units specified by
                                                ``radians``.
            'camera'                            ID of camera interior parameters (optional).
            =================================== ====================================================

        'filename', one of ('easting', 'northing', 'altitude') or ('latitude', 'longitude',
        'altitude'), and one of ('omega', 'phi', 'kappa') or ('roll', 'pitch', 'yaw') are
        required.  'camera' is required for multi-camera set ups to reference corresponding
        interior parameters, but is otherwise optional.  Other fields included in the file that
        are not in the recognised list are ignored.

        If there is no file header and ``fieldnames`` is not provided, the file is assumed to be
        in the legacy ``simple-ortho`` format::

            <filename> <easting> <northing> <altitude> <omega> <phi> <kappa>

        Comma, space, semicolon, colon and tab delimiters are supported, as are windows (\r\n)
        and unix (\n) line terminators.  Values can optionally be enclosed in single or double
        quotes.  This may be required with e.g. the 'camera' field if ID values contain the CSV
        delimiter.

        Examples
        --------

        A file specifying (easting, northing, altitude) positions, and (omega, phi,
        kappa) orientations with a comma delimiter.  The 'other' field is ignored.::

            filename,easting,northing,altitude,omega,phi,kappa,other
            image1.jpg,-523.615,-372.785,525.161,0.019,-0.080,-179.006,ignored
            image2.jpg,-537.559,-372.772,525.130,-0.080,-0.100,-179.173,ignored
            ...
            image100.jpg,-523.869,-373.883,525.553,-0.076,0.127,0.720,ignored

        A file specifying (latitude, longitude, altitude) positions, and (roll, pitch,
        yaw) orientations with a space delimiter.  A 'camera' field is included, with values
        enclosed in quotes as they contain the space delimiter.::

            filename latitude longitude altitude roll pitch yaw camera
            100_0005_0001.JPG 24.680427 120.948453 186.54 0.0 30.0 94.5 "dji fc6310r brown 0.6666"
            100_0005_0002.JPG 24.680420 120.948593 186.5 0.0 30.0 93.7 "dji fc6310r brown 0.6666"
            ...
            100_0005_0271.JPG 24.679835 120.949276 186.62 0.0 30.0 -176.8 "dji fc6310r brown 0.6666"


        Parameters
        ----------
        filename: pathlib.Path
            Path of the CSV file.
        crs: rio.crs.CRS, optional
            CRS of the world coordinate system.  If not specified and the CSV file contains (
            easting, northing, altitude) positions, ``crs`` will be read from a .prj file if it
            exists. (A prj file is a text file defining the CRS with a WKT, proj4 or EPSG string.
            It should and have the same path & stem as ``filename``, but a .prj extension).  If
            ``crs`` is not specified, and the file is in :attr:`~CsvFormat.lla_rpy` format,
            a UTM CRS will be auto-determined.  Otherwise, if the file is in
            :attr:`~CsvFormat.lla_opk` format, ``crs`` should be provided.
        lla_crs: rio.crs.CRS, optional
            CRS of (latitude, longitude, altitude) geographic camera coordinates (if any).
        fieldnames: list of str, optional
            List of recognised, and possibly other strings specifying the CSV fields.  The
            default is to determine the fields from a file header when it exists.  If
            ``fieldnames`` is passed, any existing file header is ignored.
        dialect: csv.Dialect, optional
            A :class:`csv.Dialect` object specifying CSV delimiter etc. configuration.  By default,
            this is auto-determined.
        radians: bool, optional
            Orientation angles are in radians (True), or degrees (False).
        """
        # TODO: allow other coordinate conventions for opk / rpy (bluh, odm, patb)
        Reader.__init__(self, crs, lla_crs=lla_crs)
        self._filename = Path(filename)
        if not self._filename.exists():
            raise FileNotFoundError(f"File not found: '{self._filename}'.")

        self._radians = radians
        self._fieldnames, self._dialect, self._has_header, self._format = self._parse_file(
            filename, fieldnames=fieldnames, dialect=dialect
        )
        self._crs = self._crs or self._get_crs()

    @staticmethod
    def _parse_fieldnames(fieldnames: List[str]) -> CsvFormat:
        """Validate a list of header or user field names, and return the corresponding
        :class:`CsvFormat`.
        """
        if 'filename' not in fieldnames:
            raise ParamFileError(f"Fields should include 'filename'.")

        has_xyz = {'easting', 'northing', 'altitude'}.issubset(fieldnames)
        has_lla = {'latitude', 'longitude', 'altitude'}.issubset(fieldnames)
        has_opk = {'omega', 'phi', 'kappa'}.issubset(fieldnames)
        has_rpy = {'roll', 'pitch', 'yaw'}.issubset(fieldnames)

        if not (has_xyz or has_lla):
            raise ParamFileError(
                f"Fields should include 'easting', 'northing' & 'altitude', or 'latitude', "
                f"'longitude' & 'altitude'."
            )
        if not (has_opk or has_rpy):
            raise ParamFileError(
                f"Fields should include 'omega', 'phi' & 'kappa', or 'roll', 'pitch' & 'yaw'."
            )

        # dictionary with key = (has_xyz, has_opk) and value = CsvFormat
        format_dict = {
            (True, True): CsvFormat.xyz_opk,
            (True, False): CsvFormat.xyz_rpy,
            (False, True): CsvFormat.lla_opk,
            (False, False): CsvFormat.lla_rpy,
        }
        return format_dict[(has_xyz, has_opk)]

    @staticmethod
    def _parse_file(
        filename: Path, fieldnames: List[str] = None, dialect: Dialect = None
    ) -> Tuple[List[str], Dialect, bool, CsvFormat]:
        """Determine and validate the CSV file format."""

        def strip_lower_strlist(str_list: List[str]) -> List[str]:
            """Strip and lower the case of a string list."""
            return [str_item.strip().lower() for str_item in str_list]

        # read a sample of the csv file (newline=None works around some delimiter detection
        # problems with newline='')
        with open(filename, newline=None) as f:
            sample = f.read(10000)

        # auto-detect dialect if not provided
        sniffer = Sniffer()
        if not dialect:
            dialect = sniffer.sniff(sample, delimiters=',;: \t')
            dialect.skipinitialspace = True

        # find field names and parse to determine format
        has_header = sniffer.has_header(sample)
        if not fieldnames:
            if has_header:  # read field names from header
                fieldnames = next(iter(csv.reader(sample.splitlines(), dialect=dialect)))
            else:  # assume fields in legacy format
                fieldnames = CsvReader._legacy_fieldnames
        fieldnames = strip_lower_strlist(fieldnames)
        csv_fmt = CsvReader._parse_fieldnames(fieldnames)

        return fieldnames, dialect, has_header, csv_fmt

    def _find_lla_rpy_crs(self) -> rio.CRS:
        """Return a UTM CRS covering the mean of the camera positions in a
        :attr:`~CsvFormat.lla_rpy` format file.
        """
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
        """Read / auto-determine and validate a CRS when no user CRS was supplied."""
        crs = None
        if self._format is CsvFormat.xyz_opk or self._format is CsvFormat.xyz_rpy:
            # read CRS of xyz positions / opk orientations from .prj file, if it exists
            prj_filename = self._filename.with_suffix('.prj')
            if prj_filename.exists():
                try:
                    crs = rio.CRS.from_string(prj_filename.read_text())
                except RioCrsError as ex:
                    raise CrsError(f"Could not interpret CRS in '{prj_filename.name}': '{str(ex)}'")
                if crs.is_geographic:
                    raise CrsError(
                        f"CRS in '{prj_filename.name}' should be a projected, not geographic system"
                    )
                logger.debug(f"Using '{prj_filename.name}' CRS: '{crs.to_proj4()}'")
            elif self._format is CsvFormat.xyz_rpy:
                raise CrsMissingError(
                    f"'crs' should be specified for positions in '{self._filename.name}'."
                )

        elif self._format is CsvFormat.lla_rpy:
            # find a UTM CRS to transform the lla positions & rpy orientations into
            crs = self._find_lla_rpy_crs()
            logger.debug(f"Using auto UTM CRS: '{crs.to_proj4()}'")

        elif self._format is CsvFormat.lla_opk:
            # a user-supplied opk CRS is required to project lla into
            raise CrsMissingError(
                f"'crs' should be specified for orientations in '{self._filename.name}'."
            )

        return crs

    def _convert(self, row: Dict[str, float], radians=False) -> Tuple[Tuple, Tuple]:
        """Convert a CSV row dictionary to (easting, northing, altitude) position, and (omega, phi,
        kappa) orientation.
        """
        if self._format.is_xyz:
            xyz = (row['easting'], row['northing'], row['altitude'])
        else:
            lla = (row['latitude'], row['longitude'], row['altitude'])
            xyz = transform(self._lla_crs, self._crs, [lla[1]], [lla[0]], [lla[2]])
            xyz = tuple([coord[0] for coord in xyz])

        if self._format.is_opk:
            opk = (row['omega'], row['phi'], row['kappa'])
            opk = opk if radians else tuple(np.radians(opk).tolist())
        else:
            rpy = (row['roll'], row['pitch'], row['yaw'])
            rpy = rpy if radians else tuple(np.radians(rpy).tolist())
            if self._format.is_xyz:
                lla = transform(self._crs, self._lla_crs, [xyz[0]], [xyz[1]], [xyz[2]])
                lla = (lla[1][0], lla[0][0], lla[2][0])  # x, y order -> lat, lon order
            opk = _rpy_to_opk(rpy, lla, self._crs, lla_crs=self._lla_crs)

        return xyz, opk

    def read_ext_param(self) -> Dict[str, Dict]:
        ext_param_dict = {}
        with open(self._filename, 'r', newline=None) as f:
            reader = DictReader(f, fieldnames=self._fieldnames, dialect=self._dialect)

            if self._has_header:  # read header
                _ = next(iter(reader))

            for row in reader:
                filename = Path(row['filename']).name
                row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
                xyz, opk = self._convert(row, radians=self._radians)
                ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=row.get('camera', None))

        return ext_param_dict


class OsfmReader(Reader):
    # TODO: OSfM reconstruction is in a topocentric system, so the transfer of 3D cartesian
    #  positions & rotations into a 2D+1D UTM CRS is an approximation, with similar issues to the
    #  Pix4D RPY->OPK conversion.  Ideally the orthorectification should also happen in this
    #  topocentric system, with the DEM being transformed into it. Then the orthorectified image
    #  can be reprojected to UTM.
    def __init__(
        self,
        filename: Union[str, Path],
        crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = CRS.from_epsg(4326),
    ):
        """
        Class for reading camera interior and exterior parameters from an OpenSfM
        'reconstruction.json' file.

        Parameters
        ----------
        filename: pathlib.Path
            Path of the 'reconstruction.json' file.
        crs: rio.crs.CRS, optional
            CRS of the world coordinate system.  If not specified, a UTM CRS will be
            auto-determined.
        lla_crs: rio.crs.CRS, optional
            CRS of geographic camera coordinates in the OpenSFM reconstruction file.
        """
        Reader.__init__(self, crs=crs, lla_crs=lla_crs)
        self._filename = Path(filename)
        if not self._filename.exists():
            raise FileNotFoundError(f"File not found: '{self._filename}'.")

        self._json_dict = self._read_json_dict(Path(filename))
        if not self._crs:
            self._crs = self._find_utm_crs()
            logger.debug(f"Using auto UTM CRS: '{self._crs.to_proj4()}'")

    @staticmethod
    def _read_json_dict(filename: Path) -> Dict[str, Dict]:
        """Read and validate the reconstruction json file."""
        with open(filename, 'r') as f:
            json_data = json.load(f)

        schema = [
            dict(
                cameras=dict,
                shots=dict,
                reference_lla=dict(latitude=float, longitude=float, altitude=float),
            )
        ]
        try:
            validate_collection(schema, json_data)
        except (ValueError, TypeError, KeyError) as ex:
            raise ParamFileError(
                f"'{filename.name}' is not a valid OpenSfM reconstruction file: {str(ex)}"
            )

        # keep root schema keys and delete the rest
        json_dict = {k: json_data[0][k] for k in schema[0].keys()}
        del json_data
        return json_dict

    def _find_utm_crs(self) -> rio.CRS:
        """Return a UTM CRS that covers the reconstruction reference point."""
        ref_lla = self._json_dict['reference_lla']
        ref_lla = (ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        return utm_crs_from_latlon(*ref_lla[:2])

    def read_int_param(self) -> Dict[str, Dict]:
        return _read_osfm_int_param(self._json_dict['cameras'])

    def read_ext_param(self) -> Dict[str, Dict]:
        # transform reference coordinates to the world CRS
        ref_lla = self._json_dict['reference_lla']
        ref_lla = (ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        ref_xyz = transform(
            self._lla_crs,
            self._crs,
            [ref_lla[1]],
            [ref_lla[0]],
            [ref_lla[2]],
        )
        ref_xyz = tuple([coord[0] for coord in ref_xyz])

        ext_param_dict = {}
        for filename, shot_dict in self._json_dict['shots'].items():
            # convert  reconstruction 'translation' and 'rotation' to oty exterior params,
            # adapted from ODM: https://github.com/OpenDroneMap/ODM/blob/master/opendm/shots.py
            R = cv2.Rodrigues(np.array(shot_dict['rotation']))[0]
            delta_xyz = -R.T.dot(shot_dict['translation'])
            xyz = tuple((ref_xyz + delta_xyz).tolist())
            opk = _aa_to_opk(shot_dict['rotation'])
            cam_id = shot_dict['camera']
            cam_id = cam_id[3:] if cam_id.startswith('v2 ') else cam_id
            ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=cam_id)

        return ext_param_dict


class ExifReader(Reader):
    def __init__(
        self,
        filenames: Tuple[Union[str, Path], ...],
        crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = _default_lla_crs,
    ):
        """
        Class for reading interior and exterior camera parameters from image files with EXIF / XMP
        tags.

        Parameters
        ----------
        filenames: tuple of pathlib.Path
            Paths of the image files.
        crs: rio.crs.CRS, optional
            CRS of the world coordinate system.  If not specified, a UTM CRS will be
            auto-determined.
        lla_crs: rio.crs.CRS, optional
            CRS of geographic camera coordinates in the EXIF / XMP image file tags.
        """
        Reader.__init__(self, crs, lla_crs)
        filenames = [filenames] if isinstance(filenames, (str, Path)) else filenames
        self._exif_dict = self._read_exif(filenames)

        if not self._crs and len(self._exif_dict) > 0:
            self._crs = self._find_utm_crs()
            logger.debug(f"Using auto UTM CRS: '{self._crs.to_proj4()}'")

    @staticmethod
    def _read_exif(filenames: List[Union[str, Path]]) -> Dict[str, Exif]:
        """Return a dictionary of Exif objects for the given image paths."""
        exif_dict = {}
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} files [{elapsed}<{remaining}]'
        with rio.Env(GDAL_NUM_TRHREADS='ALL_CPUS'):
            for filename in tqdm(filenames, bar_format=bar_format, dynamic_ncols=True):
                filename = Path(filename)
                exif_dict[filename.name] = Exif(filename)
        return exif_dict

    def _find_utm_crs(self) -> Optional[rio.CRS]:
        """Return a UTM CRS that covers the mean of the camera positions."""
        llas = []
        for e in self._exif_dict.values():
            if not e.lla:
                raise ParamFileError(
                    f"No latitude, longitude & altitude tags in '{e.filename.name}'."
                )
            llas.append(e.lla)

        mean_latlon = np.array(llas)[:, :2].mean(axis=0)
        return utm_crs_from_latlon(*mean_latlon)

    def read_int_param(self) -> Dict[str, Dict]:
        int_param_dict = {}
        for filename, exif in self._exif_dict.items():
            int_param = _read_exif_int_param(exif)
            int_param_dict.update(**int_param)
        return int_param_dict

    def read_ext_param(self) -> Dict[str, Dict]:
        ext_param_dict = {}
        for filename, exif in self._exif_dict.items():
            ext_param_dict[filename] = _read_exif_ext_param(
                exif, crs=self._crs, lla_crs=self._lla_crs
            )
        return ext_param_dict


class OtyReader(Reader):
    def __init__(
        self,
        filename: Union[str, Path],
        crs: Union[str, rio.CRS] = None,
        lla_crs: Union[str, rio.CRS] = _default_lla_crs,
    ):
        """
        Class for reading exterior camera parameters from an orthority format geojson file.

        Parameters
        ----------
        filename: pathlib.Path
            Paths of the geojson file.
        crs: rio.crs.CRS, optional
            CRS of the world coordinate system.  By default this is read from the file.
        lla_crs: rio.crs.CRS, optional
            CRS of geographic camera coordinates (not used).
        """
        Reader.__init__(self, crs=crs, lla_crs=lla_crs)
        self._filename = Path(filename)
        if not self._filename.exists():
            raise FileNotFoundError(f"File not found: '{self._filename}'.")

        self._crs, self._json_dict = self._read_json_dict(Path(filename), self._crs)

    @staticmethod
    def _read_json_dict(filename: Path, crs: rio.CRS) -> Tuple[rio.CRS, Dict]:
        """Read and validate the geojson file."""
        with open(filename, 'r') as f:
            json_dict = json.load(f)

        schema = dict(
            type='FeatureCollection',
            world_crs=str,
            features=[
                dict(
                    type='Feature',
                    properties=dict(filename=str, camera=None, xyz=list, opk=list),
                    geometry=dict(type='Point', coordinates=list),
                )
            ],
        )
        try:
            validate_collection(schema, json_dict)
        except (ValueError, TypeError, KeyError) as ex:
            raise ParamFileError(
                f"'{filename.name}' is not a valid GeoJSON exterior parameter file: {str(ex)}"
            )

        if not crs:
            try:
                crs = rio.CRS.from_string(json_dict['world_crs'])
            except RioCrsError as ex:
                raise ParamFileError(f"Could not interpret CRS in '{filename.name}': {str(ex)}")

        return crs, json_dict

    def read_ext_param(self) -> Dict[str, Dict]:
        ext_param_dict = {}
        for feat_dict in self._json_dict['features']:
            prop_dict = feat_dict['properties']
            filename = prop_dict['filename']
            ext_parm = dict(
                xyz=tuple(prop_dict['xyz']), opk=tuple(prop_dict['opk']), camera=prop_dict['camera']
            )
            ext_param_dict[filename] = ext_parm
        return ext_param_dict


def write_int_param(
    filename: Union[str, Path], int_param_dict: Dict[str, Dict], overwrite: bool = False
):
    """
    Write interior parameters to an orthority format yaml file.

    Parameters
    ----------
    filename: pathlib.Path
        Path of the yaml file to write.
    int_param_dict: dict
        Interior parameters to write.
    overwrite: bool, optional
        Overwrite the yaml file if it exists.
    """
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            raise FileExistsError(f"Interior parameter file exists: '{filename}'.")
        filename.unlink()

    # convert 'cam_type' key to 'type' & make the converted item the first in the dict
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
    filename: Union[str, Path],
    ext_param_dict: Dict[str, Dict],
    crs: Union[str, rio.CRS],
    overwrite: bool = False,
):
    """
    Write exterior parameters to an orthority format geojson file.

    Parameters
    ----------
    filename: pathlib.Path
        Path of the geojson file to write.
    ext_param_dict: dict
        Exterior parameters to write.
    crs: rio.crs.CRS
        CRS of the world coordinate system.
    overwrite: bool, optional
        Overwrite the geojson file if it exists.
    """
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            raise FileExistsError(f"Exterior parameter file exists: '{filename}'.")
        filename.unlink()

    feat_list = []
    lla_crs = _default_lla_crs
    crs = rio.CRS.from_string(crs) if isinstance(crs, str) else crs
    for src_file, ext_param in ext_param_dict.items():
        xyz = ext_param['xyz']
        lla = transform(crs, lla_crs, [xyz[0]], [xyz[1]], [xyz[2]])
        lla = [lla[0][0], lla[1][0], lla[2][0]]  # (lon, lat) order for geojson point
        props_dict = dict(
            filename=src_file, camera=ext_param['camera'], xyz=xyz, opk=ext_param['opk']
        )
        geom_dict = dict(type='Point', coordinates=list(lla))
        feat_dict = dict(type='Feature', properties=props_dict, geometry=geom_dict)
        feat_list.append(feat_dict)

    json_dict = dict(type='FeatureCollection', world_crs=crs.to_string(), features=feat_list)
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=4)
