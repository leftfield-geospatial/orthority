# Copyright The Orthority Contributors.
#
# This file is part of Orthority.
#
# Orthority is free software: you can redistribute it and/or modify it under the terms of the GNU
# Affero General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# Orthority is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with Orthority.
# If not, see <https://www.gnu.org/licenses/>.

"""
Interior & exterior parameter file IO and conversions.

Files are read and converted into standard format dictionaries that can be used to create camera
objects with :func:`~orthority.camera.create_camera` or the various
:class:`~orthority.camera.Camera` subclasses.

All ``crs`` and ``lla_crs`` parameters can be supplied as EPSG, proj4 or WKT strings;
or :class:`~rasterio.crs.CRS` objects.
"""
# TODO: add a note about file paths, urls & objects
# TODO: specify the dict formats with examples (maybe in its own doc)?
# TODO: could dataclasses be a better way of defining the int / ext parameter dicts?
from __future__ import annotations

import csv
import json
import logging
from csv import Dialect, DictReader, Sniffer
from io import StringIO, TextIOBase
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import rasterio as rio
import yaml
from rasterio.crs import CRS
from rasterio.errors import CRSError as RioCrsError
from rasterio.warp import transform
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from orthority import utils
from orthority.enums import CameraType, CsvFormat
from orthority.errors import CrsMissingError, ParamFileError
from orthority.exif import Exif

logger = logging.getLogger(__name__)

_optional_schema = {
    CameraType.pinhole: ['sensor_size', 'cx', 'cy'],
    # fmt: off
    CameraType.opencv: [
        'sensor_size', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2',
        's3', 's4', 'tx', 'ty',
    ],
    # fmt: on
    CameraType.brown: ['sensor_size', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'],
    CameraType.fisheye: ['sensor_size', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
}
"""Schema of valid optional parameters for each camera type."""

_default_lla_crs = CRS.from_epsg(4979)
"""Default CRS for geographic camera coordinates."""


def _read_osfm_int_param(json_dict: dict) -> dict[str, dict[str, Any]]:
    """Read camera interior parameters from an OpenDroneMap / OpenSfM JSON dictionary."""

    def parse_json_param(json_param: dict, cam_id: str) -> dict[str, Any]:
        """Validate & convert the given JSON dictionary for a single camera."""
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

        # read focal length(s) (JSON values are normalised by max of sensor width & height)
        if 'focal' in json_param:
            int_param['focal_len'] = json_param.pop('focal')
        elif 'focal_x' in json_param and 'focal_y' in json_param:
            focal_x, focal_y = json_param.pop('focal_x'), json_param.pop('focal_y')
            int_param['focal_len'] = focal_x if focal_x == focal_y else (focal_x, focal_y)
        else:
            raise ParamFileError(
                f"'focal', or 'focal_x' and 'focal_y' are missing for camera '{cam_id}'."
            )

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
        int_param_dict[cam_id] = parse_json_param(json_param.copy(), cam_id)

    return int_param_dict


def _create_exif_cam_id(exif: Exif) -> str:
    """Return a camera ID string for the given Exif object."""
    prop_list = [prop for prop in [exif.make, exif.model, exif.serial] if prop]
    cam_id = ' '.join(prop_list)
    if len(cam_id) == 0:
        cam_id = 'unknown'
    return cam_id


def _read_exif_int_param(exif: Exif) -> dict[str, dict[str, Any]]:
    """Read camera interior parameters from an Exif object."""
    # TODO: might there be cases where XMP tags CalibratedFocalLength, CalibratedOpticalCenter* are
    #  present but not DewarpData, and better than equiv EXIF tags?
    exif_name = Path(exif.filename).name
    if exif.dewarp:
        if len(exif.dewarp) != 9 or not any(exif.dewarp) or not exif.tag_im_size:
            logger.debug(f"Cannot interpret dewarp data for '{exif_name}'.")
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
        logger.debug(f"Approximating the focal length for '{exif_name}' from the 35mm equivalent.")
        if exif.sensor_size:
            # scale 35mm focal length to actual focal length in mm, assuming "35mm" = 36mm max
            # sensor dimension
            cam_dict['focal_len'] = max(exif.sensor_size) * exif.focal_len_35 / 36.0
            cam_dict['sensor_size'] = exif.sensor_size
        else:
            # normalise 35mm focal length assuming "35mm" = 36 mm max sensor dimension, and find
            # a normalised sensor size in same units, assuming square pixels
            cam_dict['focal_len'] = exif.focal_len_35 / 36.0
    else:
        raise ParamFileError(
            f"No focal length & sensor size, or 35mm focal length tags in '{exif_name}'."
        )

    return {_create_exif_cam_id(exif): cam_dict}


def _read_exif_ext_param(
    exif: Exif, crs: str | CRS, lla_crs: str | CRS
) -> dict[str, dict[str, Any]]:
    """Read camera exterior parameters from an Exif object."""
    exif_name = Path(exif.filename).name
    if not exif.lla:
        raise ParamFileError(f"No latitude, longitude & altitude tags in '{exif_name}'.")
    if not exif.rpy:
        raise ParamFileError(f"No camera / gimbal roll, pitch & yaw tags in '{exif_name}'.")
    rpy = tuple(np.radians(exif.rpy).tolist())
    opk = _rpy_to_opk(rpy, exif.lla, crs, lla_crs=lla_crs)
    xyz = transform(lla_crs, crs, [exif.lla[1]], [exif.lla[0]], [exif.lla[2]])
    xyz = tuple([coord[0] for coord in xyz])
    ext_param = dict(xyz=xyz, opk=opk, camera=_create_exif_cam_id(exif))
    return {exif_name: ext_param}


def read_oty_int_param(
    filename: str | Path | TextIOBase,
) -> dict[str, dict[str, Any]]:
    """
    Read interior parameters for one or more cameras from an :doc:`Orthority format YAML file
    <../file_formats/yaml>`.

    :param filename:
        Path / URL / file object of the YAML file to read.
    """
    with utils.text_ctx(filename) as f:
        yaml_dict = yaml.safe_load(f)

    def parse_yaml_param(yaml_param: dict, cam_id: str = None) -> dict[str, Any]:
        """Validate & convert the given YAML dictionary for a single camera."""
        # test required keys for all cameras
        for req_key in ['type', 'im_size', 'focal_len']:
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
        if 'sensor_size' in yaml_param:
            int_param['sensor_size'] = tuple(yaml_param.pop('sensor_size'))

        # validate any remaining distortion params, update param_dict & return
        err_keys = set(yaml_param.keys()).difference(_optional_schema[int_param['cam_type']])
        if len(err_keys) > 0:
            raise ParamFileError(f"Unsupported parameter(s) {err_keys} for camera '{cam_id}'")
        int_param.update(**yaml_param)
        return int_param

    # flatten if in original simple-ortho format
    if 'camera' in yaml_dict:
        logger.warning(
            "Support for the 'config.yaml' format is deprecated and will be removed in future. "
            "Please switch to the Orthority YAML format for interior parameters."
        )
        yaml_dict = yaml_dict['camera']

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


def read_osfm_int_param(filename: str | Path | TextIOBase) -> dict[str, dict[str, Any]]:
    """
    Read interior parameters for one or more camera, from an OpenDroneMap :file:`cameras.json` or
    OpenSfM :file:`reconstruction.json` file.

    See the :doc:`format documentation <../file_formats/opensfm>` for supported camera models.

    :param filename:
        Path / URL / file object of the OpenDroneMap / OpenSfM JSON file to read.
    """
    with utils.text_ctx(filename) as f:
        json_dict = json.load(f)

    return _read_osfm_int_param(json_dict)


def read_exif_int_param(filename: str | Path | rio.DatasetReader) -> dict[str, dict[str, Any]]:
    """
    Read interior parameters for a camera from an image file with EXIF / XMP tags.

    See the :doc:`format documentation <../file_formats/exif_xmp>` for required tags.

    :param filename:
        Path / URL / dataset of the image file to read.
    """
    return _read_exif_int_param(Exif(filename))


def write_int_param(
    filename: str | Path, int_param_dict: dict[str, dict[str, Any]], overwrite: bool = False
) -> None:
    """
    Write interior parameters to an :doc:`Orthority format YAML file <../file_formats/yaml>`.

    :param filename:
        Path of the file to write.
    :param int_param_dict:
        Interior parameters to write.
    :param overwrite:
        Whether to overwrite the file if it exists.
    """
    # convert 'cam_type' key to 'type' & make the converted item the first in the dict
    yaml_dict = {}
    for cam_id, int_param in int_param_dict.items():
        yaml_param = {}
        if 'cam_type' in int_param:
            yaml_param['type'] = int_param['cam_type'].value
        yaml_param.update(**{k: v for k, v in int_param.items() if k != 'cam_type'})
        yaml_dict[cam_id] = yaml_param

    mode = 'w' if overwrite else 'x'
    with open(filename, mode) as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, indent=4, default_flow_style=None)


def write_ext_param(
    filename: str | Path,
    ext_param_dict: dict[str, dict[str, Any]],
    crs: str | CRS,
    overwrite: bool = False,
) -> None:
    """
    Write exterior parameters to an :doc:`Orthority format GeoJSON file <../file_formats/geojson>`.

    :param filename:
        Path of the file to write.
    :param ext_param_dict:
        Exterior parameters to write.
    :param crs:
        CRS of the world coordinate system.
    :param overwrite:
        Whether to overwrite the file if it exists.
    """
    feat_list = []
    lla_crs = _default_lla_crs
    crs = CRS.from_string(crs) if isinstance(crs, str) else crs
    for src_file, ext_param in ext_param_dict.items():
        xyz = ext_param['xyz']
        lla = transform(crs, lla_crs, [xyz[0]], [xyz[1]], [xyz[2]])
        lla = [lla[0][0], lla[1][0], lla[2][0]]  # (lon, lat) order for GeoJSON point
        props_dict = dict(
            filename=src_file, camera=ext_param['camera'], xyz=xyz, opk=ext_param['opk']
        )
        geom_dict = dict(type='Point', coordinates=list(lla))
        feat_dict = dict(type='Feature', properties=props_dict, geometry=geom_dict)
        feat_list.append(feat_dict)

    json_dict = dict(type='FeatureCollection', world_crs=crs.to_string(), features=feat_list)
    mode = 'w' if overwrite else 'x'
    with open(filename, mode) as f:
        json.dump(json_dict, f, indent=4)


def _rpy_to_rotation(rpy: tuple[float, float, float]) -> np.ndarray:
    """Convert the given (roll, pitch, yaw) angles in radians to a rotation matrix."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    roll, pitch, yaw = rpy
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R_z.dot(R_y).dot(R_x)


def _opk_to_rotation(opk: tuple[float, float, float]) -> np.ndarray:
    """Convert the given (omega, phi, kappa) angles in radians to a rotation matrix."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    omega, phi, kappa = opk
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
    )
    R_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    R_z = np.array(
        [[np.cos(kappa), -np.sin(kappa), 0], [np.sin(kappa), np.cos(kappa), 0], [0, 0, 1]]
    )
    return R_x.dot(R_y).dot(R_z)


def _rotation_to_opk(R: np.ndarray) -> tuple[float, float, float]:
    """Convert the given rotation matrix to the (omega, phi, kappa) angles in radians."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    omega = np.arctan2(-R[1, 2], R[2, 2])
    phi = np.arcsin(R[0, 2])
    kappa = np.arctan2(-R[0, 1], R[0, 0])
    return omega, phi, kappa


def _aa_to_opk(aa: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert the given angle axis vector (OpenSfM / OpenCV convention) to (omega, phi,
    kappa) angles in radians (PATB convention)."""
    # convert ODM angle/axis to rotation matrix (see
    # https://github.com/mapillary/OpenSfM/issues/121)
    R = cv2.Rodrigues(np.array(aa))[0].T

    # rotate from OpenSfM / OpenCV to PATB convention
    R = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

    # extract OPK from R
    omega, phi, kappa = _rotation_to_opk(R)
    return omega, phi, kappa


def _rpy_to_opk(
    rpy: tuple[float, float, float],
    lla: tuple[float, float, float],
    crs: CRS,
    lla_crs: CRS,
    C_bB: list[list[float]] | np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Convert (roll, pitch, yaw) to (omega, phi, kappa) angles for a given CRS.

    Coordinate conventions are as `defined by Pix4D
    <https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.>`__

    Note this conversion assumes ``crs`` is orthogonal and uniform scale (cartesian) in the
    vicinity of ``lla``.  ``crs`` should be conformal, and have its central meridian(s) close to
    ``lla`` for this assumption to be accurate.

    :param rpy:
        (roll, pitch, yaw) angles to rotate from body to navigation coordinates (radians).
    :param lla:
        (latitude, longitude, altitude) geographic coordinates of the camera (in units of
        ``lla_crs``).
    :param crs:
        CRS of the world coordinate system.
    :param lla_crs:
        CRS of the ``lla`` geographic coordinates.
    :param C_bB:
        Optional camera to body rotation matrix.  Defaults to:
        ``[[0, 1, 0], [1, 0, 0], [0, 0, -1]]``, which describes typical drone geometry.

    :return:
        (omega, phi, kappa) angles in radians, to rotate from camera (PATB convention) to world
        coordinates.
    """
    # Adapted from the OpenSfM exif module
    # https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py and Pix4D doc
    # https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.

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

    # return OPK angles extracted from C_EB
    omega, phi, kappa = _rotation_to_opk(C_EB)
    return omega, phi, kappa


class Reader:
    """
    Abstract interior and exterior parameter reader.

    :param crs:
        CRS of the world coordinate system.
    :param lla_crs:
        CRS of input geographic coordinates (if any).
    """

    def __init__(self, crs: str | CRS = None, lla_crs: str | CRS = _default_lla_crs) -> None:
        self._crs, self._lla_crs = self._parse_crss(crs, lla_crs)

    @staticmethod
    def _parse_crss(crs: str | CRS, lla_crs: str | CRS) -> tuple:
        """Validate and convert CRSs."""
        if crs:
            crs = CRS.from_string(crs) if isinstance(crs, str) else crs
            if not crs.is_projected:
                raise ValueError(f"'crs' should be a projected system.")

        if lla_crs:
            lla_crs = CRS.from_string(lla_crs) if isinstance(lla_crs, str) else lla_crs
            if not lla_crs.is_geographic:
                raise ValueError(f"'lla_crs' should be a geographic system.")
        return crs, lla_crs

    @property
    def crs(self) -> CRS:
        """CRS of the world coordinate system."""
        return self._crs

    def read_int_param(self) -> dict[str, dict[str, Any]]:
        """Read interior camera parameters."""
        raise NotImplementedError()

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        """Read exterior camera parameters."""
        raise NotImplementedError()


class CsvReader(Reader):
    """
    Exterior parameter reader for a CSV file.

    Reads tabular data from a CSV file with a row per source image and column fields for image
    file name, camera position, orientation and ID.

    See the :doc:`CSV documentation <../file_formats/csv>` for details on supported fields and
    formats.

    :param filename:
        Path / URL / file object of the CSV file.
    :param crs:
        CRS of the world coordinate system.  If set to None (the default), and the file contains
        :attr:`~orthority.enums.CsvFormat.lla_rpy` values, a UTM CRS will be auto-determined.  If
        set to None, and the file contains (x, y, z) world coordinate positions, a CRS can
        provided via a '.prj' file (i.e. a text file defining the CRS with a WKT, proj4 or EPSG
        string, and having the same path & stem as ``filename``, but a '.prj' extension).  In all
        other situations, ``crs`` should be supplied.
    :param lla_crs:
        Geographic CRS associated with any (latitude, longitude, altitude) position and/or (roll,
        pitch, yaw) angle values in the file.
    :param fieldnames:
        List of names specifying the CSV fields.  If set to None (the default), names will be
        read from the file header if it exists.  If ``fieldnames`` is supplied, any existing file
        header is ignored.  See the :doc:`CSV documentation <../file_formats/csv>` for recognised
        and required field names.
    :param dialect:
        :class:`~csv.Dialect` object specifying the CSV delimiter, quote character etc. If set to
        None (the default), this is auto-detected from the file.
    :param radians:
        Whether orientation angles are in radians (True), or degrees (False).
    """

    _type_schema = dict(
        filename=str,
        x=float,
        y=float,
        z=float,
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
    _legacy_fieldnames = ['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa']

    def __init__(
        self,
        filename: str | Path | TextIOBase,
        crs: str | CRS = None,
        lla_crs: str | CRS = _default_lla_crs,
        fieldnames: list[str] = None,
        dialect: Dialect = None,
        radians: bool = False,
    ) -> None:
        # TODO: allow other coordinate conventions for opk / rpy (bluh, odm, patb)
        Reader.__init__(self, crs, lla_crs=lla_crs)
        self._filename = filename
        self._radians = radians

        # read file once into a buffer
        with utils.text_ctx(filename, newline=None) as f:
            self._buffer = StringIO(f.read())

        self._fieldnames, self._dialect, self._has_header, self._format = self._parse_file(
            self._buffer, fieldnames=fieldnames, dialect=dialect
        )
        self._crs = self._crs or self._get_crs()

    @staticmethod
    def _parse_fieldnames(fieldnames: list[str]) -> CsvFormat:
        """Validate a list of header or user field names, and return the corresponding
        :class:`CsvFormat`.
        """
        if 'filename' not in fieldnames:
            raise ParamFileError(f"Fields should include 'filename'.")

        has_xyz = {'x', 'y', 'z'}.issubset(fieldnames)
        has_lla = {'latitude', 'longitude', 'altitude'}.issubset(fieldnames)
        has_opk = {'omega', 'phi', 'kappa'}.issubset(fieldnames)
        has_rpy = {'roll', 'pitch', 'yaw'}.issubset(fieldnames)

        if not (has_xyz or has_lla):
            raise ParamFileError(
                f"Fields should include 'x', 'y' & 'z', or 'latitude', 'longitude' & 'altitude'."
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
        buffer: StringIO, fieldnames: list[str] = None, dialect: Dialect = None
    ) -> tuple[list[str], Dialect, bool, CsvFormat]:
        """Determine and validate the CSV file format."""

        def strip_lower_strlist(str_list: list[str]) -> list[str]:
            """Strip and lower the case of a string list."""
            return [str_item.strip().lower() for str_item in str_list]

        # read a sample of the csv file (newline=None works around some delimiter detection
        # problems with newline='')
        buffer.seek(0)
        sample = buffer.read(10000)

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

    def _find_lla_rpy_crs(self) -> CRS:
        """Return a UTM CRS covering the mean of the camera positions in a
        :attr:`~CsvFormat.lla_rpy` format file.
        """
        latlons = []
        self._buffer.seek(0)
        reader = DictReader(self._buffer, fieldnames=self._fieldnames, dialect=self._dialect)
        if self._has_header:
            _ = next(iter(reader))
        for row in reader:
            row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
            latlon = (row['latitude'], row['longitude'])
            latlons.append(latlon)

        mean_latlon = np.array(latlons).mean(axis=0)
        return utils.utm_crs_from_latlon(*mean_latlon)

    def _get_crs(self) -> CRS:
        """Read / auto-determine and validate a CRS when no user CRS was supplied."""
        # TODO: should .prj crs be read as lla_crs for lla positions?
        crs = None
        if self._format is CsvFormat.xyz_opk or self._format is CsvFormat.xyz_rpy:
            # read CRS of xyz positions / opk orientations from .prj file, if it exists
            filename = str(self._filename)
            prj_filename = filename[: filename.rfind('.')] + '.prj'
            prj_name = Path(prj_filename).name
            try:
                with utils.open_text(prj_filename) as f:
                    crs_str = f.read()
                crs = CRS.from_string(crs_str)

                if crs.is_geographic:
                    raise ValueError(f"CRS in '{prj_name}' should not be a geographic system")

                logger.debug(f"Using '{prj_name}' CRS: '{crs.to_proj4()}'")

            except (FileNotFoundError, URLError, HTTPError) as ex:
                logger.debug(f"Could not open '{prj_name}': {str(ex)}.")
                if self._format is CsvFormat.xyz_rpy:
                    raise CrsMissingError(
                        f"'crs' should be specified for positions in '{Path(self._filename).name}'."
                    )

        elif self._format is CsvFormat.lla_rpy:
            # find a UTM CRS to transform the lla positions & rpy orientations into
            crs = self._find_lla_rpy_crs()
            logger.debug(f"Using auto UTM CRS: '{crs.to_proj4()}'")

        elif self._format is CsvFormat.lla_opk:
            # a user-supplied opk CRS is required to project lla into
            raise CrsMissingError(
                f"'crs' should be specified for orientations in '{Path(self._filename).name}'."
            )

        return crs

    def _convert(self, row: dict[str, float], radians=False) -> tuple[tuple, tuple]:
        """Convert a CSV row dictionary to (x, y, z) position, and (omega, phi,
        kappa) orientation.
        """
        if self._format.is_xyz:
            xyz = (row['x'], row['y'], row['z'])
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

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        ext_param_dict = {}
        self._buffer.seek(0)
        reader = DictReader(self._buffer, fieldnames=self._fieldnames, dialect=self._dialect)

        if self._has_header:  # read header
            _ = next(iter(reader))

        for row in reader:
            filename = Path(row['filename']).name
            row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
            xyz, opk = self._convert(row, radians=self._radians)
            ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=row.get('camera', None))

        return ext_param_dict


class OsfmReader(Reader):
    """
    Interior and exterior parameter reader for an OpenSfM :file:`reconstruction.json` file.

    See the :doc:`format documentation <../file_formats/opensfm>` for supported camera models.

    :param filename:
        Path / URL / file object of the :file:`reconstruction.json` file.
    :param crs:
        CRS of the world coordinate system.  If set to None (the default), a UTM CRS will be
        auto-determined.
    :param lla_crs:
        CRS of the ``reference_lla`` value in the :file:`reconstruction.json` file.
    """

    # TODO: OSfM reconstruction is in a topocentric system, so the transfer of 3D cartesian
    #  positions & rotations into a 2D+1D UTM CRS is an approximation, with similar issues to the
    #  Pix4D RPY->OPK conversion.  Ideally the orthorectification should also happen in this
    #  topocentric system, with the DEM being transformed into it. Then the orthorectified image
    #  can be reprojected to UTM.
    def __init__(
        self,
        filename: str | Path | TextIOBase,
        crs: str | CRS = None,
        lla_crs: str | CRS = CRS.from_epsg(4326),
    ) -> None:
        Reader.__init__(self, crs=crs, lla_crs=lla_crs)
        self._json_dict = self._read_json_dict(filename)
        if not self._crs:
            self._crs = self._find_utm_crs()
            logger.debug(f"Using auto UTM CRS: '{self._crs.to_proj4()}'")

    @staticmethod
    def _read_json_dict(filename: str | Path | TextIOBase) -> dict[str, dict]:
        """Read and validate the reconstruction JSON file."""
        with utils.text_ctx(filename) as f:
            json_data = json.load(f)

        schema = [
            dict(
                cameras=dict,
                shots=dict,
                reference_lla=dict(latitude=float, longitude=float, altitude=float),
            )
        ]
        try:
            utils.validate_collection(schema, json_data)
        except (ValueError, TypeError, KeyError) as ex:
            raise ParamFileError(
                f"'{Path(filename).name}' is not a valid OpenSfM reconstruction file: {str(ex)}"
            )

        # keep root schema keys and delete the rest
        json_dict = {k: json_data[0][k] for k in schema[0].keys()}
        del json_data
        return json_dict

    def _find_utm_crs(self) -> CRS:
        """Return a UTM CRS that covers the reconstruction reference point."""
        ref_lla = self._json_dict['reference_lla']
        ref_lla = (ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        return utils.utm_crs_from_latlon(*ref_lla[:2])

    def read_int_param(self) -> dict[str, dict[str, Any]]:
        return _read_osfm_int_param(self._json_dict['cameras'])

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
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
            R = cv2.Rodrigues(np.array(shot_dict['rotation']))[0].T
            delta_xyz = -R.dot(shot_dict['translation'])
            xyz = tuple((ref_xyz + delta_xyz).tolist())
            # rotate camera coords from OpenSfM / OpenCV to PATB convention
            R_ = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            opk = _rotation_to_opk(R_)
            cam_id = shot_dict['camera']
            cam_id = cam_id[3:] if cam_id.startswith('v2 ') else cam_id
            ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=cam_id)

        return ext_param_dict


class ExifReader(Reader):
    """
    Interior and exterior parameter reader for image file(s) with EXIF / XMP tags.

    See the :doc:`format documentation <../file_formats/exif_xmp>` for required tags.

    :param filenames:
        Path / URL / datasets(s) of the image file(s).
    :param crs:
        CRS of the world coordinate system.  If set to None (the default), a UTM CRS will be
        auto-determined.
    :param lla_crs:
        CRS of geographic camera coordinates in the EXIF / XMP tags.
    """

    def __init__(
        self,
        filenames: tuple[str | Path | rio.DatasetReader, ...],
        crs: str | CRS = None,
        lla_crs: str | CRS = _default_lla_crs,
    ) -> None:
        Reader.__init__(self, crs, lla_crs)
        filenames = (
            [filenames] if isinstance(filenames, (str, Path, rio.DatasetReader)) else filenames
        )
        self._exif_dict = self._read_exif(filenames)

        if not self._crs and len(self._exif_dict) > 0:
            self._crs = self._find_utm_crs()
            logger.debug(f"Using auto UTM CRS: '{self._crs.to_proj4()}'")

    def _read_exif(self, filenames: list[str | Path | rio.DatasetReader]) -> dict[str, Exif]:
        """Return a dictionary of Exif objects for the given image paths."""
        exif_dict = {}
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} files [{elapsed}<{remaining}]'
        with logging_redirect_tqdm([logging.getLogger(__package__)], tqdm_class=tqdm):
            for filename in tqdm(filenames, bar_format=bar_format, dynamic_ncols=True):
                exif_obj = Exif(filename)
                exif_dict[Path(exif_obj.filename).name] = exif_obj
        return exif_dict

    def _find_utm_crs(self) -> CRS:
        """Return a UTM CRS that covers the mean of the camera positions."""
        # TODO: use weighted sum as in OpenSfM: https://github.com/mapillary/OpenSfM/blob/c6b5acef9376a75b87414d900c258ef876a6413a/opensfm/dataset.py#L985
        llas = []
        for e in self._exif_dict.values():
            if not e.lla:
                raise ParamFileError(
                    f"No latitude, longitude & altitude tags in '{Path(e.filename).name}'."
                )
            llas.append(e.lla)

        mean_latlon = np.array(llas)[:, :2].mean(axis=0)
        return utils.utm_crs_from_latlon(*mean_latlon)
        # return CRS.from_proj4(f'+proj=ortho +lat_0={mean_latlon[0]} +lon_0={mean_latlon[1]}'

    def read_int_param(self) -> dict[str, dict[str, Any]]:
        int_param_dict = {}
        for filename, exif in self._exif_dict.items():
            int_param = _read_exif_int_param(exif)
            int_param_dict.update(**int_param)
        return int_param_dict

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        ext_param_dict = {}
        for filename, exif in self._exif_dict.items():
            ext_param_dict.update(
                **_read_exif_ext_param(exif, crs=self._crs, lla_crs=self._lla_crs)
            )
        return ext_param_dict


class OtyReader(Reader):
    """
    Exterior parameter reader for an :doc:`Orthority format GeoJSON file <../file_formats/geojson>`.

    :param filename:
        Path / URL / file object of the GeoJSON file.
    """

    def __init__(self, filename: str | Path | TextIOBase) -> None:
        Reader.__init__(self)
        self._crs, self._json_dict = self._read_json_dict(filename, self._crs)

    @staticmethod
    def _read_json_dict(filename: str | Path | TextIOBase, crs: CRS) -> tuple[CRS, dict]:
        """Read and validate the GeoJSON file."""
        with utils.text_ctx(filename) as f:
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
            utils.validate_collection(schema, json_dict)
        except (ValueError, TypeError, KeyError) as ex:
            raise ParamFileError(
                f"'{Path(filename).name}' is not a valid GeoJSON exterior parameter file: {str(ex)}"
            )

        if not crs:
            try:
                crs = CRS.from_string(json_dict['world_crs'])
            except RioCrsError as ex:
                raise ParamFileError(
                    f"Could not interpret CRS in '{Path(filename).name}': {str(ex)}"
                )

        return crs, json_dict

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        ext_param_dict = {}
        for feat_dict in self._json_dict['features']:
            prop_dict = feat_dict['properties']
            filename = prop_dict['filename']
            ext_parm = dict(
                xyz=tuple(prop_dict['xyz']), opk=tuple(prop_dict['opk']), camera=prop_dict['camera']
            )
            ext_param_dict[filename] = ext_parm
        return ext_param_dict
