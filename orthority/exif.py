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

# Adapted from the OpenSfM exif module
# https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from xml.etree import cElementTree as ET

import numpy as np
import rasterio as rio

from orthority.utils import suppress_no_georef

logger = logging.getLogger(__name__)

xmp_schemas = dict(
    dji=dict(
        lla_keys=[
            '{http://www.dji.com/drone-dji/1.0/}GpsLatitude',
            '{http://www.dji.com/drone-dji/1.0/}GpsLongtitude',
            '{http://www.dji.com/drone-dji/1.0/}AbsoluteAltitude',
        ],
        rpy_keys=[
            '{http://www.dji.com/drone-dji/1.0/}GimbalRollDegree',
            '{http://www.dji.com/drone-dji/1.0/}GimbalPitchDegree',
            '{http://www.dji.com/drone-dji/1.0/}GimbalYawDegree',
        ],
        dewarp_key='{http://www.dji.com/drone-dji/1.0/}DewarpData',
        rpy_offsets=(0.0, 90.0, 0.0),
        rpy_gains=(1.0, 1.0, 1.0),
    ),
    # the Sensefly / Sony DSC & Pix4D / Parrot Sequoia keys may refer to the RPY of the drone,
    # not camera, but am including for now
    sensefly=dict(
        lla_keys=[],
        rpy_keys=[
            '{http://ns.sensefly.com/Camera/1.0/}Roll',
            '{http://ns.sensefly.com/Camera/1.0/}Pitch',
            '{http://ns.sensefly.com/Camera/1.0/}Yaw',
        ],
        dewarp_key=None,
        rpy_offsets=(0.0, 0.0, 0.0),
        rpy_gains=(1.0, 1.0, 1.0),
    ),
    pix4d=dict(
        lla_keys=[],
        rpy_keys=[
            '{http://pix4d.com/camera/1.0/}Roll',
            '{http://pix4d.com/camera/1.0/}Pitch',
            '{http://pix4d.com/camera/1.0/}Yaw',
        ],
        dewarp_key=None,
        rpy_offsets=(0.0, 0.0, 0.0),
        rpy_gains=(1.0, 1.0, 1.0),
    ),
)
"""
A schema of known RPY & LLA XMP keys.

Uses xml namspace qualified keys which are unique, rather than xmltodict type prefix qualified keys,
which can have different prefixes referring to the same namepace.
"""


def xml_to_flat_dict(xmp_str: str) -> Dict[str, str]:
    """Convert the given XML string to a flat dictionary."""
    etree = ET.fromstring(xmp_str)
    flat_dict = {}

    def traverse_etree(etree):
        """Traverse the given XML tree, populating flat_dict with xml element (tag, text) and
        attribute (name, value) pairs.
        """
        flat_dict[etree.tag] = etree.text
        if etree.attrib:
            flat_dict.update(**etree.attrib)
        for child in etree.findall("./*"):
            traverse_etree(child)

    traverse_etree(etree)
    return flat_dict


class Exif:
    def __init__(self, filename: Union[str, Path]):
        """
        Class to extract camera model relevant EXIF and XMP values from an image.

        Parameters
        ----------
        filename: str, Path
            Path to the image file.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File not found: '{filename}'")

        with suppress_no_georef(), rio.open(filename, 'r') as ds:
            namespaces = ds.tag_namespaces()
            exif_dict = ds.tags(ns='EXIF') if 'EXIF' in namespaces else ds.tags()
            self._im_size = ds.shape[::-1]

            if 'xml:XMP' in namespaces:
                # read XMP, stripping 'xml:XMP=' for rio copied tags
                xmp_str = ds.tags(ns='xml:XMP')['xml:XMP'].strip('xml:XMP=')
                xmp_dict = xml_to_flat_dict(xmp_str)
            else:
                logger.warning(f"'{filename.name}' contains no XMP metadata")
                xmp_dict = {}

        self._filename = filename
        self._make, self._model, self._serial = self._get_make_model_serial(exif_dict)
        self._tag_im_size = self._get_tag_im_size(exif_dict)
        self._sensor_size = self._get_sensor_size(exif_dict, self._im_size)
        self._focal_len, self._focal_len_35 = self._get_focal_len(exif_dict)
        self._orientation = self._get_orientation(exif_dict)
        self._lla = self._get_xmp_lla(xmp_dict) or self._get_lla(exif_dict)
        self._rpy = self._get_xmp_rpy(xmp_dict)
        self._dewarp = self._get_xmp_dewarp(xmp_dict)

    def __str__(self):
        lla_str = '({:.4f}, {:.4f}, {:.4f})'.format(*self._lla) if self._lla else 'None'
        rpy_str = '({:.4f}, {:.4f}, {:.4f})'.format(*self._rpy) if self._rpy else 'None'
        dewarp_str = ', '.join([f'{p:.4f}' for p in self._dewarp]) if self._dewarp else 'None'
        return (
            f'Image: {self._filename}'
            f'\nCamera: {self._make} {self._model}'
            f'\nImage size: {self.im_size}'
            f'\nFocal length: {self._focal_len}'
            f'\nFocal length (35mm): {self._focal_len_35}'
            f'\nSensor size: {self._sensor_size}'
            f'\nOrientation: {self._orientation}'
            f'\nLatitude, longitude, altitude: {lla_str}'
            f'\nRoll, pitch, yaw: {rpy_str}'
            f'\nDewarp: {dewarp_str}'
        )

    @property
    def filename(self) -> Path:
        """Path to source file."""
        return self._filename

    @property
    def make(self) -> Optional[str]:
        """Camera make."""
        return self._make

    @property
    def model(self) -> Optional[str]:
        """Camera model."""
        return self._model

    @property
    def serial(self) -> Optional[str]:
        """Camera serial number."""
        return self._serial

    @property
    def im_size(self) -> Optional[Tuple[int, int]]:
        """Actual image (width, height) in pixels."""
        return self._im_size

    @property
    def tag_im_size(self) -> Optional[Tuple[int, int]]:
        """Tagged image (width, height) in pixels."""
        return self._tag_im_size

    @property
    def sensor_size(self) -> Optional[Tuple[float, float]]:
        """Sensor (width, height) in mm."""
        return self._sensor_size

    @property
    def focal_len(self) -> Optional[float]:
        """Focal length in mm."""
        return self._focal_len

    @property
    def focal_len_35(self) -> Optional[float]:
        """35mm equivalent focal length in mm."""
        return self._focal_len_35

    @property
    def orientation(self) -> Optional[int]:
        """Image orientation code (see https://exiftool.org/TagNames/EXIF.html)."""
        return self._orientation

    @property
    def lla(self) -> Optional[Tuple[float, float, float]]:
        """(Latitude, longitude, altitude) coordinates with latitude and longitude in decimal
        degrees, and altitude in meters.
        """
        return self._lla

    @property
    def rpy(self) -> Optional[Tuple[float, float, float]]:
        """(Roll, pitch, yaw) camera/gimbal angles in degrees."""
        return self._rpy

    @property
    def dewarp(self) -> Optional[Tuple[float]]:
        """Dewarp parameters."""
        return self._dewarp

    @staticmethod
    def _get_exif_float(exif_dict: Dict[str, str], key: str) -> Optional[Union[float, Tuple]]:
        """Convert numeric EXIF tag to float(s)."""
        if key not in exif_dict:
            return None
        val_list = [
            float(val_str.strip(' (')) for val_str in exif_dict[key].split(')') if len(val_str) > 0
        ]
        return val_list[0] if len(val_list) == 1 else tuple(val_list)

    @staticmethod
    def _get_make_model_serial(exif_dict: Dict[str, str]) -> Tuple:
        """Return camera make and model string."""
        make_key = 'EXIF_Make'
        model_key = 'EXIF_Model'
        serial_key = 'EXIF_BodySerialNumber'
        make = exif_dict[make_key].lower() if make_key in exif_dict else None
        model = exif_dict[model_key].lower() if model_key in exif_dict else None
        serial = exif_dict[serial_key] if serial_key in exif_dict else None
        return make, model, serial

    @staticmethod
    def _get_tag_im_size(exif_dict: Dict[str, str]) -> Optional[Tuple]:
        """Return the tagged image (width, height) in pixels."""
        width = Exif._get_exif_float(exif_dict, 'EXIF_PixelXDimension')
        height = Exif._get_exif_float(exif_dict, 'EXIF_PixelYDimension')
        return (int(width), int(height)) if width and height else None

    @staticmethod
    def _get_sensor_size(
        exif_dict: Dict[str, str], im_size: Union[Tuple, np.ndarray]
    ) -> Optional[Tuple]:
        """Return the sensor (width, height) in mm."""

        unit_key = 'EXIF_FocalPlaneResolutionUnit'
        xres_key = 'EXIF_FocalPlaneXResolution'
        yres_key = 'EXIF_FocalPlaneYResolution'

        if unit_key not in exif_dict or xres_key not in exif_dict or yres_key not in exif_dict:
            return None

        # find mm per resolution unit
        unit_code = int(exif_dict["EXIF_FocalPlaneResolutionUnit"])
        mm_per_unit_dict = {
            # https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
            2: 25.4,  # inches
            3: 10.0,  # cm
            4: 1.0,  # mm
            5: 0.001,  # um
        }
        mm_per_unit = mm_per_unit_dict.get(unit_code, None)
        if not mm_per_unit:
            logger.warning(f'Unknown focal plane resolution unit: {unit_code}')
            return None

        # return sensor size in mm
        pixels_per_unit = np.array(
            [Exif._get_exif_float(exif_dict, xres_key), Exif._get_exif_float(exif_dict, yres_key)]
        )
        return tuple((mm_per_unit * np.array(im_size) / pixels_per_unit).tolist())

    @staticmethod
    def _get_focal_len(exif_dict: Dict[str, str]) -> Tuple:
        """Return the actual and 35mm equivalent focal lengths in mm."""
        focal_35 = Exif._get_exif_float(exif_dict, 'EXIF_FocalLengthIn35mmFilm')
        focal = Exif._get_exif_float(exif_dict, 'EXIF_FocalLength')
        return focal, focal_35

    @staticmethod
    def _get_orientation(exif_dict: Dict[str, str]) -> Optional[int]:
        ori_key = 'EXIF_Orientation'
        orientation = int(exif_dict[ori_key]) if ori_key in exif_dict else None
        return orientation

    @staticmethod
    def _get_lla(exif_dict: Dict[str, str]) -> Optional[Tuple]:
        """Return the (latitutde, longitude, altitude) EXIF image location with latitude, longitude
        in decimal degrees, and altitude in meters.
        """
        lat_ref_key = 'EXIF_GPSLatitudeRef'
        lon_ref_key = 'EXIF_GPSLongitudeRef'
        lat_key = 'EXIF_GPSLatitude'
        lon_key = 'EXIF_GPSLongitude'
        if any([key not in exif_dict for key in [lat_ref_key, lon_ref_key, lat_key, lon_key]]):
            return None

        # get latitude, longitude
        def dms_to_decimal(dms: Tuple[float, float, float], ref: str):
            """Convert (degrees, minutes, seconds) tuple to decimal degrees, applying reference
            sign.
            """
            sign = 1 if ref in 'NE' else -1
            return ((dms[2] / 60 + dms[1]) / 60 + dms[0]) * sign

        lat = dms_to_decimal(Exif._get_exif_float(exif_dict, lat_key), exif_dict[lat_ref_key])
        lon = dms_to_decimal(Exif._get_exif_float(exif_dict, lon_key), exif_dict[lon_ref_key])

        # get altitude
        alt = Exif._get_exif_float(exif_dict, 'EXIF_GPSAltitude') or 0.0
        alt_ref = int(exif_dict.get('EXIF_GPSAltitudeRef', '0x00'), 0)
        if alt_ref == 1:
            alt *= -1

        return lat, lon, alt

    @staticmethod
    def _get_xmp_lla(xmp_dict: Dict[str, str]) -> Optional[Tuple]:
        """Return the XMP (latitude, longitude, altitude) values if all of them exist. ."""
        for schema_name, xmp_schema in xmp_schemas.items():
            if all([lla_key in xmp_dict for lla_key in xmp_schema['lla_keys']]):
                lla = [float(xmp_dict[lla_key]) for lla_key in xmp_schema['lla_keys']]
                return tuple(lla)
        return None

    @staticmethod
    def _get_xmp_rpy(xmp_dict: Dict[str, str]) -> Optional[Tuple]:
        """Return the camera / gimbal (roll, pitch, yaw) angles in degrees if they exist."""
        for schema_name, xmp_schema in xmp_schemas.items():
            if all([rpy_key in xmp_dict for rpy_key in xmp_schema['rpy_keys']]):
                rpy = np.array([float(xmp_dict[rpy_key]) for rpy_key in xmp_schema['rpy_keys']])
                rpy *= np.array(xmp_schema['rpy_gains'])
                rpy += np.array(xmp_schema['rpy_offsets'])
                return tuple(rpy.tolist())
        return None

    @staticmethod
    def _get_xmp_dewarp(xmp_dict: Dict[str, str]) -> Optional[Tuple]:
        """Return the camera dewarp parameters if they exist."""
        for schema_name, xmp_schema in xmp_schemas.items():
            dewarp_str = xmp_dict.get(xmp_schema['dewarp_key'], None)
            if dewarp_str:
                return tuple([float(ps) for ps in dewarp_str.split(';')[-1].split(',')])
        return None

    def to_dict(self) -> Dict:
        """Convert to a dictionary of properties."""
        return {k: getattr(self, k) for k, v in vars(type(self)).items() if isinstance(v, property)}
