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

"""EXIF / XMP image tag decoding and reading."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from functools import cached_property
from os import PathLike
from xml.etree import ElementTree as ET

import numpy as np
import rasterio as rio
from fsspec.core import OpenFile

from orthority import common
from orthority.errors import OrthorityWarning

logger = logging.getLogger(__name__)

_xmp_schemas = dict(
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
A schema of known XMP keys.

Uses XML namespace qualified keys which are unique, rather than xmltodict type prefix qualified
keys, which can have different prefixes referring to the same namespace.
"""


def _xml_to_flat_dict(xmp_str: str) -> dict[str, str]:
    """Return a flat dictionary for the given XML string."""
    etree = ET.fromstring(xmp_str)
    flat_dict = {}

    def traverse_etree(etree: ET) -> None:
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
    # Adapted from https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py
    """
    EXIF / XMP image tag extractor for camera model related values.

    :param file:
        Image file to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile`
        object in binary mode (``'rb'``), or a dataset reader.
    """

    def __init__(self, file: str | PathLike | OpenFile | rio.DatasetReader):
        self._filename = common.get_filename(file)
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUS'), common.OpenRaster(file, 'r') as ds:
            # NB: avoid calling ds.tag_namespaces() which reads more (all?) of the dataset
            # compared to ds.tags() with known ns=
            self._exif_dict = ds.tags()
            self._exif_dict = ds.tags(ns='EXIF') if len(self._exif_dict) == 0 else self._exif_dict
            self._im_size = ds.shape[::-1]

            self._xmp_dict = ds.tags(ns='xml:XMP')
            if len(self._xmp_dict) > 0:
                xmp_str = self._xmp_dict['xml:XMP'].strip('xml:XMP=')
                self._xmp_dict = _xml_to_flat_dict(xmp_str)
            else:
                logger.debug(f"'{self._filename}' contains no XMP metadata")

    def __str__(self):
        lla_str = '({:.4f}, {:.4f}, {:.4f})'.format(*self.lla) if self.lla else 'None'
        rpy_str = '({:.4f}, {:.4f}, {:.4f})'.format(*self.rpy) if self.rpy else 'None'
        dewarp_str = ', '.join([f'{p:.4f}' for p in self.dewarp]) if self.dewarp else 'None'
        return (
            f'Image: {self.filename}'
            f'\nCamera: {self.make} {self.model}'
            f'\nActual image size: {self.im_size}'
            f'\nTagged image size: {self.tag_im_size}'
            f'\nFocal length: {self.focal_len}'
            f'\nFocal length (35mm): {self.focal_len_35}'
            f'\nSensor size: {self.sensor_size}'
            f'\nOrientation: {self.orientation}'
            f'\nLatitude, longitude, altitude: {lla_str}'
            f'\nRoll, pitch, yaw: {rpy_str}'
            f'\nDewarp: {dewarp_str}'
        )

    @property
    def filename(self) -> str:
        """Image filename."""
        return self._filename

    @cached_property
    def make(self) -> str | None:
        """Camera make."""
        make = self._exif_dict.get('EXIF_Make')
        return make.lower() if make is not None else None

    @cached_property
    def model(self) -> str | None:
        """Camera model."""
        model = self._exif_dict.get('EXIF_Model')
        return model.lower() if model is not None else None

    @cached_property
    def serial(self) -> str | None:
        """Camera serial number."""
        return self._exif_dict.get('EXIF_BodySerialNumber')

    @property
    def im_size(self) -> tuple[int, int] | None:
        """Actual image (width, height) in pixels."""
        return self._im_size

    @cached_property
    def tag_im_size(self) -> tuple[int, int] | None:
        """Tagged image (width, height) in pixels."""
        width = self._get_exif_value('EXIF_PixelXDimension')
        height = self._get_exif_value('EXIF_PixelYDimension')
        return (int(width), int(height)) if width is not None and height is not None else None

    @cached_property
    def sensor_size(self) -> tuple[float, float] | None:
        """Sensor (width, height) in mm."""
        unit_key = 'EXIF_FocalPlaneResolutionUnit'
        xres_key = 'EXIF_FocalPlaneXResolution'
        yres_key = 'EXIF_FocalPlaneYResolution'

        if (
            not {unit_key, xres_key, yres_key}.issubset(self._exif_dict.keys())
            or self.tag_im_size is None
        ):
            return None

        # find mm per resolution unit
        unit_code = int(self._exif_dict[unit_key])
        mm_per_unit_dict = {
            # https://exiftool.sourceforge.net/TagNames/EXIF.html
            2: 25.4,  # inches
            3: 10.0,  # cm
            4: 1.0,  # mm
            5: 0.001,  # um
        }
        mm_per_unit = mm_per_unit_dict.get(unit_code)
        if mm_per_unit is None:
            warnings.warn(
                f'Unknown focal plane resolution unit: {unit_code}',
                category=OrthorityWarning,
                stacklevel=2,
            )
            return None

        # return sensor size in mm
        pixels_per_unit = np.array([self._get_exif_value(xres_key), self._get_exif_value(yres_key)])
        sensor_size = mm_per_unit * np.array(self.tag_im_size) / pixels_per_unit
        return tuple(sensor_size.tolist())

    @cached_property
    def focal_len(self) -> float | None:
        """Focal length in mm."""
        return self._get_exif_value('EXIF_FocalLength')

    @cached_property
    def focal_len_35(self) -> float | None:
        """35mm equivalent focal length in mm."""
        return self._get_exif_value('EXIF_FocalLengthIn35mmFilm')

    @cached_property
    def orientation(self) -> int | None:
        """Image orientation code (see https://exiftool.sourceforge.net/TagNames/EXIF.html)."""
        orientation = self._exif_dict.get('EXIF_Orientation')
        return int(orientation) if orientation is not None else None

    @cached_property
    def lla(self) -> tuple[float, float, float] | None:
        """(latitude, longitude, altitude) coordinates with latitude and longitude in decimal
        degrees, and altitude in meters.
        """
        return self._get_xmp_lla() or self._get_exif_lla()

    @cached_property
    def rpy(self) -> tuple[float, float, float] | None:
        """(roll, pitch, yaw) camera/gimbal angles in degrees."""
        for xmp_schema in _xmp_schemas.values():
            if len(set(xmp_schema['rpy_keys']).intersection(self._xmp_dict.keys())) == 3:
                rpy = np.array([float(self._xmp_dict[key]) for key in xmp_schema['rpy_keys']])
                rpy *= np.array(xmp_schema['rpy_gains'])
                rpy += np.array(xmp_schema['rpy_offsets'])
                return tuple(rpy.tolist())
        return None

    @cached_property
    def dewarp(self) -> tuple[float, ...] | None:
        """Dewarp parameters."""
        for xmp_schema in _xmp_schemas.values():
            dewarp_str = self._xmp_dict.get(xmp_schema['dewarp_key'])
            if dewarp_str:
                return tuple([float(ps) for ps in dewarp_str.split(';')[-1].split(',')])
        return None

    def _get_exif_value(self, key: str) -> float | tuple[float, ...] | None:
        """Get the float value(s) for a numeric EXIF tag."""
        if key not in self._exif_dict:
            return None
        values = [
            float(val_str.strip(' ('))
            for val_str in self._exif_dict[key].split(')')
            if len(val_str) > 0
        ]
        return values[0] if len(values) == 1 else tuple(values)

    def _get_exif_lla(self) -> tuple[float, float, float] | None:
        """Return the (latitude, longitude, altitude) EXIF image location with latitude, longitude
        in decimal degrees, and altitude in meters.
        """
        lat_ref_key = 'EXIF_GPSLatitudeRef'
        lon_ref_key = 'EXIF_GPSLongitudeRef'
        lat_key = 'EXIF_GPSLatitude'
        lon_key = 'EXIF_GPSLongitude'
        if not {lat_ref_key, lon_ref_key, lat_key, lon_key}.issubset(self._exif_dict.keys()):
            return None

        # get latitude, longitude
        def dms_to_decimal(dms: Sequence[float], ref: str):
            """Convert (degrees, minutes, seconds) tuple to decimal degrees, applying reference
            sign.
            """
            sign = 1 if ref in 'NE' else -1
            return ((dms[2] / 60 + dms[1]) / 60 + dms[0]) * sign

        lat = dms_to_decimal(self._get_exif_value(lat_key), self._exif_dict[lat_ref_key])
        lon = dms_to_decimal(self._get_exif_value(lon_key), self._exif_dict[lon_ref_key])

        # get altitude
        alt = self._get_exif_value('EXIF_GPSAltitude') or 0.0
        alt_ref = int(self._exif_dict.get('EXIF_GPSAltitudeRef', '0x00'), base=0)
        if alt_ref == 1:
            alt *= -1

        return lat, lon, alt

    def _get_xmp_lla(self) -> tuple[float, float, float] | None:
        """Return the XMP (latitude, longitude, altitude) values if all of them exist. ."""
        for xmp_schema in _xmp_schemas.values():
            if len(set(xmp_schema['lla_keys']).intersection(self._xmp_dict.keys())) == 3:
                lla = [float(self._xmp_dict[key]) for key in xmp_schema['lla_keys']]
                return tuple(lla)
        return None

    def to_dict(self) -> dict[str, object]:
        """Convert to a property dictionary."""
        return {
            k: getattr(self, k)
            for k, v in vars(type(self)).items()
            if isinstance(v, (property, cached_property))
        }
