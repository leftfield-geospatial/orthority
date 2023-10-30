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

from __future__ import annotations
from enum import Enum

import cv2
from rasterio.enums import Resampling


class CameraType(str, Enum):
    """Enumeration for the camera model type."""

    pinhole = 'pinhole'
    """Pinhole camera model."""

    brown = 'brown'
    """
    Brown-Conrady camera model.

    Compatible with `ODM <https://docs.opendronemap.org/arguments/camera-lens/>`_ /
    `OpenSFM <https://github.com/mapillary/OpenSfM>`_
    *brown* parameter estimates; and the 4 &    5-coefficient version of the
    `general OpenCV distortion model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_.
    """

    fisheye = 'fisheye'
    """
    Fisheye camera model.

    Compatible with `ODM <https://docs.opendronemap.org/arguments/camera-lens/>`_ /
    `OpenSFM <https://github.com/mapillary/OpenSfM>`_,
    and `OpenCV
    <https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html>`_ *fisheye* parameter
    estimates.
    """

    opencv = 'opencv'
    """
    OpenCV `general camera model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_.
     
    Supports the full set of distortion coefficient estimates.
    """

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_

    @classmethod
    def from_odm(cls, cam_type: str):
        """Convert from ODM / OpenSFM projection type."""
        cam_type = 'brown' if cam_type == 'perspective' else cam_type
        if cam_type not in cls.__members__:
            raise ValueError(f"Unsupported ODM / OpenSFM camera type: '{cam_type}'")
        return cls(cam_type)


class Interp(str, Enum):
    """
    Enumeration for common `OpenCV
    <https://docs.opencv.org/4.8.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121>`_
    and `rasterio
    <https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling>`_
    interpolation types.
    """

    average = 'average'
    """Average input pixels over the corresponding output pixel area (suited to downsampling)."""
    bilinear = 'bilinear'
    """Bilinear interpolation."""
    cubic = 'cubic'
    """Bicubic interpolation."""
    lanczos = 'lanczos'
    """Lanczos windowed sinc interpolation."""
    nearest = 'nearest'
    """Nearest neighbor interpolation."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_

    @classmethod
    def cv_list(cls) -> list:
        """A list of OpenCV compatible :class:`~orthority.enums.Interp` values."""
        _cv_list = []
        for interp in list(cls):
            try:
                interp.to_cv()
                _cv_list.append(interp)
            except ValueError:
                pass
        return _cv_list

    def to_cv(self) -> int:
        """Convert to OpenCV interpolation type."""
        name_to_cv = dict(
            average=cv2.INTER_AREA,
            bilinear=cv2.INTER_LINEAR,
            cubic=cv2.INTER_CUBIC,
            lanczos=cv2.INTER_LANCZOS4,
            nearest=cv2.INTER_NEAREST,
        )
        if self._name_ not in name_to_cv:
            raise ValueError(f"OpenCV does not support '{self._name_}' interpolation.")
        return name_to_cv[self._name_]

    def to_rio(self) -> Resampling:
        """Convert to rasterio resampling type."""
        return Resampling[self._name_]


class Compress(str, Enum):
    """Enumeration for ortho compression."""

    jpeg = 'jpeg'
    """Jpeg (lossy) compression."""
    deflate = 'deflate'
    """Deflate (lossless) compression."""
    auto = 'auto'
    """Use jpeg compression if possible, otherwise deflate."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class CsvFormat(Enum):
    """Enumeration for CSV exterior parameter format."""

    xyz_opk = 1
    """Projected (easting, northing, altitude) position and (omega, phi, kappa) orientation."""
    lla_opk = 2
    """Geographic (latitude, longitude, altitude) position and (omega, phi, kappa) orientation."""
    xyz_rpy = 3
    """Projected (easting, northing, altitude) position and (roll, pitch, yaw) orientation."""
    lla_rpy = 4
    """Geographic (latitude, longitude, altitude) position and (roll, pitch, yaw) orientation."""

    @property
    def is_opk(self) -> bool:
        """True if format has an (omega, phi, kappa) orientation, False otherwise."""
        return self is CsvFormat.xyz_opk or self is CsvFormat.lla_opk

    @property
    def is_xyz(self) -> bool:
        """True if format has an (easting, northing, altitude) position, False otherwise."""
        return self is CsvFormat.xyz_opk or self is CsvFormat.xyz_rpy
