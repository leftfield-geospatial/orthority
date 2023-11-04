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
    """Camera model types."""

    pinhole = 'pinhole'
    """Pinhole camera model."""

    brown = 'brown'
    """
    Brown-Conrady camera model.

    Compatible with `OpenDroneMap <https://docs.opendronemap.org/arguments/camera-lens/>`__ and
    `OpenSfM <https://opensfm.org/docs/geometry.html#brown-camera>`__ *brown* model parameters,
    and the 4- and 5-coefficient versions of the `OpenCV general model
    <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.
    """

    fisheye = 'fisheye'
    """
    Fisheye camera model.

    Compatible with `OpenDroneMap <https://docs.opendronemap.org/arguments/camera-lens/>`__,
    `OpenSfM <https://opensfm.org/docs/geometry.html#fisheye-camera>`__, and `OpenCV
    <https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html>`__  fisheye model parameters.
    """

    opencv = 'opencv'
    """
    OpenCV `general camera model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.
    """

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_

    @classmethod
    def from_odm(cls, cam_type: str):
        """Convert from OpenDroneMap / OpenSfM projection type."""
        cam_type = 'brown' if cam_type == 'perspective' else cam_type
        if cam_type not in cls.__members__:
            raise ValueError(f"Unsupported OpenDroneMap / OpenSfM camera type: '{cam_type}'")
        return cls(cam_type)


class Interp(str, Enum):
    """Interpolation types."""

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

    def to_cv(self) -> int:
        """Convert to OpenCV interpolation type."""
        name_to_cv = dict(
            average=cv2.INTER_AREA,
            bilinear=cv2.INTER_LINEAR,
            cubic=cv2.INTER_CUBIC,
            lanczos=cv2.INTER_LANCZOS4,
            nearest=cv2.INTER_NEAREST,
        )
        return name_to_cv[self._name_]

    def to_rio(self) -> Resampling:
        """Convert to rasterio resampling type."""
        return Resampling[self._name_]


class Compress(str, Enum):
    """Compression types."""

    jpeg = 'jpeg'
    """Jpeg (lossy) compression."""
    deflate = 'deflate'
    """Deflate (lossless) compression."""
    auto = 'auto'
    """Use jpeg compression if possible, deflate otherwise."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class CsvFormat(Enum):
    """CSV exterior parameter file formats."""

    xyz_opk = 1
    """Projected (x, y, z) position and (omega, phi, kappa) orientation."""
    lla_opk = 2
    """Geographic (latitude, longitude, altitude) position and (omega, phi, kappa) orientation."""
    xyz_rpy = 3
    """Projected (x, y, z) position and (roll, pitch, yaw) orientation."""
    lla_rpy = 4
    """Geographic (latitude, longitude, altitude) position and (roll, pitch, yaw) orientation."""

    @property
    def is_opk(self) -> bool:
        """True if format has an (omega, phi, kappa) orientation, False otherwise."""
        return self is CsvFormat.xyz_opk or self is CsvFormat.lla_opk

    @property
    def is_xyz(self) -> bool:
        """True if format has an (x, y, z) position, False otherwise."""
        return self is CsvFormat.xyz_opk or self is CsvFormat.xyz_rpy
