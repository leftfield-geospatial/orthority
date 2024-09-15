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
    """Pinhole frame camera model."""

    brown = 'brown'
    """
    Brown-Conrady frame camera model.

    Compatible with `OpenDroneMap / OpenSfM 
    <https://opensfm.org/docs/geometry.html#camera-models>`__ ``perspective``, ``simple_radial``, 
    ``radial`` and ``brown`` model parameters, and the 4- and 5-coefficient versions of the 
    `OpenCV general model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__."""

    fisheye = 'fisheye'
    """
    Fisheye frame camera model.

    Compatible with `OpenDroneMap / OpenSfM 
    <https://opensfm.org/docs/geometry.html#fisheye-camera>`__ ``fisheye``, and `OpenCV 
    <https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html>`__  fisheye model 
    parameters."""

    opencv = 'opencv'
    """
    `OpenCV general frame camera model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.
    """

    rpc = 'rpc'
    """RPC camera model."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_

    @classmethod
    def from_odm(cls, cam_type: str):
        """Convert from OpenDroneMap / OpenSfM projection type."""
        cam_type = cam_type.lower().strip()
        if cam_type in ['perspective', 'simple_radial', 'radial']:
            cam_type = 'brown'
        if cam_type not in cls.__members__ or cam_type == 'rpc':
            raise ValueError(f"Unsupported OpenDroneMap / OpenSfM camera type: '{cam_type}'")
        return cls(cam_type)


class Interp(str, Enum):
    """Interpolation types."""

    nearest = 'nearest'
    """Nearest neighbor interpolation."""
    average = 'average'
    """Average input pixels over the corresponding output pixel area (suited to downsampling)."""
    bilinear = 'bilinear'
    """Bilinear interpolation."""
    cubic = 'cubic'
    """Bicubic interpolation."""
    lanczos = 'lanczos'
    """Lanczos windowed sinc interpolation."""

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
    """JPEG compression."""
    deflate = 'deflate'
    """Deflate compression."""
    lzw = 'lzw'
    """LZW compression."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class CsvFormat(Enum):
    """Type of the position and orientation values in a CSV exterior parameter file."""

    xyz_opk = 1
    """Projected (x, y, z) position and (omega, phi, kappa) orientation."""
    xyz_rpy = 2
    """Projected (x, y, z) position and (roll, pitch, yaw) orientation."""
    lla_opk = 3
    """Geographic (latitude, longitude, altitude) position and (omega, phi, kappa) orientation."""
    lla_rpy = 4
    """Geographic (latitude, longitude, altitude) position and (roll, pitch, yaw) orientation."""

    @property
    def is_xyz(self) -> bool:
        """Whether the format has (x, y, z) position."""
        return self is CsvFormat.xyz_opk or self is CsvFormat.xyz_rpy

    @property
    def is_opk(self) -> bool:
        """Whether the format has (omega, phi, kappa) orientation."""
        return self is CsvFormat.xyz_opk or self is CsvFormat.lla_opk


class RpcRefine(str, Enum):
    """RPC refinement method."""

    shift = 'shift'
    """Pixel coordinate translation."""
    shift_drift = 'shift-drift'
    """Pixel coordinate scale and translation."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class Driver(str, Enum):
    """Raster format drivers."""

    gtiff = 'gtiff'
    """GeoTIFF."""
    cog = 'cog'
    """Cloud Optimised GeoTIFF."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_
