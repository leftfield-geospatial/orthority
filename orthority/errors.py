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


class OrthorityError(Exception):
    """Base exception class."""


class ParamFileError(OrthorityError):
    """Raised when there is a problem reading interior or exterior parameter file."""


class CrsMissingError(OrthorityError):
    """Raised when a required CRS was not specified."""


class CameraInitError(OrthorityError):
    """Raised when a camera's exterior parameters have not been initialised."""
