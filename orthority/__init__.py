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

"""Orthorectification toolkit."""
import os
import pathlib
from orthority.camera import create_camera
from orthority.enums import CameraType, Interp, Compress
from orthority.ortho import Ortho

# enable on-demand download and caching of proj transformation grids
os.environ.update(PROJ_NETWORK='ON')

# path to package root TODO: remove with deprecated simple-ortho CLI
if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path.cwd().absolute()
