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

import os
import pathlib
from orthority.camera import create_camera, PinholeCamera, OpenCVCamera, BrownCamera, FisheyeCamera
from orthority.enums import CameraType, CsvFormat, Interp, Compress
from orthority.ortho import Ortho
from orthority.io import (
    read_osfm_int_param,
    read_oty_int_param,
    OtyReader,
    ExifReader,
    OsfmReader,
    CsvReader,
    write_int_param,
    write_ext_param,
)

# enable on-demand download and caching of proj transformation grids
os.environ.update(PROJ_NETWORK='ON')

# path to package root TODO: remove with deprecated simple-ortho CLI
if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path.cwd().absolute()
