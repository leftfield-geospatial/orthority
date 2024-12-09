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

# enable on-demand download and caching of proj transformation grids (NB must be done before
# importing rasterio)
# TODO: can this be called in CLI, then document and leave it up to the API user to call themselves?
#  it needs to be done before importing rasterio though...  Also, it is worth investigating, and
#  documenting if the cache is permanent and or manually downloading grids with projsync would
#  speed things up.
os.environ.update(PROJ_NETWORK='ON')

import logging
import pathlib

from orthority.enums import Compress, Interp, RpcRefine, Driver
from orthority.factory import FrameCameras, RpcCameras
from orthority.ortho import Ortho
from orthority.pan_sharp import PanSharpen


# Add a NullHandler to the package logger to hide logs by default.  Applications can then add
# their own handler(s).
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# path to package root TODO: remove with deprecated simple-ortho CLI
if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path.cwd().absolute()
