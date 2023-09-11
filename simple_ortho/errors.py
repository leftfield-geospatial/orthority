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
class OrthorityError(Exception):
    """ Base exception class. """


class ParamFileError(OrthorityError):
    """ Raised when the formatting of an interior or exterior parameter file is not supported. """


class CrsError(OrthorityError):
    """ Raised when CRS could not be interpreted. """

class CrsMissingError(OrthorityError):
    """ Raised when a required CRS was not specified. """


# TODO add dem error, & other known unknown type errors for Camera and Ortho classes
class DemBandError(OrthorityError):
    """ Raised when the DEM band parameter is out of range. """


class CameraInitError(OrthorityError):
    """ Raised when a camera has no exterior parameters. """