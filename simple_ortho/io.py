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
import json
import yaml
import logging
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict

import numpy as np
import rasterio as rio

from simple_ortho.enums import CameraType
from simple_ortho.exif import Exif

logger = logging.getLogger(__name__)


def read_yaml_int_param(filename: Path) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more cameras from a yaml file.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the yaml file to read.

    Returns
    -------
    dict
        A dictionary of camera id - camera interior parameters, key - value pairs.
    """
    with open(filename, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    def parse_cam_dict(cam_dict: Dict, cam_id: str) -> Dict:
        for req_key in ['type', 'focal_len', 'sensor_size']:
            if req_key not in cam_dict:
                raise ValueError(f'`{req_key}` is not defined for camera `{cam_id}` in {filename.name}.')

        cam_type = cam_dict.pop('type').lower()
        try:
            cam_dict['cam_type'] = CameraType(cam_type)
        except ValueError:
            raise ValueError(f'Unsupported camera type `{cam_type}` for camera `{cam_id}` in {filename.name}.')

        cam_dict.pop('name', None)
        cam_dict.pop('im_size', None)
        return cam_dict

    # flatten if in original simple-ortho format
    yaml_dict = yaml_dict.get('camera', yaml_dict)

    # convert to nested dict if in flat format
    first_value = next(iter(yaml_dict.values()))
    if not isinstance(first_value, dict):
        cam_id = yaml_dict.get('name', None) or 'default'
        yaml_dict = {cam_id: yaml_dict}

    # parse each set of camera parameters
    cams_dict = {}
    for cam_id, cam_dict in yaml_dict.items():
        cams_dict[cam_id] = parse_cam_dict(cam_dict, cam_id)
    return cams_dict


def read_json_int_param(filename: Path) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more cameras, from an ODM cameras.json or OpenSFM camera_models.json file.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the ODM / OpenSFM json file to read.

    Returns
    -------
    dict
        A dictionary of camera id - camera interior parameters, key - value pairs.
    """
    with open(filename, 'r') as f:
        json_dict = json.load(f)

    def parse_json_cam_dict(json_cam_dict: Dict, cam_id: str) -> Dict:
        """ Return a dict of interior camera parameters given an ODM / OpenSFM json dict for single camera. """
        cam_dict = {}
        for req_key in ['projection_type', 'width', 'height']:
            if req_key not in json_cam_dict:
                raise ValueError(f'`{req_key}` is not defined for camera `{cam_id}` in {filename.name}.')

        # read focal length(s) (json values are normalised by sensor width)
        if 'focal' in json_cam_dict:
            cam_dict['focal_len'] = json_cam_dict.pop('focal')
        elif 'focal_x' in json_cam_dict and 'focal_y' in json_cam_dict:
            cam_dict['focal_len'] = (json_cam_dict.pop('focal_x'), json_cam_dict.pop('focal_y'))
        else:
            raise ValueError(
                f'`focal`, or `focal_x` and `focal_y` are not defined for camera `{cam_id}` in {filename.name}.'
            )

        proj_type = json_cam_dict.pop('projection_type').lower()
        try:
            cam_dict['cam_type'] = CameraType.from_odm(proj_type)
        except ValueError:
            raise ValueError(f'Unsupported camera type `{proj_type}` for camera `{cam_id}` in {filename.name}.')

        image_size = (json_cam_dict.pop('width'), json_cam_dict.pop('height'))

        # TODO: normalised by width or max(width, height) ?
        # find a normalised sensor size in same units as focal_len, assuming square pixels (ODM / OpenSFM json files do
        # not define sensor size)
        cam_dict['sensor_size'] = (1, image_size[1] / image_size[0])

        # rename c_x->cx & c_y->cy
        for from_key, to_key in zip(['c_x', 'c_y'], ['cx', 'cy']):
            if from_key in json_cam_dict:
                cam_dict[to_key] = json_cam_dict.pop(from_key)

        # update param_dict with any remaining distortion coefficient parameters and return
        cam_dict.update(**json_cam_dict)
        return cam_dict

    cams_dict = {}
    for cam_id, json_cam_dict in json_dict.items():
        cams_dict[cam_id] = parse_json_cam_dict(json_cam_dict, cam_id)

    return cams_dict


def read_exif_int_param(exif: Exif) -> Dict[str, Dict]:
    """
    Read interior parameters for a camera from EXIF tags.

    Parameters
    ----------
    exif: simple_ortho.exif.Exif
        :class:`~simple_ortho.exif.Exif` instance to read.

    Returns
    -------
    dict
        A dictionary of a camera id - camera interior parameters, key - value pair.
    """
    cam_dict = dict(cam_type=CameraType.pinhole, name=exif.camera_name, im_size=exif.image_size)

    if exif.focal_len and exif.sensor_size:
        # prefer using focal length and sensor size directly
        cam_dict['focal_len'] = exif.focal_len
        cam_dict['sensor_size'] = exif.sensor_size
    elif exif.focal_len_35:
        logger.warning(f'Approximating the focal length for {exif.filename.name} from the 35mm equivalent.')
        if exif.sensor_size:
            # scale 35mm focal length to actual focal length in mm, assuming "35mm" = 36mm sensor width
            cam_dict['focal_len'] = exif.sensor_size[0] * exif.focal_len_35 / 36.
            cam_dict['sensor_size'] = exif.sensor_size
        else:
            # normalise 35mm focal length assuming "35mm" = 36 mm sensor width, and find a normalised sensor size
            # in same units, assuming square pixels
            cam_dict['focal_len'] = exif.focal_len_35 / 36.
            cam_dict['sensor_size'] = (1, exif.image_size[1] / exif.image_size[0])
    else:
        raise ValueError(
            f'EXIF tags in {exif.filename} should define a focal length and sensor size, or 35mm equivalent focal '
            f'length.'
        )

    cam_id = f'{exif.camera_name or "unknown"} pinhole {exif.focal_len:.4f}'
    return {cam_id: cam_dict}


def read_int_param(filename: Union[str, Path]) -> Dict[str, Dict]:
    """
    Read interior parameters for one or more cameras, yaml file, ODM / OpenSFM json file, or JPEG / TIFF file with EXIF
    tags.

    Parameters
    ----------
    filename: str, Path
        Path of file to read.

    Returns
    -------
    dict
        A dictionary of camera id - camera interior parameters, key - value pairs.
    """
    filename = Path(filename)
    if filename.suffix.lower() in ['.yaml', '.yml']:
        return read_yaml_int_param(filename)
    elif filename.suffix.lower() == '.json':
        return read_json_int_param(filename)
    elif filename.suffix.lower() in ['.jpg', '.jpeg', '.tif', '.tiff']:
        return read_exif_int_param(Exif(filename))
    else:
        raise ValueError(f'Unrecognised file type: {filename.name}')

