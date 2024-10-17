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

"""Parameter file IO and conversions."""
from __future__ import annotations

import csv
import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from csv import Dialect, DictReader, Sniffer
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any, IO, Sequence

import cv2
import fsspec
import numpy as np
import rasterio as rio
import yaml
from fsspec.core import OpenFile
from rasterio.crs import CRS
from rasterio.errors import CRSError as RioCrsError
from rasterio.transform import RPC
from rasterio.warp import transform
from tqdm.auto import tqdm

from orthority import common
from orthority.enums import CameraType, CsvFormat
from orthority.errors import CrsError, CrsMissingError, ParamError
from orthority.exif import Exif

logger = logging.getLogger(__name__)

# TODO: define custom file types for e.g. str | Path | OpenFile | IO[str] once sphinx bug with
#  linking to external type defs in __init__ type hints is fixed

_opt_frame_schema = {
    CameraType.pinhole: ['sensor_size', 'cx', 'cy'],
    # fmt: off
    CameraType.opencv: [
        'sensor_size', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2',
        's3', 's4', 'tx', 'ty',
    ],
    # fmt: on
    CameraType.brown: ['sensor_size', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'],
    CameraType.fisheye: ['sensor_size', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
}
"""Schema of valid optional parameters for each frame camera type."""

_default_lla_crs = CRS.from_epsg(4979)
"""Default CRS for geographic camera coordinates."""


def _read_osfm_int_param(json_dict: dict) -> dict[str, dict[str, Any]]:
    """Read interior parameters from the ``cameras`` section of an OpenDroneMap / OpenSfM JSON
    dictionary.
    """

    def parse_json_param(json_param: dict, cam_id: str) -> dict[str, Any]:
        """Validate & convert the given JSON dictionary for a single camera."""
        int_param = {}
        for req_key in ['projection_type', 'width', 'height']:
            if req_key not in json_param:
                raise ParamError(f"'{req_key}' is missing for camera '{cam_id}'.")

        # set 'cam_type' from 'projection_type'
        proj_type = json_param.pop('projection_type').lower()
        try:
            int_param['cam_type'] = CameraType.from_odm(proj_type)
        except ValueError:
            raise ParamError(f"Unsupported projection type '{proj_type}'.")

        im_size = (json_param.pop('width'), json_param.pop('height'))
        int_param['im_size'] = im_size

        # read focal length(s) (JSON values are normalised by max of sensor width & height)
        if 'focal' in json_param:
            int_param['focal_len'] = json_param.pop('focal')
        elif 'focal_x' in json_param and 'focal_y' in json_param:
            focal_x, focal_y = json_param.pop('focal_x'), json_param.pop('focal_y')
            int_param['focal_len'] = focal_x if focal_x == focal_y else (focal_x, focal_y)
        else:
            raise ParamError(
                f"'focal', or 'focal_x' and 'focal_y' are missing for camera '{cam_id}'."
            )

        # rename c_x->cx & c_y->cy
        for from_key, to_key in zip(['c_x', 'c_y'], ['cx', 'cy']):
            if from_key in json_param:
                int_param[to_key] = json_param.pop(from_key)

        # validate any remaining optional params, update param_dict & return
        err_keys = set(json_param.keys()).difference(_opt_frame_schema[int_param['cam_type']])
        if len(err_keys) > 0:
            raise ParamError(f"Unsupported parameter(s) {err_keys} for camera '{cam_id}'.")
        int_param.update(**json_param)
        return int_param

    # validate root dict
    try:
        common.validate_collection({str: dict}, json_dict)
    except (TypeError, KeyError, ValueError) as ex:
        # repackage all formatting errors in ParamError for callers
        raise ParamError(str(ex)) from ex

    # parse each set of interior parameters
    int_param_dict = {}
    for cam_id, json_param in json_dict.items():
        cam_id = cam_id[3:] if cam_id.startswith('v2 ') else cam_id
        int_param_dict[cam_id] = parse_json_param(json_param.copy(), cam_id)

    return int_param_dict


def _create_exif_cam_id(exif: Exif) -> str:
    """Return a camera ID string for the given Exif object."""
    prop_list = [prop for prop in [exif.make, exif.model, exif.serial] if prop]
    cam_id = ' '.join(prop_list)
    if len(cam_id) == 0:
        cam_id = 'unknown'
    return cam_id


def _read_exif_int_param(exif: Exif) -> dict[str, dict[str, Any]]:
    """Read interior parameters from an Exif object."""
    # TODO: might there be cases where XMP tags CalibratedFocalLength, CalibratedOpticalCenter* are
    #  present but not DewarpData, and are better than equiv EXIF tags?
    if exif.dewarp:
        if len(exif.dewarp) != 9 or not any(exif.dewarp) or not exif.tag_im_size:
            logger.debug(f"Cannot interpret dewarp data for '{exif.filename}'.")
        else:
            # construct brown camera parameters from dewarp data and return
            cam_dict = dict(
                cam_type=CameraType.brown,
                im_size=exif.im_size,
                focal_len=tuple(exif.dewarp[:2]),
                sensor_size=(float(exif.tag_im_size[0]), float(exif.tag_im_size[1])),
            )
            cam_dict['cx'], cam_dict['cy'] = tuple(
                (np.array(exif.dewarp[2:4]) / max(exif.tag_im_size)).tolist()
            )
            dist_params = dict(zip(['k1', 'k2', 'p1', 'p2', 'k3'], exif.dewarp[-5:]))
            cam_dict.update(**dist_params)
            return {_create_exif_cam_id(exif): cam_dict}

    # construct pinhole camera parameters
    cam_dict = dict(cam_type=CameraType.pinhole, im_size=exif.im_size)
    if exif.focal_len and exif.sensor_size:
        # prefer using focal length and sensor size directly
        cam_dict['focal_len'] = exif.focal_len
        cam_dict['sensor_size'] = exif.sensor_size
    elif exif.focal_len_35:
        logger.debug(
            f"Approximating the focal length for '{exif.filename}' from the 35mm equivalent."
        )
        if exif.sensor_size:
            # scale 35mm focal length to actual focal length in mm, assuming "35mm" = 36mm max
            # sensor dimension
            cam_dict['focal_len'] = max(exif.sensor_size) * exif.focal_len_35 / 36.0
            cam_dict['sensor_size'] = exif.sensor_size
        else:
            # normalise 35mm focal length assuming "35mm" = 36 mm max sensor dimension, and find
            # a normalised sensor size in same units, assuming square pixels
            cam_dict['focal_len'] = exif.focal_len_35 / 36.0
    else:
        raise ParamError(
            f"No focal length & sensor size, or 35mm focal length tags in '{exif.filename}'."
        )

    return {_create_exif_cam_id(exif): cam_dict}


def _read_exif_ext_param(
    exif: Exif, crs: str | CRS, lla_crs: str | CRS
) -> dict[str, dict[str, Any]]:
    """Read exterior parameters from an Exif object."""
    if not exif.lla:
        raise ParamError(f"No latitude, longitude & altitude tags in '{exif.filename}'.")
    if not exif.rpy:
        raise ParamError(f"No camera / gimbal roll, pitch & yaw tags in '{exif.filename}'.")
    rpy = tuple(np.radians(exif.rpy).tolist())
    opk = _rpy_to_opk(rpy, exif.lla, crs, lla_crs=lla_crs)
    xyz = transform(lla_crs, crs, [exif.lla[1]], [exif.lla[0]], [exif.lla[2]])
    xyz = tuple([coord[0] for coord in xyz])
    ext_param = dict(xyz=xyz, opk=opk, camera=_create_exif_cam_id(exif))
    return {exif.filename: ext_param}


def read_oty_int_param(file: str | PathLike | OpenFile | IO[str]) -> dict[str, dict[str, Any]]:
    """
    Read interior parameters for one or more cameras from an :doc:`Orthority interior parameter
    file <../file_formats/oty_int>`.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or a
        file object, opened in text mode (``'rt'``).
    """
    with common.Open(file, 'rt') as f:
        filename = common.get_filename(file)
        try:
            yaml_dict = yaml.safe_load(f)
        except Exception as ex:
            raise ParamError(f"Could not load '{filename}': {str(ex)}") from ex

    def parse_yaml_param(yaml_param: dict, cam_id: str = None) -> dict[str, Any]:
        """Validate & convert the given YAML dictionary for a single camera."""
        # test required keys for all cameras
        for req_key in ['type', 'im_size', 'focal_len']:
            if req_key not in yaml_param:
                raise ParamError(f"'{req_key}' is missing for camera '{cam_id}'.")

        # convert type -> cam_type
        cam_type = yaml_param.pop('type').lower()
        if cam_type not in CameraType.__members__ or cam_type == 'rpc':
            raise ParamError(f"Unsupported frame camera type '{cam_type}'.")
        int_param = dict(cam_type=CameraType(cam_type))

        int_param['im_size'] = tuple(yaml_param.pop('im_size'))

        # pop known legacy keys not supported by Camera.__init__
        yaml_param.pop('name', None)

        # set focal_len & sensor_size
        focal_len = yaml_param.pop('focal_len')
        int_param['focal_len'] = tuple(focal_len) if isinstance(focal_len, list) else focal_len
        if 'sensor_size' in yaml_param:
            int_param['sensor_size'] = tuple(yaml_param.pop('sensor_size'))

        # validate any remaining distortion params, update param_dict & return
        err_keys = set(yaml_param.keys()).difference(_opt_frame_schema[int_param['cam_type']])
        if len(err_keys) > 0:
            raise ParamError(f"Unsupported parameter(s) {err_keys} for camera '{cam_id}'.")
        int_param.update(**yaml_param)
        return int_param

    # warn if in original simple-ortho format
    if 'camera' in yaml_dict:
        warnings.warn(
            "Support for the 'config.yaml' format is deprecated and will be removed in future. "
            "Please switch to the Orthority interior parameter format.",
            category=FutureWarning,
        )
        yaml_dict = yaml_dict['camera']

    # convert to nested dict if in flat / simple-ortho format
    first_value = next(iter(yaml_dict.values()))
    if not isinstance(first_value, dict):
        cam_id = yaml_dict['name'] if 'name' in yaml_dict else 'unknown'
        yaml_dict = {cam_id: yaml_dict}

    # validate root dict
    try:
        common.validate_collection({str: dict}, yaml_dict)
    except (TypeError, KeyError, ValueError) as ex:
        raise ParamError(f"Could not parse '{filename}': {str(ex)}") from ex

    # parse each set of interior parameters
    int_param_dict = {}
    for cam_id, yaml_param in yaml_dict.items():
        try:
            int_param_dict[cam_id] = parse_yaml_param(yaml_param, cam_id)
        except ParamError as ex:
            # repackage error with filename
            raise ParamError(f"Could not parse '{filename}': {str(ex)}") from ex

    return int_param_dict


def read_osfm_int_param(file: str | PathLike | OpenFile | IO[str]) -> dict[str, dict[str, Any]]:
    """
    Read interior parameters for one or more cameras, from an OpenDroneMap :file:`cameras.json` or
    OpenSfM :file:`reconstruction.json` file.

    See the :doc:`format documentation <../file_formats/opensfm>` for supported camera models.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or a
        file object, opened in text mode (``'rt'``).
    """
    with common.Open(file, 'rt') as f:
        filename = common.get_filename(file)
        try:
            json_dict = json.load(f)
        except Exception as ex:
            raise ParamError(f"Could not load '{filename}': {str(ex)}") from ex

    # extract cameras section if file is an OpenSfM reconstruction.json file
    if isinstance(json_dict, list) and len(json_dict) == 1 and 'cameras' in json_dict[0]:
        json_dict = json_dict[0]['cameras']

    try:
        int_param_dict = _read_osfm_int_param(json_dict)
    except ParamError as ex:
        # repackage error with filename
        raise ParamError(f"Could not parse '{filename}': {str(ex)}") from ex

    return int_param_dict


def read_exif_int_param(
    file: str | PathLike | OpenFile | rio.DatasetReader,
) -> dict[str, dict[str, Any]]:
    """
    Read interior parameters for a camera from an image file with EXIF / XMP tags.

    See the :doc:`format documentation <../file_formats/exif_xmp>` for required tags.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object in
        binary mode (``'rb'``), or a dataset reader.
    """
    return _read_exif_int_param(Exif(file))


def read_im_rpc_param(
    files: Sequence[str | PathLike | OpenFile | rio.DatasetReader],
    progress: bool | dict = False,
) -> dict[str, dict[str, Any]]:
    """
    Read RPC camera parameters from :doc:`image file(s) with RPC tags / sidecar file(s)
    <../file_formats/image_rpc>`.

    :param files:
        File(s) to read as a list of paths or URI strings, :class:`~fsspec.core.OpenFile`
        objects in binary mode (``'rb'``), or dataset readers.
    :param progress:
        Whether to display a progress bar monitoring the portion of files read.  Can be set to a
        dictionary of arguments for a custom `tqdm <https://tqdm.github.io/docs/tqdm/>`_ bar.
    """

    def _read_im_rpc_param(
        file: str | PathLike | OpenFile | rio.DatasetReader,
    ) -> dict[str, dict[str, Any]]:
        """Read RPC camera parameters from an image file."""
        filename = common.get_filename(file)
        with common.suppress_no_georef(), rio.Env(GDAL_NUM_THREADS='ALL_CPUS'), common.OpenRaster(
            file, 'r'
        ) as im:
            # TODO: what is the speed of this for a large remote image?  does it just read the
            #  metadata, or the whole image?
            im_size = (im.width, im.height)
            rpc: RPC = im.rpcs

        if rpc is None:
            raise ParamError(f"No RPC parameters found in '{filename}'.")
        rpc_param = dict(cam_type=CameraType.rpc, im_size=im_size, rpc=rpc.to_dict())
        return {filename: rpc_param}

    # read RPC params in a thread pool, populating rpc_param_dict in same order as files
    rpc_param_dict = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_read_im_rpc_param, file) for file in files]

        # set up progress bar
        if progress is True:
            progress = common.get_tqdm_kwargs(unit='files')
        elif progress is False:
            progress = dict(disable=True, leave=False)

        for future, file in zip(tqdm(futures, **progress), files):
            try:
                rpc_param_dict.update(**future.result())
            except (FileNotFoundError, ParamError):
                executor.shutdown(wait=False)
                raise
            except Exception as ex:
                # TODO: always include text of originating exception in all raise... from ex?
                executor.shutdown(wait=False)
                filename = common.get_filename(file)
                raise RuntimeError(f"Could not read RPC tags from '{filename}'.") from ex

    return rpc_param_dict


def read_oty_rpc_param(file: str | PathLike | OpenFile | IO[str]) -> dict[str, dict[str, Any]]:
    """
    Read RPC parameters for one or more cameras from an :doc:`Orthority RPC file
    <../file_formats/oty_rpc>`.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or a
        file object, opened in text mode (``'rt'``).
    """
    with common.Open(file, 'rt') as f:
        filename = common.get_filename(file)
        try:
            yaml_dict = yaml.safe_load(f)
        except Exception as ex:
            raise ParamError(f"Could not load '{filename}': {str(ex)}") from ex

    # validate file format
    schema = {
        str: dict(
            im_size=[(float, int)] * 2,
            rpc=dict(
                height_off=float,
                height_scale=float,
                lat_off=float,
                lat_scale=float,
                line_den_coeff=[float] * 20,
                line_num_coeff=[float] * 20,
                line_off=(int, float),
                line_scale=(int, float),
                long_off=float,
                long_scale=float,
                samp_den_coeff=[float] * 20,
                samp_num_coeff=[float] * 20,
                samp_off=(int, float),
                samp_scale=(int, float),
            ),
        )
    }

    try:
        common.validate_collection(schema, yaml_dict)
    except (ValueError, TypeError, KeyError) as ex:
        raise ParamError(f"Could not parse '{filename}': {str(ex)}") from ex

    # convert to standard format dict
    rpc_param_dict = {}
    for filename, rpc_param in yaml_dict.items():
        rpc_param_dict[filename] = dict(
            cam_type=CameraType.rpc, im_size=tuple(rpc_param['im_size']), rpc=rpc_param['rpc']
        )

    return rpc_param_dict


def read_im_gcps(
    files: Sequence[str | PathLike | OpenFile | rio.DatasetReader],
    progress: bool | dict = False,
) -> dict[str, list[dict]]:
    """
    Read GCPs from :doc:`tags in image file(s) <../file_formats/image_gcps>`.

    :param files:
        File(s) to read as a list of paths or URI strings, :class:`~fsspec.core.OpenFile`
        objects in binary mode (``'rb'``), or dataset readers.
    :param progress:
        Whether to display a progress bar monitoring the portion of files read.  Can be set to a
        dictionary of arguments for a custom `tqdm <https://tqdm.github.io/docs/tqdm/>`_ bar.
    """

    def _read_im_gcps(
        file: str | PathLike | OpenFile | rio.DatasetReader,
    ) -> dict[str, dict[str, Any]]:
        """Read GCPs from an image file."""
        filename = common.get_filename(file)
        with common.suppress_no_georef(), rio.Env(GDAL_NUM_THREADS='ALL_CPUS'), common.OpenRaster(
            file, 'r'
        ) as im:
            gcps, crs = im.gcps

        if gcps is None or len(gcps) == 0:
            raise ParamError(f"No GCPs found in '{filename}'.")

        # standardise GCP world coordinates in EPSG:4979
        xyz = np.array([(gcp.x, gcp.y, gcp.z) for gcp in gcps]).T
        if crs != _default_lla_crs:
            xyz = np.array(transform(crs, _default_lla_crs, *xyz))

        # Convert to standard format dicts. GDAL leaves it to the application to interpret GCP
        # pixel coordinates as upper left or center conventions:
        # https://gdal.org/user/raster_data_model.html#gcps.  This assumes image GCPs are in
        # center of pixel coordinate convention.
        oty_gcps = []
        for gcp, xyz in zip(gcps, xyz.T):
            gcp = dict(ji=(gcp.col, gcp.row), xyz=tuple(xyz.tolist()), id=gcp.id, info=gcp.info)
            oty_gcps.append(gcp)

        return {filename: oty_gcps}

    # read GCPs in a thread pool, populating gcp_dict in same order as files
    gcp_dict = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_read_im_gcps, file) for file in files]

        # set up progress bar
        if progress is True:
            progress = common.get_tqdm_kwargs(unit='files')
        elif progress is False:
            progress = dict(disable=True, leave=False)

        for future, file in zip(tqdm(futures, **progress), files):
            try:
                gcp_dict.update(**future.result())
            except (FileNotFoundError, ParamError):
                executor.shutdown(wait=False)
                raise
            except Exception as ex:
                executor.shutdown(wait=False)
                filename = common.get_filename(file)
                raise RuntimeError(f"Could not read GCPs from '{filename}'.") from ex

    return gcp_dict


def read_oty_gcps(file: str | PathLike | OpenFile | IO[str]) -> dict[str, list[dict]]:
    """
    Read GCPs for one or more images from an :doc:`Orthority GCP file <../file_formats/oty_gcps>`.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or a
        file object, opened in text mode (``'rt'``).
    """
    with common.Open(file, 'rt') as f:
        filename = common.get_filename(file)
        try:
            json_dict = json.load(f)
        except Exception as ex:
            raise ParamError(f"Could not load '{filename}': {str(ex)}") from ex

    # validate file format
    schema = dict(
        type='FeatureCollection',
        features=[
            dict(
                type='Feature',
                properties=dict(ji=[(int, float)] * 2, filename=str, id=None, info=None),
                geometry=dict(type='Point', coordinates=[(int, float)] * 3),
            )
        ],
    )

    try:
        common.validate_collection(schema, json_dict)
    except (ValueError, TypeError, KeyError) as ex:
        raise ParamError(f"Could not parse '{filename}': {str(ex)}") from ex

    # convert to standard format dict
    gcp_dict = {}
    for feat_dict in json_dict['features']:
        prop_dict = feat_dict['properties']
        filename = prop_dict['filename']
        xyz = tuple(feat_dict['geometry']['coordinates'])
        gcp = dict(ji=tuple(prop_dict['ji']), xyz=xyz, id=prop_dict['id'], info=prop_dict['info'])

        if filename not in gcp_dict:
            gcp_dict[filename] = [gcp]
        else:
            gcp_dict[filename].append(gcp)

    return gcp_dict


def write_int_param(
    file: str | PathLike | OpenFile | IO[str],
    int_param_dict: dict[str, dict[str, Any]],
    overwrite: bool = False,
) -> None:
    """
    Write interior parameters to an :doc:`Orthority interior parameter file
    <../file_formats/oty_int>`.

    :param file:
        File to write.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or
        a file object, opened in text mode (``'wt'``).
    :param int_param_dict:
        Interior parameters to write.
    :param overwrite:
        Whether to overwrite the file if it exists.
    """
    # convert 'cam_type' key to 'type' & make the converted item the first in the dict
    yaml_dict = {}
    for cam_id, int_param in int_param_dict.items():
        yaml_param = {}
        if 'cam_type' in int_param:
            yaml_param['type'] = int_param['cam_type'].value
        yaml_param.update(**{k: v for k, v in int_param.items() if k != 'cam_type'})
        yaml_dict[cam_id] = yaml_param

    with common.Open(file, 'wt', overwrite=overwrite) as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, indent=4, default_flow_style=None)


def write_ext_param(
    file: str | PathLike | OpenFile | IO[str],
    ext_param_dict: dict[str, dict[str, Any]],
    crs: str | CRS,
    overwrite: bool = False,
) -> None:
    """
    Write exterior parameters to an :doc:`Orthority exterior parameter file
    <../file_formats/oty_ext>`.

    :param file:
        File to write.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or
        a file object, opened in text mode (``'wt'``).
    :param ext_param_dict:
        Exterior parameters to write.
    :param crs:
        CRS of the world coordinate system as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.
    :param overwrite:
        Whether to overwrite the file if it exists.
    """
    feat_list = []
    lla_crs = _default_lla_crs

    try:
        crs = CRS.from_string(crs) if isinstance(crs, str) else crs
    except RioCrsError as ex:
        raise CrsError(f"Could not interpret 'crs': {str(ex)}")
    if not crs.is_projected:
        raise CrsError(f"'crs' should be a projected system.")

    for src_file, ext_param in ext_param_dict.items():
        xyz = ext_param['xyz']
        lla = transform(crs, lla_crs, [xyz[0]], [xyz[1]], [xyz[2]])
        lla = [lla[0][0], lla[1][0], lla[2][0]]  # (lon, lat) order for GeoJSON point
        props_dict = dict(
            filename=src_file, camera=ext_param['camera'], xyz=xyz, opk=ext_param['opk']
        )
        geom_dict = dict(type='Point', coordinates=list(lla))
        feat_dict = dict(type='Feature', properties=props_dict, geometry=geom_dict)
        feat_list.append(feat_dict)

    json_dict = dict(type='FeatureCollection', world_crs=crs.to_string(), features=feat_list)

    with common.Open(file, 'wt', overwrite=overwrite) as f:
        json.dump(json_dict, f, indent=4)


def write_rpc_param(
    file: str | PathLike | OpenFile | IO[str],
    rpc_param_dict: dict[str, dict[str, Any]],
    overwrite: bool = False,
) -> None:
    """
    Write RPC parameters to an :doc:`Orthority RPC parameter file <../file_formats/oty_rpc>`.

    :param file:
        File to write.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or
        a file object, opened in text mode (``'wt'``).
    :param rpc_param_dict:
        RPC parameters to write.
    :param overwrite:
        Whether to overwrite the file if it exists.
    """
    # copy rpc_param_dict to yaml_dict without 'cam_type' item(s)
    yaml_dict = {}
    for filename, rpm_param in rpc_param_dict.items():
        yaml_dict[filename] = {k: v for k, v in rpm_param.items() if k != 'cam_type'}

    with common.Open(file, 'wt', overwrite=overwrite) as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, indent=4, default_flow_style=None)


def write_gcps(
    file: str | PathLike | OpenFile | IO[str],
    gcp_dict: dict[str, list[dict]],
    overwrite: bool = False,
) -> None:
    """
    Write GCPs to an :doc:`Orthority GCP file <../file_formats/oty_gcps>`.

    :param file:
        File to write.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or
        a file object, opened in text mode (``'wt'``).
    :param gcp_dict:
        GCPs to write.
    :param overwrite:
        Whether to overwrite the file if it exists.
    """
    feat_list = []
    for filename, gcps in gcp_dict.items():
        for gcp in gcps:
            props_dict = dict(ji=list(gcp['ji']), filename=filename, id=gcp['id'], info=gcp['info'])
            geom_dict = dict(type='Point', coordinates=list(gcp['xyz']))
            feat_dict = dict(type='Feature', properties=props_dict, geometry=geom_dict)
            feat_list.append(feat_dict)

    json_dict = dict(type='FeatureCollection', features=feat_list)

    with common.Open(file, 'wt', overwrite=overwrite) as f:
        json.dump(json_dict, f, indent=4)


def _rpy_to_rotation(rpy: tuple[float, float, float]) -> np.ndarray:
    """Convert the given (roll, pitch, yaw) angles in radians to a rotation matrix."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    roll, pitch, yaw = rpy
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R_z.dot(R_y).dot(R_x)


def _opk_to_rotation(opk: tuple[float, float, float]) -> np.ndarray:
    """Convert the given (omega, phi, kappa) angles in radians to a rotation matrix."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    omega, phi, kappa = opk
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
    )
    R_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    R_z = np.array(
        [[np.cos(kappa), -np.sin(kappa), 0], [np.sin(kappa), np.cos(kappa), 0], [0, 0, 1]]
    )
    return R_x.dot(R_y).dot(R_z)


def _rotation_to_opk(R: np.ndarray) -> tuple[float, float, float]:
    """Convert the given rotation matrix to the (omega, phi, kappa) angles in radians."""
    # see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    omega = np.arctan2(-R[1, 2], R[2, 2])
    phi = np.arcsin(R[0, 2])
    kappa = np.arctan2(-R[0, 1], R[0, 0])
    return omega, phi, kappa


def _aa_to_opk(aa: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert the given angle axis vector (OpenSfM / OpenCV convention) to (omega, phi,
    kappa) angles in radians (PATB convention)."""
    # convert ODM angle/axis to rotation matrix (see
    # https://github.com/mapillary/OpenSfM/issues/121)
    R = cv2.Rodrigues(np.array(aa))[0].T

    # rotate from OpenSfM / OpenCV to PATB convention
    R = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

    # extract OPK from R
    omega, phi, kappa = _rotation_to_opk(R)
    return omega, phi, kappa


def _rpy_to_opk(
    rpy: tuple[float, float, float],
    lla: tuple[float, float, float],
    crs: CRS,
    lla_crs: CRS,
    C_bB: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Convert (roll, pitch, yaw) to (omega, phi, kappa) angles for a given CRS.

    Coordinate conventions are as `defined by Pix4D
    <https://s3.amazonaws.com/mics.pix4d.com/KB/documents
    /Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.>`__

    Note this conversion assumes ``crs`` is orthogonal and uniform scale (cartesian) in the
    vicinity of ``lla``.  ``crs`` should be conformal, and have its central meridian(s) close to
    ``lla`` for this assumption to be accurate.

    :param rpy:
        (roll, pitch, yaw) angles to rotate from body to navigation coordinates (radians).
    :param lla:
        (latitude, longitude, altitude) geographic coordinates of the camera (in units of
        ``lla_crs``).
    :param crs:
        CRS of the world coordinate system as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.
    :param lla_crs:
        CRS of the ``lla`` geographic coordinates as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.
    :param C_bB:
        Optional camera to body rotation matrix.  Defaults to:
        ``[[0, 1, 0], [1, 0, 0], [0, 0, -1]]``, which describes typical drone geometry.

    :return:
        (omega, phi, kappa) angles in radians, to rotate from camera (PATB convention) to world
        coordinates.
    """
    # Adapted from the OpenSfM exif module
    # https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py and Pix4D doc
    # https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf.

    # Note that what the Pix4D reference calls Object (E), and Image (B) coordinates, orthority
    # calls world and camera coordinates respectively.

    # RPY rotates from body (b) to navigation (n) coordinates, and OPK from camera ('Image' B) to
    # world ('Object' E) coordinates.  Body (b) coordinates are defined as x->front, y->right,
    # z->down, and navigation (n) coordinates as x->N, y->E, z->down.  Navigation is a tangent
    # plane type system and shares its origin with body (b) at ``lla``. Camera (B) coordinates
    # are in the Pix4D / PATB convention i.e. x->right, y->top, z->back (looking through the
    # camera at the scene), and are aligned with & centered on the body (b) system.  World (E)
    # coordinates are ENU i.e. x->E, y->N, z->up.
    lla = np.array(lla)
    crs = CRS.from_string(crs) if isinstance(crs, str) else crs

    # find rotation matrix C_nb, to rotate from body (b) to navigation (n) coordinates.
    C_nb = _rpy_to_rotation(rpy)

    # find rotation matrix C_En, to rotate from navigation (n) to world (object E) coordinates
    delta = 1e-7
    lla1 = lla + (delta, 0, 0)
    lla2 = lla - (delta, 0, 0)

    # p1 & p2 must be in the world CRS, not ECEF as might be understood from the references
    p1 = np.array(transform(lla_crs, crs, [lla1[1]], [lla1[0]], [lla1[2]])).squeeze()
    p2 = np.array(transform(lla_crs, crs, [lla2[1]], [lla2[0]], [lla2[2]])).squeeze()

    # approximate the relative alignment of world and navigation systems in the vicinity of lla
    x_np = p1 - p2
    m = np.linalg.norm(x_np)
    x_np /= m  # unit vector in navigation system N direction
    z_np = np.array([0, 0, -1]).T
    y_np = np.cross(z_np, x_np)
    C_En = np.array([x_np, y_np, z_np]).T

    # C_bB is the rotation from camera (image B) to body (b) coordinates
    C_bB = np.array(C_bB) if C_bB is not None else np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    # combine C_En, C_nb, C_bB to find rotation from camera (B) to world (E) coordinates.
    C_EB = C_En.dot(C_nb).dot(C_bB)

    # return OPK angles extracted from C_EB
    omega, phi, kappa = _rotation_to_opk(C_EB)
    return omega, phi, kappa


class FrameReader(ABC):
    """
    Base frame camera parameter reader.

    :param crs:
        CRS of the world coordinate system as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.
    :param lla_crs:
        CRS of input geographic coordinates (if any), as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.
    """

    @abstractmethod
    def __init__(self, crs: str | CRS = None, lla_crs: str | CRS = _default_lla_crs) -> None:
        self._crs, self._lla_crs = self._parse_crss(crs, lla_crs)

    @staticmethod
    def _parse_crss(crs: str | CRS, lla_crs: str | CRS) -> tuple[CRS, CRS]:
        """Validate and convert CRSs."""
        if crs:
            try:
                crs = CRS.from_string(crs) if isinstance(crs, str) else crs
            except RioCrsError as ex:
                raise CrsError(f"Could not interpret 'crs': {str(ex)}")
            if not crs.is_projected:
                raise CrsError(f"'crs' should be a projected system.")

        if lla_crs:
            try:
                lla_crs = CRS.from_string(lla_crs) if isinstance(lla_crs, str) else lla_crs
            except RioCrsError as ex:
                raise CrsError(f"Could not interpret 'lla_crs': {str(ex)}")
            if not lla_crs.is_geographic:
                raise CrsError(f"'lla_crs' should be a geographic system.")
        return crs, lla_crs

    @property
    def crs(self) -> CRS:
        """CRS of the world coordinate system."""
        return self._crs

    @abstractmethod
    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        """Read exterior camera parameters."""
        pass


class CsvReader(FrameReader):
    """
    Exterior parameter reader for a CSV file.

    Reads tabular data from a CSV file with a row per source image and column fields for image
    file name, camera position, orientation and ID.

    See the :doc:`CSV documentation <../file_formats/csv>` for details on supported fields and
    formats.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or a
        file object, opened in text mode (``'rt'``).
    :param crs:
        CRS of the world coordinate system as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.  If set to ``None`` (the default), and the file
        contains (``x``, ``y``, ``z``) world coordinate positions, a CRS can be provided via a
        :file:`.prj` sidecar file.  If set to ``None``, and the file contains (``latitude``,
        ``longitude``, ``altitude``) and (``roll``, ``pitch``, ``yaw``) fields, a UTM CRS will be
        auto-determined. If the file contains (``latitude``, ``longitude``, ``altitude``) or
        (``roll``, ``pitch``, ``yaw``) fields (but not both), ``crs`` should be supplied.
    :param lla_crs:
        Geographic CRS associated with any (``latitude``, ``longitude``, ``altitude``) position
        and/or (``roll``, ``pitch``, ``yaw``) values in the file (as an EPSG, WKT or proj4
        string; or :class:`~rasterio.crs.CRS` object).
    :param fieldnames:
        List of names specifying the CSV fields.  If set to ``None`` (the default), names will be
        read from the file header if it exists.  If ``fieldnames`` is supplied, any existing file
        header is ignored.  See the :doc:`CSV documentation <../file_formats/csv>` for recognised
        and required field names.
    :param dialect:
        :class:`~csv.Dialect` object specifying the CSV delimiter, quote character etc. If set to
        ``None`` (the default), this is auto-detected from the file.
    :param radians:
        Whether orientation angles are in radians (``True``), or degrees (``False``).
    """

    _type_schema = dict(
        filename=str,
        x=float,
        y=float,
        z=float,
        latitude=float,
        longitude=float,
        altitude=float,
        omega=float,
        phi=float,
        kappa=float,
        roll=float,
        pitch=float,
        yaw=float,
        camera=lambda x: str(x) if x else x,
    )
    _legacy_fieldnames = ['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa']

    def __init__(
        self,
        file: str | PathLike | OpenFile | IO[str],
        crs: str | CRS = None,
        lla_crs: str | CRS = _default_lla_crs,
        fieldnames: Sequence[str] = None,
        dialect: Dialect = None,
        radians: bool = False,
        **kwargs,
    ) -> None:
        # TODO: allow other coordinate conventions for opk / rpy (bluh, odm, patb)
        FrameReader.__init__(self, crs, lla_crs=lla_crs)
        self._file = file
        self._radians = radians

        # get / create an OpenFile object to use when finding a .prj file
        self._ofile = None
        if isinstance(file, OpenFile):
            self._ofile = file
        elif isinstance(file, (str, PathLike)):
            self._ofile = fsspec.open(os.fspath(file), 'rt')

        # read file once into a buffer (newline=None works around some delimiter detection problems
        # with newline='')
        with common.Open(self._ofile or file, 'rt', newline=None) as f:
            self._buffer = StringIO(f.read())

        try:
            self._fieldnames, self._dialect, self._has_header, self._format = self._parse_file(
                self._buffer, fieldnames=fieldnames, dialect=dialect
            )
        except ParamError as ex:
            # repackage error with filename
            raise ParamError(f"Could not parse '{common.get_filename(file)}': {str(ex)}") from ex

        self._crs = self._crs or self._get_crs()

    @staticmethod
    def _parse_fieldnames(fieldnames: Sequence[str]) -> CsvFormat:
        """Validate a list of header or user field names, and return the corresponding
        :class:`CsvFormat`.
        """
        if 'filename' not in fieldnames:
            raise ParamError(f"Fields should include 'filename'.")

        has_xyz = {'x', 'y', 'z'}.issubset(fieldnames)
        has_lla = {'latitude', 'longitude', 'altitude'}.issubset(fieldnames)
        has_opk = {'omega', 'phi', 'kappa'}.issubset(fieldnames)
        has_rpy = {'roll', 'pitch', 'yaw'}.issubset(fieldnames)

        if not (has_xyz or has_lla):
            raise ParamError(
                f"Fields should include 'x', 'y' & 'z', or 'latitude', 'longitude' & 'altitude'."
            )
        if not (has_opk or has_rpy):
            raise ParamError(
                f"Fields should include 'omega', 'phi' & 'kappa', or 'roll', 'pitch' & 'yaw'."
            )

        # dictionary with key = (has_xyz, has_opk) and value = CsvFormat
        format_dict = {
            (True, True): CsvFormat.xyz_opk,
            (True, False): CsvFormat.xyz_rpy,
            (False, True): CsvFormat.lla_opk,
            (False, False): CsvFormat.lla_rpy,
        }
        return format_dict[(has_xyz, has_opk)]

    @staticmethod
    def _parse_file(
        buffer: StringIO, fieldnames: Sequence[str] = None, dialect: Dialect = None
    ) -> tuple[Sequence[str], Dialect, bool, CsvFormat]:
        """Determine and validate the CSV file format."""

        def strip_lower_strlist(str_list: Sequence[str]) -> list[str]:
            """Strip and lower the case of a string list."""
            return [str_item.strip().lower() for str_item in str_list]

        # read a sample of the csv file
        buffer.seek(0)
        sample = buffer.read(10000)

        # auto-detect dialect if not provided
        sniffer = Sniffer()
        if not dialect:
            dialect = sniffer.sniff(sample, delimiters=',;: \t')
            dialect.skipinitialspace = True

        # find field names and parse to determine format
        has_header = sniffer.has_header(sample)
        if not fieldnames:
            if has_header:  # read field names from header
                fieldnames = next(iter(csv.reader(sample.splitlines(), dialect=dialect)))
            else:  # assume fields in legacy format
                fieldnames = CsvReader._legacy_fieldnames
        fieldnames = strip_lower_strlist(fieldnames)
        csv_fmt = CsvReader._parse_fieldnames(fieldnames)

        return fieldnames, dialect, has_header, csv_fmt

    def _find_lla_rpy_crs(self) -> CRS:
        """Return a UTM CRS covering the mean of the camera positions in a
        :attr:`~CsvFormat.lla_rpy` format file.
        """
        latlons = []
        self._buffer.seek(0)
        reader = DictReader(self._buffer, fieldnames=self._fieldnames, dialect=self._dialect)
        if self._has_header:
            _ = next(iter(reader))
        for row in reader:
            row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
            latlon = (row['latitude'], row['longitude'])
            latlons.append(latlon)

        mean_latlon = np.array(latlons).mean(axis=0)
        return common.utm_crs_from_latlon(*mean_latlon)

    def _get_crs(self) -> CRS:
        """Read / auto-determine and validate a CRS when no user CRS was supplied."""
        crs = None
        filename = common.get_filename(self._file)
        if self._format is CsvFormat.xyz_opk or self._format is CsvFormat.xyz_rpy:
            if self._ofile:
                # read CRS of xyz positions / opk orientations from .prj file, if it exists
                prj_path = self._ofile.path[: self._ofile.path.rfind('.')] + '.prj'
                prj_name = Path(prj_path).name
                try:
                    with self._ofile.fs.open(prj_path, 'rt') as f:
                        crs_str = f.read()
                    crs = CRS.from_string(crs_str)

                    if not crs.is_projected:
                        raise ParamError(f"CRS in '{prj_name}' should be a projected system.")

                    logger.debug(f"Using '{prj_name}' CRS: '{crs.to_string()}'")
                except FileNotFoundError as ex:
                    logger.debug(f"Could not open '{prj_name}': {str(ex)}")
                except RioCrsError as ex:
                    raise ParamError(f"Could not interpret CRS in '{prj_name}': {str(ex)}")
            else:
                # a file object was passed to __init__ so the CSV file path / URI is unknown and a
                # .prj file cannot be found
                logger.debug(f"Cannot read a .prj file with a CSV file object.")

            if not crs and self._format is CsvFormat.xyz_rpy:
                raise CrsMissingError(f"'crs' should be specified for positions in '{filename}'.")

        elif self._format is CsvFormat.lla_rpy:
            # find a UTM CRS to transform the lla positions & rpy orientations into
            crs = self._find_lla_rpy_crs()
            logger.debug(f"Using auto UTM CRS: '{crs.to_proj4()}'")

        elif self._format is CsvFormat.lla_opk:
            # a user-supplied opk CRS is required to project lla into
            raise CrsMissingError(f"'crs' should be specified for orientations in '{filename}'.")

        return crs

    def _convert(self, row: dict[str, float], radians=False) -> tuple[tuple, tuple]:
        """Convert a CSV row dictionary to (x, y, z) position, and (omega, phi,
        kappa) orientation.
        """
        if self._format.is_xyz:
            xyz = (row['x'], row['y'], row['z'])
        else:
            lla = (row['latitude'], row['longitude'], row['altitude'])
            xyz = transform(self._lla_crs, self._crs, [lla[1]], [lla[0]], [lla[2]])
            xyz = tuple([coord[0] for coord in xyz])

        if self._format.is_opk:
            opk = (row['omega'], row['phi'], row['kappa'])
            opk = opk if radians else tuple(np.radians(opk).tolist())
        else:
            rpy = (row['roll'], row['pitch'], row['yaw'])
            rpy = rpy if radians else tuple(np.radians(rpy).tolist())
            if self._format.is_xyz:
                lla = transform(self._crs, self._lla_crs, [xyz[0]], [xyz[1]], [xyz[2]])
                lla = (lla[1][0], lla[0][0], lla[2][0])  # x, y order -> lat, lon order
            opk = _rpy_to_opk(rpy, lla, self._crs, lla_crs=self._lla_crs)

        return xyz, opk

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        ext_param_dict = {}
        self._buffer.seek(0)
        reader = DictReader(self._buffer, fieldnames=self._fieldnames, dialect=self._dialect)

        if self._has_header:  # read header
            _ = next(iter(reader))

        for row in reader:
            filename = Path(row['filename']).name
            row = {k: self._type_schema[k](v) for k, v in row.items() if k in self._type_schema}
            xyz, opk = self._convert(row, radians=self._radians)
            ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=row.get('camera', None))

        return ext_param_dict


class OsfmReader(FrameReader):
    """
    Interior and exterior parameter reader for an OpenSfM :file:`reconstruction.json` file.

    See the :doc:`format documentation <../file_formats/opensfm>` for supported camera models.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or a
        file object, opened in text mode (``'rt'``).
    :param crs:
        CRS of the world coordinate system as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.  If set to ``None`` (the default), a UTM CRS will be
        auto-determined.
    :param lla_crs:
        CRS of the ``reference_lla`` value in the :file:`reconstruction.json` file as an EPSG, WKT
        or proj4 string; or :class:`~rasterio.crs.CRS` object.
    """

    # TODO: OSfM reconstruction is in a topocentric system, so the transfer of 3D cartesian
    #  positions & rotations into a 2D+1D UTM CRS is an approximation, with similar issues to the
    #  Pix4D RPY->OPK conversion.  Ideally the orthorectification should also happen in this
    #  topocentric system, with the DEM being transformed into it. Then the orthorectified image
    #  can be reprojected to UTM.
    def __init__(
        self,
        file: str | PathLike | OpenFile | IO[str],
        crs: str | CRS = None,
        lla_crs: str | CRS = CRS.from_epsg(4326),
        **kwargs,
    ) -> None:
        FrameReader.__init__(self, crs=crs, lla_crs=lla_crs)
        self._json_dict = self._read_json_dict(file)
        try:
            # read interior parameters now for early validation of the 'cameras' section
            self._int_param_dict = _read_osfm_int_param(self._json_dict['cameras'])
        except ParamError as ex:
            # repackage error with filename
            raise ParamError(f"Could not parse '{common.get_filename(file)}': {str(ex)}") from ex

        if not self._crs:
            self._crs = self._find_utm_crs()
            logger.debug(f"Using auto UTM CRS: '{self._crs.to_proj4()}'")

    @staticmethod
    def _read_json_dict(file: str | PathLike | OpenFile | IO[str]) -> dict[str, dict]:
        """Read and validate the reconstruction JSON file."""
        with common.Open(file, 'rt') as f:
            filename = common.get_filename(file)
            try:
                json_data = json.load(f)
            except Exception as ex:
                raise ParamError(f"Could not load '{filename}': {str(ex)}") from ex

        schema = [
            dict(
                cameras=dict,  # items validated in _read_osfm_int_param()
                shots={str: dict(rotation=[float] * 3, translation=[float] * 3, camera=str)},
                reference_lla=dict(latitude=float, longitude=float, altitude=float),
            )
        ]
        try:
            common.validate_collection(schema, json_data)
        except (ValueError, TypeError, KeyError) as ex:
            raise ParamError(f"Could not parse '{filename}': {str(ex)}") from ex

        # keep root schema keys and delete the rest
        json_dict = {k: json_data[0][k] for k in schema[0].keys()}
        del json_data
        return json_dict

    def _find_utm_crs(self) -> CRS:
        """Return a UTM CRS that covers the reconstruction reference point."""
        ref_lla = self._json_dict['reference_lla']
        ref_lla = (ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        return common.utm_crs_from_latlon(*ref_lla[:2])

    def read_int_param(self) -> dict[str, dict[str, Any]]:
        """Read interior camera parameters."""
        return self._int_param_dict

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        # transform reference coordinates to the world CRS
        ref_lla = self._json_dict['reference_lla']
        ref_lla = (ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'])
        ref_xyz = transform(
            self._lla_crs,
            self._crs,
            [ref_lla[1]],
            [ref_lla[0]],
            [ref_lla[2]],
        )
        ref_xyz = tuple([coord[0] for coord in ref_xyz])

        ext_param_dict = {}
        for filename, shot_dict in self._json_dict['shots'].items():
            # convert  reconstruction 'translation' and 'rotation' to oty exterior params,
            # adapted from ODM: https://github.com/OpenDroneMap/ODM/blob/master/opendm/shots.py
            R = cv2.Rodrigues(np.array(shot_dict['rotation']))[0].T
            delta_xyz = -R.dot(shot_dict['translation'])
            xyz = tuple((ref_xyz + delta_xyz).tolist())
            # rotate camera coords from OpenSfM / OpenCV to PATB convention
            R_ = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            opk = _rotation_to_opk(R_)
            cam_id = shot_dict['camera']
            cam_id = cam_id[3:] if cam_id.startswith('v2 ') else cam_id
            ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=cam_id)

        return ext_param_dict


class ExifReader(FrameReader):
    """
    Interior and exterior parameter reader for image file(s) with EXIF / XMP tags.

    See the :doc:`format documentation <../file_formats/exif_xmp>` for required tags.

    :param files:
        File(s) to read as a list of paths or URI strings, :class:`~fsspec.core.OpenFile`
        objects in binary mode (``'rb'``), or dataset readers.
    :param crs:
        CRS of the world coordinate system as an EPSG, WKT or proj4 string; or
        :class:`~rasterio.crs.CRS` object.  If set to ``None`` (the default), a UTM CRS will be
        auto-determined.
    :param lla_crs:
        CRS of geographic camera coordinates in the EXIF / XMP tags as an EPSG, WKT or proj4 string;
        or :class:`~rasterio.crs.CRS` object.
    :param progress:
        Whether to display a progress bar monitoring the portion of files read.  Can be set to a
        dictionary of arguments for a custom `tqdm <https://tqdm.github.io/docs/tqdm/>`_ bar.
    """

    def __init__(
        self,
        files: Sequence[str | PathLike | OpenFile | rio.DatasetReader],
        crs: str | CRS = None,
        lla_crs: str | CRS = _default_lla_crs,
        progress: bool | dict = False,
        **kwargs,
    ) -> None:
        FrameReader.__init__(self, crs, lla_crs)
        files = files if isinstance(files, Iterable) else [files]
        self._exif_dict = self._read_exif(files, progress)

        if not self._crs and len(self._exif_dict) > 0:
            self._crs = self._find_utm_crs()
            logger.debug(f"Using auto UTM CRS: '{self._crs.to_proj4()}'")

    @staticmethod
    def _read_exif(
        files: Sequence[str | PathLike | OpenFile | rio.DatasetReader], progress: bool | dict
    ) -> dict[str, Exif]:
        """Return a dictionary of Exif objects for the given images."""
        # read exif tags in thread pool, populating exif_dict in same order as files
        exif_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(Exif, file) for file in files]

            # set up progress bar
            if progress is True:
                progress = common.get_tqdm_kwargs(unit='files')
            elif progress is False:
                progress = dict(disable=True, leave=False)

            for future, file in zip(tqdm(futures, **progress), files):
                try:
                    exif_obj = future.result()
                except FileNotFoundError:
                    executor.shutdown(wait=False)
                    raise
                except Exception as ex:
                    executor.shutdown(wait=False)
                    filename = common.get_filename(file)
                    raise RuntimeError(f"Could not read EXIF tags from '{filename}'.") from ex

                exif_dict[exif_obj.filename] = exif_obj

        return exif_dict

    def _find_utm_crs(self) -> CRS:
        """Return a UTM CRS that covers the mean of the camera positions."""
        # TODO: use weighted sum as in OpenSfM, then use ExifReader.crs in oty odm, see :
        #  https://github.com/mapillary/OpenSfM/blob/c6b5acef9376a75b87414d900c258ef876a6413a/opensfm/dataset.py#L985
        llas = []
        for e in self._exif_dict.values():
            if not e.lla:
                raise ParamError(f"No latitude, longitude & altitude tags in '{e.filename}'.")
            llas.append(e.lla)

        mean_latlon = np.array(llas)[:, :2].mean(axis=0)
        return common.utm_crs_from_latlon(*mean_latlon)
        # return CRS.from_proj4(f'+proj=ortho +lat_0={mean_latlon[0]} +lon_0={mean_latlon[1]}'

    def read_int_param(self) -> dict[str, dict[str, Any]]:
        """Read interior camera parameters."""
        int_param_dict = {}
        for filename, exif in self._exif_dict.items():
            int_param = _read_exif_int_param(exif)
            int_param_dict.update(**int_param)
        return int_param_dict

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        ext_param_dict = {}
        for filename, exif in self._exif_dict.items():
            ext_param_dict.update(
                **_read_exif_ext_param(exif, crs=self._crs, lla_crs=self._lla_crs)
            )
        return ext_param_dict


class OtyReader(FrameReader):
    """
    Exterior parameter reader for an :doc:`Orthority exterior parameter file
    <../file_formats/oty_ext>`.

    :param file:
        File to read.  Can be a path or URI string, an :class:`~fsspec.core.OpenFile` object or a
        file object, opened in text mode (``'rt'``).
    """

    def __init__(self, file: str | PathLike | OpenFile | IO[str], **kwargs) -> None:
        FrameReader.__init__(self)
        self._crs, self._json_dict = self._read_json_dict(file, self._crs)

    @staticmethod
    def _read_json_dict(file: str | PathLike | OpenFile | IO[str], crs: CRS) -> tuple[CRS, dict]:
        """Read and validate the GeoJSON file."""
        with common.Open(file, 'rt') as f:
            filename = common.get_filename(file)
            try:
                json_dict = json.load(f)
            except Exception as ex:
                raise ParamError(f"Could not load '{filename}': {str(ex)}") from ex

        schema = dict(
            type='FeatureCollection',
            world_crs=str,
            features=[
                dict(
                    type='Feature',
                    properties=dict(
                        filename=str, camera=None, xyz=[(int, float)] * 3, opk=[(int, float)] * 3
                    ),
                    geometry=dict(type='Point', coordinates=[(int, float)]),
                )
            ],
        )

        try:
            common.validate_collection(schema, json_dict)
        except (ValueError, TypeError, KeyError) as ex:
            raise ParamError(f"Could not parse '{filename}': {str(ex)}") from ex

        if not crs:
            try:
                crs = CRS.from_string(json_dict['world_crs'])
            except RioCrsError as ex:
                raise ParamError(f"Could not interpret CRS in '{filename}': {str(ex)}")

        return crs, json_dict

    def read_ext_param(self) -> dict[str, dict[str, Any]]:
        ext_param_dict = {}
        for feat_dict in self._json_dict['features']:
            prop_dict = feat_dict['properties']
            filename = prop_dict['filename']
            ext_parm = dict(
                xyz=tuple(prop_dict['xyz']), opk=tuple(prop_dict['opk']), camera=prop_dict['camera']
            )
            ext_param_dict[filename] = ext_parm
        return ext_param_dict
