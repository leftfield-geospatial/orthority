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
"""Camera model fitting and refinement."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from copy import deepcopy
from math import ceil
from typing import Any

import cv2
import numpy as np
from rasterio.crs import CRS
from rasterio.rpc import RPC
from rasterio.warp import transform

from orthority import param_io
from orthority.camera import FrameCamera, RpcCamera
from orthority.enums import CameraType, RpcRefine
from orthority.errors import OrthorityWarning

logger = logging.getLogger(__name__)

_default_rpc_refine_method = RpcRefine.shift

_frame_dist_params = {k: v[3:] for k, v in param_io._opt_frame_schema.items()}
"""Distortion coefficient names in OpenCV ordering for each frame camera model."""
_frame_num_params = {k: len(v) + 6 for k, v in _frame_dist_params.items()}
"""Number of distortion coefficient and exterior parameters for each frame camera model (excludes 
focal length(s) and principal point).
"""


def refine_rpc(
    rpc: RPC | dict, gcps: Sequence[dict], method: RpcRefine = _default_rpc_refine_method
) -> dict:
    """
    Refine an RPC model with GCPs.

    Finds the least squares solution to the 'shift' or 'shift-and-drift' bias compensation
    refinements described in https://doi.org/10.1016/j.isprsjprs.2005.11.001 for a single image.
    Refinements are incorporated into the provided RPC model.  The approach is suited to narrow
    field of view satellite imagery.

    :param rpc:
        RPC parameters as a :class:`~rasterio.rpc.RPC` object or dictionary.
    :param gcps:
        List of GCP dictionaries, e.g. an item value in a dictionary returned by
        :func:`~orthority.param_io.read_im_gcps` or :func:`~orthority.param_io.read_oty_gcps`.
    :param method:
        Refinement method.

    :return:
        Refined RPC parameters as a dictionary.
    """
    # TODO:
    #  - robustness to outliers, maybe with optional inclusion of scipy
    #  - full affine model, or the method described in https://doi.org/10.14358/PERS.75.9.1083
    #  - fit evaluation with warnings
    method = RpcRefine(method)
    camera = RpcCamera(None, rpc)
    rpc = rpc.to_dict() if isinstance(rpc, RPC) else rpc
    min_gcps = 1 if method is RpcRefine.shift else 2
    if len(gcps) < min_gcps:
        raise ValueError(f"At least {min_gcps} are required for the '{method}' method.")

    def _norm_ji(rpc: dict, ji: np.ndarray) -> np.ndarray:
        """Normalise pixel coordinates with the given RPC scale / offset parameters."""
        norm_ji = ji.T - (rpc['samp_off'], rpc['line_off'])
        norm_ji /= (rpc['samp_scale'], rpc['line_scale'])
        return norm_ji.T

    # normalised GCP pixel coordinates
    ji_gcp = np.array([gcp['ji'] for gcp in gcps]).T
    ji_gcp = _norm_ji(rpc, ji_gcp)

    # normalised RPC pixel coordinates of GCP geographic coordinates
    xyz = np.array([(gcp['xyz']) for gcp in gcps]).T
    ji_rpc = camera.world_to_pixel(xyz)
    ji_rpc = _norm_ji(rpc, ji_rpc)

    # find the (partial) affine transform to "refine" RPC pixel coordinates
    refine_tform = np.eye(2, 3)
    if method == RpcRefine.shift:
        off = ji_gcp - ji_rpc
        refine_tform[:, -1] = off.mean(axis=1)
    else:
        for axis in range(2):
            ji_rpc_ = np.vstack((ji_rpc[axis], np.ones(ji_rpc.shape[1])))
            (m, c), res, rank, s = np.linalg.lstsq(ji_rpc_.T, ji_gcp[axis], rcond=None)
            refine_tform[axis, axis] = m
            refine_tform[axis, 2] = c

    if logger.getEffectiveLevel() <= logging.DEBUG:
        # log the refinement transform and accuracy
        ji_rpc_ = np.vstack((ji_rpc, np.ones(ji_rpc.shape[1])))
        ji_refine = refine_tform.dot(ji_rpc_)
        err = ((ji_gcp - ji_refine).T * (rpc['samp_scale'], rpc['line_scale'])).T  # pixels
        err_dist = np.sum(err**2, axis=0)
        logger.debug(f"Refinement transform: \n{refine_tform}")
        logger.debug(f"Refinement RMSE (pixels): {np.sqrt(np.mean(err_dist)):.4f}")
        logger.debug(
            f"Refinement min - max error (pixels): "
            f"{np.sqrt(np.min(err_dist)):.5f} - {np.sqrt(np.max(err_dist)):.4f}"
        )

    # incorporate the refinement transform into the original RPC coefficients
    refined_rpc = deepcopy(rpc)
    for axis, num_key, den_key in zip(
        range(2), ['samp_num_coeff', 'line_num_coeff'], ['samp_den_coeff', 'line_den_coeff']
    ):
        refined_rpc[num_key] = np.array(refined_rpc[num_key]) * refine_tform[axis, axis]
        refined_rpc[num_key] += np.array(refined_rpc[den_key]) * refine_tform[axis, 2]
        refined_rpc[num_key] = refined_rpc[num_key].tolist()
    return refined_rpc


def _gcps_to_cv_coords(
    gcp_dict: dict[str, Sequence[dict]], crs: str | CRS | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Convert a GCP dictionary to list of pixel coordinate arrays, a list of world coordinate
    arrays and a reference world coordinate position which world coordinate arrays have been
    offset relative to.
    """
    crs = CRS.from_string(crs) if isinstance(crs, str) else crs
    # form lists of pixel and world coordinate arrays
    jis = []
    xyzs = []
    for gcps in gcp_dict.values():
        ji = np.array([gcp['ji'] for gcp in gcps])
        xyz = np.array([gcp['xyz'] for gcp in gcps])
        if crs:
            xyz = np.array(transform(CRS.from_epsg(4979), crs, *(xyz.T))).T
        jis.append(ji.astype('float32'))
        xyzs.append(xyz)

    # offset world coordinates and convert to float32
    ref_xyz = np.vstack(xyzs).mean(axis=0)
    xyzs = [(xyz - ref_xyz).astype('float32') for xyz in xyzs]
    return jis, xyzs, ref_xyz


def _fit_frame(
    cam_type: CameraType,
    im_size: tuple[int, int],
    gcp_dict: dict[str, Sequence[dict]],
    crs: str | CRS | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Fit a frame camera to GCPs.

    :param cam_type:
        Camera type to fit.
    :param im_size:
        Image (width, height) in pixels.
    :param gcp_dict:
        GCP dictionary e.g. as returned by :func:`~orthority.param_io.read_im_gcps` or
        :func:`~orthority.param_io.read_oty_gcps`.
    :param crs:
        CRS of the camera world coordinate system as an EPSG, proj4 or WKT string,
        or :class:`~rasterio.crs.CRS` object.  If set to ``None`` (the default), GCPs are assumed
        to be in the world coordinate CRS, and are not transformed.  Otherwise, GCPs are
        transformed from geographic WGS84 coordinates to this CRS if it is supplied.

    :return:
        Interior parameter and exterior parameter dictionaries.
    """
    # TODO: is it better to use cv2.initCameraMatrix2D and cv2.solvePnp(flags=cv2.SOLVEPNP_SQPNP)
    #  rather than cv2.calibrateCamera when num pts <=4

    # check there are at least 4 GCPs per image
    min_gcps = min(len(gcps) for gcps in gcp_dict.values())
    if min_gcps < 4:
        raise ValueError('At least four GCPs are needed per image.')

    # check the total number of GCPs is enough to fit cam_type
    ttl_gcps = sum(len(gcps) for gcps in gcp_dict.values())
    req_gcps = max(4, ceil((1 + _frame_num_params[cam_type]) / 2))
    if ttl_gcps < req_gcps:
        raise ValueError(
            f"A total of at least {req_gcps} GCPs are required to fit the '{cam_type!r}' model."
        )

    # convert GCPs to OpenCV compatible lists of arrays
    jis, xyzs, ref_xyz = _gcps_to_cv_coords(gcp_dict, crs=crs)

    # check if GCPs are co-planar (replicates OpenCV's test)
    zs = np.vstack([xyz[:, 2] for xyz in xyzs])
    z_mean, z_std = np.mean(zs), np.std(zs)
    if z_mean > 1e-5 or z_std > 1e-5:
        raise ValueError('GCPs should be co-planar to fit interior parameters.')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-15)
    warn_str = (
        "A total of at least {0} GCPs are required to estimate all '{1!r}' parameters, but there "
        "are {2}.  The initial intrinsic matrix will not be globally optimised."
    )

    # setup calibration flags & params based on cam_type and number of GCPs
    if cam_type is not CameraType.fisheye:
        calib_func = cv2.calibrateCamera
        # force square pixels always
        flags = cv2.CALIB_FIX_ASPECT_RATIO

        # fix initial intrinsic matrix if there are not enough GCPs to estimate all params (+3 is
        # for 1 focal length and 2 principal points)
        req_gcps = ceil((_frame_num_params[cam_type] + 3 + 1) / 2)
        if ttl_gcps < req_gcps:
            warnings.warn(
                warn_str.format(req_gcps, cam_type, ttl_gcps),
                category=OrthorityWarning,
                stacklevel=2,
            )
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_FOCAL_LENGTH

        if cam_type is CameraType.pinhole:
            # fix distortion at zero
            flags |= (
                cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
            )
        elif cam_type is CameraType.opencv:
            # enable full OpenCV model
            flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL

    else:
        calib_func = cv2.fisheye.calibrate
        # the oty fisheye camera does not have skew/alpha and CALIB_RECOMPUTE_EXTRINSIC improves
        # accuracy
        flags = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC

        # Fix initial intrinsic matrix if there are not enough GCPs to estimate all params (+4 is
        # for 2 focal lengths (you can't fix fisheye aspect ratio) and 2 principal points).
        # (Note that cv2.fisheye.calibrate() behaves differently to cv2.fisheye.calibrate(): it
        # still runs with ttl_gcps < req_gcps, apparently fixing K and distortion coefficients.)
        # TODO: cv2.fisheye.calibrate() seems to require a min of 5 GCPs.  confirm & change the
        #  above check for that, and consider removing the flag changes below which seem to be
        #  handled internally by cv2.fisheye.calibrate()
        req_gcps = ceil((_frame_num_params[cam_type] + 4 + 1) / 2)
        if ttl_gcps < req_gcps:
            warnings.warn(
                warn_str.format(req_gcps, cam_type, ttl_gcps),
                category=OrthorityWarning,
                stacklevel=2,
            )
            flags |= cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT | cv2.fisheye.CALIB_FIX_FOCAL_LENGTH

        # convert coords to cv2.fisheye format
        xyzs = [xyz[None, :] for xyz in xyzs]
        jis = [ji[None, :] for ji in jis]

    # calibrate
    err, K, dist_param, rs, ts = calib_func(
        xyzs, jis, im_size, None, None, flags=flags, criteria=criteria
    )
    logger.debug(
        f"RMS reprojection error for fit of '{cam_type}' model to {ttl_gcps} GCPs: {err:.4f}"
    )

    # convert opencv to oty format interior & exterior params
    cam_id = f'{cam_type!r}_fit_to_{ttl_gcps}_gcps'
    c_xy = (K[0, 2], K[1, 2]) - (np.array(im_size) - 1) / 2
    c_xy /= max(im_size)
    dist_param = dict(zip(_frame_dist_params[cam_type], dist_param.squeeze().tolist()))

    int_param = dict(
        cam_type=cam_type,
        im_size=im_size,
        focal_len=(K[0, 0], K[1, 1]),
        sensor_size=(float(im_size[0]), float(im_size[1])),
        cx=c_xy[0],
        cy=c_xy[1],
        **dist_param,
    )
    int_param_dict = {cam_id: int_param}

    ext_param_dict = {}
    for filename, t, r in zip(gcp_dict.keys(), ts, rs):
        xyz, opk = param_io._cv_ext_to_oty_ext(t, r, ref_xyz=ref_xyz)
        ext_param_dict[filename] = dict(xyz=xyz, opk=opk, camera=cam_id)

    return int_param_dict, ext_param_dict


def fit_frame_exterior(
    int_param_dict: dict[str, dict[str, Any]],
    gcp_dict: dict[str, Sequence[dict]],
    crs: str | CRS | None = None,
):
    """
    Fit frame camera exterior parameters to GCPs, given the camera's interior parameters.

    :param int_param_dict:
        Interior parameter dictionary.
    :param gcp_dict:
        GCP dictionary e.g. as returned by :func:`~orthority.param_io.read_im_gcps` or
        :func:`~orthority.param_io.read_oty_gcps`.
    :param crs:
        CRS of the camera world coordinate system as an EPSG, proj4 or WKT string,
        or :class:`~rasterio.crs.CRS` object.  If set to ``None`` (the default), GCPs are assumed
        to be in the world coordinate CRS, and are not transformed.  Otherwise, GCPs are
        transformed from geographic WGS84 coordinates to this CRS if it is supplied.

    :return:
        Exterior parameter dictionary.
    """
    if len(int_param_dict) > 1:
        warnings.warn(
            f"Refining the first of {len(int_param_dict)} cameras defined in the interior "
            f"parameter dictionary.",
            category=OrthorityWarning,
            stacklevel=2,
        )
    cam_id = next(iter(int_param_dict.keys()))
    int_param = next(iter(int_param_dict.values()))

    # check there are at least 3 GCPs per image
    min_gcps = min(len(gcps) for gcps in gcp_dict.values())
    if min_gcps < 3:
        raise ValueError('At least three GCPs are needed per image.')

    # get initial intrinsic matrix
    K = FrameCamera._get_intrinsic(
        int_param['im_size'],
        int_param['focal_len'],
        int_param.get('sensor_size'),
        int_param.get('cx', 0.0),
        int_param.get('cy', 0.0),
    )

    # get initial distortion coefficients
    dist_names = _frame_dist_params[int_param['cam_type']]
    dist_param = [int_param.get(dn, 0.0) for dn in dist_names]
    dist_param = np.array(dist_param) if dist_param else None

    # convert GCPs to OpenCV compatible lists of arrays
    jis, xyzs, ref_xyz = _gcps_to_cv_coords(gcp_dict, crs=crs)

    # fit exterior parameters (SOLVEPNP_SQPNP is globally optimal so does not need further refining)
    ext_param_dict = {}
    for filename, xyz, ji in zip(gcp_dict.keys(), xyzs, jis):
        _, r, t = cv2.solvePnP(xyz, ji, K, dist_param, flags=cv2.SOLVEPNP_SQPNP)
        xyz_, opk = param_io._cv_ext_to_oty_ext(t, r, ref_xyz=ref_xyz)
        ext_param_dict[filename] = dict(xyz=xyz_, opk=opk, camera=cam_id)

    return ext_param_dict
