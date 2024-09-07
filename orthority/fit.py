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
from copy import deepcopy
from typing import Sequence

import numpy as np
from rasterio.rpc import RPC

from orthority.camera import RpcCamera
from orthority.enums import RpcRefine

logger = logging.getLogger(__name__)

_default_rpc_refine_method = RpcRefine.shift


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
            ji_rpc_ = np.vstack((ji_rpc[axis], np.ones((ji_rpc.shape[1]))))
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
