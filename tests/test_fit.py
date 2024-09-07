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

import numpy as np
import pytest
from rasterio.rpc import RPC

from orthority import fit
from orthority.camera import RpcCamera
from orthority.enums import RpcRefine


@pytest.mark.parametrize('shift, drift', [((5.0, 10.0), None), ((5.0, 10.0), (1.2, 0.8))])
def test_refine_rpc(
    rpc: dict, im_size: tuple[int, int], shift: tuple[float, float], drift: tuple[float, float]
):
    """Test ``refine_rpc()`` correctly refines an RPC model by testing it against refinement GCPs.
    """
    # create affine transform to realise shift & drift
    method = RpcRefine.shift if not drift else RpcRefine.shift_drift
    drift = (1.0, 1.0) if not drift else drift
    refine_tform = np.eye(2, 3)
    refine_tform[:, -1] = shift
    refine_tform[:, :2] = np.diag(drift)

    # generate GCPs randomly spread over the image, with known pixel shift/drift
    n = 10
    camera = RpcCamera(im_size, rpc)
    ji_rpc = (np.random.rand(2, n) - 0.5) * np.array([im_size]).T
    z = rpc['height_off'] / 2 + (np.random.rand(ji_rpc.shape[1]) * rpc['height_off'] / 4)
    xyz = camera.pixel_to_world_z(ji_rpc, z)

    ji_rpc_ = np.vstack((ji_rpc, np.ones(ji_rpc.shape[1])))
    ji_gcp = refine_tform.dot(ji_rpc_)
    gcps = []
    for ji_pt, xyz_pt in zip(ji_gcp.T, xyz.T):
        gcp = dict(ji=tuple(ji_pt.tolist()), xyz=tuple(xyz_pt.tolist()))
        gcps.append(gcp)

    # refine RPC with GCPs
    refined_rpc = fit.refine_rpc(rpc, gcps, method=method)

    # test refined RPC model against original GCPs
    refined_camera = RpcCamera(im_size, refined_rpc)
    xyz_test = refined_camera.pixel_to_world_z(ji_gcp, z)
    assert xyz_test == pytest.approx(xyz, abs=1e-6)


def test_refine_rpc_type(rpc: dict):
    """Test ``refine_rpc()`` works with an RPC dict or :class:`~rasterio.rpc.RPC` object."""
    gcp = dict(
        ji=(rpc['samp_off'], rpc['line_off']),
        xyz=(rpc['long_off'], rpc['lat_off'], rpc['height_off']),
    )
    fit.refine_rpc(rpc, [gcp])
    fit.refine_rpc(RPC(**rpc), [gcp])


@pytest.mark.parametrize('method, min_gcps', [(RpcRefine.shift, 1), (RpcRefine.shift_drift, 2)])
def test_refine_num_gcps(rpc: dict, im_size: tuple[int, int], method: RpcRefine, min_gcps: int):
    """Test ``refine_rpc()`` works with the minimum allowed GCPs and raises an error otherwise."""
    camera = RpcCamera(im_size, rpc)
    gcps = []
    for i in range(min_gcps):
        ji = np.array([rpc['samp_off'], rpc['line_off']]) + np.random.randn(2) * 10
        xyz = camera.pixel_to_world_z(ji.reshape(-1, 1), z=rpc['height_off']).T
        gcp = dict(ji=(*ji,), xyz=(*xyz.squeeze(),))
        gcps.append(gcp)

    fit.refine_rpc(RPC(**rpc), gcps, method=method)
    with pytest.raises(ValueError) as ex:
        fit.refine_rpc(rpc, gcps[:-1], method=method)
    assert 'At least' in str(ex.value)
