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

"""Camera models for projecting between 3D world and 2D pixel coordinates."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import product
from os import PathLike

import cv2
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.crs import CRS
from rasterio.rpc import RPC
from rasterio.transform import GCPTransformer, GroundControlPoint, RPCTransformer
from rasterio.warp import transform as warp

from orthority import common
from orthority.enums import CameraType, Interp
from orthority.errors import CameraInitError, OrthorityError, OrthorityWarning
from orthority.param_io import _opk_to_rotation

logger = logging.getLogger(__name__)


class Camera(ABC):
    """Base camera class."""

    # data types accepted by cv2.remap()
    _valid_dtypes = ('uint8', 'uint16', 'int16', 'float32', 'float64')
    # cv2.remap() maximum image dimension
    _shrt_max = (1 << 15) - 1

    @abstractmethod
    def __init__(self, **kwargs):
        self._im_size = None

    @property
    def im_size(self) -> tuple[int, int]:
        """Image (width, height) in pixels."""
        return self._im_size

    @staticmethod
    def _validate_world_coords(xyz: np.ndarray) -> None:
        """Utility function to validate world coordinate dimensions."""
        if not (xyz.ndim == 2 and xyz.shape[0] == 3):
            raise ValueError("'xyz' should be a 3xN 2D array.")
        if xyz.dtype != np.float64:
            raise ValueError("'xyz' should have 'float64' data type.")

    @staticmethod
    def _validate_pixel_coords(ji: np.ndarray) -> None:
        """Utility function to validate pixel coordinate dimensions."""
        if not (ji.ndim == 2 and ji.shape[0] == 2):
            raise ValueError("'ji' should be a 2xN 2D array.")

    @staticmethod
    def _validate_z(z: np.ndarray, ji: np.ndarray) -> None:
        """Utility function to validate z against pixel coordinate dimensions."""
        if isinstance(z, np.ndarray) and (
            z.ndim != 1 or (z.shape[0] != 1 and ji.shape[1] != 1 and z.shape[0] != ji.shape[1])
        ):
            raise ValueError(
                "'z' should be a single value or 1-by-N array where 'ji' is 2-by-N or 2-by-1."
            )

    @staticmethod
    def _get_remap_slices(
        shape: tuple[int, int], maps: tuple[np.ndarray, np.ndarray], pad: int = 0
    ) -> tuple[slice, slice]:
        """Return (row, col) slices to crop an array of shape ``shape`` to the minimum and
        maximum pixel coordinates in ``maps`` with a padding of ``pad``.
        """
        slices = []
        for map_, dim in zip(maps[::-1], shape, strict=True):
            min_coord, max_coord = np.nanmin(map_), np.nanmax(map_)
            if np.isnan(min_coord):
                # return empty slices when maps are all nans
                start = stop = 0
            else:
                # convert from center-pixel coords to start/stop array indexes
                start, stop = np.round(min_coord), np.round(max_coord) + 1
                # pad and clip to image bounds
                start = np.clip(start - pad, a_min=0, a_max=dim).astype(int)
                stop = np.clip(stop + pad, a_min=0, a_max=dim).astype(int)
            slices.append(slice(start, stop))

        return slices

    @staticmethod
    def _per_band_remap(
        src: np.ndarray,
        maps: tuple[np.ndarray, np.ndarray],
        interp: Interp,
        dst: np.ndarray,
        map_offsets: tuple[float, float] | None = None,
        change_maps_inplace: bool = True,
        maps_contain_nans: bool = True,
    ) -> np.ndarray:
        """Remap ``src`` to ``dst`` band-by-band, offsetting ``maps`` with ``map_offsets`` if
        provided, and changing nans to work with ``cv2.remap()`` if ``maps_contain_nans`` is
        ``True``.
        """
        # skip remapping if dst will be unchanged
        if (np.array(src.shape) == 0).any():
            return dst

        # copy maps if they will be changed and change_maps_inplace is False
        if (not change_maps_inplace) and ((map_offsets is not None) or maps_contain_nans):
            maps = [m.copy() for m in maps]

        if map_offsets is not None:
            for map_, map_offset in zip(maps, map_offsets, strict=True):
                map_ -= map_offset

        if maps_contain_nans:
            for map_ in maps:
                map_mask = np.isnan(map_)
                if map_mask.all():
                    # skip remapping if dst will be unchanged
                    return dst
                # convert nans to -1 as cv2.remap() maps nans to 0 (the first src pixel) on some
                # packages/platforms see
                # https://answers.opencv.org/question/1057/behavior-of-not-a-number-nan-values-in-remap/
                map_[map_mask] = -1

        # cv2.remap() does not support RasterIO band ordering, does not support images with >4
        # bands, and is slower on an OpenCV ordered image (including re-ordering) than on a
        # RasterIO ordered image looping over bands
        for src_band, dst_band in zip(src, dst, strict=True):
            _ = cv2.remap(
                src_band, *maps, interp.to_cv(), dst=dst_band, borderMode=cv2.BORDER_TRANSPARENT
            )
        return dst

    @staticmethod
    def _remap(
        src: np.ndarray,
        maps: tuple[np.ndarray, np.ndarray],
        interp: Interp,
        dst: np.ndarray,
        change_maps_inplace: bool = True,
        maps_contain_nans: bool = True,
    ):
        """``cv2.remap()`` wrapper that works around the SHRT_MAX limit. ``src`` and ``dst``
        should be 3D arrays with RasterIO dimension ordering. ``maps`` should be 2D arrays.
        """
        # error checking (specific to this function and not done by cv2.remap())
        assert src.ndim == dst.ndim == 3
        assert maps[0].ndim == 2
        assert dst.shape[0] == src.shape[0]
        assert dst.shape[1:] == maps[0].shape
        src_shape = np.array(src.shape[-2:])
        map_shape = np.array(maps[0].shape)

        if (src_shape < Camera._shrt_max).all() and (map_shape < Camera._shrt_max).all():
            # remap without cropping or chunking
            Camera._per_band_remap(
                src,
                maps,
                interp,
                dst,
                change_maps_inplace=change_maps_inplace,
                maps_contain_nans=maps_contain_nans,
            )
            return dst

        # find the map coordinate ranges
        pad = 4  # worst case padding for 8x8 lanczos kernel
        src_slices = Camera._get_remap_slices(src_shape, maps, pad=pad)
        src_crop_shape = np.array([s.stop - s.start for s in src_slices])
        if (src_crop_shape == 0).any():
            # skip remapping if dst will remain unchanged
            return dst

        # find a number of chunks (along either map dimension) that will give map coordinate
        # ranges less than 32K (*5 is a heuristic to allow for non-uniform spacing in the maps
        # grids)
        dim_chunks = (src_crop_shape / (Camera._shrt_max - 1)).max()
        dim_chunks = dim_chunks * 5 if dim_chunks > 1 else 1

        # find a chunk shape that is smaller than the 32K limit, and results in a map chunk
        # coordinate range smaller than 32K (for cropping src)
        chunk_range_shape = np.ceil(map_shape / dim_chunks).astype(int)
        chunk_size_shape = np.fmin(map_shape, (Camera._shrt_max - 1, Camera._shrt_max - 1))
        chunk_shape = np.fmin(chunk_size_shape, chunk_range_shape)

        if (chunk_shape == map_shape).all():
            # remap with src cropped to map ranges without chunking
            Camera._per_band_remap(
                src[..., src_slices[0], src_slices[1]],
                maps,
                interp,
                dst,
                map_offsets=[s.start for s in src_slices[::-1]],
                change_maps_inplace=change_maps_inplace,
                maps_contain_nans=maps_contain_nans,
            )
            return dst

        num_chunks = np.ceil(map_shape / chunk_shape).prod().astype(int)
        warnings.warn(
            f'Remapping with {num_chunks} chunks of shape: {tuple(chunk_shape.tolist())}.',
            category=OrthorityWarning,
            stacklevel=2,
        )

        # remap with src cropped to chunked map ranges
        for start_i, start_j in product(
            range(0, map_shape[0], chunk_shape[0]), range(0, map_shape[1], chunk_shape[1])
        ):
            map_slices = [
                slice(start_i, min(start_i + chunk_shape[0], map_shape[0])),
                slice(start_j, min(start_j + chunk_shape[1], map_shape[1])),
            ]
            map_chunks = [m[map_slices[0], map_slices[1]] for m in maps]
            dst_chunk = dst[:, map_slices[0], map_slices[1]]
            src_slices = Camera._get_remap_slices(src_shape, map_chunks, pad=pad)
            src_crop_shape = np.array([s.stop - s.start for s in src_slices])
            if (src_crop_shape >= Camera._shrt_max).any():
                raise RuntimeError(
                    f"Cannot work around the 'cv2.remap()' {Camera._shrt_max} size limit."
                )

            Camera._per_band_remap(
                src[:, src_slices[0], src_slices[1]],
                map_chunks,
                interp,
                dst_chunk,
                map_offsets=[s.start for s in src_slices[::-1]],
                change_maps_inplace=change_maps_inplace,
                maps_contain_nans=maps_contain_nans,
            )
        return dst

    def _validate_image(self, im_array: np.ndarray) -> None:
        """Utility function to validate an image dtype and dimensions for remapping."""
        if im_array.dtype.name not in Camera._valid_dtypes:
            raise ValueError(f"'im_array' data type '{im_array.dtype.name}' not supported.")
        if not im_array.ndim == 3:
            raise ValueError("'im_array' should have 3 dimensions.")
        if im_array.shape[-2:] != self.im_size[::-1]:
            warnings.warn(
                "'im_array' does not have the same size as the camera 'im_size'.",
                category=OrthorityWarning,
                stacklevel=2,
            )

    def _pixel_to_world_surf(
        self,
        ji: np.ndarray,
        z: np.ndarray,
        transform: rio.Affine,
        interp: str | Interp = Interp.cubic,
        min_z: float | None = None,
        max_z: float | None = None,
    ) -> np.ndarray:
        """Return the world coordinate intersections of rays defined by pixel coordinates ``ji``,
        with the height array (DEM) ``z``.
        """
        if z.dtype.name not in Camera._valid_dtypes:
            raise ValueError(f"'z' data type '{z.dtype.name}' not supported.")

        # create a transform from world (x, y) to center (j, i) pixel coordinates
        inv_transform = ~(transform * rio.Affine.translation(0.5, 0.5))

        # find / initialise z surface minimum and maximum
        min_z = min_z if min_z is not None else np.nanmin(z)
        max_z = max_z if max_z is not None else np.nanmax(z)

        # find world boundary at z_min and z_max
        min_xyz = self.pixel_to_world_z(ji, min_z)
        max_xyz = self.pixel_to_world_z(ji, max_z)

        # heuristic limit on ray length to conserve memory
        max_ray_len = 2 * np.sqrt(np.square(z.shape, dtype='int64').sum()).astype('int')
        xyz = np.zeros((3, ji.shape[1]))

        # find z surface (x, y, z) world coordinate intersections for each (j, i) pixel
        # coordinate in ji
        for pi in range(0, ji.shape[1]):
            src_pt, start_xyz, stop_xyz = ji[:, pi], max_xyz[:, pi], min_xyz[:, pi]

            # create world points along the src_pt ray with (x, y) stepsize <= z resolution
            ray_len = np.abs((stop_xyz - start_xyz)[:2].squeeze() / (transform[0], transform[4]))
            ray_len = min(np.ceil(ray_len.max()).astype('int') + 1, max_ray_len)
            ray_z = np.linspace(max_z, min_z, ray_len)
            ray_xyz = self.pixel_to_world_z(src_pt.reshape(-1, 1), ray_z)

            # find the z surface pixel coordinates of the ray
            zsurf_ji = np.array(inv_transform * ray_xyz[:2]).astype('float32', copy=False)
            zsurf_z = np.full((zsurf_ji.shape[1],), dtype=z.dtype, fill_value=float('nan'))

            # find the z surface values corresponding to the ray (the remapped array will be
            # nan outside z bounds and for already masked / nan pixels)
            self._remap(
                z[np.newaxis, ...],
                [coord[np.newaxis, :] for coord in zsurf_ji],
                interp,
                zsurf_z[np.newaxis, np.newaxis, :],
                maps_contain_nans=False,
            )

            # store the first ray-z intersection point if it exists, otherwise the z_min point
            intersection_i = np.nonzero(ray_xyz[2] <= zsurf_z)[0]
            if len(intersection_i) > 0:
                xyz[:, pi] = ray_xyz[:, intersection_i[0]]
            else:
                xyz[:, pi] = ray_xyz[:, -1]
        return xyz

    @abstractmethod
    def world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        """
        Transform from 3D world to 2D pixel coordinates.

        :param xyz:
            3D world (x, y, z) coordinates to transform, as a 3-by-N array, with (x, y, z) along
            the first dimension.

        :return:
            Pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the first
            dimension.
        """

    @abstractmethod
    def pixel_to_world_z(self, ji: np.ndarray, z: float | np.ndarray) -> np.ndarray:
        """
        Transform from 2D pixel to 3D world coordinates at a specified z.

        Allows broadcasting of the pixel coordinate(s) and z value(s) i.e. can transform multiple
        pixel coordinates to a single z value, or a single pixel coordinate to multiple z values.

        :param ji:
            Pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the first
            dimension.
        :param z:
            Z values(s) to project to as a 1-by-N array.

        :return:
            3D world (x, y, z) coordinates as a 3-by-N array, with (x, y, z) along the first
            dimension.
        """

    def pixel_boundary(self, num_pts: int | None = None) -> np.ndarray:
        """
        A rectangle of 2D pixel coordinates along the image boundary.

        :param num_pts:
            Number of boundary points to include (should be even).  If set to ``None`` (the
            default), eight points are included, with points at the image corners and mid-points
            of the sides.

        :return:
            Boundary pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the
            first dimension.
        """

        def rect_boundary(im_size: np.ndarray, num_pts: int) -> np.ndarray:
            """Return a rectangular pixel coordinate boundary of ``num_pts`` ~evenly spaced points
            for the given image size ``im_size``.
            """
            if ((num_pts % 2) != 0) or (num_pts <= 0):
                raise OrthorityError("'num_pts' should be even and greater than zero.")

            br = im_size - 1
            # num of pts along width & height (fractional parts sum to 1)
            num_side_pts = (num_pts / 2) * br / br.sum()
            # round num_side_pts, dealing with the case where fractional parts are 0.5, so that
            # the rounded vals sum to num_pts / 2
            num_side_pts[:] = (
                [np.ceil(num_side_pts[0]), np.floor(num_side_pts[1])]
                if np.all(num_side_pts % 1 == 0.5)
                else np.round(num_side_pts)
            )

            num_side_pts = np.tile(num_side_pts.astype(int), 2)
            cnr_ji = np.array([[0, 0], [br[0], 0], br, [0, br[1]], [0, 0]])
            return np.vstack(
                [
                    np.linspace(cnr_ji[i], cnr_ji[i + 1], num_side_pts[i], endpoint=False)
                    for i in range(0, 4)
                ]
            ).T

        im_size = np.array(self._im_size)
        if num_pts is None:
            w, h = im_size - 1
            ji = np.array(
                [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]]
            ).T
        else:
            ji = rect_boundary(im_size, num_pts=num_pts)

        return ji

    def world_boundary(
        self,
        z: float | np.ndarray,
        num_pts: int | None = None,
        transform: rio.Affine | None = None,
        interp: str | Interp = Interp.cubic,
        **kwargs,
    ) -> np.ndarray:
        """
        A polygon of (x, y, z) world coordinates along the image boundary, at a specified z value
        or surface (DEM) intersection.

        :param z:
            Z values(s) as a single value or a 2D array (surface).
        :param num_pts:
            Number of boundary points to include (should be even).  If set to ``None`` (the
            default), eight points are included, with points at the image corners and mid-points
            of the sides.
        :param transform:
            Affine transform defining the (x, y) world coordinates of ``z`` when it is an array.
            Required when ``z`` is an array and not used otherwise.
        :param interp:
            Interpolation method to use for finding boundary intersections with ``z`` when it is an
            array.  Not used when ``z`` is a single value.
        :param kwargs:
            Not used.

        :return:
            Boundary world (x, y, z) coordinates as a 3-by-N array, with (x, y, z) along the
            first dimension.  Boundary points that lie outside ``z`` bounds, when ``z`` is an
            array, are given at the minimum of ``z``.
        """
        ji = self.pixel_boundary(num_pts=num_pts)
        if np.isscalar(z):
            xyz = self.pixel_to_world_z(ji, z)
        elif isinstance(z, np.ndarray) and z.ndim == 2:
            if transform is None:
                raise ValueError("'transform' should be supplied when 'z' is an array.")
            xyz = self._pixel_to_world_surf(ji, z, transform, interp=interp)
        else:
            raise ValueError("'z' should be a single value or 2D array.")
        return xyz

    def read(
        self,
        im_file: str | PathLike | OpenFile | rio.DatasetReader,
        indexes: Sequence[int] | int | None = None,
        dtype: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Read image band(s) from a given file.  Sub-classes may add a processing step.

        :param im_file:
            Image file to read. Can be a path or URI string, :class:`~fsspec.core.OpenFile`
            object in binary mode (``'rb'``), or dataset reader.
        :param indexes:
            Band index(es) to read (1 based).  Defaults to all bands if not supplied.
        :param dtype:
            Data type of the returned array.  If set to ``None`` (the default), the ``im_file``
            dtype is used.
        :param kwargs:
            Not used.

        :return:
            Image as 3D array with band(s) along the first dimension (Rasterio ordering).
        """
        # add an empty dimension to indexes if it is scalar so that image is read as 3D
        indexes = np.expand_dims(indexes, 0) if np.isscalar(indexes) else indexes

        # TODO: rio.Env has no effect on already opened datasets, which is I think is always how
        #  this method is used, rather leave it to user?
        env = rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False)
        with common.suppress_no_georef(), env, common.OpenRaster(im_file) as im:
            dtype = dtype or im.dtypes[0]
            return im.read(indexes, out_dtype=dtype)

    def remap(
        self,
        im_array: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        nodata: float | int | None = None,
        interp: str | Interp = Interp.cubic,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remap an image to an ortho image at the given world coordinates.

        :param im_array:
            Image to remap as a 3D array with band(s) along the first dimension (Rasterio
            ordering).  Typically, this is the image returned by :meth:`Camera.read`, with the
            same size as the camera :attr:`~Camera.im_size`.
        :param x:
            X world coordinates to remap to, as a M-by-N 2D array with ``float64`` data type. NaN
            coordinate pixels are mapped to ``nodata``.
        :param y:
            Y world coordinates to remap to, as a M-by-N 2D array with ``float64`` data type. NaN
            coordinate pixels are mapped to ``nodata``.
        :param z:
            Z world coordinates to remap to, as a M-by-N 2D array with ``float64`` data type. NaN
            coordinate pixels are mapped to ``nodata``.
        :param nodata:
            Value to use for masking invalid pixels in the remapped image.  If set to ``None`` (the
            default), a value based on the ``im_array`` data type is chosen automatically.
        :param interp:
            Interpolation method to use for remapping.
        :param kwargs:
            Not used.

        :return:
            - Remapped image as a L-by-M-by-N 3D array, where L is the number of ``im_array``
              bands.  Same data type as ``im_array``.
            - Nodata mask of the remapped image, as a M-by-N 2D boolean array.
        """
        self._validate_image(im_array)
        if not (x.shape == y.shape == z.shape) or (x.ndim != 2):
            raise ValueError("'x', 'y' and 'z' should have 2 dimensions, and the same shape.")
        if not (x.dtype == y.dtype == 'float64'):
            raise ValueError("'x' and 'y' should have 'float64' data type.")
        if not np.issubdtype(z.dtype, np.floating):
            raise ValueError("'z' should have 'float64' or 'float32' data type.")

        # initialise ortho / remapped array
        if nodata is None:
            nodata = common._nodata_vals[im_array.dtype.name]
        remap_array = np.full(
            (im_array.shape[0], *x.shape), dtype=im_array.dtype, fill_value=nodata
        )

        # skip remapping when the remapped array would be all nodata
        if np.all(np.isnan(x) | np.isnan(y) | np.isnan(z)):
            remap_mask = np.full(x.shape, dtype='bool', fill_value=True)
            return remap_array, remap_mask

        # find (j, i) image pixel coords corresponding to (x, y, z) world coords
        ji = self.world_to_pixel(np.array((x.reshape(-1), y.reshape(-1), z.reshape(-1))))

        # separate ji into (j, i) grids, converting to float32 for compatibility with cv2.remap()
        j = ji[0].reshape(*x.shape).astype('float32')
        i = ji[1].reshape(*x.shape).astype('float32')

        # remap image to ortho
        self._remap(im_array, (j, i), interp, remap_array)

        # find nodata mask
        remap_mask = np.all(common.nan_equals(remap_array, nodata), axis=0)
        return remap_array, remap_mask


class RpcCamera(Camera):
    """
    RPC camera.

    :param im_size:
        Image (width, height) in pixels.
    :param rpc:
        RPC parameters as a :class:`~rasterio.rpc.RPC` object or dictionary.
    :param rpc_options:
        Options for :cpp:func:`GDALCreateRPCTransformerV2`.  Only used for the reverse pixel to
        world coordinate transform.
    :param crs:
        World / ortho CRS as an EPSG, proj4 or WKT string, or :class:`~rasterio.crs.CRS` object.
        If its vertical CRS is defined, it should be ellipsoidal height (m), otherwise
        ellipsoidal height is assumed. If ``crs`` is set to ``None`` (the default), the 3D WGS84
        geographic CRS is used.
    """

    def __init__(
        self,
        im_size: tuple[int, int],
        rpc: RPC | dict,
        rpc_options: dict | None = None,
        crs: str | CRS = None,
    ):
        super().__init__()
        self._im_size = im_size
        self._rpc_crs = CRS.from_epsg(4979)
        # convert dict rpc to RPC object to avoid issue where RPCTransformer.__init__() raises no
        # error with rpc as a dict, but generates invalid results
        self._rpc = rpc if isinstance(rpc, RPC) else RPC(**rpc)
        self._rpc_options = rpc_options or {}
        self._crs = self._validate_crs(crs) if crs else None

    @property
    def crs(self) -> CRS | None:
        """World / ortho CRS."""
        return self._crs or self._rpc_crs

    def _validate_crs(self, crs: str | CRS) -> CRS:
        """Validate the CRS has ellipsoidal height or no vertical CRS."""
        crs = rio.CRS.from_string(crs) if isinstance(crs, str) else crs
        for z in [0, 1]:
            xyz = warp(self._rpc_crs, crs, [self._rpc.long_off], [self._rpc.lat_off], [z])
            if not xyz[2][0] == z:
                raise OrthorityError("RPC camera requires a 'crs' with ellipsoidal height (m).")
        return crs

    def world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        self._validate_world_coords(xyz)
        # TODO: make rasterio feature / pull request to release gil on crs & rpc transform,
        #  and to not copy coordinates back and forth between python/c formats in for loops (see
        #  e.g. pyproj for a way of doing this efficiently).
        if self._crs:
            # warp from world / ortho to geographic / RPC coordinates, removing, and replacing nans
            # around the warp call (which raises errors on nans)
            mask = ~np.any(np.isnan(xyz), axis=0)
            xyz_ = np.full(xyz.shape, fill_value=np.nan)
            xyz_[:, mask] = np.array(warp(self._crs, self._rpc_crs, *xyz[:, mask]))
        else:
            xyz_ = xyz.copy()

        def poly(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: Sequence[float]) -> np.ndarray:
            """Return the polynomial value for given coordinates and coefficients.  Uses a Horner
            type approach: https://en.wikipedia.org/wiki/Horner%27s_method.
            """
            res = c[0] + x * (
                c[1]
                + y * (c[4] + z * c[10])
                + z * c[5]
                + x * (c[7] + x * c[11] + y * c[14] + z * c[17])
            )
            res += y * (c[2] + c[6] * z + y * (c[8] + x * c[12] + y * c[15] + z * c[18]))
            res += z * (c[3] + z * (c[9] + x * c[13] + y * c[16] + z * c[19]))
            return res

        # RPC model evaluation based on http://geotiff.maptools.org/rpc_prop.html. Equivalent to
        # this Rasterio code, but releases the GIL:
        # ij = self._rpc_tformer.rowcol(*xyz_, op=lambda x: x)
        # ji = np.array(ij[::-1]) - 0.5
        xyz_ -= np.array([[self._rpc.long_off, self._rpc.lat_off, self._rpc.height_off]]).T
        xyz_ /= np.array([[self._rpc.long_scale, self._rpc.lat_scale, self._rpc.height_scale]]).T
        i = poly(*xyz_, self._rpc.line_num_coeff) / poly(*xyz_, self._rpc.line_den_coeff)
        j = poly(*xyz_, self._rpc.samp_num_coeff) / poly(*xyz_, self._rpc.samp_den_coeff)
        ji = np.array((j, i))
        ji *= np.array([[self._rpc.samp_scale, self._rpc.line_scale]]).T
        ji += np.array([[self._rpc.samp_off, self._rpc.line_off]]).T
        return ji

    def pixel_to_world_z(self, ji: np.ndarray, z: float | np.ndarray, **kwargs) -> np.ndarray:
        self._validate_pixel_coords(ji)
        self._validate_z(z, ji)

        # project from pixel to geographic / RPC coordinates, removing and replacing nans around
        # the .xy call (which raises warnings on nans and converts to inf)
        z = z * np.ones(ji.shape[1]) if np.isscalar(z) else z.copy()
        ji = ji * np.ones((2, z.shape[0])) if ji.shape[1] == 1 else ji
        mask = ~(np.any(np.isnan(ji), axis=0) | np.isnan(z))
        xy = np.full(ji.shape, fill_value=np.nan)
        z[~mask] = np.nan
        with RPCTransformer(self._rpc, **self._rpc_options) as tform:
            # TODO: the center offset in .xy below & in GcpCamera is inefficient
            xy[:, mask] = tform.xy(ji[1, mask], ji[0, mask], zs=z[mask], offset='center')

        xyz = np.array([*xy, z])
        if self._crs:
            # warp from geographic / RPC to world / ortho  coordinates, removing, and replacing nans
            # around the warp call (which raises errors on nans)
            xyz[:, mask] = np.array(warp(self._rpc_crs, self._crs, *xyz[:, mask]))
        return xyz


class GcpCamera(Camera):  # pragma: no cover
    """
    GCP camera (UNTESTED).

    :param im_size:
        Image (width, height) in pixels.
    :param gcps:
        GCPs as a sequence of :class:`~rasterio.control.GroundControlPoint` objects or dictionaries.
    """

    def __init__(
        self,
        im_size: tuple[int, int],
        gcps: Sequence[GroundControlPoint, dict],
    ):
        super().__init__()
        self._im_size = im_size
        self._gcps = [GroundControlPoint(gcp) if isinstance(gcp, dict) else gcp for gcp in gcps]

    def world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        self._validate_world_coords(xyz)
        with GCPTransformer(self._gcps) as tform:
            ij = tform.rowcol(*xyz, op=lambda x: x)
        # flip i & j and convert UL to center pixel coordinates
        ji = np.array(ij[::-1]) - 0.5
        return ji

    def pixel_to_world_z(self, ji: np.ndarray, z: float | np.ndarray, **kwargs) -> np.ndarray:
        self._validate_pixel_coords(ji)
        self._validate_z(z, ji)

        # project from pixel to GCP coordinates, removing and replacing nans around the .xy call
        # (which raises warnings on nans and converts to inf)
        z = z * np.ones(ji.shape[1]) if np.isscalar(z) else z.copy()
        ji = ji * np.ones((2, z.shape[0])) if ji.shape[1] == 1 else ji
        mask = ~(np.any(np.isnan(ji), axis=0) | np.isnan(z))
        xy = np.full(ji.shape, fill_value=np.nan)
        z[~mask] = np.nan
        with GCPTransformer(self._gcps) as tform:
            xy[:, mask] = tform.xy(ji[1, mask], ji[0, mask], zs=z[mask], offset='center')

        xyz = np.array([*xy, z])
        return xyz


class FrameCamera(Camera):
    """
    Pinhole camera with no distortion.

    The ``xyz`` and ``opk`` exterior parameters must be supplied here, or via
    :meth:`~FrameCamera.update`, before calling any methods that generate or require world
    coordinates.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value
        or (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to ``None``
         (the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
         ``focal_len`` = focal length / max(sensor width & height)).
    :param cx:
        Principal point offsets in `normalised image coordinates
        <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`__.
    :param cy:
    :param xyz:
        Camera (x, y, z) position in world coordinates.
    :param opk:
        Camera (omega, phi, kappa) angles in radians to rotate from camera (PATB convention) to
        world coordinates.
    :param distort:
        Not used for the pinhole camera model.
    :param alpha:
        Not used for the pinhole camera model.
    """

    _default_alpha: float = 1.0
    _default_distort: bool = True

    def __init__(
        self,
        im_size: tuple[int, int],
        focal_len: float | tuple[float, float],
        sensor_size: tuple[float, float] | None = None,
        cx: float = 0.0,
        cy: float = 0.0,
        xyz: tuple[float, float, float] | None = None,
        opk: tuple[float, float, float] | None = None,
        distort: bool = _default_distort,
        alpha: float = _default_alpha,
    ) -> None:
        super().__init__()
        self._im_size = (int(im_size[0]), int(im_size[1]))
        self._K = self._get_intrinsic(self._im_size, focal_len, sensor_size, cx, cy)
        self._R, self._T = self._get_extrinsic(xyz, opk)
        self._K_undistort, self._K_undistort_inv = self._K, np.linalg.inv(self._K)

        self._undistort_maps = None
        self._distort = distort
        self._alpha = alpha

    @property
    def pos(self) -> tuple[float, float, float] | None:
        """Camera (x, y, z) position in units of the world / ortho CRS."""
        return tuple(self._T.reshape(-1)) if self._T is not None else None

    @property
    def distort(self) -> bool:
        """Include distortion in the camera model, and return the original (distorted) image from
        :meth:`~FrameCamera.read` (``True``).  Or, exclude distortion from the camera model,
        and return an undistorted image from :meth:`~FrameCamera.read` (``False``).
        :meth:`~FrameCamera.remap` of an image returned by :meth:`~FrameCamera.read` is faster
        with ``distort=False``, but may reduce remap quality.
        """
        return self._distort

    @distort.setter
    def distort(self, value: bool) -> None:
        self._distort = value

    @property
    def alpha(self) -> float:
        """Scaling (``0``-``1``) of the undistorted image returned by :meth:`~FrameCamera.read` when
        :attr:`~FrameCamera.distort` is ``False``.  ``0`` includes the largest portion of the source
        image that allows all undistorted pixels to be valid.  ``1`` includes all source pixels in
        the undistorted image. Its value affects scaling of the camera model intrinsic matrix.
        Not used when :attr:`~FrameCamera.distort` is ``True``.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if type(self) is not PinholeCamera and value != self._alpha:
            self._K_undistort, self._K_undistort_inv = self._get_undistort_intrinsic(value)
            self._undistort_maps = None
        self._alpha = value

    @staticmethod
    def _get_intrinsic(
        im_size: tuple[int, int],
        focal_len: float | tuple[float, float],
        sensor_size: tuple[float, float] | None,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """Return the intrinsic matrix and its inverse, for the given interior parameters."""
        # Adapted from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        # and https://en.wikipedia.org/wiki/Camera_resectioning
        # TODO: incorporate orientation from exif

        if len(im_size) != 2:
            raise ValueError("'im_size' should contain 2 values: (width, height).")
        im_size = np.array(im_size)
        if sensor_size is not None and len(sensor_size) != 2:
            raise ValueError("'sensor_size' should contain 2 values: (width, height).")
        focal_len = np.array(focal_len)
        if focal_len.size > 2:
            raise ValueError("'focal_len' should contain at most 2 values.")

        # find the xy focal lengths in pixels
        if sensor_size is None:
            warnings.warn(
                "'sensor_size' not specified, assuming square pixels and 'focal_len' normalised by "
                "sensor width.",
                category=OrthorityWarning,
                stacklevel=2,
            )
            sigma_xy = (focal_len * im_size[0]) * np.ones(2)
        else:
            sensor_size = np.array(sensor_size)
            sigma_xy = focal_len * im_size / sensor_size

        # find principal point in pixels
        c_xy = (im_size - 1) / 2
        c_xy += im_size.max() * np.array((cx, cy))

        # intrinsic matrix to convert from camera co-ords in OpenSfM / OpenCV convention
        # (x->right, y->down, z->forwards, looking through the camera at the scene) to pixel
        # co-ords in standard convention (x->right, y->down).
        K = np.array([[sigma_xy[0], 0, c_xy[0]], [0, sigma_xy[1], c_xy[1]], [0, 0, 1]])
        return K

    @staticmethod
    def _get_extrinsic(
        xyz: tuple[float, float, float],
        opk: tuple[float, float, float],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the rotation matrix and translation vector for the given exterior parameters."""
        if xyz is None or opk is None:
            return None, None
        elif len(xyz) != 3 or len(opk) != 3:
            raise ValueError("'xyz' and 'opk' should contain 3 values.")

        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        T = np.array(xyz).reshape(-1, 1)
        R = _opk_to_rotation(opk)

        # rotate from PATB (x->right, y->up, z->backwards looking through the camera at the
        # scene) to OpenSfM / OpenCV convention (x->right, y->down, z->forwards, looking through
        # the camera at the scene)
        R = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        return R, T

    def _test_init(self) -> None:
        """Utility function to test if exterior parameters are initialised."""
        if self._R is None or self._T is None:
            raise CameraInitError('Exterior parameters not initialised.')

    def _horizon_fov(self) -> bool:
        """Whether this camera's field of view includes, or is above, the horizon."""
        # TODO: actually a world horizontal plane in line with camera, not a horizon
        self._test_init()
        # camera coords for image boundary
        w, h = np.array(self._im_size) - 1
        src_ji = np.array(
            [[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]]
        ).T
        xyz_ = self._pixel_to_camera(src_ji)

        # rotate camera to world alignment & test if any z vals are above the camera / origin
        xyz_r = self._R.dot(xyz_)
        return np.any(xyz_r[2] >= 0)

    def _get_undistort_intrinsic(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a new camera intrinsic matrix, and its inverse, for an undistorted image that is
        the same size as the source image.

        ``alpha`` (``0``-``1``) controls the portion of the source included in the distorted
        image. 0 includes the largest portion of the source image that allows all undistorted
        pixels to be valid.  ``1`` includes all source pixels in the undistorted image.
        """

        # Adapted from and equivalent to:
        # K_undistort, _ = cv2.getOptimalNewCameraMatrix(K, dist_param, im_size, alpha).
        # See https://github.com/opencv/opencv/blob/4790a3732e725b102f6c27858e7b43d78aee2c3e/modules/calib3d/src/calibration.cpp#L2772
        # Note that cv2.fisheye.estimateNewCameraMatrixForUndistortRectify() does not include all
        # source pixels for balance=1.  This method works for all subclasses including fisheye.
        def _get_rectangles(
            im_size: tuple[int, int],
        ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
            """Return inner and outer rectangles for distorted image grid points."""
            w, h = np.array(im_size) - 1
            n = 9
            scale_j, scale_i = np.meshgrid(range(0, n), range(0, n))
            scale_j, scale_i = scale_j.ravel(), scale_i.ravel()
            ji = np.vstack([scale_j * w / (n - 1), scale_i * h / (n - 1)])
            xy = self._pixel_to_camera(ji)[:2]
            outer = xy.min(axis=1), xy.max(axis=1) - xy.min(axis=1)
            inner_ul = np.array((xy[0][scale_j == 0].max(), xy[1][scale_i == 0].max()))
            inner_br = np.array((xy[0][scale_j == n - 1].min(), xy[1][scale_i == n - 1].min()))
            inner = inner_ul, inner_br - inner_ul
            return inner, outer

        alpha = np.clip(alpha, a_min=0, a_max=1)
        (inner_off, inner_size), (outer_off, outer_size) = _get_rectangles(self._im_size)

        im_size = np.array(self._im_size)
        f0 = (im_size - 1) / inner_size
        c0 = -f0 * inner_off
        f1 = (im_size - 1) / outer_size
        c1 = -f1 * outer_off
        f = f0 * (1 - alpha) + f1 * alpha
        c = c0 * (1 - alpha) + c1 * alpha

        K_undistort = np.eye(3)
        K_undistort[[0, 1], [0, 1]] = f
        K_undistort[:2, 2] = c
        return K_undistort, np.linalg.inv(K_undistort)

    def _get_undistort_maps(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return cv2.remap() maps for undistorting an image, and intrinsic matrix for undistorted
        image.
        """
        return None

    def _camera_to_pixel(self, xyz_: np.ndarray) -> np.ndarray:
        """Transform from homogenous 3D camera to 2D pixel coordinates."""
        ji = self._K_undistort.dot(xyz_)[:2]
        return ji

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        """Transform 2D pixel to homogenous 3D camera coordinates."""
        ji_ = np.vstack([ji.astype('float64', copy=False), np.ones((1, ji.shape[1]))])
        xyz_ = self._K_undistort_inv.dot(ji_)
        return xyz_

    def update(
        self,
        xyz: tuple[float, float, float],
        opk: tuple[float, float, float],
    ) -> None:
        """
        Update exterior parameters.

        :param xyz:
            Camera (x, y, z) position in world coordinates.
        :param opk:
            Camera (omega, phi, kappa) angles in radians to rotate camera (PATB convention) to
            world coordinates.
        """
        self._R, self._T = self._get_extrinsic(xyz, opk)

    def world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        """
        Transform from 3D world to 2D pixel coordinates.

        :param xyz:
            3D world (x, y, z) coordinates to transform, as a 3-by-N array, with (x, y, z) along
            the first dimension.

        :return:
            Pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the first
            dimension.
        """
        self._test_init()
        self._validate_world_coords(xyz)

        # transform from world to camera coordinates & scale to origin
        xyz_ = self._R.T.dot(xyz - self._T)
        xyz_ = xyz_ / xyz_[2]
        # transform from camera to pixel coordinates, including the distortion model if
        # distort==True
        ji = (
            self._camera_to_pixel(xyz_)
            if self._distort
            else FrameCamera._camera_to_pixel(self, xyz_)
        )
        return ji

    def pixel_to_world_z(self, ji: np.ndarray, z: float | np.ndarray) -> np.ndarray:
        """
        Transform from 2D pixel to 3D world coordinates at a specified z.

        Allows broadcasting of the pixel coordinate(s) and z value(s) i.e. can transform multiple
        pixel coordinates to a single z value, or a single pixel coordinate to multiple z values.

        :param ji:
            Pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the first
            dimension.
        :param z:
            Z values(s) to project to as a 1-by-N array.

        :return:
            3D world (x, y, z) coordinates as a 3-by-N array, with (x, y, z) along the first
            dimension.
        """
        # TODO: consider only returning (x, y).  the z dimension is redundant, and it is used this
        #  way in most (all?) places.
        # TODO: i have noticed that the results with e.g. z=0 sometimes have z close to but not
        #  equal 0.  is there a way of re-organising this so that doesn't happen?
        self._test_init()
        self._validate_pixel_coords(ji)
        self._validate_z(z, ji)

        # transform pixel coordinates to camera coordinates
        xyz_ = (
            self._pixel_to_camera(ji) if self._distort else FrameCamera._pixel_to_camera(self, ji)
        )
        # rotate first (camera to world) to get world aligned axes with origin on the camera
        xyz_r = self._R.dot(xyz_)

        # find scales to reach z (offset for camera z)
        scales = (z - self.pos[2]) / xyz_r[2]

        # scale to z with origin on camera, then offset to world
        xyz = (xyz_r * scales) + self._T
        return xyz

    def _distort_pixel(self, ji: np.ndarray, clip: bool = False) -> np.ndarray:
        """Return distorted pixel coordinates with the same shape as ``ji``, clipping to
        :attr:`~Camera.im_size` if ``clip==True``.
        """
        self._validate_pixel_coords(ji)

        xyz_ = FrameCamera._pixel_to_camera(self, ji)
        ji = self._camera_to_pixel(xyz_)

        if clip:
            ji = np.clip(ji.T, a_min=(0, 0), a_max=np.array(self._im_size) - 1).T
        return ji

    def _undistort_pixel(self, ji: np.ndarray, clip: bool = False) -> np.ndarray:
        """Return undistorted pixel coordinates with the same shape as ``ji``, clipping to
        :attr:`~Camera.im_size` if ``clip==True``.
        """
        self._validate_pixel_coords(ji)

        xyz_ = self._pixel_to_camera(ji)
        ji = FrameCamera._camera_to_pixel(self, xyz_)

        if clip:
            ji = np.clip(ji.T, a_min=(0, 0), a_max=np.array(self._im_size) - 1).T
        return ji

    def _undistort_im(
        self,
        im_array: np.ndarray,
        nodata: float | int | None = None,
        interp: str | Interp = Interp.cubic,
    ) -> np.ndarray:
        """Return an undistorted image as a 3D array with the same number of bands as
        ``im_array``, and the same band size as :attr:`~Camera.im_size`.
        """
        self._validate_image(im_array)

        # find undistort maps once on first use
        self._undistort_maps = self._undistort_maps or self._get_undistort_maps()

        if self._undistort_maps is None:
            return im_array

        if nodata is None:
            nodata = common._nodata_vals[im_array.dtype.name]
        remap_array = np.full(im_array.shape, dtype=im_array.dtype, fill_value=nodata)

        # remap image
        self._remap(
            im_array,
            self._undistort_maps,
            interp,
            remap_array,
            change_maps_inplace=False,
            maps_contain_nans=False,
        )
        return remap_array

    def pixel_boundary(self, num_pts: int | None = None) -> np.ndarray:
        """
        A polygon of 2D pixel coordinates along the image boundary.  If
        :attr:`~FrameCamera.distort` is ``False``, coordinates will be along the boundary of the
        valid area in the undistorted image returned by :meth:`~FrameCamera.read`.

        :param num_pts:
            Number of boundary points to include (should be even).  If set to ``None`` (the
            default), eight points are included, with points at the image corners and mid-points
            of the sides.

        :return:
            Boundary pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the
            first dimension.
        """
        ji = super().pixel_boundary(num_pts=num_pts)

        if not self._distort:
            ji = self._undistort_pixel(ji, clip=True)
        return ji

    def world_boundary(
        self,
        z: float | np.ndarray,
        num_pts: int | None = None,
        transform: rio.Affine | None = None,
        interp: str | Interp = Interp.cubic,
        clip: bool = True,
    ) -> np.ndarray:
        """
        A polygon of (x, y, z) world coordinates along the image boundary, at a specified z value
        or surface (DEM).

        :param z:
            Z values(s) as a single value or a 2D array (surface).
        :param num_pts:
            Number of boundary points to include (should be even).  If set to ``None`` (the
            default), eight points are included, with points at the image corners and mid-points
            of the sides.
        :param transform:
            Affine transform defining the (x, y) world coordinates of ``z`` when it is an array.
            Required when ``z`` is an array and not used otherwise.
        :param interp:
            Interpolation method to use for finding boundary intersections with ``z`` when it is an
            array.  Not used when ``z`` is a single value.
        :param clip:
            Clip the z coordinate of boundary points to the camera height.

        :return:
            Boundary world (x, y, z) coordinates as a 3-by-N array, with (x, y, z) along the
            first dimension.  Boundary points that lie outside ``z`` bounds, when ``z`` is an
            array, are given at the minimum of ``z``.
        """
        self._test_init()
        if self._horizon_fov():
            raise OrthorityError(
                "Camera has a field of view that includes, or is above, the horizon."
            )

        ji = self.pixel_boundary(num_pts=num_pts)
        if np.isscalar(z):
            # clip z to camera height
            z = z if not clip else min(z, self.pos[2])
            xyz = self.pixel_to_world_z(ji, z)
        elif isinstance(z, np.ndarray) and z.ndim == 2:
            if transform is None:
                raise ValueError("'transform' should be supplied when 'z' is an array.")

            # find / test / clip dem minimum and maximum
            min_z = np.nanmin(z)
            max_z = np.nanmax(z)
            if min_z > self.pos[2]:
                raise ValueError('The DEM is higher than the camera.')
            max_z = max_z if not clip else min(max_z, self.pos[2])

            xyz = self._pixel_to_world_surf(
                ji, z, transform, interp=interp, min_z=min_z, max_z=max_z
            )
        else:
            raise ValueError("'z' should be a single value or 2D array.")
        return xyz

    def read(
        self,
        im_file: str | PathLike | OpenFile | rio.DatasetReader,
        indexes: Sequence[int] | None = None,
        dtype: str | None = None,
        nodata: float | int | None = None,
        interp: str | Interp = Interp.cubic,
    ) -> np.ndarray:
        """
        Read image band(s) from a given file, undistorting when :attr:`~FrameCamera.distort` is
        ``False``.

        :param im_file:
            Image file to read.  Can be a path or URI string, :class:`~fsspec.core.OpenFile`
            object in binary mode (``'rb'``), or dataset reader.
        :param indexes:
            Band index(es) to read (1 based).
        :param dtype:
            Data type of the returned array.  If set to ``None`` (the default), the ``im_file``
            dtype is used.
        :param nodata:
            Value to use for masking invalid pixels in the undistorted image.  If set to ``None``
            (the default), a value based on ``dtype`` is chosen automatically.  Not used if
            :attr:`~FrameCamera.distort` is ``True``.
        :param interp:
            Interpolation method to use when undistorting the image.  Not used if
            :attr:`~FrameCamera.distort` is ``True``.

        :return:
            Image as 3D array with band(s) along the first dimension (Rasterio ordering).
        """
        image = super().read(im_file, indexes=indexes, dtype=dtype)

        if not self._distort:
            if nodata is None:
                nodata = common._nodata_vals[image.dtype.name]
            image = self._undistort_im(image, nodata=nodata, interp=interp)
        return image

    def remap(
        self,
        image: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        nodata: float | int | None = None,
        interp: str | Interp = Interp.cubic,
        kernel_size: tuple[int, int] = (3, 3),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remap an image to an ortho image at the given world coordinates.

        :param im_array:
            Image to remap as a 3D array with band(s) along the first dimension (Rasterio
            ordering).  Typically, this is the image returned by :meth:`Camera.read`, with the
            same size as the camera :attr:`~Camera.im_size`.
        :param x:
            X world coordinates to remap to, as a M-by-N 2D array with ``float64`` data type. NaN
            coordinate pixels are mapped to ``nodata``.
        :param y:
            Y world coordinates to remap to, as a M-by-N 2D array with ``float64`` data type. NaN
            coordinate pixels are mapped to ``nodata``.
        :param z:
            Z world coordinates to remap to, as a M-by-N 2D array with ``float64`` data type. NaN
            coordinate pixels are mapped to ``nodata``.
        :param nodata:
            Value to use for masking invalid pixels in the remapped image.  If set to ``None`` (the
            default), a value based on the ``im_array`` data type is chosen automatically.
        :param interp:
            Interpolation method to use for remapping.
        :param kernel_size:
            Kernel (width, height) size in pixels, used for dilating the nodata mask.  Removes
            blurring of boundary pixels with nodata areas in an undistorted ``im_array``.  Not used
            if blurring could not have occurred (e.g. if :attr:`~FrameCamera.distort` is ``True``).

        :return:
            - Remapped image as a L-by-M-by-N 3D array, where L is the number of ``im_array``
              bands.  Same data type as ``im_array``.
            - Nodata mask of the remapped image, as a M-by-N 2D boolean array.
        """
        remap, mask = super().remap(image, x, y, z, nodata=nodata, interp=interp)

        # remove blurring with nodata pixels when necessary
        if (
            not self.distort
            and Interp[interp] != Interp.nearest
            and not np.isnan(nodata)
            and type(self) is not PinholeCamera
        ):
            kernel = np.ones(kernel_size[::-1], np.uint8)
            mask = cv2.dilate(mask.view(np.uint8), kernel).view(bool)

            if nodata is None:
                nodata = common._nodata_vals[image.dtype.name]
            remap[:, mask] = nodata

        return remap, mask


# alias FrameCamera as PinholeCamera
PinholeCamera = FrameCamera


class OpenCVCamera(FrameCamera):
    """
    OpenCV camera model.

    This is a wrapper around the `OpenCV general model
    <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.  Partial or special cases can be
    specified by omitting some or all of the distortion coefficients.  E.g. if no distortion
    coefficients are specified, this model corresponds to :class:`PinholeCamera`, or if the first 5
    distortion coefficients are specified, this model corresponds to :class:`BrownCamera`.

    The ``xyz`` and ``opk`` exterior parameters must be supplied here, or via
    :meth:`~FrameCamera.update`, before calling any methods that generate or require world
    coordinates.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or
        (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to ``None``
         (the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
         ``focal_len`` = focal length / max(sensor width & height)).
    :param cx:
        Principal point offsets in `normalised image coordinates
        <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`__.
    :param cy:
    :param k1:
        Distortion coefficients - see the `OpenCV docs
        <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.
    :param k2:
    :param k3:
    :param p1:
    :param p2:
    :param k4:
    :param k5:
    :param k6:
    :param s1:
    :param s2:
    :param s3:
    :param s4:
    :param tx:
    :param ty:
    :param xyz:
        Camera (x, y, z) position in world coordinates.
    :param opk:
        Camera (omega, phi, kappa) angles in radians to rotate from camera (PATB convention) to
        world coordinates.
    :param distort:
        Include distortion in the camera model, and return the original (distorted) image from
        :meth:`~FrameCamera.read` (``True``).  Or, exclude distortion from the camera model,
        and return an undistorted image from :meth:`~FrameCamera.read` (``False``).
        :meth:`~FrameCamera.remap` of an image returned by :meth:`~FrameCamera.read` is faster
        with ``distort=False``, but may reduce remap quality.
    :param alpha:
        Scaling (``0``-``1``) of the undistorted image returned by :meth:`~FrameCamera.read` when
        ``distort`` is ``False``.  ``0`` includes the largest portion of the source image that
        allows all undistorted pixels to be valid.  ``1`` includes all source pixels in the
        undistorted image. Its value affects scaling of the camera model intrinsic matrix.  Not
        used when ``distort`` is ``True``.
    """

    def __init__(
        self,
        im_size: tuple[int, int],
        focal_len: float | tuple[float, float],
        sensor_size: tuple[float, float] | None = None,
        cx: float = 0.0,
        cy: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        k4: float = 0.0,
        k5: float = 0.0,
        k6: float = 0.0,
        s1: float = 0.0,
        s2: float = 0.0,
        s3: float = 0.0,
        s4: float = 0.0,
        tx: float = 0.0,
        ty: float = 0.0,
        xyz: tuple[float, float, float] | None = None,
        opk: tuple[float, float, float] | None = None,
        distort: bool = FrameCamera._default_distort,
        alpha: float = FrameCamera._default_alpha,
    ):
        super().__init__(
            im_size,
            focal_len,
            sensor_size=sensor_size,
            cx=cx,
            cy=cy,
            xyz=xyz,
            opk=opk,
            alpha=alpha,
            distort=distort,
        )

        # order _dist_param & truncate zeros according to OpenCV docs
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        self._dist_param = np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty])
        for dist_len in (4, 5, 8, 12, 14):
            if np.all(self._dist_param[dist_len:] == 0.0):
                self._dist_param = self._dist_param[:dist_len]
                break
        self._K_undistort, self._K_undistort_inv = self._get_undistort_intrinsic(alpha)

    def _get_undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        im_size = np.array(self._im_size)
        # Note: float32 maps should be used (here and throughout) for cv2.remap.  The map type
        # can be specified as cv2.CV_16SC2 below, or in as a conversion step with cv2.convertMaps
        # to reduce map memory.  But CV_16SC2 maps create artefacts with nearest interpolation.
        # (The OpenCV docs say CV_16SC2 maps also speed up remapping, but tests don't support this).
        undistort_maps = cv2.initUndistortRectifyMap(
            self._K, self._dist_param, np.eye(3), self._K_undistort, im_size, cv2.CV_32FC1
        )
        # equivalent to the above, but using Camera methods (works out slower):
        # j = np.arange(0, self.im_size[0], dtype='int32')
        # i = np.zeros(self.im_size[0], dtype='int32')
        # ji = np.vstack((j, i))
        # undistort_maps = (
        #     np.zeros(self.im_size[::-1], dtype='float32'),
        #     np.zeros(self.im_size[::-1], dtype='float32'),
        # )
        # for ii in range(0, self.im_size[1]):
        #     ji[1].fill(ii)
        #     ji_ = self._distort_pixel(ji, clip=False).astype('float32')
        #     undistort_maps[0][ii] = ji_[0]
        #     undistort_maps[1][ii] = ji_[1]
        #
        return undistort_maps

    def _camera_to_pixel(self, xyz_: np.ndarray) -> np.ndarray:
        # omit world to camera rotation & translation to transform from camera to pixel coords
        ji, _ = cv2.projectPoints(xyz_.T, np.zeros(3), np.zeros(3), self._K, self._dist_param)
        return ji[:, 0, :].T

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T.astype('float64', copy=False)
        xyz_ = cv2.undistortPoints(ji_cv, self._K, self._dist_param)
        xyz_ = np.vstack([xyz_[:, 0, :].T, np.ones((1, ji.shape[1]))])
        return xyz_


class BrownCamera(OpenCVCamera):
    """
    Brown-Conrady camera model.

    Compatible with `OpenDroneMap / OpenSfM
    <https://opensfm.org/docs/geometry.html#camera-models>`__ ``perspective``, ``simple_radial``,
    ``radial`` and ``brown`` model parameters, and the 4- and 5-coefficient versions of the
    `OpenCV general model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.

    The ``xyz`` and ``opk`` exterior parameters must be supplied here, or via
    :meth:`~FrameCamera.update`, before calling any methods that generate or require world
    coordinates.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or
        (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to ``None``
         (the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
         ``focal_len`` = focal length / max(sensor width & height)).
    :param cx:
        Principal point offsets in `normalised image coordinates
        <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`__.
    :param cy:
    :param k1:
        Distortion coefficients.
    :param k2:
    :param p1:
    :param p2:
    :param k3:
    :param xyz:
        Camera (x, y, z) position in world coordinates.
    :param opk:
        Camera (omega, phi, kappa) angles in radians to rotate from camera (PATB convention) to
        world coordinates.
    :param distort:
        Include distortion in the camera model, and return the original (distorted) image from
        :meth:`~FrameCamera.read` (``True``).  Or, exclude distortion from the camera model,
        and return an undistorted image from :meth:`~FrameCamera.read` (``False``).
        :meth:`~FrameCamera.remap` of an image returned by :meth:`~FrameCamera.read` is faster
        with ``distort=False``, but may reduce remap quality.
    :param alpha:
        Scaling (``0``-``1``) of the undistorted image returned by :meth:`~FrameCamera.read` when
        ``distort`` is ``False``.  ``0`` includes the largest portion of the source image that
        allows all undistorted pixels to be valid.  ``1`` includes all source pixels in the
        undistorted image. Its value affects scaling of the camera model intrinsic matrix.  Not
        used when ``distort`` is ``True``.
    """

    def __init__(
        self,
        im_size: tuple[int, int],
        focal_len: float | tuple[float, float],
        sensor_size: tuple[float, float] | None = None,
        cx: float = 0.0,
        cy: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        k3: float = 0.0,
        xyz: tuple[float, float, float] | None = None,
        opk: tuple[float, float, float] | None = None,
        distort: bool = FrameCamera._default_distort,
        alpha: float = FrameCamera._default_alpha,
    ):
        # fmt: off
        super().__init__(
            im_size, focal_len, sensor_size=sensor_size, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, cx=cx,
            cy=cy, xyz=xyz, opk=opk, alpha=alpha, distort=distort
        )
        # fmt: on
        # overwrite possibly truncated _dist_param for use in _camera_to_pixel
        self._dist_param = np.array([k1, k2, p1, p2, k3])

    def _camera_to_pixel(self, xyz_: np.ndarray) -> np.ndarray:
        # Brown model adapted from OpenSfM:
        # https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle.h#LL299C25-L299C25.
        # Works out faster than the opencv equivalent in OpenCVCamera._camera_to_pixel().
        k1, k2, p1, p2, k3 = self._dist_param
        x2, y2 = np.square(xyz_[:2])
        xy = xyz_[0] * xyz_[1]
        r2 = x2 + y2

        radial_dist = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x_tangential_dist = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
        y_tangential_dist = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy

        xyz_[0] = xyz_[0] * radial_dist + x_tangential_dist
        xyz_[1] = xyz_[1] * radial_dist + y_tangential_dist

        # transform from distorted camera to pixel coordinates
        ji = self._K.dot(xyz_)[:2]
        return ji


class FisheyeCamera(FrameCamera):
    """
    Fisheye camera model.

    Compatible with `OpenDroneMap / OpenSfM
    <https://opensfm.org/docs/geometry.html#fisheye-camera>`__, and `OpenCV
    <https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html>`__  ``fisheye`` model
    parameters.

    The ``xyz`` and ``opk`` exterior parameters must be supplied here, or via
    :meth:`~FrameCamera.update`, before calling any methods that generate or require world
    coordinates.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or
        (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to ``None``
         (the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
         ``focal_len`` = focal length / max(sensor width & height)).
    :param cx:
        Principal point offsets in `normalised image coordinates
        <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`__.
    :param cy:
    :param k1:
        Distortion coefficients.
    :param k2:
    :param k3:
    :param k4:
    :param xyz:
        Camera (x, y, z) position in world coordinates.
    :param opk:
        Camera (omega, phi, kappa) angles in radians to rotate from camera (PATB convention) to
        world coordinates.
    :param distort:
        Include distortion in the camera model, and return the original (distorted) image from
        :meth:`~FrameCamera.read` (``True``).  Or, exclude distortion from the camera model,
        and return an undistorted image from :meth:`~FrameCamera.read` (``False``).
        :meth:`~FrameCamera.remap` of an image returned by :meth:`~FrameCamera.read` is faster
        with ``distort=False``, but may reduce remap quality.
    :param alpha:
        Scaling (``0``-``1``) of the undistorted image returned by :meth:`~FrameCamera.read` when
        ``distort`` is ``False``.  ``0`` includes the largest portion of the source image that
        allows all undistorted pixels to be valid.  ``1`` includes all source pixels in the
        undistorted image. Its value affects scaling of the camera model intrinsic matrix.  Not
        used when ``distort`` is ``True``.
    """

    def __init__(
        self,
        im_size: tuple[int, int],
        focal_len: float | tuple[float, float],
        sensor_size: tuple[float, float] | None = None,
        cx: float = 0.0,
        cy: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        xyz: tuple[float, float, float] | None = None,
        opk: tuple[float, float, float] | None = None,
        distort: bool = FrameCamera._default_distort,
        alpha: float = FrameCamera._default_alpha,
    ):
        super().__init__(
            im_size,
            focal_len,
            sensor_size=sensor_size,
            cx=cx,
            cy=cy,
            xyz=xyz,
            opk=opk,
            distort=distort,
            alpha=alpha,
        )

        self._dist_param = np.array([k1, k2, k3, k4])
        self._K_undistort, self._K_undistort_inv = self._get_undistort_intrinsic(alpha)

    def _get_undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        im_size = np.array(self._im_size)
        # unlike cv2.initUndistortRectifyMap(), cv2.fisheye.initUndistortRectifyMap() requires
        # default R & P (new camera matrix) params to be specified
        undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            self._K, self._dist_param, np.eye(3), self._K_undistort, im_size, cv2.CV_32FC1
        )
        return undistort_maps

    def _camera_to_pixel(self, xyz_: np.ndarray) -> np.ndarray:
        # Fisheye distortion adapted from OpenSfM:
        # https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle.h#L365.
        # and OpenCV docs: https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html.
        # Works out faster than the opencv equivalent:
        #   x_cv = np.expand_dims((x - self._T).T, axis=0)
        #   ji, _ = cv2.fisheye.projectPoints(
        #       x_cv, self._inv_aa, np.zeros(3), self._K, self._dist_param
        #   )
        #   ji = np.squeeze(ji).T

        k1, k2, k3, k4 = self._dist_param
        r = np.sqrt(np.square(xyz_[:2]).sum(axis=0))
        theta = np.arctan(r)
        theta2 = theta * theta
        if k3 == k4 == 0.0:
            # odm / opensfm 2 parameter version
            theta_d = theta * (1.0 + theta2 * (k1 + theta2 * k2))
        else:
            # opencv 4 parameter version
            theta_d = theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
        xyz_[:2] *= theta_d / r

        # transform from distorted camera to pixel coordinates
        ji = self._K.dot(xyz_)[:2]
        return ji

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T[None, :].astype('float64', copy=False)
        xyz_ = cv2.fisheye.undistortPoints(ji_cv, self._K, self._dist_param, None, None)
        xyz_ = np.vstack([xyz_[0].T, np.ones((1, ji.shape[1]))])
        return xyz_


def create_camera(cam_type: str | CameraType, *args, **kwargs) -> FrameCamera | RpcCamera:
    """
    Create a camera object given a camera type and parameters.

    :param cam_type: Camera type.
    :param args: Positional arguments to pass to camera constructor.
    :param kwargs: Keyword arguments to pass to camera constructor.
    """
    cam_type = CameraType(cam_type)
    if cam_type == CameraType.brown:
        cam_class = BrownCamera
    elif cam_type == CameraType.fisheye:
        cam_class = FisheyeCamera
    elif cam_type == CameraType.opencv:
        cam_class = OpenCVCamera
    elif cam_type == CameraType.rpc:
        cam_class = RpcCamera
    else:
        cam_class = PinholeCamera

    return cam_class(*args, **kwargs)


##
