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
from os import PathLike
from typing import Sequence

import cv2
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from rasterio.crs import CRS
from rasterio.transform import (
    GCPTransformer,
    GroundControlPoint,
    RPC,
    RPCTransformer,
)
from rasterio.warp import transform

from orthority import utils
from orthority.enums import CameraType, Interp
from orthority.errors import CameraInitError, OrthorityWarning
from orthority.param_io import _opk_to_rotation

logger = logging.getLogger(__name__)


class Camera(ABC):
    """Base camera class."""

    _valid_dtypes = ['uint8', 'uint16', 'int16', 'float32', 'float64']

    @abstractmethod
    def __init__(
        self,
        **kwargs,
    ):
        self._im_size = None

    @property
    def im_size(self) -> tuple[int, int]:
        """Image (width, height) in pixels."""
        return self._im_size

    @staticmethod
    def _validate_world_coords(xyz: np.ndarray) -> None:
        """Utility function to validate world coordinate dimensions."""
        if not (xyz.ndim == 2 and xyz.shape[0] == 3):
            raise ValueError(f"'xyz' should be a 3xN 2D array.")
        if xyz.dtype != np.float64:
            raise ValueError(f"'xyz' should have float64 data type.")

    @staticmethod
    def _validate_pixel_coords(ji: np.ndarray) -> None:
        """Utility function to validate pixel coordinate dimensions."""
        if not (ji.ndim == 2 and ji.shape[0] == 2):
            raise ValueError(f"'ji' should be a 2xN 2D array.")

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Utility function to validate an image dtype and dimensions for remapping."""
        if str(image.dtype) not in Camera._valid_dtypes:
            raise ValueError(f"'image' data type '{image.dtype}' not supported.")
        if not image.ndim == 3:
            raise ValueError("'image' should have 3 dimensions.")

    @staticmethod
    def _get_dtype_nodata(dtype: str) -> float | int:
        """Return a sensible nodata value for the given ``dtype``."""
        return np.nan if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).min

    def _pixel_to_world_surf(
        self,
        ji: np.ndarray,
        z: float | np.ndarray,
        transform: rio.Affine,
        interp: str | Interp = Interp.cubic,
        min_z: float = None,
        max_z: float = None,
    ) -> np.ndarray:
        """Return the world coordinate intersections of rays defined by pixel coordinates ``ji``,
        with the height array (DEM) ``z``.
        """
        # create a transform from world (x, y) to center (j, i) pixel coordinates
        inv_transform = ~(transform * rio.Affine.translation(0.5, 0.5))

        # find / initialise z surface minimum and maximum
        min_z = min_z if min_z is not None else np.nanmin(z)
        max_z = max_z if max_z is not None else np.nanmax(z)

        # find world boundary at z_min and z_max
        min_xyz = self.pixel_to_world_z(ji, min_z)
        max_xyz = self.pixel_to_world_z(ji, max_z)

        # heuristic limit on ray length to conserve memory
        max_ray_steps = 2 * np.sqrt(np.square(z.shape, dtype='int64').sum()).astype('int')
        xyz = np.zeros((3, ji.shape[1]))

        # find z surface (x, y, z) world coordinate intersections for each (j, i) pixel
        # coordinate in ji
        for pi in range(0, ji.shape[1]):
            src_pt, start_xyz, stop_xyz = ji[:, pi], max_xyz[:, pi], min_xyz[:, pi]

            # create world points along the src_pt ray with (x, y) stepsize <= z resolution,
            # if num points <= max_ray_steps, else max_ray_steps points
            ray_steps = np.abs((stop_xyz - start_xyz)[:2].squeeze() / (transform[0], transform[4]))
            ray_steps = min(np.ceil(ray_steps.max()).astype('int') + 1, max_ray_steps)
            ray_z = np.linspace(max_z, min_z, ray_steps)
            # TODO: for frame cameras, linspace rather than pixel_to_world_z can be used to form
            #  the ray
            ray_xyz = self.pixel_to_world_z(src_pt.reshape(-1, 1), ray_z)

            # find the z surface values corresponding to the ray (the remapped surface will be
            # nan outside its bounds and for already masked / nan pixels)
            zsurf_ji = np.array(inv_transform * ray_xyz[:2]).astype('float32', copy=False)
            zsurf_z = np.full((zsurf_ji.shape[1],), dtype=z.dtype, fill_value=float('nan'))
            # dem_ji = cv2.convertMaps(*dem_ji, cv2.CV_16SC2)
            cv2.remap(z, *zsurf_ji, interp.to_cv(), dst=zsurf_z, borderMode=cv2.BORDER_TRANSPARENT)

            # store the first ray-z intersection point if it exists, otherwise the z_min point
            zsurf_min_xyz = ray_xyz[:, -1]
            intersection_i = np.nonzero(ray_xyz[2] <= zsurf_z)[0]
            if len(intersection_i) > 0:
                xyz[:, pi] = ray_xyz[:, intersection_i[0]]
            else:
                xyz[:, pi] = zsurf_min_xyz
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

    def pixel_boundary(self, num_pts: int = None) -> np.ndarray:
        """
        A rectangle of 2D pixel coordinates along the image boundary.

        :param num_pts:
            Number of boundary points to include.  If set to None (the default), eight points are
            included, with points at the image corners and mid-points of the sides.

        :return:
            Boundary pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the
            first dimension.
        """

        def rect_boundary(im_size: np.ndarray, num_pts: int) -> np.ndarray:
            """Return a rectangular pixel coordinate boundary of ``num_pts`` ~evenly spaced points
            for the given image size ``im_size``.
            """
            br = im_size - 1
            perim = 2 * br.sum()
            cnr_ji = np.array([[0, 0], [br[0], 0], br, [0, br[1]], [0, 0]])
            dist = np.sum(np.abs(np.diff(cnr_ji, axis=0)), axis=1)
            return np.row_stack(
                [
                    np.linspace(
                        cnr_ji[i],
                        cnr_ji[i + 1],
                        np.round(num_pts * dist[i] / perim).astype('int'),
                        endpoint=False,
                    )
                    for i in range(0, 4)
                ]
            ).T

        im_size = np.array(self._im_size)
        if not num_pts:
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
        transform: rio.Affine = None,
        interp: str | Interp = Interp.cubic,
        num_pts: int = None,
        **kwargs,
    ) -> np.ndarray:
        """
        A polygon of (x, y, z) world coordinates along the image boundary, at a specified z value
        or surface (DEM).

        :param z:
            Z value(s) as a scalar float or a 2D array (surface).
        :param transform:
            Affine transform defining the (x, y) world coordinates of ``z``.  Required when ``z``
            is an array and not used otherwise.
        :param interp:
            Interpolation method to use for finding boundary intersections with ``z`` when it is an
            array.  Not used when ``z`` is scalar.
        :param num_pts:
            Number of boundary points to include.  If set to None (the default), eight points are
            included, with points at the image corners and mid-points of the sides.
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
            xyz = self._pixel_to_world_surf(ji, z, transform, interp=interp, num_pts=num_pts)
        else:
            raise ValueError("'z' should be a scalar float or 2D array.")
        return xyz

    def read(
        self,
        im_file: str | PathLike | OpenFile | rio.DatasetReader,
        indexes: Sequence[int] | int = None,
        dtype: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Read image band(s) from a given file.  Sub-classes may add a processing

        :param im_file:
            Image file to read from.
        :param indexes:
            Band index(es) to read (1 based).
        :param dtype:
            Data type of the returned array.  If set to None (the default), the ``im_file``
            dtype is used.
        :param kwargs:
            Not used.

        :return:
            Image as 3D array with band(s) along the first dimension (Rasterio ordering).
        """
        env = rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False)
        with utils.suppress_no_georef(), env, utils.OpenRaster(im_file) as im:
            dtype = dtype or im.dtypes[0]
            return im.read(indexes, out_dtype=dtype)

    def remap(
        self,
        im_array: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        nodata: float | int = None,
        interp: str | Interp = Interp.cubic,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remap image to ortho image at given world / ortho coordinates.

        :param im_array:
            Image to remap as a 3D array with band(s) along the first dimension (Rasterio
            ordering).  Typically, this should be the image returned by :meth:`Camera.read`.
        :param x:
            X world / ortho coordinates to remap to, as a M-by-N 2D array.
        :param y:
            Y world / ortho coordinates to remap to, as a M-by-N 2D array.
        :param z:
            Z world / ortho coordinates to remap to, as a M-by-N 2D array.
        :param nodata:
            Value to use for masking invalid pixels in the remapped image.  If set to None
            (the default), a value based on the ``image`` data type is chosen automatically.
        :param interp:
            Interpolation method to use for remapping.
        :param kwargs:
            Not used.

        :return:
            - Remapped image, as a 3D array with band(s) along the first dimension (Rasterio
              ordering).
            - Nodata mask of the remapped image, as a 2D boolean array.
        """
        # TODO: is there a neater or more efficient way to package x, y & z?
        self._validate_image(im_array)
        if not (x.shape == y.shape == z.shape) or (x.ndim != 2):
            raise ValueError("'x', 'y' and 'z' should have 2 dimensions, and the same shape.")
        if nodata is None:
            nodata = self._get_dtype_nodata(im_array.dtype)

        # find (j, i) image pixel coords corresponding to (x, y, z) world coords
        ji = self.world_to_pixel(np.row_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1))))

        # separate ji into (j, i) grids, converting to float32 for compatibility with
        # cv2.remap (nans are converted to -1 as cv2.remap maps nans to 0 (the first src pixel)
        # on some packages/platforms see
        # https://answers.opencv.org/question/1057/behavior-of-not-a-number-nan-values-in-remap/)
        ji[np.isnan(ji)] = -1
        j = ji[0].reshape(*x.shape).astype('float32')
        i = ji[1].reshape(*x.shape).astype('float32')
        # j, i = cv2.convertMaps(j, i, cv2.CV_16SC2)

        # initialise ortho / remapped array
        remap_array = np.full(
            (im_array.shape[0], *x.shape), dtype=im_array.dtype, fill_value=nodata
        )

        # remap image to ortho, looping over band(s) (cv2.remap execution time depends on array
        # ordering)
        # TODO: test speed if src and tile are in cv ordering and this done all bands at once
        for oi in range(0, im_array.shape[0]):
            cv2.remap(
                im_array[oi],
                j,
                i,
                Interp[interp].to_cv(),
                dst=remap_array[oi],
                borderMode=cv2.BORDER_TRANSPARENT,
            )

        # find nodata mask
        remap_mask = np.all(utils.nan_equals(remap_array, nodata), axis=0)
        return remap_array, remap_mask


class RpcCamera(Camera):
    def __init__(self, rpc: RPC | dict, rpc_options: dict | None = None, crs: str | CRS = None):
        super().__init__()
        self._rpc_crs = CRS.from_epsg(4979)
        self._crs = CRS.from_string(crs) if isinstance(crs, str) else crs
        self._transformer = RPCTransformer(rpc, **rpc_options)

    @property
    def crs(self) -> CRS | None:
        return self._crs or self._rpc_crs

    def world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        self._validate_world_coords(xyz)
        if self._crs:
            xyz = transform(self._crs, self._rpc_crs, [xyz[0]], [xyz[1]], [xyz[2]])
        return np.array(self._transformer.rowcol(*xyz))

    @abstractmethod
    def pixel_to_world_z(self, ji: np.ndarray, z: float | np.ndarray, **kwargs) -> np.ndarray:
        self._validate_pixel_coords(ji)
        # TODO: sort out pixel offset here and in world_to_pixel
        xy = self._transformer.xy(ji[1], ji[0], zs=z, offset='ul')
        xyz = np.row_stack([xy, z * np.ones(xy.shape[1])])
        if self._crs:
            xyz = np.array(transform(self._rpc_crs, self._crs, *xyz))
        return xyz


class GcpCamera(Camera):
    def __init__(
        self,
        gcps: Sequence[GroundControlPoint, dict],
    ):
        super().__init__()
        gcps = [GroundControlPoint(gcp) if isinstance(gcp, dict) else gcp for gcp in gcps]
        self._transformer = GCPTransformer(gcps)

    def world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        self._validate_world_coords(xyz)
        return np.array(self._transformer.rowcol(*xyz))

    @abstractmethod
    def pixel_to_world_z(self, ji: np.ndarray, z: float | np.ndarray, **kwargs) -> np.ndarray:
        self._validate_pixel_coords(ji)
        # TODO: sort out pixel offset here and in world_to_pixel
        xy = self._transformer.xy(ji[1], ji[0], zs=z, offset='ul')
        xyz = np.row_stack([xy, z * np.ones(xy.shape[1])])
        return xyz


class FrameCamera(Camera):
    """
    Pinhole camera with no distortion.

    The ``xyz`` and ``opk`` exterior parameters must be supplied here, or via
    :meth:`~FrameCamera.update`, before calling :meth:`~Camera.world_to_pixel` or
    :meth:`~Camera.pixel_to_world_z`.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value
        or (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to None (
         the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
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
        :meth:`read` (True).  Or, exclude distortion from the camera model, and return an
        undistorted image from :meth:`read` (False).
        """
        return self._distort

    @distort.setter
    def distort(self, value: bool) -> None:
        if value:
            self._undistort_maps = None
        self._distort = value

    @property
    def alpha(self) -> float:
        """Scaling (0-1) of the undistorted image returned by :meth:`FrameCamera.read` when
        :attr:`~FrameCamera.distort` is False.  0 includes the largest portion of the source
        image that allows all undistorted pixels to be valid.  1 includes all source pixels in
        the undistorted image. Affects scaling of the camera model intrinsic matrix.  No effect
        when :attr:`~FrameCamera.distort` is True.
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
            raise CameraInitError(f'Exterior parameters not initialised.')

    def _horizon_fov(self) -> bool:
        """Whether this camera's field of view includes, or is above, the horizon."""
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

        ``alpha`` (0-1) controls the portion of the source included in the distorted image. 0
        includes the largest portion of the source image that allows all undistorted pixels to be
        valid.  1 includes all source pixels in the undistorted image.
        """

        # Adapted from and equivalent to:
        # K_undistort, _ = cv2.getOptimalNewCameraMatrix(K, dist_param, im_size, alpha).
        # See https://github.com/opencv/opencv/blob/4790a3732e725b102f6c27858e7b43d78aee2c3e/modules/calib3d/src/calibration.cpp#L2772
        # Note that cv2.fisheye.estimateNewCameraMatrixForUndistortRectify() does not include all
        # source pixels for balance=1.  This method works for all subclasses including fisheye.
        def _get_rectangles(
            im_size: tuple[int, int]
        ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
            """Return inner and outer rectangles for distorted image grid points."""
            w, h = np.array(im_size) - 1
            n = 9
            scale_j, scale_i = np.meshgrid(range(0, n), range(0, n))
            scale_j, scale_i = scale_j.ravel(), scale_i.ravel()
            ji = np.row_stack([scale_j * w / (n - 1), scale_i * h / (n - 1)])
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
        ji_ = np.row_stack([ji.astype('float64', copy=False), np.ones((1, ji.shape[1]))])
        xyz_ = self._K_undistort_inv.dot(ji_)
        return xyz_

    def _undistort_im(
        self, image: np.ndarray, nodata: float | int, interp: str | Interp
    ) -> np.ndarray:
        """Undistort an image using ``interp`` interpolation and setting invalid pixels to
        ``nodata``.
        """
        self._validate_image(image)

        # find undistort maps once on first use
        self._undistort_maps = self._undistort_maps or self._get_undistort_maps()

        if self._undistort_maps is None:
            return image

        def undistort_band(src_array: np.ndarray, dst_array: np.ndarray):
            """Undistort a 2D band array."""
            # equivalent without stored _undistort_maps:
            # return cv2.undistort(band_array, self._K, self._dist_param)
            cv2.remap(
                src_array,
                *self._undistort_maps,
                Interp[interp].to_cv(),
                dst=dst_array,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

        out_image = np.full(image.shape, dtype=image.dtype, fill_value=nodata)
        # TODO: see if using cv ordering throughout (read, undistort, remap) speeds things
        #  up, and or undistort bands concurrently in thread pool
        for bi in range(image.shape[0]):
            undistort_band(image[bi], out_image[bi])

        return out_image

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
        self._test_init()
        self._validate_pixel_coords(ji)

        if isinstance(z, np.ndarray) and (
            z.ndim != 1 or (z.shape[0] != 1 and ji.shape[1] != 1 and z.shape[0] != ji.shape[1])
        ):
            raise ValueError(
                f"'z' should be a single value or 1-by-N array where 'ji' is 2-by-N or 2-by-1."
            )

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

    def undistort_pixel(self, ji: np.ndarray, clip: bool = False) -> np.ndarray:
        """
        Undistort pixel coordinates.

        :param ji:
            Pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the first
            dimension.
        :param clip:
            Whether to clip the undistorted coordinates to the image dimensions.

        :return:
            Undistorted pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along
            the first dimension.
        """
        self._validate_pixel_coords(ji)

        xyz_ = self._pixel_to_camera(ji)
        ji = FrameCamera._camera_to_pixel(self, xyz_)

        if clip:
            ji = np.clip(ji.T, a_min=(0, 0), a_max=np.array(self._im_size) - 1).T
        return ji

    def pixel_boundary(self, num_pts: int = None) -> np.ndarray:
        """
        A polygon of 2D pixel coordinates along the image boundary.  If
        :attr:`~FrameCamera.distort` is False, coordinates will be along the boundary of the
        valid area in the undistorted image returned by :meth:`~FrameCamera.read`.

        :param num_pts:
            Number of boundary points to include in the polygon.  If set to None (the default),
            eight points are included, with points at the image corners and mid-points of the sides.

        :return:
            Boundary pixel (j=column, i=row) coordinates as a 2-by-N array, with (j, i) along the
            first dimension.
        """
        ji = super().pixel_boundary(num_pts=num_pts)

        if not self._distort:
            ji = self.undistort_pixel(
                ji, clip=True
            )  # TODO: decide whether we need undistort_pixel, or just do it inline below  # #
            # undistort ji and clip to image bounds  # xyz_ = self._pixel_to_camera(ji)  # ji =
            # FrameCamera._camera_to_pixel(self, xyz_)  # ji = np.clip(ji.T, a_min=(0, 0),
            # a_max=np.array(self._im_size) - 1).T
        return ji

    def world_boundary(
        self,
        z: float | np.ndarray,
        num_pts: int = None,
        transform: rio.Affine = None,
        interp: str | Interp = Interp.cubic,
        clip: bool = True,
    ) -> np.ndarray:
        """
        A polygon of (x, y, z) world coordinates along the image boundary, at a specified z value
        or surface (DEM).

        :param z:
            Z value(s) as a scalar float or a 2D array (surface).
        :param transform:
            Affine transform defining the (x, y) world coordinates of ``z``.  Required when ``z``
            is an array and not used otherwise.
        :param interp:
            Interpolation method to use for finding boundary intersections with ``z`` when it is an
            array.  Not used when ``z`` is scalar.
        :param num_pts:
            Number of boundary points to include.  If set to None (the default), eight points are
            included, with points at the image corners and mid-points of the sides.
        :param clip:
            Clip the z coordinate of boundary points to the camera height.

        :return:
            Boundary world (x, y, z) coordinates as a 3-by-N array, with (x, y, z) along the
            first dimension.  Boundary points that lie outside ``z`` bounds, when ``z`` is an
            array, are given at the minimum of ``z``.
        """
        self._test_init()
        # TODO: is this check too restrictive?  this would prevent horizontal views e.g. inside
        #  rugged terrain being orthorectified
        if self._horizon_fov():
            raise ValueError("Camera has a field of view that includes, or is above, the horizon.")

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
            raise ValueError("'z' should be a scalar float or 2D array.")
        return xyz

    def read(
        self,
        im_file: str | PathLike | OpenFile | rio.DatasetReader,
        indexes: Sequence[int] = None,
        dtype: str = None,
        nodata: float | int = None,
        interp: str | Interp = Interp.cubic,
    ) -> np.ndarray:
        """
        Read image band(s) from a given file, undistorting when :attr:`~FrameCamera.distort` is
        False.

        :param im_file:
            Image file to read from.
        :param indexes:
            Band index(es) to read (1 based).
        :param dtype:
            Data type of the returned array.  If set to None (the default), the ``im_file``
            dtype is used.
        :param nodata:
            Value to use for masking invalid pixels in the undistorted image.  If set to None
            (the default), a value based on ``dtype`` is chosen automatically.  Not used if
            :attr:`~FrameCamera.distort` is True.
        :param interp:
            Interpolation method to use when undistorting the image.  Not used if
            :attr:`~FrameCamera.distort` is True.

        :return:
            Image as 3D array with band(s) along the first dimension (Rasterio ordering).
        """
        image = super().read(im_file, indexes=indexes, dtype=dtype)

        if not self._distort:
            if nodata is None:
                nodata = self._get_dtype_nodata(image.dtype)
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
        Remap image to ortho image at given world / ortho coordinates.

        :param image:
            Image to remap as a 3D array with band(s) along the first dimension (Rasterio
            ordering).  Typically, this should be the image returned by :meth:`FrameCamera.read`.
        :param x:
            X world / ortho coordinates to remap to, as a M-by-N 2D array.
        :param y:
            Y world / ortho coordinates to remap to, as a M-by-N 2D array.
        :param z:
            Z world / ortho coordinates to remap to, as a M-by-N 2D array.
        :param nodata:
            Value to use for masking invalid pixels in the remapped image.  If set to None
            (the default), a value based on the ``image`` data type is chosen automatically.
        :param interp:
            Interpolation method to use for remapping.
        :param kernel_size:
            Kernel (width, height) size in pixels, used for dilating the nodata mask.  Removes
            blurring of boundary pixels with nodata areas in an undistorted ``image``.  Not used
            if blurring could not have occurred (e.g. if :attr:`~FrameCamera.distort` is True).

        :return:
            - Remapped image, as a 3D array with band(s) along the first dimension (Rasterio
              ordering).
            - Nodata mask of the remapped image, as a 2D boolean array.
        """
        remap, mask = super().remap(image, x, y, z, nodata=nodata, interp=interp)

        # remove blurring with nodata pixels when necessary
        if (
            not self.distort
            and Interp[interp] != Interp.nearest
            and not np.isnan(nodata)
            and not type(self) == PinholeCamera
        ):
            kernel = np.ones(kernel_size[::-1], np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8, copy=False), kernel)
            mask = mask.astype(bool, copy=False)
            if nodata is None:
                nodata = self._get_dtype_nodata(image.dtype)
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
    :meth:`~FrameCamera.update`, before calling :meth:`~Camera.world_to_pixel` or
    :meth:`~Camera.pixel_to_world_z`.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or
        (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to None (
         the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
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
        :meth:`read` (True).  Or, exclude distortion from the camera model, and return an
        undistorted image from :meth:`read` (False).  :meth:`~FrameCamera.remap` of a
        :meth:`~FrameCamera.read` image is faster with ``distort=False``, but may reduce remap
        quality.  Can be read or changed after initialisation with :attr:`~FrameCamera.distort`.
    :param alpha:
        Scaling (0-1) of the undistorted image returned by :meth:`FrameCamera.read` when
        ``distort`` is False.  0 includes the largest portion of the source image that allows all
        undistorted pixels to be valid.  1 includes all source pixels in the undistorted image.
        Affects scaling of the camera model intrinsic matrix.  Not used when ``distort`` is True.
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
        # TODO: experiment with different map types and undistort speed
        undistort_maps = cv2.initUndistortRectifyMap(
            self._K, self._dist_param, np.eye(3), self._K_undistort, im_size, cv2.CV_16SC2
        )
        # undistort_maps = cv2.convertMaps(*undistort_maps, cv2.CV_16SC2)
        return undistort_maps

    def _camera_to_pixel(self, xyz_: np.ndarray) -> np.ndarray:
        # omit world to camera rotation & translation to transform from camera to pixel coords
        ji, _ = cv2.projectPoints(xyz_.T, np.zeros(3), np.zeros(3), self._K, self._dist_param)
        return ji[:, 0, :].T

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T.astype('float64', copy=False)
        xyz_ = cv2.undistortPoints(ji_cv, self._K, self._dist_param)
        xyz_ = np.row_stack([xyz_[:, 0, :].T, np.ones((1, ji.shape[1]))])
        return xyz_


class BrownCamera(OpenCVCamera):
    """
    Brown-Conrady camera model.

    Compatible with `OpenDroneMap / OpenSfM
    <https://opensfm.org/docs/geometry.html#camera-models>`__ ``perspective``, ``simple_radial``,
    ``radial`` and ``brown`` model parameters, and the 4- and 5-coefficient versions of the
    `OpenCV general model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.

    The ``xyz`` and ``opk`` exterior parameters must be supplied here, or via
    :meth:`~FrameCamera.update`, before calling :meth:`~Camera.world_to_pixel` or
    :meth:`~Camera.pixel_to_world_z`.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or
        (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to None (
         the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
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
        :meth:`read` (True).  Or, exclude distortion from the camera model, and return an
        undistorted image from :meth:`read` (False).  :meth:`~FrameCamera.remap` of a
        :meth:`~FrameCamera.read` image is faster with ``distort=False``, but may reduce remap
        quality.  Can be read or changed after initialisation with :attr:`~FrameCamera.distort`.
    :param alpha:
        Scaling (0-1) of the undistorted image returned by :meth:`FrameCamera.read` when
        ``distort`` is False.  0 includes the largest portion of the source image that allows all
        undistorted pixels to be valid.  1 includes all source pixels in the undistorted image.
        Affects scaling of the camera model intrinsic matrix.  Not used when ``distort`` is True.
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
    :meth:`~FrameCamera.update`, before calling :meth:`~Camera.world_to_pixel` or
    :meth:`~Camera.pixel_to_world_z`.

    :param im_size:
        Image (width, height) in pixels.
    :param focal_len:
        Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or
        (x, y) tuple.
    :param sensor_size:
         Sensor (width, height) with the same units/scale as ``focal_len``.  If set to None (
         the default), pixels are assumed square and ``focal_len`` normalised and unitless (i.e.
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
        :meth:`read` (True).  Or, exclude distortion from the camera model, and return an
        undistorted image from :meth:`read` (False).  :meth:`~FrameCamera.remap` of a
        :meth:`~FrameCamera.read` image is faster with ``distort=False``, but may reduce remap
        quality.  Can be read or changed after initialisation with :attr:`~FrameCamera.distort`.
    :param alpha:
        Scaling (0-1) of the undistorted image returned by :meth:`FrameCamera.read` when
        ``distort`` is False.  0 includes the largest portion of the source image that allows all
        undistorted pixels to be valid.  1 includes all source pixels in the undistorted image.
        Affects scaling of the camera model intrinsic matrix.  Not used when ``distort`` is True.
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
            self._K, self._dist_param, np.eye(3), self._K_undistort, im_size, cv2.CV_16SC2
        )
        # undistort_maps = cv2.convertMaps(*undistort_maps, cv2.CV_16SC2)
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
        xyz_ = np.row_stack([xyz_[0].T, np.ones((1, ji.shape[1]))])
        return xyz_


def create_camera(cam_type: str | CameraType, *args, **kwargs) -> Camera | FrameCamera:
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
    else:
        cam_class = PinholeCamera

    return cam_class(*args, **kwargs)


##
