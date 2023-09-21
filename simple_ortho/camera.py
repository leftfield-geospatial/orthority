"""
   Copyright 2021 Dugal Harris - dugalh@gmail.com

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

import logging
from typing import Union, Tuple, Optional

import cv2
import numpy as np

from simple_ortho.enums import CameraType
from simple_ortho.errors import CameraInitError
from simple_ortho.io import _opk_to_rotation

logger = logging.getLogger(__name__)


class Camera:
    _default_alpha: float = 1.
    # TODO: only pass intrinsic param on __init__, then extrinsic on update or similar (?)

    def __init__(
        self, im_size: Union[Tuple[int, int], np.ndarray], focal_len: Union[float, Tuple[float, float], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None, cx: float = 0, cy: float = 0,
        xyz: Union[Tuple[float, float, float], np.ndarray] = None,
        opk: Union[Tuple[float, float, float], np.ndarray] = None, alpha: float = _default_alpha
    ):  # yapf: disable
        """
        Pinhole camera class, without any distortion model, for transforming between 3D world and 2D pixel coordinates.

        Parameters
        ----------
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        cx, cy: float, optional
            Principal point offsets in `normalised image coordinates
            <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`_.
        xyz: tuple of float, ndarray, optional
            Camera position (x=easting, y=northing, z=altitude) in world coordinates.
        opk: tuple of float, ndarray, optional
            Camera (omega, phi, kappa) angles in radians to rotate camera (PATB convention) to world coordinates.
        alpha: float
            Undistorted image scaling (0-1).  0 results in an undistorted image with all valid pixels.  1 results in an
            undistorted image that keeps all source pixels.  Not used for the pinhole camera model.
        """
        self._im_size = im_size
        self._K = self._get_intrinsic(im_size, focal_len, sensor_size, cx, cy)
        self._R, self._T = self._get_extrinsic(xyz, opk)
        self._undistort_maps = None
        self._K_undistort = self._K

    @staticmethod
    def _get_intrinsic(
        im_size: Union[Tuple[int, int], np.ndarray], focal_len: Union[float, Tuple[float, float], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]], cx: float, cy: float
    ) -> np.ndarray:
        """ Return the camera intrinsic matrix for the given interior parameters. """
        # Adapted from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera
        # -Parameters-defined and https://en.wikipedia.org/wiki/Camera_resectioning
        # TODO: incorporate orientation from exif

        if len(im_size) != 2:
            raise ValueError('`im_size` should contain 2 values: (width, height).')
        im_size = np.array(im_size)
        if sensor_size is not None and len(sensor_size) != 2:
            raise ValueError('`sensor_size` should contain 2 values: (width, height).')
        focal_len = np.array(focal_len)
        if focal_len.size > 2:
            raise ValueError('`focal_len` should contain at most 2 values.')

        # find the xy focal lengths in pixels
        if sensor_size is None:
            logger.warning(
                '`sensor_size` not specified, assuming square pixels and `focal_len` normalised by sensor width.'
            )
            sigma_xy = (focal_len * im_size[0]) * np.ones(2)
        else:
            sensor_size = np.array(sensor_size)
            sigma_xy = focal_len * im_size / sensor_size

        # principal point
        c_xy = (im_size - 1) / 2
        c_xy += im_size.max() * np.array((cx, cy))

        # intrinsic matrix to convert from camera co-ords in OpenSfM / OpenCV convention (x->right, y->down,
        # z->forwards, looking through the camera at the scene) to pixel co-ords in standard convention (x->right,
        # y->down).
        K = np.array([[sigma_xy[0], 0, c_xy[0]], [0, sigma_xy[1], c_xy[1]], [0, 0, 1]])
        return K

    @staticmethod
    def _get_extrinsic(
        xyz: Union[Tuple[float, float, float], np.ndarray], opk: Union[Tuple[float, float, float], np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """ Return the rotation matrix and translation vector for the given exterior parameters. """
        if xyz is None or opk is None:
            return None, None
        elif len(xyz) != 3 or len(opk) != 3:
            raise ValueError('`xyz` and `opk` should contain 3 values.')

        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters
        # -defined
        T = np.array(xyz).reshape(-1, 1)
        R = _opk_to_rotation(opk)

        # rotate from PATB (x->right, y->up, z->backwards looking through the camera at the scene) to OpenSfM / OpenCV
        # convention (x->right, y->down, z->forwards, looking through the camera at the scene)
        R = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        return R, T

    @staticmethod
    def _check_world_coordinates(xyz: np.ndarray):
        """ Utility function to check world coordinate dimensions. """
        if not (xyz.ndim == 2 and xyz.shape[0] == 3):
            raise ValueError(f'`xyz` should be a 3xN 2D array.')
        if xyz.dtype != np.float64:
            raise ValueError(f'`xyz` should have float64 data type.')

    @staticmethod
    def _check_pixel_coordinates(ji: np.ndarray):
        """ Utility function to check pixel coordinate dimensions. """
        if not (ji.ndim == 2 and ji.shape[0] == 2):
            raise ValueError(f'`ji` should be a 2xN 2D array.')

    def _check_init(self):
        """ Utility function to check if exterior parameters are initialised. """
        if self._R is None or self._T is None:
            raise CameraInitError(f'Exterior parameters not initialised.')

    def _horizon_fov(self) -> bool:
        """ Return True if this camera's field of view includes, or is above, the horizon; otherwise False. """
        # camera coords for image boundary
        # TODO: check image world / camera coord bounds are found similarly elsewhere
        w, h = np.array(self._im_size) - 1
        src_ji = np.array([
            [0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]
        ]).T  # yapf: disable
        xyz_ = self._pixel_to_camera(src_ji)

        # rotate camera to world alignment & test if any z vals are above the camera / origin
        xyz_r = self._R.dot(xyz_)
        return np.any(xyz_r[2] >= 0)

    def _get_undistort_intrinsic(self, alpha: float):
        """
        Return a new camera intrinsic matrix for an undistorted image that is the same size as the source image.
        `alpha` (0-1) controls the portion of the source included in the distorted image.
        For `alpha`=1, the undistorted image includes all source pixels and some invalid (nodata) areas. For `alpha`=0,
        the undistorted image includes the largest portion of the source image that allows all undistorted pixels to be
        valid.
        """
        # Adapted from and equivalent to cv2.getOptimalNewCameraMatrix(newImageSize=self._im_size,
        # centerPrincipalPoint=False).
        # See https://github.com/opencv/opencv/blob/4790a3732e725b102f6c27858e7b43d78aee2c3e/modules/calib3d/src
        # /calibration.cpp#L2772
        def _get_rectangles(im_size: Union[Tuple[int, int], np.ndarray]):
            """ Return inner and outer rectangles for distorted image grid points. """
            w, h = np.array(im_size) - 1
            n = 9
            scale_j, scale_i = np.meshgrid(range(0, n), range(0, n))
            scale_j, scale_i = scale_j.flatten(), scale_i.flatten()
            ji = np.row_stack([scale_j * w / (n - 1), scale_i * h / (n - 1)])
            xy = self._pixel_to_camera(ji)[:2]
            outer = xy.min(axis=1), xy.max(axis=1) - xy.min(axis=1)
            inner_ul = np.array((xy[0][scale_j == 0].max(), xy[1][scale_i == 0].max()))
            inner_br = np.array((xy[0][scale_j == n - 1].min(), xy[1][scale_i == n - 1].min()))
            inner = inner_ul, inner_br - inner_ul
            return inner, outer

        alpha = np.clip(alpha, a_min=0, a_max=1)
        im_size = np.array(self._im_size)
        (inner_off, inner_size), (outer_off, outer_size) = _get_rectangles(im_size)

        f0 = (im_size - 1) / inner_size
        c0 = -f0 * inner_off
        f1 = (im_size - 1) / outer_size
        c1 = -f1 * outer_off
        f = f0 * (1 - alpha) + f1 * alpha
        c = c0 * (1 - alpha) + c1 * alpha

        K_undistort = np.eye(3)
        K_undistort[[0, 1], [0, 1]] = f
        K_undistort[:2, 2] = c
        return K_undistort

    def _get_undistort_maps(self, alpha: float) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        # TODO: make a design decision if internal methods can have keyword args with default values or should be
        #  forced to positional args
        """" Return cv2.remap() maps for undistorting an image, and intrinsic matrix for undistorted image. """
        return None, self._K

    def _world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        """ Transform from 3D world to 2D pixel coordinates. """
        # transform from world to camera coordinates
        xyz_ = self._R.T.dot(xyz - self._T)
        # normalise, and transform to pixel coordinates
        ji = self._K_undistort.dot(xyz_ / xyz_[2, :])[:2, :]
        return ji

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        """ Transform 2D pixel to 3D camera coordinates. """
        ji_ = np.row_stack([ji.astype('float64', copy=False), np.ones((1, ji.shape[1]))])
        xyz_ = np.linalg.inv(self._K_undistort).dot(ji_)
        return xyz_

    def update(
        self, xyz: Union[Tuple[float, float, float], np.ndarray], opk: Union[Tuple[float, float, float], np.ndarray]
    ):
        """
        Update exterior parameters.

        Parameters
        ----------
        xyz: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world coordinates.
        opk: tuple of float, ndarray
            Camera (omega, phi, kappa) angles in radians to rotate camera (PATB convention) to world coordinates.
        """
        self._R, self._T = self._get_extrinsic(xyz, opk)

    def world_to_pixel(self, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
        """
        Transform from 3D world to 2D pixel coordinates.

        Parameters
        ----------
        xyz : ndarray
            3D world (x=easting, y=northing, z=altitude) coordinates to transform, as a 3-by-N array with (x, y, z)
            along the first dimension.
        distort : bool (optional)
            Whether to include the distortion model.

        Returns
        -------
        ndarray
            Pixel (j=column, i=row) coordinates, as a 2-by-N array with (j, i) along the first dimension.
        """
        self._check_init()
        self._check_world_coordinates(xyz)
        return self._world_to_pixel(xyz) if distort else Camera._world_to_pixel(self, xyz)

    def pixel_to_world_z(self, ji: np.ndarray, z: Union[float, np.ndarray], distort: bool = True) -> np.ndarray:
        """
        Transform from 2D pixel to 3D world coordinates at a specified z (altitude).

        Parameters
        ----------
        ji: ndarray
            Pixel (j=column, i=row) coordinates, as a 2-by-N array with (j, i) along the first dimension.
        z: float, ndarray
            Z altitude(s) to project to, as a single value, or 1-by-N array where ``ji`` is 2-by-N or 2-by-1.
        distort : bool (optional)
            Whether to include the distortion model.

        Returns
        -------
        ndarray
            3D world (x=easting, y=northing, z=altitude) coordinates, as a 3-by-N array with (x, y, z) along the first
            dimension.
        """
        self._check_init()
        self._check_pixel_coordinates(ji)

        if (
            isinstance(z, np.ndarray) and
            (z.ndim != 1 or (z.shape[0] != 1 and ji.shape[1] != 1 and z.shape[0] != ji.shape[1]))
        ):  # yapf: disable
            raise ValueError(f'`z` should be a single value or 1-by-N array where `ji` is 2-by-N or 2-by-1.')

        # transform pixel coordinates to camera coordinates
        xyz_ = self._pixel_to_camera(ji) if distort else Camera._pixel_to_camera(self, ji)
        # rotate first (camera to world) to get world aligned axes with origin on the camera
        xyz_r = self._R.dot(xyz_)
        # scale to z (offset for camera z) with origin on camera, then offset to world
        xyz = (xyz_r * (z - self._T[2]) / xyz_r[2]) + self._T
        return xyz

    def undistort(self, ji: np.ndarray, clip: bool = False) -> np.ndarray:
        """
        Undistort pixel coordinates.

        Parameters
        ----------
        ji: ndarray
            Pixel (j=column, i=row) coordinates, as a 2-by-N array with (j, i) along the first dimension.
        clip: bool
            Whether to clip the undistorted coordinates to the image dimensions.

        Returns
        -------
        ndarray
            Undistorted pixel (j=column, i=row) coordinates, as a 2-by-N array with (j, i) along the first dimension.
        """
        self._check_init()
        self._check_pixel_coordinates(ji)

        xyz_ = self._pixel_to_camera(ji)
        ji = self._K_undistort.dot(xyz_ / xyz_[2, :])[:2, :]

        if clip:
            ji = np.clip(ji.T, a_min=(0, 0), a_max=np.array(self._im_size) - 1).T

        return ji


class PinholeCamera(Camera):
    """"""


class OpenCVCamera(Camera):

    def __init__(
        self, im_size: Union[Tuple[int, int], np.ndarray], focal_len: Union[float, Tuple[float, float], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None, cx: float = 0, cy: float = 0,
        k1: float = 0., k2: float = 0., k3: float = 0., p1: float = 0., p2: float = 0., k4: float = 0., k5: float = 0.,
        k6: float = 0., s1: float = 0., s2: float = 0., s3: float = 0., s4: float = 0., t1: float = 0., t2: float = 0.,
        xyz: Union[Tuple[float, float, float], np.ndarray] = None,
        opk: Union[Tuple[float, float, float], np.ndarray] = None, alpha: float = Camera._default_alpha
    ):  # yapf: disable
        """
        Camera class with OpenCV distortion model, for transforming between 3D world and 2D pixel coordinates, and
        undistorting images.

        This is a wrapper around the `OpenCV general model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_
        that includes radial, tangential & thin prism distortion components.  Partial or special cases of the model can
        be computed by omitting some or all of the distortion coefficients; e.g. if no distortion coefficients are
        specified, this model corresponds to :class:`PinholeCamera`, or if the first 5 distortion coefficients are
        specified, this model corresponds to :class:`BrownCamera` with (cx, cy) = (0, 0).

        Parameters
        ----------
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        cx, cy: float, optional
            Principal point offsets in `normalised image coordinates
            <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`_.
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty: float, optional
            OpenCV distortion coefficients - see their `docs <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_
            for details.
        xyz: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world coordinates.
        opk: tuple of float, ndarray
            Camera (omega, phi, kappa) angles in radians to rotate camera (PATB convention) to world coordinates.
        alpha: float
            Undistorted image scaling (0-1).  0 results in an undistorted image with all valid pixels.  1 results in an
            undistorted image that keeps all source pixels.
        """
        Camera.__init__(self, im_size, focal_len, sensor_size=sensor_size, cx=cx, cy=cy, xyz=xyz, opk=opk)

        # order _dist_param & truncate zeros according to OpenCV docs
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        self._dist_param = np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, t1, t2])
        for dist_len in (4, 5, 8, 12, 14):
            if np.all(self._dist_param[dist_len:] == 0.):
                self._dist_param = self._dist_param[:dist_len]
                break
        self._undistort_maps, self._K_undistort = self._get_undistort_maps(alpha)

    def _get_undistort_maps(self, alpha: float) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        # yapf:
        # disable
        # TODO: expose alpha to the API
        # TODO: is centerPrincipalPoint=True necessary to avoid nodata ortho borders?
        # TODO: add tests for alpha (e.g. 1~=full_remap=True, 0<full_remap=True (both valid and entire image areas)
        # TODO: test --no-full-remap with fisheye
        im_size = np.array(self._im_size).astype(int)
        # K_undistort, _ = cv2.getOptimalNewCameraMatrix(K, dist_param, im_size, alpha)  # cv2 equivalent
        K_undistort = self._get_undistort_intrinsic(alpha)
        undistort_maps = cv2.initUndistortRectifyMap(
            self._K, self._dist_param, np.eye(3), K_undistort, im_size, cv2.CV_16SC2
        )
        return undistort_maps, K_undistort

    def _world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        inv_aa = cv2.Rodrigues(self._R.T)[0]  # inverse rotation in angle axis format
        ji, _ = cv2.projectPoints((xyz - self._T).T, inv_aa, np.zeros(3), self._K, self._dist_param)
        return ji[:, 0, :].T

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T.astype('float64', copy=False)
        xyz_ = cv2.undistortPoints(ji_cv, self._K, self._dist_param)
        xyz_ = np.row_stack([xyz_[:, 0, :].T, np.ones((1, ji.shape[1]))])
        return xyz_


class BrownCamera(OpenCVCamera):

    def __init__(
        self, im_size: Union[Tuple[int, int], np.ndarray], focal_len: Union[float, Tuple[float, float], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None, cx: float = 0., cy: float = 0.,
        k1: float = 0., k2: float = 0., p1: float = 0., p2: float = 0., k3: float = 0.,
        xyz: Union[Tuple[float, float, float], np.ndarray] = None,
        opk: Union[Tuple[float, float, float], np.ndarray] = None, alpha: float = Camera._default_alpha
    ):  # yapf: disable
        """
        Camera class with Brown-Conrady distortion for transforming between 3D world and 2D pixel coordinates, and
        undistorting images.

        The distortion model is compatible with `OpenDroneMap (ODM)
        <https://docs.opendronemap.org/arguments/camera-lens/>`_ / `OpenSFM <https://github.com/mapillary/OpenSfM>`_
        *brown* parameter estimates; and the 4- & 5-coefficient version of the `general OpenCV distortion model
        <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_.  ODM places their estimates in
        *<dataset path>/cameras.json*, and OpenSFM in *<dataset path>/camera_models.json*.

        Parameters
        ----------
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        cx, cy: float, optional
            Principal point offsets in `normalised image coordinates
            <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`_.
        k1, k2, p1, p2, k3: float, optional
            Brown model distortion coefficients.
        xyz: tuple of float, ndarray, optional
            Camera position (x=easting, y=northing, z=altitude) in world coordinates.
        opk: tuple of float, ndarray, optional
            Camera (omega, phi, kappa) angles in radians to rotate camera (PATB convention) to world coordinates.
        alpha: float
            Undistorted image scaling (0-1).  0 results in an undistorted image with all valid pixels.  1 results in an
            undistorted image that keeps all source pixels.
        """
        Camera.__init__(self, im_size, focal_len, sensor_size=sensor_size, cx=cx, cy=cy, xyz=xyz, opk=opk)

        self._dist_param = np.array([k1, k2, p1, p2, k3])
        self._undistort_maps, self._K_undistort = self._get_undistort_maps(alpha)

    def _world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        # transform from world to camera coordinates, and normalise
        xyz_ = self._R.T.dot(xyz - self._T)
        xyz_ = xyz_ / xyz_[2, :]

        # Brown model adapted from the OpenSFM implementation:
        # https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle.h
        # #LL299C25-L299C25.
        # Works out faster than the opencv equivalent in OpenCVCamera.world_to_pixel().
        k1, k2, p1, p2, k3 = self._dist_param
        x2, y2 = np.square(xyz_[:2, :])
        xy = xyz_[0, :] * xyz_[1, :]
        r2 = x2 + y2

        radial_dist = 1. + r2 * (k1 + r2 * (k2 + r2 * k3))
        x_tangential_dist = 2. * p1 * xy + p2 * (r2 + 2.0 * x2)
        y_tangential_dist = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy

        xyz_[0, :] = xyz_[0, :] * radial_dist + x_tangential_dist
        xyz_[1, :] = xyz_[1, :] * radial_dist + y_tangential_dist

        # transform from distorted camera to pixel coordinates
        ji = self._K.dot(xyz_)[:2, :]
        return ji

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T.astype('float64', copy=False)
        xyz_ = cv2.undistortPoints(ji_cv, self._K, self._dist_param)
        xyz_ = np.row_stack([xyz_[:, 0, :].T, np.ones((1, ji.shape[1]))])
        return xyz_


class FisheyeCamera(Camera):

    def __init__(
        self, im_size: Union[Tuple[int, int], np.ndarray], focal_len: Union[float, Tuple[float, float], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None, cx: float = 0., cy: float = 0.,
        k1: float = 0., k2: float = 0., k3: float = 0., k4: float = 0.,
        xyz: Union[Tuple[float, float, float], np.ndarray] = None,
        opk: Union[Tuple[float, float, float], np.ndarray] = None, alpha: float = Camera._default_alpha
    ):  # yapf: disable
        """
        Camera class with fisheye distortion for transforming between 3D world and 2D pixel coordinates, and
        undistorting images.

        The distortion model is compatible with `OpenDroneMap (ODM)
        <https://docs.opendronemap.org/arguments/camera-lens/>`_ / `OpenSFM <https://github.com/mapillary/OpenSfM>`_
        *fisheye* parameter estimates; and the `OpenCV fisheye distortion model
        <https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html>`_.  ODM places their estimates in
        *<dataset path>/cameras.json*, and OpenSFM in *<dataset path>/camera_models.json*.

        Parameters
        ----------
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        cx, cy: float, optional
            Principal point offsets in `normalised image coordinates
            <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`_.
        k1, k2, k3, k4: float, optional
            Fisheye distortion coefficients.  OpenCV uses all coefficients, while ODM and OpenSFM use k1 & k2 only.
        xyz: tuple of float, ndarray, optional
            Camera position (x=easting, y=northing, z=altitude) in world coordinates.
        opk: tuple of float, ndarray, optional
            Camera (omega, phi, kappa) angles in radians to rotate camera (PATB convention) to world coordinates.
        alpha: float
            Undistorted image scaling (0-1).  0 results in an undistorted image with all valid pixels.  1 results in an
            undistorted image that keeps all source pixels.
        """
        Camera.__init__(self, im_size, focal_len, sensor_size=sensor_size, cx=cx, cy=cy, xyz=xyz, opk=opk)

        self._dist_param = np.array([k1, k2, k3, k4])
        self._undistort_maps, self._K_undistort = self._get_undistort_maps(alpha)

    def _get_undistort_maps(self, alpha: float) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        im_size = np.array(self._im_size).astype(int)

        # use internal method to get K_undistort as cv2.fisheye.estimateNewCameraMatrixForUndistortRectify() does not
        # include all source pixels for balance=1.
        K_undistort = self._get_undistort_intrinsic(alpha)

        # unlike cv2.initUndistortRectifyMap(), cv2.fisheye.initUndistortRectifyMap() requires default R & P
        # (new camera matrix) params to be specified
        undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            self._K, self._dist_param, np.eye(3), K_undistort, im_size, cv2.CV_16SC2
        )
        return undistort_maps, K_undistort

    def _world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        # transform from world to camera coordinates, and normalise
        xyz_ = self._R.T.dot(xyz - self._T)
        xyz_ = xyz_ / xyz_[2, :]
        # Fisheye distortion adapted from the OpenSFM implementation:
        # https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle.h
        # #L365.
        # and OpenCV docs: https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html.
        # Works out faster than the opencv equivalent:
        #   x_cv = np.expand_dims((x - self._T).T, axis=0)
        #   ji, _ = cv2.fisheye.projectPoints(x_cv, self._inv_aa, np.zeros(3), self._K, self._dist_param)
        #   ji = np.squeeze(ji).T

        k1, k2, k3, k4 = self._dist_param
        r = np.sqrt(np.square(xyz_[:2, :]).sum(axis=0))
        theta = np.arctan(r)
        theta2 = theta * theta
        if k3 == k4 == 0.:
            # odm 2 parameter version
            theta_d = theta * (1.0 + theta2 * (k1 + theta2 * k2))
        else:
            # opencv 4 parameter version
            theta_d = theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
        xyz_[:2, :] *= theta_d / r
        # transform from distorted camera to pixel coordinates
        ji = self._K.dot(xyz_)[:2, :]
        return ji

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T[None, :].astype('float64', copy=False)
        xyz_ = cv2.fisheye.undistortPoints(ji_cv, self._K, self._dist_param, None, None)
        xyz_ = np.row_stack([xyz_[0].T, np.ones((1, ji.shape[1]))])
        return xyz_


def create_camera(cam_type: CameraType, *args, **kwargs) -> Camera:
    """
    Create a camera object given a camera type and its parameters.

    Parameters
    ----------
    cam_type: CameraType
        Camera type (pinhole, brown, fisheye, opencv).
    args:
        Positional arguments to pass to camera constructor.
    kwargs:
        Keyword arguments to pass to camera constructor.

    Returns
    -------
    Camera
        The created camera object.
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

# TODO: rotation should be specified in a more general way e.g. angle axis in the ODM co-ordinate
#  convention.  OPK etc conversions can be done externally.
# TODO: Allow the intrinsic principal point to be specified in a way that doesn't conflict with the ODM brown model
#  principal point (e.g. all cameras take cx, cy offset from center in pixels, and ODM cx, cy are converted to pixels
#  before passing to *Camera, or all cameras take normalised cx, cy offsets?)
# TODO: call ortho coordinates everywhere world coordinates
# TODO: rename distort and or full_remap to something consistent and meaningful to user
# TODO: user super rather than Camera.
# TODO: normalise by width or by max(width, height)
##
