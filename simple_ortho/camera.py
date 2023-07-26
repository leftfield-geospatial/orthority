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

from simple_ortho.enums import CameraType, Interp

logger = logging.getLogger(__name__)


class Camera:

    def __init__(
        self, position: Union[Tuple[float, float, float], np.ndarray],
        rotation: Union[Tuple[float, float, float], np.ndarray],
        focal_len: Union[float, Tuple[float, float], np.ndarray], im_size: Union[Tuple[int, int], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None,
    ):  # yapf: disable
        """
        Pinhole camera class, without any distortion model, for transforming between 3D world and 2D pixel co-ordinates.

        Parameters
        ----------
        position: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world co-ordinates.  If the Camera object is being
            used to generate an ortho image with :class:`~simple_ortho.ortho.Ortho`, ``position`` should be in the
            ortho image CRS.
        rotation: tuple of float, ndarray
            Camera (omega, phi, kappa) angles in radians to rotate camera to world co-ordinates, where camera
            co-ordinates are in PATB convention (i.e. x->right, y->up and z->backwards, looking through the camera at
            the scene).
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        """

        self._T, self._R, self._Rtv = self._create_extrinsic(position, rotation)

        self._focal_len, self._im_size, self._sensor_size, self._K = self._create_intrinsic(
            focal_len, im_size, sensor_size
        )
        self._undistort_maps = None
        self._dist_coeff = None

        config_dict = dict(focal_len=focal_len, im_size=im_size, sensor_size=sensor_size)
        logger.debug(f'Camera configuration: {config_dict}')
        logger.debug(f'Position: {position}')
        logger.debug(f'Orientation: {rotation}')

    @staticmethod
    def _create_intrinsic(
        focal_len: Union[float, Tuple[float, float], np.ndarray], im_size: Union[Tuple[int, int], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create camera intrinsic parameters.
        """
        # Adapted from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera
        # -Parameters-defined and https://en.wikipedia.org/wiki/Camera_resectioning

        if len(im_size) != 2:
            raise ValueError('`im_size` should contain 2 values: (width, height).')
        if sensor_size is not None and len(sensor_size) != 2:
            raise ValueError('`sensor_size` should contain 2 values: (width, height).')
        focal_len = np.array(focal_len)
        if focal_len.size > 2:
            raise ValueError('`focal_len` should contain at most 2 values.')

        im_size = np.array(im_size)

        # find the xy focal lengths in pixels
        if sensor_size is None:
            logger.warning(
                '`sensor_size` not specified, assuming square pixels and `focal_len` normalised by sensor width.'
            )
            sigma_xy = (focal_len * im_size[0]) * np.ones(2)
        else:
            sensor_size = np.array(sensor_size)
            sigma_xy = focal_len * im_size / sensor_size

        # xy offsets
        c_xy = (im_size - 1) / 2

        # intrinsic matrix to convert from camera co-ords in ODM convention (x->right, y->down, z->forwards,
        # looking through the camera at the scene) to pixel co-ords in standard convention (x->right, y->down).
        K = np.array([[sigma_xy[0], 0, c_xy[0]], [0, sigma_xy[1], c_xy[1]], [0, 0, 1]])
        return focal_len, im_size, sensor_size, K

    @staticmethod
    def _create_extrinsic(
        position: Union[Tuple[float, float, float], np.ndarray], rotation: Union[Tuple[float, float, float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create camera extrinsic parameters.
        """
        if len(position) != 3 or len(rotation) != 3:
            raise ValueError('`position` and `rotation` should contain 3 values.')
        T = np.array(position).reshape(-1, 1)

        omega, phi, kappa = rotation

        # Find rotation matriz from OPK in PATB convention
        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera
        # -Parameters-defined
        omega_r = np.array(
            [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
        )  # yapf: disable

        phi_r = np.array(
            [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]
        )  # yapf: disable

        kappa_r = np.array(
            [[np.cos(kappa), -np.sin(kappa), 0], [np.sin(kappa), np.cos(kappa), 0], [0, 0, 1]]
        )  # yapf: disable

        R = omega_r.dot(phi_r).dot(kappa_r)

        # rotate from PATB (x->right, y->up, z->backwards looking through the camera at the scene) to ODM convention
        # (x->right, y->down, z->forwards, looking through the camera at the scene)
        R = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        # store angle axis format for opencv
        Rtv = cv2.Rodrigues(R.T)[0]
        return T, R, Rtv

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int, int], np.ndarray], dist_coeff: np.ndarray
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
        """"
        Create cv2.remap() maps for undistorting an image.
        """
        return None

    @staticmethod
    def _check_world_coordinates(xyz: np.ndarray):
        """
        Utility function to check world coordinate dimensions.
        """
        if not (xyz.ndim == 2 and xyz.shape[0] == 3):
            raise ValueError(f'`xyz` should be a 3xN 2D array.')
        if xyz.dtype != np.float64:
            raise ValueError(f'`xyz` should have float64 data type.')

    def _horizon_fov(self) -> bool:
        """ Whether this camera's field of view includes, or is above, the horizon. """
        # corner pixel coordinates of source image
        src_ji = np.array([[0, 0], [self._im_size[0], 0], self._im_size, [0, self._im_size[1]]]).T
        xyz_ = self._pixel_to_camera(src_ji)
        xyz_r = self._R.dot(xyz_)
        return np.any(xyz_r[2] >= 0)

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        """
        Transform 2D pixel to normalised 3D camera co-ordinates.
        """
        ji_ = np.row_stack([ji.astype('float64', copy=False), np.ones((1, ji.shape[1]))])
        xyz_ = np.linalg.inv(self._K).dot(ji_)
        return xyz_

    def update_extrinsic(
        self, position: Union[Tuple[float, float, float], np.ndarray],
        rotation: Union[Tuple[float, float, float], np.ndarray]
    ):
        """
        Update extrinsic parameters.

        Parameters
        ----------
        position: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world co-ordinates.
        rotation: tuple of float, ndarray
            Camera (omega, phi, kappa) angles in radians to rotate camera (PATB convention) to world co-ordinates.
        """
        self._T, self._R, self._Rtv = self._create_extrinsic(position, rotation)

    def world_to_pixel(self, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
        """
        Transform from 3D world to 2D pixel co-ordinates.

        Parameters
        ----------
        xyz : ndarray
            3D world (x=easting, y=northing, z=altitude) co-ordinates to transform, as a 3-by-N array with (x, y, z)
            along the first dimension.
        distort : bool (optional)
            Whether to include the distortion model.

        Returns
        -------
        ndarray
            Pixel (j=column, i=row) co-ordinates, as a 2-by-N array with (j, i) along the first dimension.
        """
        self._check_world_coordinates(xyz)
        # transform from world to camera co-ordinates
        xyz_ = self._R.T.dot(xyz - self._T)
        # normalise, and transform to pixel co-ordinates
        ji = self._K.dot(xyz_ / xyz_[2, :])[:2, :]
        return ji

    def pixel_to_world_z(self, ji: np.ndarray, z: Union[float, np.ndarray], distort: bool = True) -> np.ndarray:
        """
        Transform from 2D pixel to 3D world co-ordinates at a specified Z (altitude).

        Parameters
        ----------
        ji: ndarray
            Pixel (j=column, i=row) co-ordinates, as a 2-by-N array with (j, i) along the first dimension.
        z: float, ndarray
            Z altitude(s) to project to, as a single value, or 1-by-N array where ``ji`` is 2-by-N or 2-by-1.
        distort : bool (optional)
            Whether to include the distortion model.

        Returns
        -------
        ndarray
            3D world (x=easting, y=northing, z=altitude) co-ordinates, as a 3-by-N array with (x, y, z) along the first
            dimension.
        """
        if not (ji.ndim == 2 and ji.shape[0] == 2):
            raise ValueError(f'`ji` should be a 2xN 2D array.')

        if (
            isinstance(z, np.ndarray) and
            (z.ndim != 1 or (z.shape[0] != 1 and ji.shape[1] != 1 and z.shape[0] != ji.shape[1]))
        ):  # yapf: disable
            raise ValueError(f'`z` should be single value or 1-by-N array where `ji` is 2-by-N or 2-by-1.')

        # transform pixel co-ordinates to camera co-ordinates
        xyz_ = self._pixel_to_camera(ji) if distort else PinholeCamera._pixel_to_camera(self, ji)
        # rotate first (camera to world) to get world aligned axes with origin on the camera
        xyz_r = self._R.dot(xyz_)
        # scale to desired z (offset for camera z) with origin on camera, then offset to world
        xyz = (xyz_r * (z - self._T[2]) / xyz_r[2]) + self._T
        return xyz

    def undistort(
        self, image: np.ndarray, nodata: Union[float, int] = 0, interp: Union[str, Interp] = Interp.bilinear
    ) -> np.ndarray:
        """
        Undistort an image in-place.

        Parameters
        ----------
        image: ndarray
            Image array to undistort, as a single 2D band, or multiple bands in rasterio format i.e. with bands
            along the first dimension.
        nodata: float, int, optional
            Fill invalid areas in the undistorted image with this value.
        interp: str, Interp, optional
            Interpolation type to use when undistorting.

        Returns
        -------
        ndarray
            Undistorted array with the same shape and data type as ``image``.
        """
        if self._undistort_maps is None:
            return image

        def undistort_band(band: np.array) -> np.array:
            """ Undistort a 2D band array. """

            # equivalent without stored _undistort_maps:
            # return cv2.undistort(band_array, self._K, self._dist_coeff)
            return cv2.remap(
                band, *self._undistort_maps, Interp[interp].to_cv(), borderMode=cv2.BORDER_CONSTANT, borderValue=nodata
            )

        if image.ndim > 2:
            # undistort by band so that output data stays in the rasterio ordering
            out_image = np.full(image.shape, fill_value=nodata, dtype=image.dtype)
            for bi in range(image.shape[0]):
                out_image[bi] = undistort_band(image[bi])
        else:
            out_image = undistort_band(image)

        return out_image


class PinholeCamera(Camera):
    """"""


class OpenCVCamera(Camera):

    def __init__(
        self, position: Union[Tuple[float, float, float], np.ndarray],
        rotation: Union[Tuple[float, float, float], np.ndarray],
        focal_len: Union[float, Tuple[float, float], np.ndarray], im_size: Union[Tuple[int, int], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None, k1: float = 0., k2: float = 0.,
        k3: float = 0., p1: float = 0., p2: float = 0., k4: float = 0., k5: float = 0., k6: float = 0., s1: float = 0.,
        s2: float = 0., s3: float = 0., s4: float = 0., t1: float = 0., t2: float = 0.,
    ):  # yapf: disable
        """
        Camera class with OpenCV distortion model, for transforming between 3D world and 2D pixel co-ordinates, and
        undistorting images.

        This is a wrapper around the `OpenCV general model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_
        that includes radial, tangential & thin prism distortion components.  Partial or special cases of the model can
        be computed by omitting some or all of the distortion coefficients; e.g. if no distortion coefficients are
        specified, this model corresponds to :class:`PinholeCamera`, or if the first 5 distortion coefficients are
        specified, this model corresponds to :class:`BrownCamera` with (cx, cy) = (0, 0).

        Parameters
        ----------
        position: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world co-ordinates.  If the Camera object is being
            used to generate an ortho image with :class:`~simple_ortho.ortho.Ortho`, ``position`` should be in the
            ortho image CRS.
        rotation: tuple of float, ndarray
            Camera (omega, phi, kappa) angles in radians to rotate camera to world co-ordinates, where camera
            co-ordinates are in PATB convention (i.e. x->right, y->up and z->backwards, looking through the camera at
            the scene).
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty: float, optional
            OpenCV distortion coefficients - see their `docs <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_
            for details.
        """
        Camera.__init__(self, position, rotation, focal_len, im_size, sensor_size=sensor_size)

        # order _dist_coeff & truncate zeros according to OpenCV docs
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        self._dist_coeff = np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, t1, t2])
        for dist_len in (4, 5, 8, 12, 14):
            if np.all(self._dist_coeff[dist_len:] == 0.):
                self._dist_coeff = self._dist_coeff[:dist_len]
                break

        self._undistort_maps = self._create_undistort_maps(self._K, self._im_size, self._dist_coeff)

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int, int], np.ndarray], dist_coeff: np.ndarray
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:  # yapf: disable
        undistort_maps = cv2.initUndistortRectifyMap(
            K, dist_coeff, None, None, np.array(im_size).astype(int), cv2.CV_16SC2
        )
        return undistort_maps

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T.astype('float64', copy=False)
        xyz_ = cv2.undistortPoints(ji_cv, self._K, self._dist_coeff)
        xyz_ = np.row_stack([xyz_[:, 0, :].T, np.ones((1, ji.shape[1]))])
        return xyz_

    def world_to_pixel(self, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
        self._check_world_coordinates(xyz)
        if not distort:
            return PinholeCamera.world_to_pixel(self, xyz)
        ji, _ = cv2.projectPoints((xyz - self._T).T, self._Rtv, np.zeros(3), self._K, self._dist_coeff)
        ji = ji[:, 0, :].T
        return ji


class BrownCamera(OpenCVCamera):

    def __init__(
        self, position: Union[Tuple[float, float, float], np.ndarray],
        rotation: Union[Tuple[float, float, float], np.ndarray],
        focal_len: Union[float, Tuple[float, float], np.ndarray], im_size: Union[Tuple[int, int], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None, k1: float = 0., k2: float = 0.,
        p1: float = 0., p2: float = 0., k3: float = 0., cx: float = 0., cy: float = 0.,
    ):  # yapf: disable
        """
        Camera class with Brown-Conrady distortion for transforming between 3D world and 2D pixel co-ordinates, and
        undistorting images.

        The distortion model is compatible with `OpenDroneMap (ODM)
        <https://docs.opendronemap.org/arguments/camera-lens/>`_ / `OpenSFM <https://github.com/mapillary/OpenSfM>`_
        *brown* parameter estimates; and the 4- & 5-coefficient version of the `general OpenCV distortion model
        <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_.  ODM places their estimates in
        *<dataset path>/cameras.json*, and OpenSFM in *<dataset path>/camera_models.json*.

        Parameters
        ----------
        position: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world co-ordinates.  If the Camera object is being
            used to generate an ortho image with :class:`~simple_ortho.ortho.Ortho`, ``position`` should be in the
            ortho image CRS.
        rotation: tuple of float, ndarray
            Camera (omega, phi, kappa) angles in radians to rotate camera to world co-ordinates, where camera
            co-ordinates are in PATB convention (i.e. x->right, y->up and z->backwards, looking through the camera at
            the scene).
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        k1, k2, p1, p2, k3: float, optional
            Brown model distortion coefficients.
        cx, cy: float, optional
            ODM / OpenSFM brown model principal point.
        """
        Camera.__init__(self, position, rotation, focal_len, im_size, sensor_size=sensor_size)

        self._dist_coeff = np.array([k1, k2, p1, p2, k3])

        # incorporate ODM/OpenSFM brown model offsets into self._Koff
        self._Koff = self._offset_intrinsic(self._K, self._im_size, cx=cx, cy=cy)

        # create undistort maps with self._Koff
        self._undistort_maps = self._create_undistort_maps(self._Koff, self._im_size, self._dist_coeff)

    @staticmethod
    def _offset_intrinsic(K: np.ndarray, im_size: np.ndarray, cx: float = 0., cy: float = 0.) -> np.ndarray:
        """
        Incorpotate ODM / OpenSFM Brown model principal point.
        """
        # Following the radial/tangential distortion, ODM has an affine transform as part of their Brown model
        # (see https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle
        # .h#LL299C25-L299C25):
        #     xni = fx xu + cx
        #     yni = fy yu + cy
        # where (xni, yni) are ODM "normalised image co-ordinates"
        # (https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates), and (fx, fy) & (cx, cy)
        # are the parameters in `cameras.json`.
        #
        # To get to pixel co-ordinates, another affine transform is applied (see
        # https://opensfm.readthedocs.io/en/latest/geometry.html#pixel-coordinates):
        #     u = max(w, h) * xni + (w - 1) / 2
        #     v = max(w, h) * yni + (h - 1) / 2
        # where (w, h) are the image (width, height) in pixels.
        #
        # The effective (additional) pixel offsets are then: max(w, h) * (cx, cy).
        # TODO: test if the OpenCV estimate of the intrinsic principal point incorporates these values.
        Koff = K.copy()
        # Koff[:2, 2] += np.diag(self._K)[:2] * np.array([cx, cy]) / self._focal_len  # equivalent to below
        Koff[:2, 2] += im_size.max() * np.array([cx, cy])
        return Koff

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T.astype('float64', copy=False)
        xyz_ = cv2.undistortPoints(ji_cv, self._Koff, self._dist_coeff)
        xyz_ = np.row_stack([xyz_[:, 0, :].T, np.ones((1, ji.shape[1]))])
        return xyz_

    def world_to_pixel(self, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
        self._check_world_coordinates(xyz)

        # transform from world to camera co-ordinates, and normalise
        xyz_ = self._R.T.dot(xyz - self._T)
        xyz_ = xyz_ / xyz_[2, :]

        if distort:
            # Brown model adapted from the OpenSFM implementation:
            # https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle.h
            # #LL299C25-L299C25.
            # Works out faster than the opencv equivalent in OpenCVCamera.world_to_pixel().
            k1, k2, p1, p2, k3 = self._dist_coeff
            x2, y2 = np.square(xyz_[:2, :])
            xy = xyz_[0, :] * xyz_[1, :]
            r2 = x2 + y2

            radial_dist = 1. + r2 * (k1 + r2 * (k2 + r2 * k3))
            x_tangential_dist = 2. * p1 * xy + p2 * (r2 + 2.0 * x2)
            y_tangential_dist = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy

            xyz_[0, :] = xyz_[0, :] * radial_dist + x_tangential_dist
            xyz_[1, :] = xyz_[1, :] * radial_dist + y_tangential_dist

            # transform from distorted camera to pixel co-ordinates, using Koff
            ji = self._Koff.dot(xyz_)[:2, :]
        else:
            # transform from camera to pixel co-ordinates, using K
            ji = self._K.dot(xyz_)[:2, :]

        return ji


class FisheyeCamera(Camera):

    def __init__(
        self, position: Union[Tuple[float, float, float], np.ndarray],
        rotation: Union[Tuple[float, float, float], np.ndarray],
        focal_len: Union[float, Tuple[float, float], np.ndarray], im_size: Union[Tuple[int, int], np.ndarray],
        sensor_size: Optional[Union[Tuple[float, float], np.ndarray]] = None, k1: float = 0., k2: float = 0.,
        k3: float = 0., k4: float = 0.,
    ):  # yapf: disable
        """
        Camera class with fisheye distortion for transforming between 3D world and 2D pixel co-ordinates, and
        undistorting images.

        The distortion model is compatible with `OpenDroneMap (ODM)
        <https://docs.opendronemap.org/arguments/camera-lens/>`_ / `OpenSFM <https://github.com/mapillary/OpenSfM>`_
        *fisheye* parameter estimates; and the `OpenCV fisheye distortion model
        <https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html>`_.  ODM places their estimates in
        *<dataset path>/cameras.json*, and OpenSFM in *<dataset path>/camera_models.json*.

        Parameters
        ----------
        position: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world co-ordinates.  If the Camera object is being
            used to generate an ortho image with :class:`~simple_ortho.ortho.Ortho`, ``position`` should be in the
            ortho image CRS.
        rotation: tuple of float, ndarray
            Camera (omega, phi, kappa) angles in radians to rotate camera to world co-ordinates, where camera
            co-ordinates are in PATB convention (i.e. x->right, y->up and z->backwards, looking through the camera at
            the scene).
        focal_len: float, tuple of float, ndarray
            Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or (x, y)
            tuple/ndarray pair.
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) with the same units/scale as ``focal_len``.  If not specified, pixels are
            assumed square, and ``focal_len`` should be a normalised & unitless value:
            ``focal_len`` = (focal length) / (sensor width).
        k1, k2, k3, k4: float, optional
            Fisheye distortion coefficients.  OpenCV uses all coefficients, while ODM and OpenSFM use k1 & k2 only.
        """
        Camera.__init__(self, position, rotation, focal_len, im_size, sensor_size=sensor_size)

        self._dist_coeff = np.array([k1, k2, k3, k4])
        self._undistort_maps = self._create_undistort_maps(self._K, self._im_size, self._dist_coeff)

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int, int], np.ndarray], dist_coeff: np.ndarray
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
        # unlike cv2.initUndistortRectifyMap(), cv2.fisheye.initUndistortRectifyMap() requires default R & P
        # (new camera matrix) params to be specified
        undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            K, dist_coeff, np.eye(3), K, np.array(im_size).astype(int), cv2.CV_16SC2
        )
        return undistort_maps

    def _pixel_to_camera(self, ji: np.ndarray) -> np.ndarray:
        ji_cv = ji.T[None, :].astype('float64', copy=False)
        xyz_ = cv2.fisheye.undistortPoints(ji_cv, self._K, self._dist_coeff, None, None)
        xyz_ = np.row_stack([xyz_[0].T, np.ones((1, ji.shape[1]))])
        return xyz_

    def world_to_pixel(self, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
        self._check_world_coordinates(xyz)

        # transform from world to camera co-ordinates, and normalise
        xyz_ = self._R.T.dot(xyz - self._T)
        xyz_ = xyz_ / xyz_[2, :]

        if distort:
            # Fisheye distortion adapted from the OpenSFM implementation:
            # https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle.h
            # #L365.
            # and OpenCV docs: https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html.
            # Works out faster than the opencv equivalent:
            #   x_cv = np.expand_dims((x - self._T).T, axis=0)
            #   ji, _ = cv2.fisheye.projectPoints(x_cv, self._Rtv, np.zeros(3), self._K, self._dist_coeff)
            #   ji = np.squeeze(ji).T

            k1, k2, k3, k4 = self._dist_coeff
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

        # transform from camera to pixel co-ordinates
        ji = self._K.dot(xyz_)[:2, :]
        return ji


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
#  principal point
##
