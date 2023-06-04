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

import cv2
import numpy as np
from typing import Union, Tuple, Optional
from enum import Enum
from simple_ortho import get_logger


# from scipy.ndimage import map_coordinates

logger = get_logger(__name__)


class CameraType(Enum):
    pinhole = 'pinhole'
    brown = 'brown'
    fisheye = 'fisheye'
    opencv = 'opencv'


class Camera:
    def __init__(
        self, position: Union[Tuple[float], np.ndarray], rotation: Union[Tuple[float], np.ndarray],
        focal_len: Union[float, Tuple[float], np.ndarray], im_size: Union[Tuple[int], np.ndarray],
        sensor_size: Optional[Union[Tuple[float], np.ndarray]] = None,
    ):
        """
        Pinhole camera class for projecting between 3D world and 2D pixel co-ordinates.

        Parameters
        ----------
        position: tuple of float, ndarray
            Camera position (x=easting, y=northing, z=altitude) in world co-ordinates.  The CRS of these values must be
            the same as any ortho image CRS being created with :class:`~simple_ortho.ortho.OrthoIm`.
        rotation: tuple of float, ndarray
            Camera rotation (omega, phi, kappa) angles in radians (PATB convention i.e. x->right, y->up and z->
            backwards, looking through the camera at the scene).
        focal_len: float, tuple of float, ndarray
            Focal length(s). Can be a scalar or length 2 (x, y) tuple/ndarray. Must be in the same units as
            ``sensor_size``.
        im_size: tuple of int, ndarray
            Image (width, height) in pixels.
        sensor_size: tuple of float, ndarray, optional
            Sensor (ccd) (width, height) in the same units as ``focal_len``.  If not specified, ``focal_len`` should be
            normalised by the sensor width i.e. ``focal_len`` = (focal length) / (sensor width).
        """

        self._T, self._R, self._Rtv = self._create_extrinsic(position, rotation)

        self._focal_len, self._im_size, self._sensor_size, self._K = self._create_intrinsic(
            focal_len, im_size, sensor_size
        )
        self._undistort_maps = None

        config_dict = dict(focal_len=focal_len, im_size=im_size, sensor_size=sensor_size)
        logger.debug(f'Camera configuration: {config_dict}')
        logger.debug(f'Position: {position}')
        logger.debug(f'Orientation: {rotation}')

    @staticmethod
    def _create_intrinsic(
        focal_len: Union[float, Tuple[float], np.ndarray], im_size: Union[Tuple[int], np.ndarray],
        sensor_size: Optional[Union[Tuple[float], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create camera intrinsic parameters.
        """
        # Adapted from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera
        # -Parameters-defined
        # and https://en.wikipedia.org/wiki/Camera_resectioning

        if len(im_size) != 2:
            raise ValueError('`im_size` must contain 2 values: (width, height).')
        if sensor_size and len(sensor_size) != 2:
            raise ValueError('`sensor_size` must contain 2 values: (width, height).')
        focal_len = np.array(focal_len)
        if focal_len.size > 2:
            raise ValueError('`focal_len` must contain at most 2 values.')

        im_size = np.array(im_size)

        # find the xy focal lengths in pixels
        if not sensor_size:
            logger.warning('`sensor_size` not specified, assuming square pixels and `focal_len` normalised.')
            sigma_xy = (focal_len * im_size[0]) * np.ones((1, 2))
        else:
            sensor_size = np.array(sensor_size)
            sigma_xy = focal_len * im_size / sensor_size

        # xy offsets
        c_xy = (im_size - 1) / 2

        # intrinsic matrix to convert from camera co-ords in ODM convention (x->right, y->down, z->forwards,
        # looking through the camera at the scene) to pixel co-ords in standard convention (x->right, y->down).
        K = np.array(
            [[sigma_xy[0], 0, c_xy[0]],
             [0, sigma_xy[1],  c_xy[1]],
             [0, 0, 1]]
        )
        return focal_len, im_size, sensor_size, K

    @staticmethod
    def _create_extrinsic(
        position: Union[Tuple[float], np.ndarray], rotation: Union[Tuple[float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create camera extrinsic parameters.
        """
        if len(position) != 3 or len(rotation) != 3:
            raise ValueError('`position` and `rotation` must contain 3 values')
        T = np.array(position).reshape(3, 1)

        omega, phi, kappa = rotation

        # Find rotation matriz from OPK in PATB convention
        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        omega_r = np.array(
            [[1, 0, 0],
             [0, np.cos(omega), -np.sin(omega)],
             [0, np.sin(omega), np.cos(omega)]]
        )

        phi_r = np.array(
            [[np.cos(phi), 0, np.sin(phi)],
             [0, 1, 0],
             [-np.sin(phi), 0, np.cos(phi)]]
        )

        kappa_r = np.array(
            [[np.cos(kappa), -np.sin(kappa), 0],
             [np.sin(kappa), np.cos(kappa), 0],
             [0, 0, 1]]
        )

        R = omega_r.dot(phi_r).dot(kappa_r)

        # rotate from PATB (x->right, y->up, z->backwards looking through the camera at the scene) to ODM convention
        # (x->right, y->down, z->forwards, looking through the camera at the scene)
        R = R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        # store angle axis format for opencv
        Rtv = cv2.Rodrigues(R.T)[0]
        return T, R, Rtv

    def world_to_pixel(self, x: np.ndarray, distort: bool=False) -> np.ndarray:
        """
        Transform from 3D world to 2D pixel co-ordinates.

        Parameters
        ----------
        x : ndarray
            3-by-N array of 3D world (x=easting, y=northing, z=altitude) co-ordinates to unproject, with (x, y, z)
            along the first dimension.
        distort : bool (optional)
            Whether to include the distortion model (default: False).

        Returns
        -------
        ndarray
            2-by-N array of pixel (i=row, j=column) co-ordinates, with (i, j) along the first dimension.
        """
        if not (x.shape[0] == 3 and x.shape[1] > 0):
            raise ValueError('x must have 3 rows and more than one column')

        # reshape to xyz along 1st dimension, and broadcast rotation and translation for each xyz vector
        x_ = self._R.T.dot(x - self._T)
        # normalise xyz/z and apply intrinsic matrix, then discard the 3rd dimension
        ij = self._K.dot(x_ / x_[2, :])[:2, :]
        return ij

    def _pixel_to_camera(self, ij: np.ndarray) -> np.ndarray:
        """
        Transform 2D pixel co-ordinates to normalised 3D camera co-ordinates.
        """
        ij_ = np.row_stack([ij, np.ones((1, ij.shape[1]))])
        # TODO: store inverse rather than recalculate each time
        x_ = np.dot(np.linalg.inv(self._K), ij_)
        return x_

    def pixel_to_world_z(self, ij: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Transform from 2D pixel to 3D world co-ordinates at a specified Z.

        Parameters
        ----------
        ij: ndarray
            2-by-N array of image (i=row, j=column) co-ordinates, with (i, j) along the first dimension.
        z: ndarray
            1-by-1 or 1-by-N array of Z (altitude) value(s) to project to.

        Returns
        -------
        ndarray
            3-by-N array of 3D world (x=easting, y=northing, z=altitude) co-ordinates, with (x, y, z) along the first
            dimension.
        """
        if not (ij.shape[0] == 2 and ij.shape[1] > 0):
            raise ValueError('`ij` must have 2 rows and one or more columns.')

        # transform pixel co-ordinates to camera co-ordinates
        x_ = self._pixel_to_camera(ij)
        # rotate first (camera to world) to get world aligned axes with origin on the camera
        x_r = self._R.dot(x_)
        # scale to desired z (offset for camera z) with origin on camera, then offset to world
        x = (x_r * (z - self._T[2]) / x_r[2, :]) + self._T
        return x

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int], np.ndarray], dist_coeff: np.ndarray
    )->Union[None, Tuple[np.ndarray, np.ndarray]]:
        """"
        Create cv2.remap() maps for undistorting an image.
        """
        return None

    def undistort(self, image: np.ndarray, nodata: Union[float, int]=0, interp: int=cv2.INTER_LINEAR) -> np.ndarray:
        """
        Undistort an image.

        Parameters
        ----------
        image: ndarray
            Image array to undistort.  Can be a single 2D band, or multiple bands in rasterio format i.e. with bands
            along the first dimension.
        nodata: float, int, optional
            Fill invalid areas in the undistorted image with this value.
        interp: int, optional
            OpenCV interpolation type to use when undistorting.

        Returns
        -------
        ndarray
            Undistorted array with the same shape and data type as ``image``.
        """
        if self._undistort_maps is None:
            return image

        def undistort_band(band: np.array) -> np.array:
            """ Undistort a 2D band array. """
            return cv2.remap(
                band, *self._undistort_maps, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=nodata
            )
            # return cv2.undistort(band_array, self._K, self._dist_coeff)

        if image.ndim > 2:
            # Undistorting the 3D image in a single call requires conversion between rasterio and opencv 3D array
            # ordering.  This leaves the output array ordering in a form that works with cv2.remap, but is slow.
            # This can be fixed with an extra copy to repack the array, but we rather undistort by band here to avoid
            # that and save some memory.
            # TODO: can we make this method take images in opencv format? that would make better sense as rio is not
            #  used in this module
            out_image = np.full(image.shape, fill_value=nodata, dtype=image.dtype)
            for bi in range(out_image.shape[0]):
                out_image[bi] = undistort_band(image[bi])
        else:
            out_image = undistort_band(image)

        return out_image


class PinholeCamera(Camera):
    """ Pinhole camera model for transforming between 2D pixel and 3D world co-ordinates. """


class BrownCamera(Camera):
    """
    Brown camera model for transforming between 2D pixel and 3D world co-ordinates.  Compatible with ODM Brown model
    coefficients.
    """
    def __init__(self, *args, dist_coeff=None, **kwargs):
        Camera.__init__(self, *args, **kwargs)
        self._dist_coeff = np.array(dist_coeff) if dist_coeff else None
        self._undistort_maps = self._create_undistort_maps(self._K, self._im_size, self._dist_coeff)

    def world_to_pixel(self, x: np.ndarray, distort: bool = False) -> np.ndarray:
        if not distort:
            return PinholeCamera.world_to_pixel(self, x)
        ij, _ = cv2.projectPoints(
            x - self._T, self._Rtv, np.array([0., 0., 0.]), self._K, distCoeffs=self._dist_coeff
        )
        ij = np.squeeze(ij).T
        return ij

    def _pixel_to_camera(self, ij: np.ndarray) -> np.ndarray:
        x_ = cv2.undistortPoints(ij.astype('float64'), self._K, self._dist_coeff)
        x_ = np.row_stack([x_.squeeze().T, np.ones((1, ij.shape[1]))])
        return x_

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int], np.ndarray], dist_coeff: np.ndarray
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
        undistort_maps = cv2.initUndistortRectifyMap(
            K, dist_coeff, None, None, np.array(im_size).astype(int), cv2.CV_16SC2  # cv2.CV_32FC1
        )
        return undistort_maps


class FisheyeCamera(Camera):
    """
    Fisheye camera model for transforming between 2D pixel and 3D world co-ordinates.  Compatible with ODM and OpenCV
    fisheye coefficients.
    """
    def __init__(self, *args, dist_coeff=None, **kwargs):
        Camera.__init__(self, *args, **kwargs)
        self._dist_coeff = np.array(dist_coeff) if dist_coeff else None
        self._undistort_maps = self._create_undistort_maps(self._K, self._im_size, self._dist_coeff)

    def world_to_pixel(self, x: np.ndarray, distort: bool=False) -> np.ndarray:
        if not distort:
            return PinholeCamera.world_to_pixel(self, x)
        x_cv = np.expand_dims((x - self._T).T, axis=0)
        ij, _ = cv2.fisheye.projectPoints(x_cv, self._Rtv, np.array([0., 0., 0.]), self._K, self._dist_coeff)
        ij = np.squeeze(ij).T
        return ij

    def _pixel_to_camera(self, ij: np.ndarray) -> np.ndarray:
        ij_cv = np.expand_dims(ij.T, axis=0).astype('float64')
        x_ = cv2.fisheye.undistortPoints(ij_cv, self._K, self._dist_coeff, np.eye(3), np.eye(3))
        x_ = np.row_stack([x_.squeeze().T, np.ones((1, ij.shape[1]))])
        return x_

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int], np.ndarray], dist_coeff: np.ndarray
    )->Union[None, Tuple[np.ndarray, np.ndarray]]:

        undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            K, dist_coeff, np.eye(3), K, np.array(im_size).astype(int), cv2.CV_16SC2
        )
        return undistort_maps


class _Camera:
    def __init__(
        self, focal_len: float, sensor_size: Union[Tuple[float], np.array], im_size: Union[Tuple[int], np.array],
        position: Union[Tuple[float], np.array], orientation: Union[Tuple[float], np.array],
        dist_coeff: Union[Tuple[float], np.array]=None, dist_type: CameraType = CameraType.pinhole
    ):
        """
        Camera class to project from 2D camera (i,j) pixel co-ordinates to 3D world (x,y,z) co-ordinates,
        and vice-versa

        Parameters
        ----------
        focal_len: float
            Focal length in mm.
        sensor_size: np.array, tuple of float
            Sensor (ccd) (width, height) in mm.
        im_size: np.array, tuple of int
            Image (width, height) in pixels
        position: np.array, tuple of float
            Camera location (x=easting, y=northing, z=altitude) co-ordinates, in image CRS.
        orientation: np.array, tuple of float
            Camera orientation (omega, phi, kappa) angles in radians (PATB convention).
        dist_coeff: np.array, tuple of float, optional
            Lens distortion coefficients in opencv order - see
            https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d.
        """
        # TODO: the sensor size could be omitted if the focal len is given in normalised (focal len / sensor width)
        #  "units", and the pixels are square (i.e. the aspect ratio can be derived from the image pixel dims).  In
        #  this case the sensor size is [1 h/w], (w, h) are image dims in pixels and w>h.
        self.update_extrinsic(position, orientation)

        if np.size(sensor_size) != 2 or np.size(im_size) != 2:
            raise Exception('len(sensor_size) != 2 or len(image_size) != 2')

        self._sensor_size = np.array(sensor_size)
        self._im_size = np.array(im_size)
        self._focal_len = focal_len
        self._dist_coeff = np.array(dist_coeff) if dist_coeff else None
        # TODO: add some error checking
        dist_type = DistortionType(dist_type) if dist_coeff else DistortionType.pinhole
        self._cv = cv2.fisheye if dist_type == DistortionType.fisheye else cv2
        self._undistort_maps = None

        self._create_intrinsic()
        config_dict = dict(focal_len=focal_len, sensor_size=sensor_size, im_size=im_size, dist_coeff=dist_coeff)
        logger.debug(f'Camera configuration: {config_dict}')
        logger.debug(f'Position: {position}')
        logger.debug(f'Orientation: {orientation}')

    def update_extrinsic(self, position, orientation):
        """
        Update camera extrinsic parameters

        Parameters
        ----------
        position :      numpy.array_like
                        list of [x=easting, y=northing, z=altitude] camera location co-ordinates, in image CRS
        orientation :   numpy.array_like
                        camera orientation [omega, phi, kappa] angles in degrees
        """
        if np.size(position) != 3 or np.size(orientation) != 3:
            raise Exception('len(position) != 3 or len(orientation) != 3')
        self._T = np.array(position).reshape(3, 1)

        self._omega, self._phi, self._kappa = orientation

        # Find rotation matriz from OPK in PATB convention
        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        omega_r = np.array(
            [[1, 0, 0],
             [0, np.cos(self._omega), -np.sin(self._omega)],
             [0, np.sin(self._omega), np.cos(self._omega)]]
        )

        phi_r = np.array(
            [[np.cos(self._phi), 0, np.sin(self._phi)],
             [0, 1, 0],
             [-np.sin(self._phi), 0, np.cos(self._phi)]]
        )

        kappa_r = np.array(
            [[np.cos(self._kappa), -np.sin(self._kappa), 0],
             [np.sin(self._kappa), np.cos(self._kappa), 0],
             [0, 0, 1]]
        )

        self._R = np.dot(np.dot(omega_r, phi_r), kappa_r)

        # rotate from PATB (x->right, y->up, z->backwards looking through the camera at the scene) to ODM convention
        # (x->right, y->down, z->forwards, looking through the camera at the scene)
        self._R = self._R.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        # store angle axis format for opencv
        self._Rtv = cv2.Rodrigues(self._R.T)[0]
        return

    def _create_intrinsic(self):
        """
        Update camera intrinsic parameters
        """
        # TODO: this should be renamed and hidden from the user, I think the intrinsics will remain fixed for a
        #  specific camera.  Consider different zooms and focal lengths for same camera though?
        # Adapted from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera
        # -Parameters-defined
        # and https://en.wikipedia.org/wiki/Camera_resectioning
        sigma_xy = self._focal_len * self._im_size / self._sensor_size  # xy focal lengths in pixels
        c_xy = (self._im_size - 1) / 2

        # Intrinsic matrix to convert from camera co-ords in PATB convention (x->right, y->up, z->backwards,
        # looking through the camera at the scene) to pixel co-ords in standard convention (x->right, y->down).
        # self._K = np.array(
        #     [[-sigma_xy[0], 0, c_xy[0]],
        #      [0, sigma_xy[1],  c_xy[1]],
        #      [0, 0, 1]]
        # )
        # Intrinsic matrix to convert from camera co-ords in ODM convention (x->right, y->down, z->forwards,
        # looking through the camera at the scene) to pixel co-ords in standard convention (x->right, y->down).
        self._K = np.array(
            [[sigma_xy[0], 0, c_xy[0]],
             [0, sigma_xy[1],  c_xy[1]],
             [0, 0, 1]]
        )
        return

    def unproject(self, x: np.array, distort: bool=False):
        """
        Unproject from 3D world co-ordinates to 2D image co-ordinates

        When using this method to build orthorectification maps for use in e.g. ``cv2.remap()``, it is faster to
        leave ``distort=False``, and do the remap with an undistorted image (e.g. from ``Camera.undistort()``).
        This is faster than remapping the distorted image with ``distort=True`` produced maps, as it avoids calling
        ``cv2.projectPoints()`` (see below).

        Parameters
        ----------
        x : np.array
            3-by-N array of 3D world (x=easting, y=northing, z=altitude) co-ordinates to unproject.
            (x,y,z) along the first dimension.
        distort : bool (optional)
            Whether to include the distortion model (default: False).

        Returns
        -------
        ij : numpy.array_like
             2-by-N array of image (i=row, j=column) co-ordinates. (i, j) along the first dimension.
        """
        if not (x.shape[0] == 3 and x.shape[1] > 0):
            raise Exception('x must have 3 rows and more than one column')

        if (distort and (self._dist_coeff is not None) and (not np.all(self._dist_coeff == 0))):
            # include the distortion model via opencv
            # ij, _ = cv2.projectPoints(
            #     x - self._T, self._Rtv, np.array([0., 0., 0.]), self._K, distCoeffs=self._dist_coeff
            # )
            x_cv = np.expand_dims((x - self._T).T, axis=0)
            ij, _ = self._cv.projectPoints(x_cv, self._Rtv, np.array([0., 0., 0.]), self._K, self._dist_coeff)
            ij = np.squeeze(ij).T
        else:
            # reshape/transpose to xyz along 1st dimension, and broadcast rotation and translation for each xyz vector
            x_ = np.dot(self._R.T, (x - self._T))
            # homogenise xyz/z and apply intrinsic matrix, then discard the 3rd dimension
            ij = (np.dot(self._K, x_ / x_[2, :]))[:2, :]

        return ij

    def project_to_z(self, ij, z):
        """
        Project from 2D image co-ordinates to 3D world co-ordinates at a specified Z

        Parameters
        ----------
        ij : numpy.array_like
             2-by-N array of image (i=row, j=column) co-ordinates. (i, j) along the first dimension.
        z :  numpy.array_like
             1-by-N array of Z (altitude) values to project to

        Returns
        -------
        x : numpy.array_like
            3-by-N array of 3D world (x=easting, y=northing, z=altitude) co-ordinates.
            (x,y,z) along the first dimension.
        """
        if not (ij.shape[0] == 2 and ij.shape[1] > 0):
            raise Exception('not(ij.shape[0] == 2 and ij.shape[1] > 0)')

        if (self._dist_coeff is not None) and (not np.all(self._dist_coeff == 0)):
            # transform pixel co-ordinates to distortion corrected camera co-ordinates
            # x_ = cv2.undistortPoints(ij.astype('float64'), self._K, self._dist_coeff)
            ij_cv = np.expand_dims(ij.T, axis=0).astype('float64')
            x_ = self._cv.undistortPoints(
                ij_cv, self._K, self._dist_coeff, np.eye(3), np.eye(3)
            )
            x_ = np.row_stack([x_.squeeze().T, np.ones((1, ij.shape[1]))])
        else:
            # transform pixel co-ordinates to camera co-ordinates
            # TODO: store inverse rather than recalculate each time
            ij_ = np.row_stack([ij, np.ones((1, ij.shape[1]))])
            x_ = np.dot(np.linalg.inv(self._K), ij_)
        # rotate first (camera to world) to get world aligned axes with origin on the camera
        x_r = np.dot(self._R, x_)
        # scale to desired z (offset for camera z) with origin on camera, then offset to world
        x = (x_r * (z - self._T[2]) / x_r[2, :]) + self._T
        return x

    def undistort(self, array: np.array, nodata: Union[float, int]=0, interp: int=cv2.INTER_LINEAR) -> np.array:
        """
        Undistort an image.

        Parameters
        ----------
        array: np.array
            Image array to undistort.  Can be a single 2D band, or multiple bands in 3D rasterio format i.e. with bands
            along the first dimension.

        Returns
        -------
        np.array
            Undistorted array with the same shape and data type as ``array``.
        """
        if self._dist_coeff is None or np.all(self._dist_coeff == 0):
            return array

        if self._undistort_maps is None:
            # Find undistortion maps once off, and store for repeat use. (OpenCV docs say that using
            # cv2.initUndistortRectifyMap(..., m1type=cv2.CV_16SC2) speeds up cv2.remap - that isn't the case here, but
            # it is still used to save some memory).
            # self._undistort_maps = cv2.initUndistortRectifyMap(
            #     self._K, self._dist_coeff, None, None, self._im_size.astype(int), cv2.CV_16SC2  # cv2.CV_32FC1
            # )
            # TODO: why do we need newCamerMatrix = K below, but newCamerMatrix = eye(3) in cv2.fisheye.undistortPoints ?
            self._undistort_maps = self._cv.initUndistortRectifyMap(
                self._K, self._dist_coeff, np.eye(3), self._K, (self._im_size).astype(int), cv2.CV_32FC1
            )

            # debug code for testing odm brown distortion
            # f = 0.6134313085596745
            # c_x = -0.004891928269862716
            # c_y = 0.001106653485186689
            # # Knew = cv2.getOptimalNewCameraMatrix(self._K, self._dist_coeff, self._im_size.astype(int), 1)[0]
            # Kodm = self._K.copy()
            # Kodm[:2, 2] += np.array([c_x, c_y]) * np.abs(np.diag(self._K)[:2] / f)
            # Kodm[0, 0] *= -1
            # self._undistort_maps = self._cv.initUndistortRectifyMap(
            #     Kodm, self._dist_coeff, np.eye(3), None, (self._im_size).astype(int), cv2.CV_32FC1
            # )

        def undistort_band(band_array: np.array) -> np.array:
            """ Undistort a 2D band array. """
            return cv2.remap(
                band_array, *self._undistort_maps, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=nodata
            )
            # return cv2.undistort(band_array, self._K, self._dist_coeff)

        if array.ndim > 2:
            # Undistorting the 3D image in a single call requires conversion between rasterio and opencv 3D array
            # ordering.  This leaves the output array ordering in a form that works with cv2.remap, but is slow.
            # This can be fixed with an extra copy to repack the array, but we rather undistort by band here to avoid
            # that and save some memory.
            out_array = np.full(array.shape, fill_value=nodata, dtype=array.dtype)
            for bi in range(out_array.shape[0]):
                out_array[bi] = undistort_band(array[bi])
        else:
            out_array = undistort_band(array)

        return out_array


def create_camera(cam_type: CameraType, *args, **kwargs) -> Camera:
    """
    Create a camera object given a type and parameters.

    Parameters
    ----------
    cam_type: CameraType
        Camera type (pinhole, brown, fisheye, opencv).
    args:
        Positional arguments to pass to camera constructor.
    kwargs:
        Keyword argument to pass to camera constructor.

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
    # elif type == CameraType.opencv:
    #     cam_class = OpencvCamera
    else:
        cam_class = PinholeCamera

    return cam_class(*args, **kwargs)

##
