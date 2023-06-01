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
from typing import Union, Tuple
from simple_ortho import get_logger


# from scipy.ndimage import map_coordinates

logger = get_logger(__name__)


class Camera:
    def __init__(
        self, focal_len: float, sensor_size: Union[Tuple[float], np.array], im_size: Union[Tuple[int], np.array],
        position: Union[Tuple[float], np.array], orientation: Union[Tuple[float], np.array],
        dist_coeff: Union[Tuple[float], np.array]=None
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

        self.update_extrinsic(position, orientation)

        if np.size(sensor_size) != 2 or np.size(im_size) != 2:
            raise Exception('len(sensor_size) != 2 or len(image_size) != 2')

        self._sensor_size = np.array(sensor_size)
        self._im_size = np.array(im_size)
        self._focal_len = focal_len
        self._dist_coeff = np.array(dist_coeff) if dist_coeff else None
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

        # PATB convention
        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera
        # -Parameters-defined
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

        # Intrinsic matrix to convert from camera co-ords in PATB convention (x->right, y->up, z->backwards,
        # looking through the camera at the scene) to pixel co-ords in standard convention (x->right, y->down).
        self._K = np.array(
            [[-sigma_xy[0], 0, self._im_size[0] / 2],
             [0, sigma_xy[1], self._im_size[1] / 2],
             [0, 0, 1]]
        )
        return

    def unproject(self, x: np.array, distort: bool=False):
        """
        Unproject from 3D world co-ordinates to 2D image co-ordinates

        When using this method to build orthorectification maps for use in e.g. ``cv2.remap()``, it is best to
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

        if distort and (self._dist_coeff is not None) and (not np.all(self._dist_coeff == 0)):
            # include the distortion model via opencv
            ij, _ = cv2.projectPoints(
                x - self._T, self._Rtv, np.array([0., 0., 0.]), self._K, distCoeffs=self._dist_coeff
            )
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
            x_ = cv2.undistortPoints(ij.astype('float64'), self._K, self._dist_coeff)
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
            # Find undistortion maps once off, and store for repeat use. (Opencv claims that using
            # cv2.initUndistortRectifyMap(...m1type=cv2.CV_16SC2) speeds up cv2.remap - that isn't the case here, but
            # it is still used to save some memory).
            self._undistort_maps = cv2.initUndistortRectifyMap(
                self._K, self._dist_coeff, None, None, self._im_size.astype(int), cv2.CV_16SC2  # cv2.CV_32FC1
            )
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
            # that and save a little memory.
            out_array = np.full(array.shape, fill_value=nodata, dtype=array.dtype)
            for bi in range(out_array.shape[0]):
                out_array[bi] = undistort_band(array[bi])
        else:
            out_array = undistort_band(array)

        return out_array