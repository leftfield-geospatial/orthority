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

from simple_ortho import get_logger

# from scipy.ndimage import map_coordinates

logger = get_logger(__name__)


class Camera:
    def __init__(
        self, focal_len, sensor_size, im_size, geo_transform, position, orientation, dtype='float32', dist_coeff=None
    ):
        """
        Camera class to project from 2D camera (i,j) pixel co-ordinates to 3D world (x,y,z) co-ordinates,
        and vice-versa

        Parameters
        ----------
        focal_len :     float
                        focal length in mm
        sensor_size :   numpy.array_like
                        sensor (ccd) [width, height] in mm
        im_size :       numpy.array_like
                        image [width, height]] in pixels
        geo_transform :     numpy.array_like
                            gdal or rasterio 6 element image transform (only the pixel scale is used)
        position :      numpy.array_like
                        column vector of [x=easting, y=northing, z=altitude] camera location co-ordinates, in image CRS
        orientation :   numpy.array_like
                        camera orientation [omega, phi, kappa] angles in radians
        dtype :         numpy.dtype, Type
                        Data type to use for camera parameters (to avoid e.g. unproject forcing float32 to 64)
        """

        self._dtype = dtype
        self.update_extrinsic(position, orientation)

        if np.size(sensor_size) != 2 or np.size(im_size) != 2:
            raise Exception('len(sensor_size) != 2 or len(image_size) != 2')

        self._sensor_size = np.array(sensor_size)
        self._im_size = np.array(im_size)
        self._focal_len = focal_len
        self._dist_coeff = np.array(dist_coeff) if dist_coeff else None

        self.update_intrinsic()
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
        self._T = np.array(position, dtype=self._dtype).reshape(3, 1)

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

        self._R = np.dot(np.dot(omega_r, phi_r), kappa_r).astype(self._dtype)
        self._Rtv = cv2.Rodrigues(self._R.T)[0]
        return

    def update_intrinsic(self):
        """
        Update camera intrinsic parameters
        """

        # Adapted from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera
        # -Parameters-defined
        # and https://en.wikipedia.org/wiki/Camera_resectioning
        sigma_xy = self._focal_len * self._im_size / self._sensor_size  # xy focal lengths in pixels

        # Intrinsic matrix to convert from camera co-ords in PATB convention (x->right, y->up, z->backwards,
        # looking through the camera at the scene) to pixel co-ords in standard convention (x->right, y->down).
        self._K = np.array(
            [[-sigma_xy[0], 0, self._im_size[0] / 2],
             [0, sigma_xy[1], self._im_size[1] / 2],
             [0, 0, 1]], dtype=self._dtype
        )
        return

    def unproject(self, x, use_cv=True):
        """
        Unproject from 3D world co-ordinates to 2D image co-ordinates

        Parameters
        ----------
        x : numpy.array_like
            3-by-N array of 3D world (x=easting, y=northing, z=altitude) co-ordinates to unproject.
            (x,y,z) along the first dimension.
        use_cv : bool (optional)
                 False = use the numpy implementation (faster - recommended)
                 True = use the opencv implementation

        Returns
        -------
        ij : numpy.array_like
             2-by-N array of image (i=row, j=column) co-ordinates. (i, j) along the first dimension.
        """
        # x,y,z down 1st dimension
        if not (x.shape[0] == 3 and x.shape[1] > 0):
            raise Exception('x must have 3 rows and more than one column')

        if use_cv:  # use opencv
            ij, _ = cv2.projectPoints(
                x - self._T, self._Rtv, np.array([0., 0., 0.], dtype=self._dtype), self._K, distCoeffs=self._dist_coeff
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

        ij_ = np.row_stack([ij, np.ones((1, ij.shape[1]))])
        # TODO: store inverse rather than recalculate each time
        x_ = np.dot(np.linalg.inv(self._K), ij_)
        # rotate first (camera to world) to get world aligned axes with origin on the camera
        x_r = np.dot(self._R, x_)
        # scale to desired z (offset for camera z) with origin on camera, then offset to world
        x = (x_r * (z - self._T[2]) / x_r[2, :]) + self._T
        return x
