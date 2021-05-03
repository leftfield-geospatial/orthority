from simple_ortho import get_logger
import numpy as np
import cv2
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling,  transform
import pandas as pd
import pathlib
import datetime

logger = get_logger(__name__)

class Camera():
    def __init__(self, focal_len, sensor_size, im_size, geo_transform, position, orientation):

        self.update_extrinsic(position, orientation)

        if len(sensor_size) != 2 or len(im_size) != 2:
            raise Exception('len(sensor_size) != 2 or len(image_size) != 2')

        self._sensor_size = sensor_size
        self._im_size = im_size
        self._focal_len = focal_len

        self.update_intrinsic(geo_transform)

    def update_extrinsic(self, position, orientation):
        if len(position) != 3 or len(orientation) != 3:
            raise Exception('len(position) != 3 or len(orientation) != 3')
        self._T = position.copy()

        self._omega, self._phi, self._kappa = orientation

        # PATB convention
        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        omega_r = np.array([[1, 0, 0],
                            [0, np.cos(self._omega), -np.sin(self._omega)],
                            [0, np.sin(self._omega), np.cos(self._omega)]])

        phi_r = np.array([[np.cos(self._phi), 0, np.sin(self._phi)],
                          [0, 1, 0],
                          [-np.sin(self._phi), 0, np.cos(self._phi)]])

        kappa_r = np.array([[np.cos(self._kappa), -np.sin(self._kappa), 0],
                            [np.sin(self._kappa), np.cos(self._kappa), 0],
                            [0, 0, 1]])

        self._R = np.dot(np.dot(omega_r, phi_r), kappa_r)
        return


    def update_intrinsic(self, geo_transform, kappa=None):

        # See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        # and https://en.wikipedia.org/wiki/Camera_resectioning
        if len(geo_transform) != 6:
            raise Exception('len(geo_transform) != 6')

        if kappa is None:
            kappa = self._kappa

        # image signed dimensions for orientation (origin and kappa)
        image_size_s = -np.sign(np.cos(kappa)) * np.float64(
            [np.sign(geo_transform[0]) * self._im_size[0], np.sign(geo_transform[4]) * self._im_size[0]])
        sigma_xy = self._focal_len * image_size_s / self._sensor_size   # x,y signed focal lengths in pixels

        self._K = np.array([[sigma_xy[0], 0, self._im_size[0] / 2],
                            [0, sigma_xy[1], self._im_size[1] / 2],
                            [0, 0, 1]])
        return

    def unproject(self, X):
        # Unproject from 3D world to 2D image co-ordinates
        # x,y,z down 1st dimension
        if not(X.shape[0] == 3 and X.shape[1] > 0):
            raise Exception('not(X.shape[0] == 3 and X.shape[1] > 0')
        # reshape/transpose to xyz along 1st dimension, and broadcast rotation and translation for each xyz vector
        X_ = np.dot(self._R.T, (X - self._T))
        # homogenise xyz/z and apply intrinsic matrix, discarding 3rd dimension
        ij = np.dot(self._K, X_/X_[2, :])[:2, :]
        return ij

    def project_to_z(self, ij, Z):
        # Reproject from 2D image to 3D world co-ordinates with known Z
        # x,y,z down 1st dimension
        if not(ij.shape[0] == 2 and ij.shape[1] > 0):
            raise Exception('not(ij.shape[0] == 2 and ij.shape[1] > 0)')
        # TODO: check Z dimensions
        ij_ = np.row_stack([ij, np.ones((1, ij.shape[1]))])
        X_ = np.dot(np.linalg.inv(self._K), ij_)
        X_R = np.dot(self._R, X_) # rotate first (camera to world)
        X = (X_R * (Z - self._T[2])/X_R[2,:]) + self._T  # scale to desired Z and offset to world
        return X
