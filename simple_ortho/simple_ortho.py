from simple_ortho import get_logger
import numpy as np
import os
import cv2
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling,  transform
import pandas as pd
import pathlib
import datetime

logger = get_logger(__name__)

class Camera():
    def __init__(self, focal_len, sensor_size, im_size, geo_transform, position, orientation):
        """
        Camera class to project from camera 2D pixel co-ordinates to world 3D (x,y,z) co-ordinates, and vice-versa

        Parameters
        ----------
        focal_len :     float
                        focal length in mm
        sensor_size :   numpy.array_like
                        sensor (ccd) [width, height] in mm
        im_size :       numpy.array_like
                        image [width, height]] in pixels
        geo_transform :     numpy.array_like
                            gdal or rasterio 6 element image transform
        position :      numpy.array_like
                        column vector of [x=easting, y=northing, z=altitude] camera location co-ordinates, in image CRS
        orientation :   numpy.array_like
                        camera orientation [omega, phi, kappa] angles in degrees
        """

        self.update_extrinsic(position, orientation)

        if np.size(sensor_size) != 2 or np.size(im_size) != 2:
            raise Exception('len(sensor_size) != 2 or len(image_size) != 2')

        self._sensor_size = sensor_size
        self._im_size = im_size
        self._focal_len = focal_len

        self.update_intrinsic(geo_transform)


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
        """
        Update camera instrinsic parameters

        Parameters
        ----------
        geo_transform : numpy.array_like
                        gdal or rasterio 6 element image transform
        kappa :         float
                        (optional) kappa angle in degrees - if not specified kappa from last call of update_extrinsic()
                        is used
        """

        # Adapted from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        # and https://en.wikipedia.org/wiki/Camera_resectioning
        if np.size(geo_transform) < 6:
            raise Exception('len(geo_transform) < 6')

        if kappa is None:
            kappa = self._kappa

        # image signed dimensions for orientation (origin and kappa)
        image_size_s = -np.sign(np.cos(kappa)) * np.float64(
            [np.sign(geo_transform[0]) * self._im_size[0], np.sign(geo_transform[4]) * self._im_size[1]])
        sigma_xy = self._focal_len * image_size_s / self._sensor_size   # x,y signed focal lengths in pixels

        self._K = np.array([[sigma_xy[0], 0, self._im_size[0] / 2],
                            [0, sigma_xy[1], self._im_size[1] / 2],
                            [0, 0, 1]])
        return

    def unproject(self, X):
        """
        Unproject from 3D world co-ordinates to 2D image co-ordinates

        Parameters
        ----------
        X : numpy.array_like
            3-by-N array of 3D world (x=easting, y=northing, z=altitude) co-ordinates to unproject.
            (x,y,z) along the first dimension.

        Returns
        -------
        ij : numpy.array_like
             2-by-N array of image (i=row, j=column) co-ordinates. (i, j) along the first dimension.
        """
        # x,y,z down 1st dimension
        if not(X.shape[0] == 3 and X.shape[1] > 0):
            raise Exception('not(X.shape[0] == 3 and X.shape[1] > 0')

        # reshape/transpose to xyz along 1st dimension, and broadcast rotation and translation for each xyz vector
        X_ = np.dot(self._R.T, (X - self._T))
        # homogenise xyz/z and apply intrinsic matrix, discarding 3rd dimension
        ij = np.dot(self._K, X_/X_[2, :])[:2, :]

        return ij

    def project_to_z(self, ij, Z):
        """
        Project from 2D image co-ordinates to 3D world co-ordinates at a specified Z

        Parameters
        ----------
        ij : numpy.array_like
             2-by-N array of image (i=row, j=column) co-ordinates. (i, j) along the first dimension.
        Z :  numpy.array_like
             1-by-N array of Z (altitude) values to project to

        Returns
        -------
        X : numpy.array_like
            3-by-N array of 3D world (x=easting, y=northing, z=altitude) co-ordinates.
            (x,y,z) along the first dimension.
        """
        if not(ij.shape[0] == 2 and ij.shape[1] > 0):
            raise Exception('not(ij.shape[0] == 2 and ij.shape[1] > 0)')

        ij_ = np.row_stack([ij, np.ones((1, ij.shape[1]))])
        X_ = np.dot(np.linalg.inv(self._K), ij_)
        X_R = np.dot(self._R, X_) # rotate first (camera to world)
        X = (X_R * (Z - self._T[2])/X_R[2, :]) + self._T  # scale to desired Z and offset to world

        return X

class OrthoIm():
    def __init__(self, raw_im_filename, dem_filename, camera, config=None, ortho_im_filename=None, ):
        """
        Class to orthorectify image with known DEM and camera model

        Parameters
        ----------
        raw_im_filename :   str
                            Filename of raw image to orthorectified
        dem_filename :      str
                            Filename of DEM covering raw image
        camera :            simple_orth.Camera
                            camera object relevant to raw image
        ortho_im_filename : str
                            (optional) specify the filename of the orthorectified image to create.  If not specified,
                            appends '_ORTHO' to the raw_im_filename
        config :            dict
                            (optional) dictionary of configuration parameters.  With key, value pairs as follows:
                                'dem_interp': Interpolation type for resampling DEM (average, bilinear, cubic,
                                              cubic_spline, gauss, lanczos) default = 'cubic_spline'
                                'ortho_interp': Interpolation type for ortho-image (average, bilinear, cubic, lanczos,
                                                nearest), default = 'bilinear'
                                'resolution':  Output pixel size [x, y] in m, default = [0.5, 0.5]
                                'compression': GeoTIFF compression type (deflate, jpeg, jpeg2000, lzw, zstd, none),
                                                default = 'deflate'
                                'tile_size': default = Tile/block [x, y] size in pixels, default = [512, 512]
        """
        if not os.path.exists(raw_im_filename):
            raise Exception(f"Raw image file {raw_im_filename} does not exist")

        if not os.path.exists(dem_filename):
            raise Exception(f"DEM file {dem_filename} does not exist")

        self._raw_im_filename = pathlib.Path(raw_im_filename)
        self._dem_filename = pathlib.Path(dem_filename)

        self._camera = camera

        if ortho_im_filename is None:
            self._ortho_im_filename = self._raw_im_filename.parent.joinpath(self._raw_im_filename.stem + '_ORTHO.TIF')
        else:
            self._ortho_im_filename = pathlib.Path(ortho_im_filename)

        if config is None: # set defaults:
            config = dict(dem_interp='cubic_spline', ortho_interp='bilinear', resolution=[0.5, 0.5],
                          compression='deflate', tile_size=[512, 512])

        self._parse_config(config)
        self.dem_min = 0.
        # TODO: some error checking like does DEM cover raw image

    def _parse_config(self, config):
        """
        Parse dict config items where necessary

        Parameters
        ----------
        config :  dict
                  e.g. dict(dem_interp='cubic_spline', ortho_interp='bilinear', resolution=[0.5, 0.5],
                          compression='deflate', tile_size=[512, 512])
        """

        for key, value in config.items():
            setattr(self, key, value)

        try:
            self.dem_interp = Resampling[config['dem_interp']]
        except :
            logger.error(f'Unknown dem_interp configuration type: {config["dem_interp"]}')
            raise

        cv_interp_dict = dict(average=cv2.INTER_AREA, bilinear=cv2.INTER_LINEAR, cubic=cv2.INTER_CUBIC,
                              lanczos=cv2.INTER_LANCZOS4, nearest=cv2.INTER_NEAREST)

        if self.ortho_interp not in cv_interp_dict:
            raise Exception(f'Unknown ortho_interp configuration type: {config["ortho_interp"]}')
        else:
            self.ortho_interp = cv_interp_dict[self.ortho_interp]

    def _get_dem_min(self):
        """
        Find minimum of the DEM over the bounds of the raw image
        """

        dem_min = 0.
        with rio.Env():
            with rio.open(self._raw_im_filename, 'r') as raw_im:
                with rio.open(self._dem_filename, 'r') as dem_im:
                    # find raw image bounds in DEM CRS
                    [dem_xbounds, dem_ybounds] = transform(raw_im.crs, dem_im.crs,
                                                           [raw_im.bounds.left, raw_im.bounds.right],
                                                           [raw_im.bounds.top, raw_im.bounds.bottom])
                    dem_win = rio.windows.from_bounds(dem_xbounds[0], dem_ybounds[1], dem_xbounds[1], dem_ybounds[0],
                                                      transform=dem_im.transform)

                    # read DEM in raw image ROI and find minimum
                    dem_im_array = dem_im.read(1, window=dem_win)
                    dem_min = np.max([dem_im_array.min(), 0])

        return dem_min

    def _get_ortho_bounds(self, dem_min=0):
        """
        Get the bounds of the output ortho image in its CRS

        Parameters
        ----------
        dem_min : (optional) minimum altitude Z value over the image area in m, default=0

        Returns
        -------
        ortho_bl: numpy.array_like
                  [x, y] co-ordinates of the bottom left corner
        ortho_tr: numpy.array_like
                  [x, y] co-ordinates of the top right corner
        """

        with rio.Env():
            with rio.open(self._raw_im_filename, 'r') as raw_im:
                # find the bounds of the ortho by projecting 2D image pixel corners onto 3D Z plane = dem_min
                ortho_cnrs = self._camera.project_to_z(
                    np.array([[0, 0], [raw_im.width, 0], [raw_im.width, raw_im.height], [0, raw_im.height]]).T,
                    dem_min)[:2, :]

                raw_cnrs = np.array(
                    [[raw_im.bounds.left, raw_im.bounds.bottom], [raw_im.bounds.right, raw_im.bounds.bottom],
                     [raw_im.bounds.right, raw_im.bounds.top], [raw_im.bounds.left, raw_im.bounds.top]]).T

                ortho_cnrs = np.column_stack([ortho_cnrs, raw_cnrs])  # make doubl sure we encompass the raw image
                ortho_bl = ortho_cnrs.min(axis=1)   # bottom left
                ortho_tr = ortho_cnrs.max(axis=1)   # top right
                # ortho_wh = np.ceil(np.abs((ortho_bl - ortho_tr).squeeze()[:2] / self.resolution)) # ortho im width, height
                # ortho_transform = rio.transform.from_origin(ortho_bl[0], ortho_tr[1], *self.resolution)

        return ortho_bl, ortho_tr

