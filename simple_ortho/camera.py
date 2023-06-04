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

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int], np.ndarray], dist_coeff: np.ndarray
    )->Union[None, Tuple[np.ndarray, np.ndarray]]:
        """"
        Create cv2.remap() maps for undistorting an image.
        """
        return None

    def _pixel_to_camera(self, ij: np.ndarray) -> np.ndarray:
        """
        Transform 2D pixel to normalised 3D camera co-ordinates.
        """
        ij_ = np.row_stack([ij, np.ones((1, ij.shape[1]))])
        # TODO: store inverse rather than recalculate each time
        x_ = np.dot(np.linalg.inv(self._K), ij_)
        return x_

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

    def undistort(self, image: np.ndarray, nodata: Union[float, int] = 0, interp: int=cv2.INTER_LINEAR) -> np.ndarray:
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
    parameters.
    """
    def __init__(self, *args, dist_coeff: np.ndarray = None, cx: float = 0., cy: float = 0., **kwargs):
        Camera.__init__(self, *args, **kwargs)
        self._dist_coeff = np.array(dist_coeff) if dist_coeff else None
        # cx = -0.004891928269862716, cy = 0.001106653485186689   # test code for eg3
        self._Kd = self._offset_intrinsic(self._K, self._im_size, cx=cx, cy=cy)
        # create undistort maps after offsetting K
        self._undistort_maps = self._create_undistort_maps(self._Kd, self._im_size, self._dist_coeff)

    @staticmethod
    def _offset_intrinsic(K: np.ndarray, im_size: np.ndarray, cx: float = 0., cy: float = 0.) -> np.ndarray:
        """
        Incorpotate ODM Brown model offsets not included in the OpenCV generic model.  See
        https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates and
        https://github.com/mapillary/OpenSfM/blob/7e393135826d3c0a7aa08d40f2ccd25f31160281/opensfm/src/bundle.h#LL299C25-L299C25.

        Following the radial/tangential distortion, ODM has an affine transform as part of their Brown model:
            xni = fx xu + cx
            yni = fy yu + cy
        where (xni, yni) are ODM "normalised image co-ordinates", and (fx, fy) & (cx, cy) are parameters specified in
        their `cameras.json` output.

        To get to pixel co-ordinates, they apply another affine transform (see
        https://opensfm.readthedocs.io/en/latest/geometry.html#pixel-coordinates):
            u = max(w, h) * xni + (w - 1) / 2
            v = max(w, h) * yni + (h - 1) / 2
        where (w, h) are the image (width, height) in pixels.

        So the effective pixel offsets become: max(w, h) * (cx, cy).
        """
        Kd = K.copy()
        # Kd[:2, 2] += np.diag(self._K)[:2] * np.array([cx, cy]) / self._focal_len  # equivalent to below
        Kd[:2, 2] += im_size.max() * np.array([cx, cy])
        return Kd

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int], np.ndarray], dist_coeff: np.ndarray
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
        undistort_maps = cv2.initUndistortRectifyMap(
            K, dist_coeff, None, None, np.array(im_size).astype(int), cv2.CV_16SC2
        )
        return undistort_maps

    def _pixel_to_camera(self, ij: np.ndarray) -> np.ndarray:
        x_ = cv2.undistortPoints(ij.astype('float64'), self._Kd, self._dist_coeff)
        x_ = np.row_stack([x_.squeeze().T, np.ones((1, ij.shape[1]))])
        return x_

    def world_to_pixel(self, x: np.ndarray, distort: bool = False) -> np.ndarray:
        if not distort:
            # using original K (Kd is incorporated in self._undistort_maps)
            # TODO: can we make K/Kd less cryptic? if undistort and _undistort_maps were omitted, we could have K=Kd
            return PinholeCamera.world_to_pixel(self, x)
        # using Kd
        ij, _ = cv2.projectPoints(x - self._T, self._Rtv, np.array([0., 0., 0.]), self._Kd, self._dist_coeff)
        ij = np.squeeze(ij).T
        return ij


class FisheyeCamera(Camera):
    """
    Fisheye camera model for transforming between 2D pixel and 3D world co-ordinates.  Compatible with ODM and OpenCV
    fisheye parameters.
    """
    def __init__(self, *args, dist_coeff=None, **kwargs):
        Camera.__init__(self, *args, **kwargs)
        self._dist_coeff = np.array(dist_coeff) if dist_coeff else None
        self._undistort_maps = self._create_undistort_maps(self._K, self._im_size, self._dist_coeff)

    @staticmethod
    def _create_undistort_maps(
        K: np.ndarray, im_size: Union[Tuple[int], np.ndarray], dist_coeff: np.ndarray
    )->Union[None, Tuple[np.ndarray, np.ndarray]]:
        # specify default R & P (new camera matrix) params
        undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            K, dist_coeff, np.eye(3), K, np.array(im_size).astype(int), cv2.CV_16SC2
        )
        return undistort_maps

    def _pixel_to_camera(self, ij: np.ndarray) -> np.ndarray:
        ij_cv = np.expand_dims(ij.T, axis=0).astype('float64')
        x_ = cv2.fisheye.undistortPoints(ij_cv, self._K, self._dist_coeff)
        x_ = np.row_stack([x_.squeeze().T, np.ones((1, ij.shape[1]))])
        return x_

    def world_to_pixel(self, x: np.ndarray, distort: bool=False) -> np.ndarray:
        if not distort:
            return PinholeCamera.world_to_pixel(self, x)
        x_cv = np.expand_dims((x - self._T).T, axis=0)
        ij, _ = cv2.fisheye.projectPoints(x_cv, self._Rtv, np.array([0., 0., 0.]), self._K, self._dist_coeff)
        ij = np.squeeze(ij).T
        return ij


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
