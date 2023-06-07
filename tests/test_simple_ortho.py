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

import os
import unittest
from typing import Tuple

import numpy as np
import rasterio as rio

from simple_ortho import root_path
from simple_ortho.camera import Camera, create_camera
from simple_ortho.enums import CameraType
from simple_ortho.ortho import OrthoIm


def _create_camera(cam_type: CameraType = CameraType.fisheye):
    """
    Create a camera for downsampled NGI 3324c_2015_1004_05_0182_RGB image (as in data/inputs/test_sample)
    """

    # hard code camera parameters
    position = np.array([-55094.504480, -3727407.037480, 5258.307930])
    orientation = np.array([-0.349216, 0.298484, -179.086702]) * np.pi / 180.

    # create camera
    return create_camera(cam_type, position, orientation, 120, (640, 1152), sensor_size=(92.160, 165.888))


class TestSimpleOrthoModule(unittest.TestCase):
    def _test_camera(self, cam_type: CameraType = CameraType.fisheye):
        """
        Test camera functionality including projection
        """

        camera = _create_camera(cam_type=cam_type)

        # check camera params have been created
        self.assertTrue(hasattr(camera, '_K'), msg="Intrinsic params created")
        self.assertTrue(hasattr(camera, '_R') and hasattr(camera, '_T'), msg="Extrinsic params created")
        self.assertTrue(np.allclose(np.dot(camera._R, camera._R.T), np.eye(3), atol=1e-6), msg="Rotation matrix valid")

        # create arbitrary world co-ords and unproject to image co-ords
        start_x = np.array([-52359.614680, -3727390.785280, 500])
        x = (start_x + np.random.rand(100, 3) * np.array([5000, 5000, 500])).T
        ji = camera.world_to_pixel(x)

        # re-project image co-ords to world space at original z
        x2 = camera.pixel_to_world_z(ji, x[2, :])

        # check original and re-projected co-ords are approx equal
        self.assertTrue(np.allclose(x, x2, atol=1e-4), msg="Image <-> world projections ok")

    def test_camera(self):
        """ Test all camera types with no distortion parameters. """
        for cam_type in CameraType:
            self._test_camera(cam_type)


    # TODO: add tests for partial and no DEM coverage
    def _test_ortho_im_class(
        self, cam_type: CameraType = CameraType.pinhole,
        ortho_bounds: Tuple[float]=(-57132.499, -3731010.241, -53107.291, -3723903.939)
    ):
        """
        Test ortho_im support functionality and orthorectify the test_example data
        """
        # hard code camera and config
        camera = _create_camera(cam_type=cam_type)
        config = dict(dem_interp='cubic_spline', dem_band=1, interp='bilinear', resolution=[5, 5],
                      compress=None, tile_size=[256, 256], interleave='pixel', photometric=None, nodata=0,
                      per_band=False, driver='GTiff', dtype=None, build_ovw=True, overwrite=True, write_mask=True)

        # point to the test_example data
        src_im_filename = root_path.joinpath('data/inputs/test_example/3324c_2015_1004_05_0182_RGB.tif')
        dem_filename = root_path.joinpath('data/inputs/test_example/dem.tif')
        ortho_im_filename = root_path.joinpath('data/outputs/test_example/3324c_2015_1004_05_0182_RGB_ORTHO_TEST.tif')

        if ortho_im_filename.exists():
            os.remove(ortho_im_filename)

        ortho_im = OrthoIm(src_im_filename, dem_filename, camera, config=config,
                                        ortho_im_filename=ortho_im_filename)            # create OrthoIm object

        # test config set correctly
        for k, v in config.items():
            self.assertTrue(hasattr(ortho_im, k), msg=f'OrthoIm has {k} config attribute')
            if 'interp' not in k:
                self.assertEqual(getattr(ortho_im, k), config[k], msg=f'OrthoIm {k} config attribute set ok')

        # test _get_ortho_bounds() with hard coded vals
        _ortho_bounds = ortho_im._get_ortho_bounds()

        self.assertTrue(np.allclose(_ortho_bounds, ortho_bounds, atol=1e-2), msg=f"Ortho bounds OK: {cam_type}")

        try:
            ortho_im.orthorectify()         # run the orthorectification
            ortho_im.build_ortho_overviews()

            # do some sparse checks on ortho_im
            self.assertTrue(ortho_im_filename.exists(), msg="Ortho file exists")
            with rio.open(ortho_im_filename, 'r', num_threads='all_cpus') as o_im:
                self.assertEqual(o_im.res, tuple(config['resolution']), 'Ortho resolution ok')
                self.assertEqual(o_im.block_shapes[0], tuple(config['tile_size']), 'Tile size ok')
                self.assertEqual(o_im.nodata, config['nodata'], 'Nodata ok')
                self.assertTrue(np.allclose(
                    [o_im.bounds.left, o_im.bounds.top], [_ortho_bounds[0], _ortho_bounds[3]], atol=1e-2), 'TL cnr ok'
                )
                self.assertTrue(np.allclose([*o_im.bounds], _ortho_bounds, atol=ortho_im.resolution[0]), 'Bounds ok')

                # check the ortho and source image means and sizes in same order of magnitude
                o_band = o_im.read(1)
                o_band = o_band[o_band != config['nodata']]
                with rio.open(src_im_filename, 'r', num_threads='all_cpus') as s_im:
                    s_band = s_im.read(1)
                    self.assertAlmostEqual(o_band.mean() / 10, s_band.mean() / 10, places=0,
                                           msg='Ortho and source means in same order of magnitude')
                    self.assertAlmostEqual(o_band.size / s_band.size, 1, places=0,
                                           msg='Ortho and source sizes in same order of magnitude')

        finally:
            if ortho_im_filename.exists():
                os.remove(ortho_im_filename)  # tidy up

    def test_ortho_im_class(self):
        """ Test OrthoIm with all camera types. """

        # Create list of camera types with bounds to test against.  The fisheye camera does not have the pinhole camera
        # as a special case (with distortion coefficients==0), so fisheye has its own bounds.
        pinhole_bounds = (-57132.499, -3731010.241, -53107.291, -3723903.939)
        fisheye_bounds = (-57692.861, -3732012.478, -52546.357, -3722926.400)
        cam_list = [dict(cam_type=ct, ortho_bounds=pinhole_bounds) for ct in CameraType if ct != CameraType.fisheye]
        cam_list.append(dict(cam_type=CameraType.fisheye, ortho_bounds=fisheye_bounds))

        for cam_dict in cam_list:
            self._test_ortho_im_class(**cam_dict)


if __name__ == '__main__':
    unittest.main()

##
