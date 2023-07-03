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
from simple_ortho.ortho import Ortho


def _create_camera(cam_type: CameraType = CameraType.fisheye, **kwargs):
    """
    Create a camera for downsampled NGI 3324c_2015_1004_05_0182_RGB image (as in data/inputs/test_sample)
    """

    # hard code camera parameters
    position = np.array([-55094.504480, -3727407.037480, 5258.307930])
    orientation = np.array([-0.349216, 0.298484, -179.086702]) * np.pi / 180.

    # create camera
    return create_camera(cam_type, position, orientation, 120., (640., 1152.), sensor_size=(92.160, 165.888), **kwargs)


class TestSimpleOrthoModule(unittest.TestCase):
    def _test_camera(self, camera: Camera):
        """
        Test camera functionality including projection
        """

        # check camera params have been created
        self.assertTrue(hasattr(camera, '_K'), msg="Intrinsic params created")
        self.assertTrue(hasattr(camera, '_R') and hasattr(camera, '_T'), msg="Extrinsic params created")
        self.assertTrue(np.allclose(np.dot(camera._R, camera._R.T), np.eye(3), atol=1e-6), msg="Rotation matrix valid")

        # create arbitrary world co-ords and unproject to image co-ords
        start_x = np.array([-57132.499, -3723903.939, 500])
        x = (start_x + np.random.rand(100, 3) * np.array([3500, -7000, 500])).T
        ji = camera.world_to_pixel(x)

        # re-project image co-ords to world space at original z
        x2 = camera.pixel_to_world_z(ji, x[2, :])

        # check original and re-projected co-ords are approx equal
        self.assertTrue(np.allclose(x, x2, atol=1e-2), msg=f"Image <-> world projections ok: {camera}")

    def test_camera(self):
        """ Test all camera types with distortion parameters. """
        cam_list = [
            _create_camera(cam_type=CameraType.pinhole),
            _create_camera(cam_type=CameraType.fisheye, k1=-0.0525, k2=-0.0098),
            _create_camera(
                cam_type=CameraType.opencv,  k1=-0.0093, k2=0.0075, p1=-0.0004, p2=-0.0004, k3=0.0079,
            ),
            _create_camera(
                cam_type=CameraType.brown, k1=-0.0093, k2=0.0075, p1=-0.0004, p2=-0.0004, k3=0.0079, cx=-0.0049,
                cy=0.0011
            ),
        ]
        for camera in cam_list:
            self._test_camera(camera)

    # TODO: add tests for partial and no DEM coverage
    def _test_ortho_im_class(
        self, camera: Camera, ortho_bounds: Tuple[float] = (-57132.499, -3731010.241, -53107.291, -3723903.939),
        max_size_diff: float = 0.5
    ):
        """
        Test ortho_im support functionality and orthorectify the test_example data
        """
        # hard code camera and config
        dem_band = 1
        resolution = (5, 5)
        config = dict(
            dem_interp='cubic_spline', interp='bilinear', compress=None, blockxsize=256, blockysize=256,
            interleave='pixel', photometric=None, nodata=0, per_band=False, driver='GTiff', dtype=None, build_ovw=True,
            overwrite=True, write_mask=True
        )

        # point to the test_example data
        src_filename = root_path.joinpath('data/inputs/test_example/3324c_2015_1004_05_0182_RGB.tif')
        dem_filename = root_path.joinpath('data/inputs/test_example/dem.tif')
        ortho_filename = root_path.joinpath('data/outputs/test_example/3324c_2015_1004_05_0182_RGB_ORTHO_TEST.tif')

        if ortho_filename.exists():
            os.remove(ortho_filename)

        # create Ortho object
        ortho_im = Ortho(src_filename, dem_filename, camera, dem_band=dem_band)

        # test config set correctly
        # for k, v in config.items():
        #     self.assertTrue(hasattr(ortho_im, k), msg=f'Ortho has {k} config attribute')
        #     if 'interp' not in k and v is not None:
        #         self.assertEqual(getattr(ortho_im, k), v, msg=f'Ortho {k} config attribute set ok')

        # test _get_ortho_bounds() with hard coded vals
        # _ortho_bounds = ortho_im._get_ortho_bounds()
        #
        # self.assertTrue(np.allclose(_ortho_bounds, ortho_bounds, atol=1e-2), msg=f"Ortho bounds OK: {camera}")

        try:
            ortho_im.process(ortho_filename, resolution, **config)         # run the orthorectification

            # do some sparse checks on ortho_im
            self.assertTrue(ortho_filename.exists(), msg="Ortho file exists")
            with rio.open(ortho_filename, 'r', num_threads='all_cpus') as o_im:
                self.assertEqual(o_im.res, resolution, 'Ortho resolution ok')
                self.assertEqual(
                    o_im.block_shapes[0], (config['blockysize'], config['blockxsize']), 'Tile size ok'
                )
                self.assertEqual(o_im.nodata, config['nodata'], 'Nodata ok')
                # self.assertTrue(np.allclose(
                #     [o_im.bounds.left, o_im.bounds.top], [_ortho_bounds[0], _ortho_bounds[3]], atol=1e-2), 'TL cnr ok'
                # )
                # self.assertTrue(np.allclose([*o_im.bounds], _ortho_bounds, atol=ortho_im.resolution[0]), 'Bounds ok')

                # check the ortho and source image means and sizes in same order of magnitude
                o_band = o_im.read(1)
                o_band = o_band[o_band != config['nodata']]
                with rio.open(src_filename, 'r', num_threads='all_cpus') as s_im:
                    s_band = s_im.read(1)
                    self.assertAlmostEqual(o_band.mean() / 10, s_band.mean() / 10, places=0,
                                           msg='Ortho and source means in same order of magnitude')
                    self.assertLess(
                        np.abs(o_band.size - s_band.size) / np.max((o_band.size, s_band.size)), max_size_diff,
                        msg='Ortho and source sizes similar'
                    )

        finally:
            if ortho_filename.exists():
                os.remove(ortho_filename)  # tidy up

    def test_ortho_im_class(self):
        """ Test Ortho with all camera types. """

        # Create list of camera types with bounds to test against.  The fisheye camera does not have the pinhole camera
        # as a special case (with distortion coefficients==0), so fisheye has its own bounds.
        pinhole_bounds = (-57132.499, -3731010.241, -53107.291, -3723903.939)
        fisheye_bounds = (-57692.861, -3732012.478, -52546.357, -3722926.400)
        cam_list = [
            dict(camera=_create_camera(ct), ortho_bounds=pinhole_bounds) for ct in CameraType
            if ct != CameraType.fisheye
        ]
        cam_list.append(dict(camera=_create_camera(CameraType.fisheye), ortho_bounds=fisheye_bounds))

        for cam_dict in cam_list:
            self._test_ortho_im_class(**cam_dict)


if __name__ == '__main__':
    unittest.main()

##
