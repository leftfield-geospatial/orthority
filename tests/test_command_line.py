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

import glob
import os
import unittest
from pathlib import Path

import numpy as np
import rasterio as rio

from simple_ortho import root_path, command_line


class TestCommandLine(unittest.TestCase):
    def _test_ortho_im(
        self, input_dir='data/inputs/test_example', output_dir='data/outputs/test_example', input_wildcard='*_RGB.tif'
    ):
        """
        Test ortho_im script on images in data/inputs/test_example
        """
        input_path = root_path.joinpath(input_dir)
        output_path = root_path.joinpath(output_dir)
        # construct script args to orthorectify images in data/inputs/test_example
        args = dict(src_im_file=[str(input_path.joinpath(input_wildcard))],
                    dem_file=str(input_path.joinpath('dem.tif')),
                    pos_ori_file=str(input_path.joinpath('camera_pos_ori.txt')),
                    read_conf=str(input_path.joinpath('config.yaml')),
                    ortho_dir=str(output_path), verbosity=2, write_conf=None)

        # delete the ortho files if they exist
        ortho_im_wildcard = str(output_path.joinpath('*_ORTHO.tif'))
        for ortho_filename in glob.glob(ortho_im_wildcard):
            os.remove(ortho_filename)

        # run the script
        command_line.main(**args)
        try:
            # check there are the same number of ortho files as source files
            self.assertEqual(len(glob.glob(args['src_im_file'][0])), len(glob.glob(ortho_im_wildcard)),
                             msg='Number of ortho files == number of source files')

            # compare source and ortho files to check their means are similar and they overlap
            for src_filename, ortho_filename in zip(glob.glob(args['src_im_file'][0]),
                                                          glob.glob(ortho_im_wildcard)):
                with rio.open(src_filename, 'r', num_threads='all_cpus') as src_im:
                    src_array = src_im.read(1)
                    with rio.open(ortho_filename, 'r', num_threads='all_cpus') as ortho_im:
                        ortho_array = ortho_im.read(1)
                        ortho_mask = ortho_im.dataset_mask().astype('bool')
                        ortho_array_masked = ortho_array[ortho_mask]

                        # compare source and ortho means
                        source_mean = src_array.mean()
                        self.assertAlmostEqual(ortho_array_masked.mean(), source_mean, delta=source_mean/10,
                                               msg='Ortho and source means in same order of magnitude')

                        if src_im.crs and not src_im.transform == rio.Affine.identity():
                            # compare source and ortho bounds
                            self.assertTrue(not rio.coords.disjoint_bounds(src_im.bounds, ortho_im.bounds),
                                            msg='Ortho and source bounds overlap')

            # check overlapping regions between pairwise combinations of ortho-images are roughly similar
            ortho_im_list = glob.glob(ortho_im_wildcard)
            for ortho_i1, ortho_filename1 in enumerate(ortho_im_list):
                for ortho_filename2 in ortho_im_list[ortho_i1 + 1:]:
                    with rio.open(ortho_filename1, 'r', num_threads='all_cpus') as ortho_im1, \
                            rio.open(ortho_filename2, 'r', num_threads='all_cpus') as ortho_im2:

                        # find windows for the overlap area in each image
                        common_bl = np.array([ortho_im1.bounds[:2], ortho_im2.bounds[:2]]).max(axis=0)
                        common_tr = np.array([ortho_im1.bounds[2:], ortho_im2.bounds[2:]]).min(axis=0)
                        if np.any(common_bl >= common_tr):
                            continue  # no overlap
                        common_bounds = [*common_bl, *common_tr]

                        win1 = ortho_im1.window(*common_bounds)
                        win2 = ortho_im2.window(*common_bounds)

                        # test overlap similarity when overlap area > 20%
                        if (win1.width * win1.height) / (ortho_im1.width * ortho_im1.height) > 0.2:
                            # read overlap area in each image
                            ortho_data1 = ortho_im1.read(1, window=win1)
                            ortho_data2 = ortho_im2.read(1, window=win2)

                            # common valid data mask
                            common_mask = (ortho_im1.read_masks(1, window=win1) &
                                           ortho_im2.read_masks(1, window=win2)).astype(bool, copy=False)

                            # find R2 corr coefficient between the valid data in the overlapping image regions
                            c = np.corrcoef(ortho_data1[common_mask], ortho_data2[common_mask])
                            ortho_im_filestem1 = Path(ortho_filename1).stem
                            ortho_im_filestem2 = Path(ortho_filename2).stem

                            print(f'Overlap similarity of {ortho_im_filestem1} and {ortho_im_filestem2}: {c[0, 1]:.4f}')
                            self.assertTrue(c[0, 1] > 0.6,
                                            msg=f'Overlap similarity of {ortho_im_filestem1} and {ortho_im_filestem2}')

        finally:
            pass  # leave the ortho images in the outputs dir so they can be manually checked if necessary

    def test_ortho_im(self):
        self._test_ortho_im(input_dir='data/inputs/test_example', output_dir='data/outputs/test_example')
        self._test_ortho_im(input_dir='data/inputs/test_example2', output_dir='data/outputs/test_example2')
        self._test_ortho_im(
            input_dir='data/inputs/test_example3', output_dir='data/outputs/test_example3', input_wildcard='*.JPG'
        )
        self._test_ortho_im(
            input_dir='data/inputs/test_example4', output_dir='data/outputs/test_example4', input_wildcard='*GRE.TIF'
        )


if __name__ == '__main__':
    unittest.main()
