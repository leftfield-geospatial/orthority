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
import pathlib
import unittest

import numpy as np
import rasterio as rio
import yaml
from rasterio import windows
from shapely.geometry import box

from simple_ortho import root_path, command_line


class TestCommandLine(unittest.TestCase):
    def test_ortho_im(self):
        """
        Test ortho_im script on images in data/inputs/test_example
        """

        # construct script args to orthorectify images in data/inputs/test_example
        args = dict(src_im_file=[str(root_path.joinpath('data/inputs/test_example/*_RGB.tif'))],
                    dem_file=str(root_path.joinpath('data/inputs/test_example/dem.tif')),
                    pos_ori_file=str(root_path.joinpath('data/inputs/test_example/camera_pos_ori.txt')),
                    read_conf=str(root_path.joinpath('data/inputs/test_example/config.yaml')),
                    ortho_dir=str(root_path.joinpath('data/outputs/test_example')), verbosity=2, write_conf=None)

        # delete the ortho files if they exist
        ortho_im_wildcard = str(root_path.joinpath('data/outputs/test_example/*_ORTHO.tif'))
        for ortho_im_filename in glob.glob(ortho_im_wildcard):
            os.remove(ortho_im_filename)

        # run the script
        command_line.main(**args)
        try:
            # check there are the same number of ortho files as source files
            self.assertEqual(len(glob.glob(args['src_im_file'][0])), len(glob.glob(ortho_im_wildcard)),
                             msg='Number of ortho files == number of source files')

            # load the config so we know nodata
            with open(root_path.joinpath('data/inputs/test_example/config.yaml')) as config_f:
                config = yaml.safe_load(config_f)
                nodata = config['ortho']['nodata']

            # compare source and ortho files to check their means are similar and they overlap
            for src_im_filename, ortho_im_filename in zip(glob.glob(args['src_im_file'][0]),
                                                          glob.glob(ortho_im_wildcard)):
                with rio.open(src_im_filename, 'r', num_threads='all_cpus') as src_im:
                    src_array = src_im.read(1)
                    with rio.open(ortho_im_filename, 'r', num_threads='all_cpus') as ortho_im:
                        ortho_array = ortho_im.read(1)
                        ortho_array_masked = ortho_array[ortho_array != nodata]

                        # compare source and ortho means
                        self.assertAlmostEqual(ortho_array_masked.mean() / 10, src_array.mean() / 10, places=0,
                                               msg='Ortho and source means in same order of magnitude')

                        # compare source and ortho bounds
                        self.assertTrue(not rio.coords.disjoint_bounds(src_im.bounds, ortho_im.bounds),
                                        msg='Ortho and source bounds overlap')

            # check overlapping regions between pairwise combinations of ortho-images are roughly similar
            ortho_im_list = glob.glob(ortho_im_wildcard)
            for ortho_i1, ortho_im_filename1 in enumerate(ortho_im_list):
                for ortho_im_filename2 in ortho_im_list[ortho_i1 + 1:]:
                    with rio.open(ortho_im_filename1, 'r', num_threads='all_cpus') as ortho_im1, \
                            rio.open(ortho_im_filename2, 'r', num_threads='all_cpus') as ortho_im2:

                        box1 = box(*ortho_im1.bounds)
                        box2 = box(*ortho_im2.bounds)

                        if box1.intersects(box2):  # the images overlap
                            common_geom = box1.intersection(box2)

                            # find windows for the overlap area in each image and read
                            win1 = windows.from_bounds(*common_geom.bounds, transform=ortho_im1.transform)
                            win2 = windows.from_bounds(*common_geom.bounds, transform=ortho_im2.transform)
                            ortho_data1 = ortho_im1.read(1, window=win1)
                            ortho_data2 = ortho_im2.read(1, window=win2)

                            # common valid data mask
                            common_mask = (ortho_im1.read_masks(1, window=win1) &
                                           ortho_im2.read_masks(1, window=win2)).astype(bool, copy=False)

                            # find R2 corr coefficient between the valid data in the overlapping image regions
                            c = np.corrcoef(ortho_data1[common_mask], ortho_data2[common_mask])
                            ortho_im_filestem1 = pathlib.Path(ortho_im_filename1).stem
                            ortho_im_filestem2 = pathlib.Path(ortho_im_filename2).stem

                            print(f'Overlap similarity of {ortho_im_filestem1} and {ortho_im_filestem2}: {c[0, 1]:.4f}')
                            self.assertTrue(c[0, 1] > 0.5,
                                            msg=f'Overlap similarity of {ortho_im_filestem1} and {ortho_im_filestem2}')

        finally:
            pass  # leave the ortho images in the outputs dir so they can be manually checked if necessary


if __name__ == '__main__':
    unittest.main()
