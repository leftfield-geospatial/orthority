import cv2
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling,  transform
import numpy as np
import pandas as pd
import pathlib
from simple_ortho import get_logger
from simple_ortho import simple_ortho
import datetime
from scripts import root_path
import yaml

logger = get_logger(__name__)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_07_0298_RGB_PRE.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_08_0321_RGB_PRE.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_11_0451_RGB_PRE.tif")
dem_filename = pathlib.Path(r"D:\Data\Development\Projects\PhD GeoInformatics\Data\CGA\SUDEM\SUDEM_3318B_D_5m.tif")
ext_filename = pathlib.Path(r"C:\Data\Development\Projects\PhD GeoInformatics\Docs\PCI\NGI Orthorectification\extori3318D_2016_1143_lo19wgs84n_e_rect.txt")

# TODO require cmd line spec of dem, raw, and ext_ori filenames, optional out file and res, overwrite in cfg if they are specified
# also, make a default config file?
with open(root_path.joinpath('config.yml'), 'r') as f:
    config = yaml.safe_load(f)

with rio.open(im_filename) as raw_im:
    geo_transform = raw_im.transform
    im_size = np.float64([raw_im.width, raw_im.height])

cam_extorid = pd.read_csv(ext_filename, header=None, sep=' ', index_col=0,
                   names=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])

im_filestem = pathlib.Path(im_filename).stem[:-4]
im_ext = cam_extorid.loc[im_filestem]

orientation = np.pi * im_ext[['omega', 'phi', 'kappa']] / 180.
position = np.array([im_ext['easting'], im_ext['northing'], im_ext['altitude']])

camera_config = config['camera']
camera = simple_ortho.Camera(camera_config['focal_len'], camera_config['sensor_size'], camera_config['im_size'],
                             geo_transform, position, orientation)

X = np.array([[-22544.1, -3736338.7, 200], [-20876.7, -3739374.3, 200], [-19430.2, -3742345.8, 200],
              [-19430.2, -3742345.8, 200]]).T
ij = camera.unproject(X)
X2 = camera.project_to_z(ij, X[2, :])
print(ij)
print(X - X2)

ortho_im = simple_ortho.OrthoIm(im_filename, dem_filename, camera, config=config['ortho'])
ortho_im.orthorectify()