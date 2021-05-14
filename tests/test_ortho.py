import rasterio as rio
import numpy as np
import pandas as pd
import pathlib
from simple_ortho import get_logger
from simple_ortho import simple_ortho
import datetime
from simple_ortho import root_path
import yaml
import tracemalloc

logger = get_logger(__name__)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_07_0298_RGB_PRE.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_08_0321_RGB_PRE.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_11_0451_RGB_PRE.tif")
im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3323D_2015_1001\RGBN\3323d_2015_1001_02_0078_RGBN_CMP.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_11_0449_RGB_PRE.tif")

# dem_filename = pathlib.Path(r"D:\Data\Development\Projects\PhD GeoInformatics\Data\CGA\SUDEM\SUDEM_3318B_D_5m.tif")
dem_filename = pathlib.Path(
    r"D:\Data\Development\Projects\PhD GeoInformatics\Data\CGA\SUDEM L3 Unedited\x3323db_2015_L3a.tif")
# ext_filename = pathlib.Path(r"C:\Data\Development\Projects\PhD GeoInformatics\Docs\PCI\NGI Orthorectification\extori3318D_2016_1143_lo19wgs84n_e_rect.txt")
ext_filename = pathlib.Path(
    r"C:\Data\Development\Projects\PhD GeoInformatics\Docs\PCI\NGI Orthorectification\extori3323D_2015_1001_lo23wgs84n_e_rect.txt")

# also, make a default config file?
with open(root_path.joinpath('config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

with open(root_path.joinpath('config_out.yml'), 'w') as f:
    yaml.dump(config, stream=f)

with rio.open(im_filename) as raw_im:
    geo_transform = raw_im.transform
    im_size = np.float64([raw_im.width, raw_im.height])

cam_extorid = pd.read_csv(ext_filename, header=None, sep=' ', index_col=0,
                          names=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])

im_filestem = pathlib.Path(im_filename).stem[:-4]
im_ext = cam_extorid.loc[im_filestem]

orientation = np.array(np.pi * im_ext[['omega', 'phi', 'kappa']] / 180.)
position = np.array([im_ext['easting'], im_ext['northing'], im_ext['altitude']])

camera_config = config['camera']
camera = simple_ortho.Camera(camera_config['focal_len'], camera_config['sensor_size'], camera_config['im_size'],
                             geo_transform, position, orientation, dtype=np.float32)

X = np.array([[-22544.1, -3736338.7, 200], [-20876.7, -3739374.3, 200], [-19430.2, -3742345.8, 200],
              [-19430.2, -3742345.8, 200]], dtype='float32').T
start = datetime.datetime.now()
for j in range(0, 10000):
    ij = camera.unproject(X, use_cv=True)
print(f'unproject {datetime.datetime.now() - start}')

X2 = camera.project_to_z(ij, X[2, :])
print(ij)
print(X - X2)

# ij2, _ = cv2.projectPoints(X-camera._T, camera._R, np.array([0.,0.,0.], dtype='float32'), camera._K, distCoeffs=np.array([0], dtype='float32'))

tracemalloc.start()
ortho_im = simple_ortho.OrthoIm(im_filename, dem_filename, camera, config=config['ortho'])
ortho_im.orthorectify()
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
tracemalloc.stop()
