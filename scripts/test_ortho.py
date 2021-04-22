import cv2
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import pathlib
from simple_ortho import get_logger

logger = get_logger(__name__)

im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_07_0298_RGB_PRE.tif")
dem_filename = pathlib.Path(r"D:\Data\Development\Projects\PhD GeoInformatics\Data\CGA\SUDEM\SUDEM_3318B_D_5m.tif")
ext_filename = pathlib.Path(r"C:\Data\Development\Projects\PhD GeoInformatics\Docs\PCI\NGI Orthorectification\extori3318D_2016_1143_lo19wgs84n_e_rect.txt")
ortho_filename = im_filename.parent.joinpath(im_filename.stem + '_ORTHO.TIF')
ortho_res = [0.5, 0.5]

# construct intrinsic matrix
focal_length = 120. # mm
ccd_xysize = np.array([92.160, 165.888])  # mm
im_xydim = [0, 0]   # pixels
with rio.open(im_filename) as raw_im:
    im_xydim = np.float64([raw_im.width, raw_im.height])
sigma_xy = focal_length * im_xydim / ccd_xysize

K = np.array([[-sigma_xy[0], 0, im_xydim[0]/2], [0, sigma_xy[1], im_xydim[1]/2], [0, 0, 1]])

# construct extrinsic  matrix

extd = pd.read_csv(ext_filename, header=None, sep=' ', index_col=0,
                   names=['file', 'easting', 'northing', 'altitude', 'roll', 'pitch', 'yaw'])
im_filestem = pathlib.Path(im_filename).stem[:-4]
im_ext = extd.loc[im_filestem]
roll, pitch, yaw = np.pi * im_ext[['roll','pitch','yaw']] / 180.
# See https://stackoverflow.com/questions/21412169/creating-a-rotation-matrix-with-pitch-yaw-roll-using-eigen/21412445#21412445
pitch_r = np.array([[1, 0, 0],
                    [0, np.cos(pitch), np.sin(pitch)],
                    [0, -np.sin(pitch), np.cos(pitch)]])

yaw_r = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                  [0, 1, 0],
                  [np.sin(yaw), 0, np.cos(yaw)]])

roll_r = np.array([[np.cos(roll), np.sin(roll), 0],
                   [-np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])

R = (roll_r * yaw_r) * pitch_r
T = np.array([im_ext['easting'], im_ext['northing'], im_ext['altitude']])

if False:
    res = cv2.projectPoints(np.array([[-21010.5,-3739755.8, 500], [-20369.6, -3741246.6, 700], [-22738.7,-3736138.1, 500]]),
                            np.linalg.inv(R), -T, K, None)

# with rio.open(im_filename) as raw_im:
raw_im = rio.open(im_filename)
dem_im = rio.open(dem_filename)
if raw_im.crs != dem_im.crs:
    logger.warning('DEM and image co-ordinate systems are different')

with rio.Env():
    with rio.open(im_filename, 'r') as raw_im:
        ortho_win_off = np.array([0, 0])  # test
        ortho_profile = raw_im.profile

        ortho_profile.update(nodata=0, compress='deflate', tiled=True, blockxsize=512, blockysize=512)

        with rio.open(ortho_filename, 'w', **ortho_profile) as ortho_im:
            for ji, block_win in ortho_im.block_windows(1):
                print((ji, block_win))
                win_transform = ortho_im.window_transform(block_win)
                print(win_transform)

                with rio.open(dem_filename, 'r') as dem_im:
                    dem_win_reproj = np.zeros((block_win.height, block_win.width), dem_im.dtypes[0])
                    # calculate_default_transform(ortho_im.crs, dem_im.crs, block_win.width, block_win.height)
                    reproject(rio.band(dem_im, 1), dem_win_reproj, dst_transform=win_transform, dst_crs=ortho_im.crs,
                              resampling=Resampling.bilinear)


for band_i in range(1, raw_im.count+1):
    logger.info(f'Reading band {band_i} from {im_filestem}')
    raw_band = raw_im.read(band_i)
    dem_roi

raw_im.close()
im = rio.open(im_filename)

# memory planning
# we want to keep the whole unrect image in memory, perhaps one band at a time, so we can freely do "lookups"/ remaps
# then it probably makes sense to force a tiled output file, and remap one tile at a time.
# possibly also resampling one dem roi corresponding to a tile at a time
