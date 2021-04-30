import cv2
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling,  transform
import numpy as np
import pandas as pd
import pathlib
from simple_ortho import get_logger
import datetime
# See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
# and https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf


logger = get_logger(__name__)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
# 3318D_2016_1143_11_0450_RGB.tif
im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_07_0298_RGB_PRE.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_08_0321_RGB_PRE.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_11_0451_RGB_PRE.tif")

dem_filename = pathlib.Path(r"D:\Data\Development\Projects\PhD GeoInformatics\Data\CGA\SUDEM\SUDEM_3318B_D_5m.tif")
ext_filename = pathlib.Path(r"C:\Data\Development\Projects\PhD GeoInformatics\Docs\PCI\NGI Orthorectification\extori3318D_2016_1143_lo19wgs84n_e_rect.txt")
ortho_filename = im_filename.parent.joinpath(im_filename.stem + '_ORTHO.TIF')
ortho_res = [0.5, 0.5]
block_size = [512, 512]

def unproject(X, R, T, K):
    # Unproject from 3D world to 2D image co-ordinates
    # x,y,z down 1st dimension
    if T.shape != (3, 1):
        T = T.reshape(3, 1)
    assert(R.shape == (3, 3))
    assert(K.shape == (3, 3))
    assert(X.shape[0] == 3 and X.shape[1] > 0)
    # reshape/transpose to xyz along 1st dimension, and broadcast rotation and translation for each xyz vector
    X_ = np.dot(R.T, (X - T))
    # homogenise xyz/z and apply intrinsic matrix, discarding 3rd dimension
    ij = np.dot(K, X_/X_[2, :])[:2, :]
    return ij

def project_to_z(ij, Z, R, T, K):
    # Reproject from 2D image to 3D world co-ordinates with known Z
    # x,y,z down 1st dimension
    if T.shape != (3, 1):
        T = T.reshape(3, 1)
    assert(R.shape == (3, 3))
    assert(K.shape == (3, 3))
    assert(ij.shape[0] == 2 and ij.shape[1] > 0)
    ij_ = np.row_stack([ij, np.ones((1, ij.shape[1]))])
    X_ = np.dot(np.linalg.inv(K), ij_)
    X_R = np.dot(R, X_) # rotate first (camera to world)
    X = (X_R * (Z - T[2])/X_R[2,:]) + T  # scale to desired Z and offset to world
    return X

# construct extrinsic  matrix
# see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
extd = pd.read_csv(ext_filename, header=None, sep=' ', index_col=0,
                   names=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])
im_filestem = pathlib.Path(im_filename).stem[:-4]
im_ext = extd.loc[im_filestem]
omega, phi, kappa = np.pi * im_ext[['omega', 'phi', 'kappa']] / 180.
# omega, phi, kappa = 0,0,0
# if np.abs(kappa) > np.pi/2:
#     kappa -= np.pi

# PATB convention
omega_r = np.array([[1, 0, 0],
                    [0, np.cos(omega), -np.sin(omega)],
                    [0, np.sin(omega), np.cos(omega)]])

phi_r = np.array([[np.cos(phi), 0, np.sin(phi)],
                  [0, 1, 0],
                  [-np.sin(phi), 0, np.cos(phi)]])

kappa_r = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                   [np.sin(kappa), np.cos(kappa), 0],
                   [0, 0, 1]])

R = np.dot(np.dot(omega_r, phi_r), kappa_r)
T = np.array([im_ext['easting'], im_ext['northing'], im_ext['altitude']])

# construct intrinsic matrix - see https://en.wikipedia.org/wiki/Camera_resectioning
focal_length = 120. # mm
ccd_xysize = np.array([92.160, 165.888])  # mm
im_xydim = [0, 0]   # pixels
with rio.open(im_filename) as raw_im:
    im_transform = raw_im.transform
    im_xydim = np.float64([raw_im.width, raw_im.height])
    # im_xydim_s = -np.sign(np.cos(kappa)) * np.float64([np.sign(im_transform[0]) * raw_im.width, np.sign(im_transform[4]) * raw_im.height])
    im_xydim_s = -np.sign(np.cos(kappa)) * np.float64(
        [np.sign(im_transform[0]) * raw_im.width, np.sign(im_transform[4]) * raw_im.height])
sigma_xy = focal_length * im_xydim_s / ccd_xysize
# TODO: automate the minus below?
# K inferred from https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
K = np.array([[sigma_xy[0], 0, im_xydim[0]/2], [0, sigma_xy[1], im_xydim[1]/2], [0, 0, 1]])


if True:
    X = np.array([[-22544.1, -3736338.7, 200], [-20876.7, -3739374.3, 200], [-19430.2, -3742345.8, 200], [-19430.2, -3742345.8, 200]]).T
    # X = np.array([[-3962.2, -3755180.2, 200], [-2274.3, -3758214.2, 500], [-728.8, -3761226.9, 1400], [-728.8, -3761226.9, 1400]]).T
    # X = X[:, 0].reshape(-1, 1)
    ij = unproject(X, R, T, K)

    X2 = project_to_z(ij, X[2,:], R, T, K)
    # ij = unproject(X, R, T, K)
    print(ij)
    print(X - X2)

# with rio.open(im_filename) as raw_im:
raw_im = rio.open(im_filename)
dem_im = rio.open(dem_filename)
if raw_im.crs != dem_im.crs:
    logger.warning('DEM and image co-ordinate systems are different')

time_rec = dict(dem_min=datetime.timedelta(0), raw_im_read=datetime.timedelta(0), grid_creation=datetime.timedelta(0),
                dem_reproject=datetime.timedelta(0), unproject=datetime.timedelta(0), raw_remap=datetime.timedelta(0),
                write=datetime.timedelta(0))
start_ttl = start = datetime.datetime.now()
with rio.Env():
    with rio.open(im_filename, 'r') as raw_im:
        # find min of dem over raw_im bounds
        dem_min = 0
        start = datetime.datetime.now()
        with rio.open(dem_filename, 'r') as dem_im:
            # option 1: transform raw_im bounds to dem_im bounds to dem im window
            # use reproject or vrt to read directly into raw_im ROI
            [dem_xbounds, dem_ybounds] = transform(raw_im.crs, dem_im.crs, [raw_im.bounds.left, raw_im.bounds.right],
                               [raw_im.bounds.top, raw_im.bounds.bottom])
            dem_win = rio.windows.from_bounds(dem_xbounds[0], dem_ybounds[1], dem_xbounds[1], dem_ybounds[0],
                                              transform=dem_im.transform)
            dem_im_array = dem_im.read(1, window=dem_win)
            dem_min = np.max([dem_im_array.min(), 0])
        time_rec['dem_min'] += (datetime.datetime.now()-start)

        # read the whole raw im (band) into memory for use in remap
        # TODO: loop through bands to save mem
        start = datetime.datetime.now()
        raw_bands = list(range(1, raw_im.count+1))
        raw_im_array = raw_im.read(raw_bands)
        time_rec['raw_im_read'] += (datetime.datetime.now()-start)

        ortho_win_off = np.array([0, 0])  # test
        ortho_profile = raw_im.profile

        # find the bounds of the ortho by projecting image corners onto Z plane = dem_min
        ortho_cnrs = project_to_z(np.array([[0, 0], [raw_im.width, 0], [raw_im.width, raw_im.height], [0, raw_im.height]]).T,
                                  dem_min, R, T, K)[:2, :]
        raw_cnrs = np.array([[raw_im.bounds.left, raw_im.bounds.bottom], [raw_im.bounds.right, raw_im.bounds.bottom],
                             [raw_im.bounds.right, raw_im.bounds.top], [raw_im.bounds.left, raw_im.bounds.top]]).T
        ortho_cnrs = np.column_stack([ortho_cnrs, raw_cnrs])    # probably unnecessary, but ensure we encompass the raw image
        ortho_bl = ortho_cnrs.min(axis=1)   # TODO is min also west & north for any projection?
        ortho_tr = ortho_cnrs.max(axis=1)
        ortho_wh = np.ceil(np.abs((ortho_bl - ortho_tr).squeeze()[:2]/ortho_res))

        ortho_transform = rio.transform.from_origin(ortho_bl[0], ortho_tr[1], ortho_res[0], ortho_res[1])
        ortho_profile.update(nodata=0, compress='deflate', tiled=True, blockxsize=512, blockysize=512, transform=ortho_transform,
                             width=ortho_wh[0], height=ortho_wh[1], num_threads='all_cpus')  #, count=1, dtype='float32')

        with rio.open(ortho_filename, 'w', **ortho_profile) as ortho_im:
            for ji, ortho_win in ortho_im.block_windows(1):
                print((ji, ortho_win))
                # print(win_transform)
                # TODO: move this outside the loop if possible
                start = datetime.datetime.now()
                j_range = np.arange(ortho_win.col_off, ortho_win.col_off + ortho_win.width)
                i_range = np.arange(ortho_win.row_off, ortho_win.row_off + ortho_win.height)
                ortho_jj, ortho_ii = np.meshgrid(j_range, i_range, indexing='xy')
                ortho_xx, ortho_yy = ortho_im.transform * [ortho_jj, ortho_ii]
                time_rec['grid_creation'] += (datetime.datetime.now() - start)

                with rio.open(dem_filename, 'r') as dem_im:
                    start = datetime.datetime.now()
                    dem_win_transform = ortho_im.window_transform(ortho_win)
                    ortho_zz = np.zeros((ortho_win.height, ortho_win.width), dem_im.dtypes[0])
                    # dem_win_transform = calculate_default_transform(ortho_im.crs, dem_im.crs, block_win.width, block_win.height, left=win_transform.xoff, top=win_transform.yoff)

                    # reproject and resample the DEM to ortho CRS and resolution
                    # TODO: check if this is reading the whole DEM every time or just win_transform
                    reproject(rio.band(dem_im, 1), ortho_zz, dst_transform=dem_win_transform, dst_crs=ortho_im.crs,
                              resampling=Resampling.cubic_spline, src_transform=dem_im.transform, src_crs=dem_im.crs)
                    time_rec['dem_reproject'] += (datetime.datetime.now() - start)

                    start = datetime.datetime.now()
                    im_ji = unproject(np.array([ortho_xx.flatten(), ortho_yy.flatten(), ortho_zz.flatten()]), R, T, K)
                    # ortho_im.write(dem_win_reproj, indexes=1, window=block_win)
                    im_jj = np.float32(im_ji[0, :].reshape(ortho_win.height, ortho_win.width))
                    im_ii = np.float32(im_ji[1, :].reshape(ortho_win.height, ortho_win.width))
                    time_rec['unproject'] += (datetime.datetime.now() - start)

                    start = datetime.datetime.now()
                    ortho_im_win_array = np.zeros((raw_im.count, ortho_win.height, ortho_win.width), dtype=raw_im.dtypes[0])
                    for band_i in range(0, raw_im.count):
                        ortho_im_win_array[band_i, :, :] = cv2.remap(raw_im_array[band_i, :, :], im_jj, im_ii,
                                                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    time_rec['raw_remap'] += (datetime.datetime.now() - start)
                    start = datetime.datetime.now()
                    ortho_im.write(ortho_im_win_array, indexes=raw_bands, window=ortho_win)
                    time_rec['write'] += (datetime.datetime.now() - start)
                    start = datetime.datetime.now()

time_rec['write'] += (datetime.datetime.now() - start)
time_rec['ttl'] = (datetime.datetime.now() - start_ttl)
print(time_rec)
timed = pd.DataFrame.from_dict(time_rec, orient='index')
print(timed.sort_values(by=0))
# TODO can we enable multithreading to write out raster / do compression
# memory planning
# we want to keep the whole unrect image in memory, perhaps one band at a time, so we can freely do "lookups"/ remaps
# then it probably makes sense to force a tiled output file, and remap one tile at a time.
# possibly also resampling one dem roi corresponding to a tile at a time
