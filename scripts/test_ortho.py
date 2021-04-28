import cv2
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import pathlib
from simple_ortho import get_logger
# See https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined



logger = get_logger(__name__)
# 3318D_2016_1143_11_0450_RGB.tif
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_07_0298_RGB_PRE.tif")
# im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_08_0321_RGB_PRE.tif")
im_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3318D_2016_1143\3318D_2016_1143_11_0451_RGB_PRE.tif")

dem_filename = pathlib.Path(r"D:\Data\Development\Projects\PhD GeoInformatics\Data\CGA\SUDEM\SUDEM_3318B_D_5m.tif")
ext_filename = pathlib.Path(r"C:\Data\Development\Projects\PhD GeoInformatics\Docs\PCI\NGI Orthorectification\extori3318D_2016_1143_lo19wgs84n_e_rect.txt")
ortho_filename = im_filename.parent.joinpath(im_filename.stem + '_ORTHO_B.TIF')
ortho_res = [0.5, 0.5]
block_size = [512, 512]
conv = 'patb'

# construct intrinsic matrix - see https://en.wikipedia.org/wiki/Camera_resectioning
focal_length = 120. # mm
ccd_xysize = np.array([92.160, 165.888])  # mm
im_xydim = [0, 0]   # pixels
with rio.open(im_filename) as raw_im:
    im_transform = raw_im.transform
    im_xydim = np.float64([raw_im.width, raw_im.height])
    im_xydim_s = np.float64([np.sign(im_transform[0]) * raw_im.width, np.sign(im_transform[4]) * raw_im.height])
sigma_xy = focal_length * im_xydim_s / ccd_xysize
# TODO: automate the minus below?
K = np.array([[sigma_xy[0], 0, im_xydim[0]/2], [0, sigma_xy[1], im_xydim[1]/2], [0, 0, 1]]) # from https://en.wikipedia.org/wiki/Camera_resectioning

def unproject(X, R, T, K):
    assert(R.shape == (3,3))
    assert(T.shape == (3,1))
    assert(K.shape == (3,3))
    X_ = np.dot(R.T, (X-T))
    ij = (np.array([-K[0, 0] * X_[0, :]/X_[2, :], -K[1, 1] * X_[1, :]/X_[2, :]])) + K[0:2, -1].reshape(-1,1)
    return ij

# construct extrinsic  matrix
# see https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
extd = pd.read_csv(ext_filename, header=None, sep=' ', index_col=0,
                   names=['file', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])
im_filestem = pathlib.Path(im_filename).stem[:-4]
im_ext = extd.loc[im_filestem]
omega, phi, kappa = np.pi * im_ext[['omega', 'phi', 'kappa']] / 180.
# omega, phi, kappa = 0,0,0
# kappa += np.pi

if str.lower(conv) == 'patb':   # pix4d
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
    # R = np.dot(kappa_r, phi_r, omega_r)
    # R = np.dot(omega_r, kappa_r, phi_r)
else:   #BLUH
    omega_r = np.array([[1, 0, 0],
                        [0, np.cos(omega), np.sin(omega)],
                        [0, -np.sin(omega), np.cos(omega)]])

    phi_r = np.array([[np.cos(phi), 0, -np.sin(phi)],
                      [0, 1, 0],
                      [np.sin(phi), 0, np.cos(phi)]])

    kappa_r = np.array([[np.cos(kappa), np.sin(kappa), 0],
                        [-np.sin(kappa), np.cos(kappa), 0],
                        [0, 0, 1]])

    # R = np.dot(kappa_r, omega_r, phi_r)
    R = np.dot(np.dot(phi_r, omega_r), kappa_r)

T = np.array([im_ext['easting'], im_ext['northing'], im_ext['altitude']]).reshape(-1, 1)

if True:
    res = cv2.projectPoints(np.array([[-22544.1, -3736338.7, 200], [-20876.7,-3739374.3, 200], [-19430.2,-3742345.8, 200]]),
                             R.T, -T, K, None)  # 298

    # res = cv2.projectPoints(np.array([[-3318.4, -3744541.5, 200], [-2057.6,-3747647.2, 200], [-4971.0,-3741514.1, 200]]),
    #                          R.T, -T, K, None)  # 321
    #
    # res = cv2.projectPoints(np.array([[-2133.9, -3758225.1, 1400], [-727.6, -3761237.7, 400], [-3847.2,-3755255.4, 300]]),
    #     R.T, -T, K, None)
    X = np.array([[-3962.2, -3755180.2, 200], [-2274.3,-3758214.2, 500], [-728.8,-3761226.9, 1400]]).T
    print(res[0])

# with rio.open(im_filename) as raw_im:
raw_im = rio.open(im_filename)
dem_im = rio.open(dem_filename)
if raw_im.crs != dem_im.crs:
    logger.warning('DEM and image co-ordinate systems are different')

with rio.Env():
    with rio.open(im_filename, 'r') as raw_im:
        # TODO: loop through bands to save mem
        raw_bands = list(range(1, raw_im.count+1))
        raw_im_array = raw_im.read(raw_bands)
        ortho_win_off = np.array([0, 0])  # test
        ortho_profile = raw_im.profile
        # TODO: change/round resolution here
        # TODO: change size to fit spatial extents here, raw res may not be enough, or may be too much
        ortho_profile.update(nodata=0, compress='deflate', tiled=True, blockxsize=512, blockysize=512)  #, count=1, dtype='float32')
        dem_max = 0
        with rio.open(ortho_filename, 'w', **ortho_profile) as ortho_im:
            for ji, ortho_win in ortho_im.block_windows(1):
                print((ji, ortho_win))
                # print(win_transform)
                # TODO: move this outside the loop if possible
                j_range = np.arange(ortho_win.col_off, ortho_win.col_off + ortho_win.width)
                i_range = np.arange(ortho_win.row_off, ortho_win.row_off + ortho_win.height)
                ortho_jj, ortho_ii = np.meshgrid(j_range, i_range, indexing='xy')
                ortho_xx, ortho_yy = ortho_im.transform * [ortho_jj, ortho_ii]

                with rio.open(dem_filename, 'r') as dem_im:
                    dem_win_transform = ortho_im.window_transform(ortho_win)
                    ortho_zz = np.zeros((ortho_win.height, ortho_win.width), dem_im.dtypes[0])
                    # dem_win_transform = calculate_default_transform(ortho_im.crs, dem_im.crs, block_win.width, block_win.height, left=win_transform.xoff, top=win_transform.yoff)

                    # reproject and resample the DEM to ortho CRS and resolution
                    # TO DO: check if this is reading the whole DEM every time or just win_transform
                    reproject(rio.band(dem_im, 1), ortho_zz, dst_transform=dem_win_transform, dst_crs=ortho_im.crs,
                              resampling=Resampling.cubic_spline, src_transform=dem_im.transform, src_crs=dem_im.crs)

                    if False:
                        im_ji, _ = cv2.projectPoints(np.array([ortho_xx.reshape(-1, 1), ortho_yy.reshape(-1, 1), ortho_zz.reshape(-1, 1)]),
                                                R.T, -T, K, None)
                        # ortho_im.write(dem_win_reproj, indexes=1, window=block_win)
                        im_jj = np.float32(im_ji[:, 0, 0].reshape(ortho_win.height, ortho_win.width))
                        im_ii = np.float32(im_ji[:, 0, 1].reshape(ortho_win.height, ortho_win.width))
                    else:
                        im_ji = unproject(np.array([ortho_xx.flatten(), ortho_yy.flatten(), ortho_zz.flatten()]), R, T, K)
                        # ortho_im.write(dem_win_reproj, indexes=1, window=block_win)
                        im_jj = np.float32(im_ji[0, :].reshape(ortho_win.height, ortho_win.width))
                        im_ii = np.float32(im_ji[1, :].reshape(ortho_win.height, ortho_win.width))

                    ortho_im_win_array = np.zeros((raw_im.count, ortho_win.height, ortho_win.width), dtype=raw_im.dtypes[0])
                    for band_i in range(0, raw_im.count):
                        ortho_im_win_array[band_i, :, :] = cv2.remap(raw_im_array[band_i, :, :], im_jj, im_ii,
                                                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    ortho_im.write(ortho_im_win_array, indexes=raw_bands, window=ortho_win)
# memory planning
# we want to keep the whole unrect image in memory, perhaps one band at a time, so we can freely do "lookups"/ remaps
# then it probably makes sense to force a tiled output file, and remap one tile at a time.
# possibly also resampling one dem roi corresponding to a tile at a time
