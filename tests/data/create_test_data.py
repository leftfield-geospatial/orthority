"""Functions to create NGI & ODM test data sets."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import GCPTransformer, GroundControlPoint
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from orthority import common
from orthority import param_io
from orthority.enums import Compress
from orthority.exif import Exif
from orthority.factory import RpcCameras

ngi_src_root = Path('V:/Data/SimpleOrthoEgs/NGI_3324C_2015_Baviaans/')
ngi_test_root = Path('C:/Data/Development/Projects/orthority/tests/data/ngi')

odm_src_root = Path('V:/Data/SimpleOrthoEgs/20190411_Miaoli_Toufeng_Tuniu-River')
odm_test_root = Path('C:/Data/Development/Projects/orthority/tests/data/odm')

rpc_src_root = Path('V:/Data/SimpleOrthoEgs/QB2_Nov_2003_MpSite')
rpc_test_root = Path('C:/Data/Development/Projects/orthority/tests/data/rpc')

pan_sharp_test_root = Path('C:/Data/Development/Projects/orthority/tests/data/pan_sharp')

io_root = Path('C:/Data/Development/Projects/orthority/tests/data/io')


def downsample_image(
    src_file: Path,
    dst_file: Path,
    src_indexes: int | list[int] = None,
    src_win: Window = None,
    ds_fact: float = 4.0,
    crs: str | rio.CRS = None,
    dtype: str | np.dtype = None,
    compress: str | Compress = None,
    scale: float = None,
    copy_tags: bool = False,
    **kwargs,  # destination creation options
):
    """Read and reproject / downsample ``src_file``, and write to ``dst_file``."""
    dst_file = Path(dst_file)
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False), rio.open(
        src_file, 'r'
    ) as src_im:
        # set up WarpedVRT params
        src_indexes = src_indexes or src_im.indexes
        src_win = src_win or Window(0, 0, src_im.width, src_im.height)
        crs = crs or src_im.crs
        dtype = dtype or src_im.dtypes[0]
        transform = (
            src_im.transform
            * rio.Affine.translation(src_win.col_off, src_win.row_off)
            * rio.Affine.scale(ds_fact)
        )

        # create initial destination profile (to get nodata value)
        colorinterp = [src_im.colorinterp[ci - 1] for ci in src_indexes]
        profile, _ = common.create_profile(
            dtype, compress=compress, write_mask=False, colorinterp=colorinterp
        )
        if src_im.nodata is None:
            profile.update(nodata=None)
        if profile['compress'] == 'deflate':
            profile.update(predictor=2, zlevel=9)

        # read, crop and reproject source (use WarpedVRT, rather than
        # DatasetReader.read(out_shape=) which uses overviews possibly resampled with a different
        # method and/or on a different grid)
        with WarpedVRT(
            src_im,
            crs=crs,
            transform=transform,
            width=int(np.ceil(src_win.width / ds_fact)),
            height=int(np.ceil(src_win.height / ds_fact)),
            nodata=profile['nodata'],
            dtype='float64',
            resampling=Resampling.average,
            num_threads=os.cpu_count(),
        ) as src_im_:
            array = src_im_.read(indexes=src_indexes)

        # update profile with spatial / dimensional items
        profile.update(
            crs=crs,
            transform=transform if not src_im.transform.is_identity else None,
            count=array.shape[0],
            width=array.shape[2],
            height=array.shape[1],
            blockxsize=256,  # use original tile config
            blockysize=256,
        )

        # scale and clip the image array
        if scale:
            array *= scale
        if np.issubdtype(dtype, np.integer):
            array = array.round()
            info = np.iinfo(dtype)
            array = array.clip(info.min, info.max)
        array = array.astype(dtype, copy=False)

        # write destination file
        dst_file.unlink(missing_ok=True)
        dst_file.with_suffix(dst_file.suffix + '.aux.xml').unlink(missing_ok=True)
        with common.OpenRaster(dst_file, 'w', **profile, **kwargs) as dst_im:
            if copy_tags:
                # copy metadata
                dst_im.update_tags(**src_im.tags())
                for namespace in src_im.tag_namespaces():
                    # note there is an apparent rio/gdal bug with ':' in the 'xml:XMP' namespace/ tag
                    # name, where 'xml:XMP=' gets prefixed to the value
                    ns_dict = src_im.tags(ns=namespace)
                    dst_im.update_tags(ns=namespace, **ns_dict)
                for index in dst_im.indexes:
                    dst_im.update_tags(index, **src_im.tags(index))

            dst_im.write(array)


def create_ngi_test_data():
    src_rgb_files = [
        '3324c_2015_1004_05_0182_RGBN_CMP.tif',
        '3324c_2015_1004_05_0184_RGBN_CMP.tif',
        '3324c_2015_1004_06_0251_RGBN_CMP.tif',
        '3324c_2015_1004_06_0253_RGBN_CMP.tif',
    ]
    dem_file = 'sudem_l3a_clip.tif'
    ds_fact = 12

    # downsample rgb images
    ngi_test_root.mkdir(exist_ok=True, parents=True)
    for src_rgb_file in src_rgb_files:
        src_rgb_file = ngi_src_root.joinpath('Source', src_rgb_file)
        dst_rgb_file = ngi_test_root.joinpath(src_rgb_file.name[:-9]).with_suffix('.tif')
        downsample_image(
            src_rgb_file,
            dst_rgb_file,
            src_indexes=[1, 2, 3],
            ds_fact=ds_fact,
            dtype='uint8',
            scale=255 / 3000,
        )

    # downsample dem
    with rio.open(ngi_src_root.joinpath('Source', src_rgb_files[0]), 'r') as src_im:
        # DEM heights (and NGI external params) are EGM2008 but lack a vertical CRS. This adds a
        # vertical component to the 2D CRS, which is used for converting to ellipsoidal heights
        # when this DEM is used for RPC orthorectification.
        dst_wkt = (
            f'COMPD_CS["Lo25 WGS84 + EGM2008 height", {src_im.crs.to_wkt()}, VERT_CS['
            f'"EGM2008 height", VERT_DATUM["EGM2008 geoid",2005], UNIT["metre",1]]]'
        )
        dst_crs = rio.CRS.from_wkt(dst_wkt)

    src_dem_file = ngi_src_root.joinpath('DEM', dem_file)
    dst_dem_file = ngi_test_root.joinpath('dem.tif')
    downsample_image(src_dem_file, dst_dem_file, ds_fact=ds_fact, crs=dst_crs, dtype='float32')

    # copy & convert csv exterior params
    src_ext_file = ngi_src_root.joinpath('Parameters/camera_pos_ori.txt')
    dst_ext_file = ngi_test_root.joinpath('camera_pos_ori.txt')
    if dst_ext_file.exists():
        dst_ext_file.unlink()
    with open(src_ext_file, 'r', newline=None) as fin, open(dst_ext_file, 'w', newline='') as fout:
        reader = csv.DictReader(
            fin,
            fieldnames=['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa'],
            delimiter=' ',
        )
        writer = csv.DictWriter(
            fout,
            fieldnames=['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa'],
            delimiter=' ',
        )
        for row in reader:
            if row['filename'] + '.tif' in src_rgb_files:
                row['filename'] = row['filename'][:-5]
                writer.writerow(row)


def create_odm_test_data():
    src_rgb_files = [
        '100_0005_0018.jpg',
        '100_0005_0136.jpg',
        '100_0005_0140.jpg',
        '100_0005_0142.jpg',
    ]
    ortho_files = [
        '100_0005_0018_ORTHO.tif',
        '100_0005_0136_ORTHO.tif',
        '100_0005_0140_ORTHO.tif',
        '100_0005_0142_ORTHO.tif',
    ]

    # downsample rgb images
    odm_test_root.joinpath('images').mkdir(exist_ok=True, parents=True)
    for src_rgb_file in src_rgb_files:
        dst_rgb_file = odm_test_root.joinpath('images', src_rgb_file).with_suffix('.tif')
        src_rgb_file = odm_src_root.joinpath('images', src_rgb_file)
        downsample_image(src_rgb_file, dst_rgb_file, ds_fact=4, dtype='uint8', copy_tags=True)

    # get bounds covering orthos of src files
    def bounds_union(bounds1, bounds2):
        return *np.min((bounds1[:2], bounds2[:2]), axis=0), *np.max(
            (bounds1[-2:], bounds2[-2:]), axis=0
        )

    bounds = None
    for ortho_file in ortho_files:
        with rio.open(odm_src_root.joinpath('orthority', ortho_file), 'r') as ortho_im:
            bounds = ortho_im.bounds if not bounds else bounds_union(bounds, ortho_im.bounds)

    # crop and downsample dem
    odm_test_root.joinpath('odm_dem').mkdir(exist_ok=True, parents=True)
    dst_dem_file = odm_test_root.joinpath('odm_dem', 'dsm.tif')
    src_dem_file = odm_src_root.joinpath('odm_dem', 'dsm.tif')
    with rio.open(src_dem_file, 'r') as dem_im:
        win = dem_im.window(*bounds)
    win = common.expand_window_to_grid(win)
    downsample_image(src_dem_file, dst_dem_file, src_win=win, ds_fact=16, dtype='float32')

    # copy relevant parts of opensfm reconstruction file with image size conversion
    src_rec_file = odm_src_root.joinpath('opensfm/reconstruction.json')
    dst_rec_file = odm_test_root.joinpath('opensfm/reconstruction.json')
    odm_test_root.joinpath('opensfm').mkdir(exist_ok=True, parents=True)
    with rio.open(dst_rgb_file, 'r') as dst_im:
        im_size = (dst_im.width, dst_im.height)
    if dst_rec_file.exists():
        dst_rec_file.unlink()
    with open(src_rec_file, 'r') as f:
        json_obj = json.load(f)
    json_obj = [
        {k: v for k, v in json_obj[0].items() if k in ['cameras', 'shots', 'reference_lla']}
    ]
    json_obj[0]['shots'] = {
        k[:-4]: v for k, v in json_obj[0]['shots'].items() if k.lower() in src_rgb_files
    }
    for camera in json_obj[0]['cameras'].values():
        camera['width'], camera['height'] = im_size[0], im_size[1]
    with open(dst_rec_file, 'w') as f:
        json.dump(json_obj, f, indent=4)


def create_rpc_test_data():
    src_file = '03NOV18082012-P1BS-056844553010_01_P001.TIF'
    gcp_file = 'pan_gcps.geojson'
    rpc_test_root.mkdir(exist_ok=True)
    dem_file = ngi_test_root.joinpath('dem.tif')
    test_file = 'qb2_basic1b.tif'

    # read ngi dem bounds & crs
    with rio.open(dem_file, 'r') as dem_im:
        dem_bounds = dem_im.bounds
        dem_crs = dem_im.crs

    # find window corresponding to dem bounds (with an inner buffer)
    ds_fact = 10.0
    rpc_src_file = rpc_src_root.joinpath('Source', src_file)
    with rio.open(rpc_src_file, 'r') as src_im:
        rpcs = src_im.rpcs
        bounds = np.array(transform_bounds(dem_crs, src_im.gcps[1], *dem_bounds))
        buf_bounds = (*(bounds[:2] + 0.012), *(bounds[2:] - 0.012))
        with GCPTransformer(src_im.gcps[0]) as tform:
            ul = np.round(tform.rowcol(buf_bounds[0], buf_bounds[3])[::-1], -2)
            br = np.round(tform.rowcol(buf_bounds[2], buf_bounds[1])[::-1], -2)
        win = Window(*ul, *(br - ul))
        win = win.intersection(Window(0, 0, src_im.width, src_im.height))
        win = common.expand_window_to_grid(win)

    # read field GCPs (center pixel coord convention)
    gcps = param_io.read_oty_gcps(rpc_src_root.joinpath('GCP', gcp_file))
    gcps = next(iter(gcps.values()))

    # choose GCP inliers (to improve refinement and prevent weird QGIS renderings)
    cameras = RpcCameras.from_images((rpc_src_file,))
    camera = cameras.get(rpc_src_file)
    xyz = np.array([gcp_dict['xyz'] for gcp_dict in gcps]).T
    ji_rpc = camera.world_to_pixel(xyz)
    ji_gcp = np.array([gcp_dict['ji'] for gcp_dict in gcps]).T
    off = ji_gcp - ji_rpc
    off_med = np.median(off, axis=1)
    off_dist = np.sum((off - off_med.reshape(-1, 1)) ** 2, axis=0)
    inlier_idx = np.argsort(off_dist)[:5]
    gcps = [gcps[i] for i in inlier_idx]

    # adjust GCPs for crop and downsample
    for gcp in gcps:
        # +0.5 converts center to UL pixel coords so that they can be scaled.  then -0.5
        # converts back from UL to center pixel coords as expected by param_io.write_gcps()
        gcp['ji'] = (gcp['ji'] - np.array((win.col_off, win.row_off)) + 0.5) / ds_fact - 0.5

    # convert GCPs to rasterio format for storing in image metadata
    rio_gcps = []
    for gcp in gcps:
        # +0.5 converts from center to UL pixel convention as used by oty/QGIS/QB2 for image
        # metadata GCPs
        ij = (gcp['ji'][1] + 0.5, gcp['ji'][0] + 0.5)
        rio_gcps.append(GroundControlPoint(*ij, *gcp['xyz'], gcp['id'], gcp['info']))

    # adjust existing metadata GCPs for crop and downsample
    # rio_gcps = src_im.gcps[0]
    # for gcp in rio_gcps:
    #     gcp.col = (gcp.col - win.col_off) / ds_fact
    #     gcp.row = (gcp.row - win.row_off) / ds_fact

    # adjust RPCs for crop and downsample (see GCP comments for +-0.5 notes)
    rpcs.line_off = (rpcs.line_off - win.row_off + 0.5) / ds_fact - 0.5
    rpcs.samp_off = (rpcs.samp_off - win.col_off + 0.5) / ds_fact - 0.5
    rpcs.line_scale /= ds_fact
    rpcs.samp_scale /= ds_fact

    # crop and downsample image, and write to test_file with RPC and GPC metadata
    rpc_test_file = rpc_test_root.joinpath(test_file)
    downsample_image(
        rpc_src_file,
        rpc_test_file,
        src_win=win,
        ds_fact=ds_fact,
        crs='EPSG:4979',
        dtype='uint8',
        compress='jpeg',
        scale=255 / 700,
        rpcs=rpcs,
        gcps=rio_gcps,
    )

    # create oty format GCP file for test image
    param_io.write_gcps(
        rpc_test_root.joinpath('gcps.geojson'), {rpc_test_file.name: gcps}, overwrite=True
    )

    # create oty format rpc param file for test image
    rpc_param_dict = param_io.read_im_rpc_param([rpc_test_file])
    param_io.write_rpc_param(
        rpc_test_root.joinpath('rpc_param.yaml'), rpc_param_dict, overwrite=True
    )


def create_pan_sharp_data():
    pan_sharp_test_root.mkdir(exist_ok=True)
    src_file = odm_src_root.joinpath('images/100_0005_0140.jpg')

    # dowsample the source image to temporary pan res RGB
    temp_file = pan_sharp_test_root.joinpath('temp.tif')
    downsample_image(src_file, temp_file, ds_fact=4, compress='deflate')

    # convert pan res RGB to pan
    pan_test_file = pan_sharp_test_root.joinpath('pan.tif')
    with rio.open(temp_file, 'r') as temp_im:
        profile = temp_im.profile
        profile.update(count=1, photometric=None, interleave='pixel', compress='jpeg')
        temp_array = temp_im.read()
        pan_array = temp_array.mean(axis=0).round().astype('uint8')
        with rio.open(pan_test_file, 'w', **profile) as pan_im:
            pan_im.write(pan_array, indexes=1)

    # downsample the source image to ms res (deflate compression gives more accurate ms to pan
    # weights)
    ms_test_file = pan_sharp_test_root.joinpath('ms.tif')
    downsample_image(src_file, ms_test_file, ds_fact=16, compress='deflate')


def create_io_test_data():
    # create lla_rpy csv file for odm data
    io_root.mkdir(exist_ok=True)

    osfm_reader = param_io.OsfmReader(odm_test_root.joinpath('opensfm', 'reconstruction.json'))
    cam_id = next(iter(osfm_reader.read_int_param().keys()))
    exif_list = [Exif(sf) for sf in odm_test_root.joinpath('images').glob('*.tif')]
    with open(io_root.joinpath('odm_lla_rpy.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ', quotechar='"')
        writer.writerow(
            [
                'filename',
                'latitude',
                'longitude',
                'altitude',
                'roll',
                'pitch',
                'yaw',
                'camera',
                'other',
            ]
        )
        for exif in exif_list:
            writer.writerow([Path(exif.filename).name, *exif.lla, *exif.rpy, cam_id, 'ignored'])

    # create xyz_opk csv file for odm data
    with open(io_root.joinpath('odm_xyz_opk.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ', quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa', 'camera'])
        ext_param_dict = osfm_reader.read_ext_param()
        for filename, ext_param in ext_param_dict.items():
            xyz = np.round(ext_param['xyz'], 3)
            opk = np.round(np.degrees(ext_param['opk']), 3)
            writer.writerow([filename, *xyz, *opk, cam_id])

    # create xyz_opk csv file for ngi data
    src_csv_file = ngi_test_root.joinpath('camera_pos_ori.txt')
    reader = param_io.CsvReader(src_csv_file)
    ext_param_dict = reader.read_ext_param()
    with open(io_root.joinpath('ngi_xyz_opk.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa'])
        for filename, ext_param in ext_param_dict.items():
            xyz = np.round(ext_param['xyz'], 3)
            opk = np.round(np.degrees(ext_param['opk']), 3)
            writer.writerow([filename, *xyz, *opk])

    # create proj file for above
    src_im_file = next(iter(ngi_test_root.glob('*RGB.tif')))
    with rio.open(src_im_file, 'r') as src_im:
        crs = src_im.crs

    with open(io_root.joinpath('ngi_xyz_opk.prj'), 'w', newline='') as f:
        f.write(crs.to_proj4())

    # create oty format interior and exterior param files for ngi data
    int_param_dict = param_io.read_oty_int_param(ngi_test_root.joinpath('config.yaml'))
    param_io.write_int_param(io_root.joinpath('ngi_int_param.yaml'), int_param_dict, overwrite=True)
    cam_id = next(iter(int_param_dict.keys()))
    for ext_params in ext_param_dict.values():
        ext_params.update(camera=cam_id)
    ngi_image_file = ngi_test_root.joinpath(next(iter(ext_param_dict.keys()))).with_suffix('.tif')
    with rio.open(ngi_image_file, 'r') as im:
        ngi_crs = im.crs
    param_io.write_ext_param(
        io_root.joinpath('ngi_ext_param.geojson'), ext_param_dict, crs=ngi_crs, overwrite=True
    )

    # create oty format rpc param file for rpc image
    rpc_param_dict = param_io.read_im_rpc_param((rpc_test_root.joinpath('qb2_basic1b.tif'),))
    param_io.write_rpc_param(
        rpc_test_root.joinpath('rpc_param.yaml'), rpc_param_dict, overwrite=True
    )


if __name__ == '__main__':
    create_odm_test_data()
    create_ngi_test_data()
    create_rpc_test_data()
    create_pan_sharp_data()
    create_io_test_data()
