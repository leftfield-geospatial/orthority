"""Functions to create test data."""

from __future__ import annotations

import csv
import json
import os
from inspect import getsourcefile
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import GCPTransformer, GroundControlPoint
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from orthority import common, param_io
from orthority.enums import Compress
from orthority.exif import Exif

src_data_root = Path('D:/OneDrive/Data/Leftfield/test/orthority')
test_data_root = Path(getsourcefile(lambda: None)).parent


def downsample_image(
    src_file: Path | rio.MemoryFile,
    dst_file: Path | rio.MemoryFile,
    src_indexes: int | list[int] | None = None,
    src_win: Window = None,
    ds_fact: float = 4.0,
    crs: str | rio.CRS = None,
    dtype: str | np.dtype = None,
    compress: str | Compress = None,
    scale: float | None = None,
    copy_tags: bool = False,
    **kwargs,  # destination creation options
):
    """Read and reproject / downsample ``src_file``, and write to ``dst_file``."""
    with (
        rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False),
        rio.open(src_file, 'r') as src_im,
    ):
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
        nodata = None if src_im.nodata is None else common._nodata_vals[dtype]

        # read, crop and reproject source (use WarpedVRT, rather than
        # DatasetReader.read(out_shape=) which uses overviews possibly resampled with a different
        # method and/or on a different grid)
        with WarpedVRT(
            src_im,
            crs=crs,
            transform=transform,
            width=int(np.ceil(src_win.width / ds_fact)),
            height=int(np.ceil(src_win.height / ds_fact)),
            nodata=nodata,
            dtype='float64',
            resampling=Resampling.average,
            num_threads=os.cpu_count(),
        ) as src_im_:
            array = src_im_.read(indexes=src_indexes)

        # create destination profile
        profile, _ = common.create_profile(
            'gtiff', array.shape, dtype, compress=compress, write_mask=False
        )
        if profile['compress'] == 'deflate':
            profile.update(predictor=2, zlevel=9)

        profile.update(
            crs=crs,
            transform=transform if not src_im.transform.is_identity else None,
            blockxsize=256,
            blockysize=256,
            colorinterp=[src_im.colorinterp[ci - 1] for ci in src_indexes],
            nodata=nodata,
        )
        profile.update(**kwargs)

        # scale and clip the image array
        if scale:
            array *= scale
        if np.issubdtype(dtype, np.integer):
            array = array.round()
            info = np.iinfo(dtype)
            array = array.clip(info.min, info.max)
        array = array.astype(dtype, copy=False)

        # write destination file
        with rio.open(dst_file, 'w', **profile) as dst_im:
            if copy_tags:
                # copy metadata
                dst_im.update_tags(**src_im.tags())
                for namespace in src_im.tag_namespaces():
                    # note there is an apparent rio/gdal bug with ':' in the 'xml:XMP' namespace/
                    # tag name, where 'xml:XMP=' gets prefixed to the value
                    ns_dict = src_im.tags(ns=namespace)
                    dst_im.update_tags(ns=namespace, **ns_dict)
                for index in dst_im.indexes:
                    dst_im.update_tags(index, **src_im.tags(index))

            dst_im.write(array)


def create_ngi_test_data():
    ngi_src_root = src_data_root.joinpath('ngi')
    ngi_test_root = test_data_root.joinpath('ngi')
    ds_fact = 12

    # downsample and convert images from RGBN to RGB
    src_rgb_files = [
        '3324c_2015_1004_05_0182_RGBN.tif',
        '3324c_2015_1004_05_0184_RGBN.tif',
        '3324c_2015_1004_06_0251_RGBN.tif',
        '3324c_2015_1004_06_0253_RGBN.tif',
    ]
    ngi_test_root.mkdir(exist_ok=True, parents=True)
    for src_rgb_file in src_rgb_files:
        src_rgb_file = ngi_src_root.joinpath(src_rgb_file)
        dst_rgb_file = ngi_test_root.joinpath(src_rgb_file.stem[:-1]).with_suffix('.tif')
        downsample_image(
            src_rgb_file,
            dst_rgb_file,
            src_indexes=[1, 2, 3],
            ds_fact=ds_fact,
            dtype='uint8',
            scale=255 / 3000,
        )

    # downsample and crop dem
    src_dem_file = ngi_src_root.joinpath('x3324cb_2015_L3a.tif')
    dst_dem_file = ngi_test_root.joinpath('dem.tif')
    bounds = [-60454, -3735692, -52606, -3723500]  # bounds of the orthos with a buffer
    with rio.open(src_dem_file) as dem_im:
        win = dem_im.window(*bounds).round()
    downsample_image(src_dem_file, dst_dem_file, src_win=win, ds_fact=ds_fact, dtype='float32')

    # create exterior parameters for src_rgb_files
    src_ext_file = ngi_src_root.joinpath('ext_param.csv')
    dst_ext_file = ngi_test_root.joinpath('camera_pos_ori.txt')
    if dst_ext_file.exists():
        dst_ext_file.unlink()
    with open(src_ext_file, newline=None) as fin, open(dst_ext_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin, delimiter=' ')
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter=' ')
        for row in reader:
            if row['filename'] + '.tif' in src_rgb_files:
                row['filename'] = row['filename'][:-1]
                writer.writerow(row)


def create_odm_test_data():
    odm_src_root = src_data_root.joinpath('odm')
    odm_test_root = test_data_root.joinpath('odm')

    # downsample rgb images
    src_rgb_files = [
        '100_0005_0018.jpg',
        '100_0005_0136.jpg',
        '100_0005_0140.jpg',
        '100_0005_0142.jpg',
    ]
    odm_test_root.joinpath('images').mkdir(exist_ok=True, parents=True)
    for src_rgb_file in src_rgb_files:
        dst_rgb_file = odm_test_root.joinpath('images', src_rgb_file).with_suffix('.tif')
        src_rgb_file = odm_src_root.joinpath('images', src_rgb_file)
        downsample_image(src_rgb_file, dst_rgb_file, ds_fact=4, dtype='uint8', copy_tags=True)

    # crop and downsample dem
    odm_test_root.joinpath('odm_dem').mkdir(exist_ok=True, parents=True)
    src_dem_file = odm_src_root.joinpath('odm_dem/dsm.tif')
    dst_dem_file = odm_test_root.joinpath('odm_dem/dsm.tif')
    bounds = [292540.292, 2730869.049, 292930.692, 2731225.049]  # bounds of src_rgb_files orthos
    with rio.open(src_dem_file, 'r') as dem_im:
        win = dem_im.window(*bounds).round()
    downsample_image(src_dem_file, dst_dem_file, src_win=win, ds_fact=16, dtype='float32')

    # copy relevant parts of opensfm reconstruction file with image size conversion
    odm_test_root.joinpath('opensfm').mkdir(exist_ok=True, parents=True)
    src_rec_file = odm_src_root.joinpath('opensfm/reconstruction.json')
    dst_rec_file = odm_test_root.joinpath('opensfm/reconstruction.json')
    if dst_rec_file.exists():
        dst_rec_file.unlink()

    with open(src_rec_file) as f:
        src_rec = json.load(f)
    dst_rec = [{k: v for k, v in src_rec[0].items() if k in ['cameras', 'shots', 'reference_lla']}]
    dst_rec[0]['shots'] = {
        k[:-4]: v for k, v in dst_rec[0]['shots'].items() if k.lower() in src_rgb_files
    }

    with rio.open(dst_rgb_file, 'r') as dst_im:
        im_size = (dst_im.width, dst_im.height)
    for camera in dst_rec[0]['cameras'].values():
        camera['width'], camera['height'] = im_size[0], im_size[1]

    with open(dst_rec_file, 'w') as f:
        json.dump(dst_rec, f, indent=4)


def create_rpc_test_data():
    rpc_src_root = src_data_root.joinpath('rpc')
    rpc_test_root = test_data_root.joinpath('rpc')
    rpc_test_root.mkdir(exist_ok=True)

    # read ngi dem bounds & crs
    with rio.open(test_data_root.joinpath('ngi/dem.tif'), 'r') as dem_im:
        dem_bounds = dem_im.bounds
        dem_crs = dem_im.crs

    # find window corresponding to dem bounds (with an inner buffer)
    ds_fact = 10.0
    src_image_file = rpc_src_root.joinpath('03NOV18082012-P1BS-056844553010_01_P001.TIF')
    with rio.open(src_image_file, 'r') as src_im:
        rpcs = src_im.rpcs
        bounds = np.array(transform_bounds(dem_crs, src_im.gcps[1], *dem_bounds))
        buf_bounds = (*(bounds[:2] + 0.012), *(bounds[2:] - 0.012))
        with GCPTransformer(src_im.gcps[0]) as tform:
            ul = np.round(tform.rowcol(buf_bounds[0], buf_bounds[3])[::-1], -2)
            br = np.round(tform.rowcol(buf_bounds[2], buf_bounds[1])[::-1], -2)
        win = Window(*ul, *(br - ul))
        win = win.intersection(Window(0, 0, src_im.width, src_im.height))

    # read inlier field GCPs (centre pixel coord convention)
    src_gcp_file = rpc_src_root.joinpath('inlier_gcps.geojson')
    gcps = param_io.read_oty_gcps(src_gcp_file)
    gcps = next(iter(gcps.values()))

    # adjust GCPs for crop and downsample
    for gcp in gcps:
        # +0.5 converts centre to UL pixel coords so that they can be scaled.  then -0.5
        # converts back from UL to centre pixel coords as expected by param_io.write_gcps()
        gcp['ji'] = (gcp['ji'] - np.array((win.col_off, win.row_off)) + 0.5) / ds_fact - 0.5

    # convert GCPs to rasterio format for storing in image metadata (leave in center pixel
    # coordinate convention)
    rio_gcps = []
    for gcp in gcps:
        rio_gcps.append(GroundControlPoint(*gcp['ji'][::-1], *gcp['xyz'], gcp['id'], gcp['info']))

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

    # crop and downsample image, and write with RPCs and GCPs
    dst_image_file = rpc_test_root.joinpath('qb2_basic1b.tif')
    downsample_image(
        src_image_file,
        dst_image_file,
        src_win=win,
        ds_fact=ds_fact,
        crs='EPSG:4979',
        dtype='uint8',
        compress='jpeg',
        scale=255 / 700,
        rpcs=rpcs,
        gcps=rio_gcps,
    )

    # create oty format GCP file
    param_io.write_gcps(
        rpc_test_root.joinpath('gcps.geojson'), {dst_image_file.name: gcps}, overwrite=True
    )

    # create oty format RPC param file
    rpc_param_dict = param_io.read_im_rpc_param([dst_image_file])
    param_io.write_rpc_param(
        rpc_test_root.joinpath('rpc_param.yaml'), rpc_param_dict, overwrite=True
    )


def create_pan_sharp_data():
    pan_sharp_test_root = test_data_root.joinpath('pan_sharp')
    pan_sharp_test_root.mkdir(exist_ok=True)

    # dowsample the source image to temporary pan res RGB
    src_image_file = src_data_root.joinpath('odm/images/100_0005_0140.jpg')
    mem_file = rio.MemoryFile()
    downsample_image(src_image_file, mem_file, ds_fact=4, compress='deflate')

    # convert pan res RGB to pan
    dst_pan_file = pan_sharp_test_root.joinpath('pan.tif')
    with mem_file, rio.open(mem_file, 'r') as mem_im:
        profile = mem_im.profile
        profile.update(count=1, photometric=None, interleave='pixel', compress='jpeg')
        temp_array = mem_im.read()
        pan_array = temp_array.mean(axis=0).round().astype('uint8')
        with rio.open(dst_pan_file, 'w', **profile) as pan_im:
            pan_im.write(pan_array, indexes=1)

    # downsample the source image to ms res (deflate compression gives more accurate ms to pan
    # weights)
    dst_ms_file = pan_sharp_test_root.joinpath('ms.tif')
    downsample_image(src_image_file, dst_ms_file, ds_fact=16, compress='deflate')


def create_io_test_data():
    io_test_root = test_data_root.joinpath('io')
    io_test_root.mkdir(exist_ok=True)

    # create lla_rpy csv file for odm data
    osfm_reader = param_io.OsfmReader(test_data_root.joinpath('odm/opensfm/reconstruction.json'))
    cam_id = next(iter(osfm_reader.read_int_param().keys()))
    exif_list = [Exif(sf) for sf in test_data_root.joinpath('odm/images').glob('*.tif')]
    with open(io_test_root.joinpath('odm_lla_rpy.csv'), 'w', newline='') as f:
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
    with open(io_test_root.joinpath('odm_xyz_opk.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ', quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa', 'camera'])
        ext_param_dict = osfm_reader.read_ext_param()
        for filename, ext_param in ext_param_dict.items():
            xyz = np.round(ext_param['xyz'], 3)
            opk = np.round(np.degrees(ext_param['opk']), 3)
            writer.writerow([filename, *xyz, *opk, cam_id])

    # create xyz_opk csv file for ngi data
    reader = param_io.CsvReader(test_data_root.joinpath('ngi/camera_pos_ori.txt'))
    ext_param_dict = reader.read_ext_param()
    with open(io_test_root.joinpath('ngi_xyz_opk.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(['filename', 'x', 'y', 'z', 'omega', 'phi', 'kappa'])
        for filename, ext_param in ext_param_dict.items():
            xyz = np.round(ext_param['xyz'], 3)
            opk = np.round(np.degrees(ext_param['opk']), 3)
            writer.writerow([filename, *xyz, *opk])

    # create proj file for above
    src_im_file = next(iter(test_data_root.joinpath('ngi').glob('*RGB.tif')))
    with rio.open(src_im_file, 'r') as src_im:
        crs = src_im.crs
    with open(io_test_root.joinpath('ngi_xyz_opk.prj'), 'w', newline='') as f:
        # use WKT to avoid GDAL issues with proj4 Lo* WGS84 representation
        f.write(crs.to_wkt())

    # create oty format interior and exterior param files for ngi data
    int_param_dict = param_io.read_oty_int_param(test_data_root.joinpath('ngi/config.yaml'))
    param_io.write_int_param(
        io_test_root.joinpath('ngi_int_param.yaml'), int_param_dict, overwrite=True
    )
    cam_id = next(iter(int_param_dict.keys()))
    for ext_params in ext_param_dict.values():
        ext_params.update(camera=cam_id)
    with rio.open(test_data_root.joinpath('ngi/3324c_2015_1004_05_0182_RGB.tif'), 'r') as im:
        ngi_crs = im.crs
    param_io.write_ext_param(
        io_test_root.joinpath('ngi_ext_param.geojson'), ext_param_dict, crs=ngi_crs, overwrite=True
    )

    # create oty format rpc param file for rpc image
    rpc_param_dict = param_io.read_im_rpc_param((test_data_root.joinpath('rpc/qb2_basic1b.tif'),))
    param_io.write_rpc_param(
        test_data_root.joinpath('rpc/rpc_param.yaml'), rpc_param_dict, overwrite=True
    )


if __name__ == '__main__':
    create_ngi_test_data()
    create_odm_test_data()
    create_rpc_test_data()
    create_pan_sharp_data()
    create_io_test_data()
