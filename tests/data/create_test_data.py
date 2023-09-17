""" Functions to create NGI & ODM test data sets. """
from pathlib import Path
from typing import Tuple
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from simple_ortho.utils import expand_window_to_grid
from simple_ortho.exif import Exif
from simple_ortho import io
import csv
import re
import numpy as np
import json


def downsample_rgb(
    src_file: Path, dst_file: Path, ds_fact: int = 4, scale_clip: float = None, strip_dewarp: bool = False
):
    """ Downsample `src_file` by `ds_fact`, scale & clip to `scale_clip` and write to uint8 jpeg geotiff `dst_file`. """
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False), rio.open(src_file, 'r') as src_im:
        # raise error if aspect ratio will not be maintained
        if not np.all(np.mod(src_im.shape, ds_fact) == 0):
            raise ValueError(f'Source dimensions {src_im.shape} are not a multiple of {ds_fact}.')

        # create destination profile (note: copying xmp metadata requires driver='GTiff' (possible rasterio bug ?))
        dst_shape = tuple((np.array(src_im.shape) / ds_fact).astype('int'))
        dst_profile = src_im.profile.copy()
        if dst_profile.get('crs', None):
            dst_profile['transform'] *= rio.Affine.scale((src_im.width / dst_shape[1]), (src_im.height / dst_shape[0]))
        dst_profile.update(
            width=dst_shape[1], height=dst_shape[0], count=3, compress='jpeg', interleave='pixel', photometric='ycbcr',
            tiled=True, blockxsize=256, blockysize=256, dtype='uint8', driver='GTiff'
        )

        with rio.open(dst_file, 'w', **dst_profile) as dst_im:
            # copy metadata
            dst_im.update_tags(**src_im.tags())
            for namespace in src_im.tag_namespaces():
                # note there is an apparent rio/gdal bug with ':' in the 'xml:XMP' namspace/ tag name,
                # where 'xml:XMP=' gets prefixed to the value
                ns_dict = src_im.tags(ns=namespace)
                if strip_dewarp and namespace == 'xml:XMP':
                    ns_dict[namespace] = re.sub(r'[ ]*?drone-dji:DewarpData(.*?)"\n', '', ns_dict[namespace])
                dst_im.update_tags(ns=namespace, **ns_dict)

            for index in dst_im.indexes:
                dst_im.update_tags(index, **src_im.tags(index))
            # copy image data, scaling and clipping if required
            array = src_im.read(indexes=dst_im.indexes, out_shape=dst_shape, resampling=rio.enums.Resampling.cubic)
            if scale_clip:
                array = np.clip(array * scale_clip, 0, 255)
            dst_im.write(array)


def downsample_dem(src_file: Path, dst_file: Path, ds_fact: int = 8, bounds: Tuple = None):
    """
    Downsample `src_file` by `ds_fact`, cropping to `bounds` if specified.  Write to float32 deflate geotiff `dst_file`.
    """
    nodata = float('nan')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False), rio.open(src_file, 'r') as src_im:
        if bounds:
            src_win = expand_window_to_grid(src_im.window(*bounds))
            src_transform = src_im.window_transform(src_win)
        else:
            src_win = rio.windows.Window(0, 0, *src_im.shape[::-1])
            src_transform = src_im.transform

        src_array = src_im.read(window=src_win)

        # reproject rather than read(out_shape=...) to ensure square pixels
        dst_array, dst_transform = reproject(
            src_array, src_crs=src_im.crs, src_transform=src_transform, src_nodata=src_im.nodata, dst_crs=src_im.crs,
            dst_nodata=nodata, dst_resolution=(np.abs(src_transform.a) * ds_fact, ) * 2, resampling=Resampling.cubic,
            init_dest_nodata=True
        )
        dst_profile = src_im.profile.copy()
        dst_profile.update(
            width=dst_array.shape[-1], height=dst_array.shape[-2], transform=dst_transform, compress='deflate',
            interleave='band', tiled=True, blockxsize=256, blockysize=256, dtype='float32', driver='GTiff',
            predictor=2, zlevel=9, nodata=nodata
        )
        with rio.open(dst_file, 'w', **dst_profile) as dst_im:
            dst_im.write(dst_array)


def create_ngi_test_data():
    src_rgb_files = [
        '3324c_2015_1004_05_0182_RGBN_CMP.tif',
        '3324c_2015_1004_05_0184_RGBN_CMP.tif',
        '3324c_2015_1004_06_0251_RGBN_CMP.tif',
        '3324c_2015_1004_06_0253_RGBN_CMP.tif',
    ]
    dem_file = 'sudem_l3a_clip.tif'
    src_root = Path('V:/Data/SimpleOrthoEgs/NGI_3324C_2015_Baviaans/')
    dst_root = Path('C:/Data/Development/Projects/simple-ortho/tests/data/ngi')
    ds_fact = 12

    # downsample rgb images
    dst_root.mkdir(exist_ok=True, parents=True)
    for src_rgb_file in src_rgb_files:
        dst_rgb_file = dst_root.joinpath(src_rgb_file[:-9]).with_suffix('.tif')
        src_rgb_file = src_root.joinpath(src_rgb_file)
        if dst_rgb_file.exists():
            dst_rgb_file.unlink()
            dst_rgb_file.with_suffix(dst_rgb_file.suffix + '.aux.xml').unlink(missing_ok=True)
        downsample_rgb(src_rgb_file, dst_rgb_file, ds_fact=ds_fact, scale_clip=(255 / 3000))

    # downsample dem
    src_dem_file = src_root.joinpath(dem_file)
    dst_dem_file = dst_root.joinpath('dem.tif')
    if dst_dem_file.exists():
        dst_dem_file.unlink()
        dst_dem_file.with_suffix(dst_dem_file.suffix + '.aux.xml').unlink(missing_ok=True)
    downsample_dem(src_dem_file, dst_dem_file, bounds=None, ds_fact=ds_fact)

    # copy & convert csv exterior params
    src_ext_file = src_root.joinpath('camera_pos_ori.txt')
    dst_ext_file = dst_root.joinpath('camera_pos_ori.txt')
    if dst_ext_file.exists():
        dst_ext_file.unlink()
    with open(src_ext_file, 'r', newline=None) as fin, open(dst_ext_file, 'w', newline='') as fout:
        reader = csv.DictReader(
            fin, fieldnames=['filename', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'], delimiter=' '
        )
        writer = csv.DictWriter(
            fout, fieldnames=['filename', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'], delimiter=' '
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
        '100_0005_0142_ORTHO.tif'
    ]

    src_root = Path('V:/Data/SimpleOrthoEgs/20190411_Miaoli_Toufeng_Tuniu-River')
    dst_root = Path('C:/Data/Development/Projects/simple-ortho/tests/data/odm')

    # downsample rgb images
    dst_root.joinpath('images').mkdir(exist_ok=True, parents=True)
    for src_rgb_file in src_rgb_files:
        dst_rgb_file = dst_root.joinpath('images', src_rgb_file).with_suffix('.tif')
        src_rgb_file = src_root.joinpath('images', src_rgb_file)
        if dst_rgb_file.exists():
            dst_rgb_file.unlink()
            dst_rgb_file.with_suffix(dst_rgb_file.suffix + '.aux.xml').unlink(missing_ok=True)
        downsample_rgb(src_rgb_file, dst_rgb_file, ds_fact=4)

    # get bounds covering orthos of src files
    def bounds_union(bounds1, bounds2):
        return *np.min((bounds1[:2], bounds2[:2]), axis=0), *np.max((bounds1[-2:], bounds2[-2:]), axis=0)

    bounds = None
    for ortho_file in ortho_files:
        with rio.open(src_root.joinpath('orthority', ortho_file), 'r') as ortho_im:
            bounds = ortho_im.bounds if not bounds else bounds_union(bounds, ortho_im.bounds)

    # crop and downsample dem
    dst_root.joinpath('odm_dem').mkdir(exist_ok=True, parents=True)
    dst_dem_file = dst_root.joinpath('odm_dem', 'dsm.tif')
    if dst_dem_file.exists():
        dst_dem_file.unlink()
        dst_dem_file.with_suffix(dst_dem_file.suffix + '.aux.xml').unlink(missing_ok=True)
    downsample_dem(src_root.joinpath('odm_dem', 'dsm.tif'), dst_dem_file, bounds=bounds, ds_fact=16)

    # copy relevant parts of opensfm reconstruction file
    src_rec_file = src_root.joinpath('opensfm/reconstruction.json')
    dst_rec_file = dst_root.joinpath('opensfm/reconstruction.json')
    dst_root.joinpath('opensfm').mkdir(exist_ok=True, parents=True)
    if dst_rec_file.exists():
        dst_rec_file.unlink()
    with open(src_rec_file, 'r') as f:
        json_obj = json.load(f)
    json_obj = [{k: v for k, v in json_obj[0].items() if k in['cameras', 'shots', 'reference_lla']}]
    json_obj[0]['shots'] = {k[:-4]: v for k, v in json_obj[0]['shots'].items() if k.lower() in src_rgb_files}
    with open(dst_rec_file, 'w') as f:
        json.dump(json_obj, f, indent=4)


def create_io_test_data():
    # create lla_rpy csv file for odm data
    odm_root = Path('C:/Data/Development/Projects/simple-ortho/tests/data/odm')
    dst_root = Path('C:/Data/Development/Projects/simple-ortho/tests/data/io')
    dst_root.mkdir(exist_ok=True)

    osfm_reader = io.OsfmReader(odm_root.joinpath('opensfm', 'reconstruction.json'))
    cam_id = next(iter(osfm_reader.read_int_param().keys()))
    exif_list = [Exif(sf) for sf in odm_root.joinpath('images').glob('*.tif')]
    with open(dst_root.joinpath('odm_lla_rpy.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ', quotechar='"')
        writer.writerow(['filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'camera', 'other'])
        for exif in exif_list:
            writer.writerow([exif.filename.name, *exif.lla, *exif.rpy, cam_id, 'ignored'])

    # create xyz_opk csv file for ngi data
    ngi_root = Path('C:/Data/Development/Projects/simple-ortho/tests/data/ngi')
    src_csv_file = ngi_root.joinpath('camera_pos_ori.txt')
    reader = io.CsvReader(src_csv_file)
    ext_param_dict = reader.read_ext_param()
    with open(dst_root.joinpath('ngi_xyz_opk.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(['filename', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'])
        for filename, ext_param in ext_param_dict.items():
            writer.writerow([filename, *ext_param['xyz'], *np.degrees(ext_param['opk'])])

    # create proj file for above
    src_im_file = next(iter(ngi_root.glob('*RGB.tif')))
    with rio.open(src_im_file, 'r') as src_im:
        crs = src_im.crs

    with open(dst_root.joinpath('ngi_xyz_opk.proj'), 'w', newline='') as f:
        f.write(crs.to_proj4())

    # create an image w/o xmp DeWarpData tag
    src_im_file = Path('V:/Data/SimpleOrthoEgs/20190411_Miaoli_Toufeng_Tuniu-River/images/100_0005_0140.jpg')
    dst_im_file = dst_root.joinpath(src_im_file.name[:-4] + '.tif')
    if dst_im_file.exists():
        dst_im_file.unlink()
    downsample_rgb(src_im_file, dst_root.joinpath(src_im_file.name).with_suffix('.tif'), ds_fact=16, strip_dewarp=True)


if __name__ == '__main__':
    # create_odm_test_data()
    # create_ngi_test_data()
    create_io_test_data()
