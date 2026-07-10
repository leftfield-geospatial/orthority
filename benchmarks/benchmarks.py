"""Performance and memory benchmarking."""

import functools
import subprocess
import sys
from collections.abc import Sequence
from inspect import getsourcefile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import yappi
from rasterio.windows import Window

from orthority import cli, common

bench_path = Path(getsourcefile(lambda: None)).parent
sys.path.append(bench_path.joinpath('../tests/data').absolute().as_posix())
from create_test_data import downsample_image

yappi.LINESEP = '\n'  # prevent double newlines in yappi output
mag_src_path = Path('D:/OneDrive/Data/Leftfield/test/orthority')
ssd_src_path = Path('C:/Temp/Leftfield/test/orthority')
src_path: Path = mag_src_path
mag_out_path = Path('D:/Temp')
out_path: Path = mag_out_path
common_ortho_func_names = [
    '_ortho',
    'Ortho._get_init_dem',
    'Ortho.process',
    'Ortho._reproject_dem',
    'Ortho._mask_dem',
    'Ortho._remap',
    'Ortho._remap_tile',
    'Ortho._read_remap_tile',
    'build_overviews',
    '_remap',
    '_per_band_remap',
    '_get_remap_slices',
]
untiled_src_files = False


def _yappi_filter_by_names(x: yappi.YFuncStat, names: Sequence[str] = ()) -> bool:
    """Filtering callback for yappi.get_func_stats()."""
    return any(name == x.name for name in names)


def _purge_ram():
    """Purge RAM with RAMMap (requires admin privileges)."""
    # don't use the -Ew option which writes working sets to pagefile.sys and results
    # in some apps reading their working sets back while benchmarks are running
    rammap = str(bench_path.joinpath('rammap.exe'))
    for option in ['-Em', '-E0', '-Et']:
        subprocess.run([rammap, option])


def _oty_cli(**kwargs) -> None:
    """Orthority CLI wrapper."""
    kwargs.setdefault('standalone_mode', False)
    cli.cli.main(**kwargs)


def ngi_bench_func():
    """Benchmarking func() to orthorectify an NGI image."""
    ngi_path = src_path.joinpath('ngi')
    if untiled_src_files:
        im_file = Path(
            'V:/Data/NGI/Unrectified/3324C_2015_1004/RGBN/3324c_2015_1004_05_0182_RGBN.tif'
        )
    else:
        im_file = ngi_path.joinpath('3324c_2015_1004_05_0182_RGBN.tif')

    cli_str = (
        f'-v frame --dem {ngi_path.joinpath("x3324cb_2015_L3a.tif")} '
        f'--int-param {ngi_path.joinpath("int_param.yaml")} '
        f'--ext-param {ngi_path.joinpath("ext_param.csv")} -od {out_path} -o '
    )
    # purge cached disk reads etc
    _purge_ram()
    # delete any previous ortho file
    out_path.joinpath(f'{im_file.stem}_ORTHO.tif').unlink(missing_ok=True)
    yield
    # benchmark
    _oty_cli(args=[*cli_str.split(), str(im_file)])
    yield


def ngi_bench_params() -> dict[str, Any]:
    """Return bench_func() parameters to orthorectify an NGI image."""
    func_names = [
        *common_ortho_func_names,
        'frame',
        'FrameCamera.read',
        'FrameCamera.remap',
        'FrameCamera.world_to_pixel',
    ]
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(
        func=ngi_bench_func, name='NGI', loops=5, filter_callback=filter_callback, write_pstat=True
    )


def odm_bench_func() -> dict[str, Any]:
    """Benchmarking func() to orthorectify a drone image."""
    odm_path = src_path.joinpath('odm')
    im_file = odm_path.joinpath("images/100_0005_0140.JPG")
    cli_str = (
        f'-v frame --dem {odm_path.joinpath("odm_dem/dsm.tif")} '
        f'--int-param {odm_path.joinpath("opensfm/reconstruction.json")} '
        f'--ext-param {odm_path.joinpath("opensfm/reconstruction.json")} -od {out_path} -o '
        f'{im_file}'
    )
    # purge cached disk reads etc
    _purge_ram()
    # delete any previous ortho file
    out_path.joinpath(f'{im_file.stem}_ORTHO.tif').unlink(missing_ok=True)
    yield
    # benchmark
    _oty_cli(args=cli_str.split())
    yield


def odm_bench_params() -> dict[str, Any]:
    """Return bench_func() parameters to orthorectify a drone image."""
    func_names = [
        *common_ortho_func_names,
        'frame',
        'BrownCamera.read',
        'BrownCamera.remap',
        'BrownCamera.world_to_pixel',
    ]
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(
        func=odm_bench_func, name='ODM', loops=5, filter_callback=filter_callback, write_pstat=True
    )


def rpc_bench_func():
    """Benchmarking func() to orthorectify a satellite image using RPC tags."""
    src_im_file = src_path.joinpath("rpc/03NOV18082012-P1BS-056844553010_01_P001.TIF")
    if untiled_src_files:
        crop_im_file = out_path.joinpath(f'{src_im_file.stem}_CROP_UNTILED.tif')
    else:
        crop_im_file = out_path.joinpath(f'{src_im_file.stem}_CROP.tif')

    if not crop_im_file.exists():
        # create a cropped version of src_im_file to reduce processing time
        ds_fact = 1
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUS'), rio.open(src_im_file, 'r') as src_im:
            rpcs = src_im.rpcs
            win = Window(10500, 8000, 8000, 6000)

            # adjust RPCs for crop
            rpcs.line_off = (rpcs.line_off - win.row_off + 0.5) / ds_fact - 0.5
            rpcs.samp_off = (rpcs.samp_off - win.col_off + 0.5) / ds_fact - 0.5
            rpcs.line_scale /= ds_fact
            rpcs.samp_scale /= ds_fact

            # write the cropped image
            array = src_im.read(window=win)
            profile, _ = common.create_profile(
                'gtiff', array.shape, array.dtype, compress='deflate', write_mask=False
            )
            if untiled_src_files:
                profile.update(
                    crs=src_im.crs, rpcs=rpcs, tiled=False, blockxsize=None, blockysize=None
                )
            else:
                profile.update(crs=src_im.crs, rpcs=rpcs, overviews='none')

            with rio.open(crop_im_file, 'w', **profile) as dst_im:
                dst_im.write(array)

    cli_str = f'-v rpc --dem {src_path.joinpath("ngi/x3324cb_2015_L3a.tif")} -od {out_path} -o '
    # purge cached disk reads etc
    _purge_ram()
    # delete any previous ortho file
    out_path.joinpath(f'{crop_im_file.stem}_ORTHO.tif').unlink(missing_ok=True)
    yield
    _oty_cli(args=[*cli_str.split(), str(crop_im_file)])
    yield


def rpc_bench_params() -> dict[str, Any]:
    """Return bench_func() parameters to orthorectify a satellite image using RPC tags."""
    func_names = [
        *common_ortho_func_names,
        'rpc',
        'RpcCamera.read',
        'RpcCamera.remap',
        'RpcCamera.world_to_pixel',
    ]
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(
        func=rpc_bench_func, name='RPC', loops=5, filter_callback=filter_callback, write_pstat=True
    )


def pan_sharp_bench_func() -> dict[str, Any]:
    """Benchmarking func() to pan sharpen simulated pan / MS drone images."""
    src_file = src_path.joinpath('odm/images/100_0005_0140.JPG')
    postfix = '_untiled' if untiled_src_files else ''
    ms_file = out_path.joinpath(f'ms{postfix}.tif')
    pan_file = out_path.joinpath(f'pan{postfix}.tif')

    if not pan_file.exists() or not ms_file.exists():
        # convert RGB source image to pan
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUS'), rio.open(src_file, 'r') as src_im:
            src_array = src_im.read()
            pan_array = src_array.mean(axis=0).round().astype('uint8')
            pan_array = np.expand_dims(pan_array, axis=0)
            profile, _ = common.create_profile(
                'gtiff', pan_array.shape, pan_array.dtype, compress='deflate', write_mask=False
            )
            profile.update(nodata=None)

            if untiled_src_files:
                tile_kwargs = dict(tiled=False, blockxsize=None, blockysize=None)
            else:
                tile_kwargs = dict()
            profile.update(**tile_kwargs)

            with rio.open(pan_file, 'w', **profile) as pan_im:
                pan_im.write(pan_array)

            # downsample RGB source image to MS resolution
            downsample_image(src_file, ms_file, ds_fact=4, compress='deflate', **tile_kwargs)

    pan_sharp_file = out_path.joinpath("pan_sharp.tif")
    cli_str = f'sharpen -p {pan_file} -ms {ms_file} --compress deflate -of {pan_sharp_file} -o '
    # purge cached disk reads etc
    _purge_ram()
    # delete any previous pan sharpened file
    pan_sharp_file.unlink(missing_ok=True)
    yield
    _oty_cli(args=cli_str.split())
    yield


def pan_sharp_bench_params() -> dict[str, Any]:
    """Return bench_func() parameters to pan sharpen simulated pan / MS drone images."""
    func_names = [
        'sharpen',
        'PanSharpen._get_stats',
        'get_tile_stats',
        'PanSharpen._get_params',
        'PanSharpen._process_tile_array',
        'PanSharpen._process_tile',
        'PanSharpen.process',
    ]
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(
        func=pan_sharp_bench_func,
        name='Sharpen',
        loops=5,
        filter_callback=filter_callback,
        write_pstat=True,
    )


if __name__ == '__main__':
    params = [ngi_bench_params(), odm_bench_params(), rpc_bench_params(), pan_sharp_bench_params()]
    common.run_benchmarks(params)

# TODO:
#  - why does the RPC benchmark not have a higher CPU usage during remapping?
#  - benchmark using numexpr for camera models
#  - does limiting openmp threads affect rasterio write performance with conda-forge rasterio
#  which uses vcomp?
