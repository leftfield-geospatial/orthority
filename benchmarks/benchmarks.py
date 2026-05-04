"""Performance and memory benchmarking."""

import functools
import os
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Sequence
from datetime import datetime
from inspect import getsourcefile
from multiprocessing import Process
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rasterio as rio
import yappi
from tqdm import tqdm

from orthority import cli, common, version

bench_path = Path(getsourcefile(lambda: None)).parent
sys.path.append(bench_path.joinpath('../tests/data').absolute().as_posix())
from create_test_data import downsample_image

yappi.LINESEP = '\n'  # prevent double newlines in yappi output
mag_src_path = Path('D:/OneDrive/Data/Leftfield/test/orthority')
ssd_src_path = Path('C:/Temp/Leftfield/test/orthority')
src_path: Path = mag_src_path
mag_out_path = Path('D:/Temp')
out_path: Path = bench_path
ortho_func_names = [
    '_ortho',
    'Ortho._get_init_dem',
    'Ortho.process',
    'Ortho._reproject_dem',
    'Ortho._mask_dem',
    'Ortho._remap',
    'Ortho._remap_tile',
    'Ortho._read_remap_tile',
    'build_overviews',
    'write_tile',
    '_write_tile',
    'im_read',
]
untiled_src_files = False


def _yappi_filter_by_names(x: yappi.YFuncStat, names: Sequence[str] = ()) -> bool:
    """Filtering callback for yappi.get_func_stats()."""
    return any(name == x.name for name in names)


def _purge_ram():
    """Purge RAM with RAMMap (requires admin privileges)."""
    rammap = str(bench_path.joinpath('rammap.exe'))
    for option in ['-Ew', '-Es', '-Em', '-Et', '-E0']:
        subprocess.run([rammap, option])


def _oty_cli(**kwargs) -> None:
    """Orthority CLI wrapper."""
    kwargs.setdefault('standalone_mode', False)
    cli.cli.main(**kwargs)


def _bench_func(
    func: Callable,
    name: str | None = None,
    loops: int = 1,
    filter_callback: Callable[[yappi.YFuncStat], bool] | None = None,
    write_pstat: bool = False,
):
    """Report func() performance and memory usage."""
    proc = psutil.Process()
    proc.nice(psutil.HIGH_PRIORITY_CLASS)
    dt = datetime.now()
    yappi.set_clock_type('wall')
    wall_times, cpu_times = [], []
    for _ in range(loops):
        func_gen = func()
        next(func_gen)  # setup
        yappi.start()
        # note that time.process_time() has a resolution of 16ms on windows, so cpu_time should
        # be >> 16ms for it to be accurate
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        try:
            next(func_gen)  # benchmark
            wall_end, cpu_end = time.perf_counter(), time.process_time()
        finally:
            yappi.stop()
        try:
            next(func_gen)  # teardown
        except StopIteration:
            pass
        wall_times.append(wall_end - wall_start)
        cpu_times.append(cpu_end - cpu_start)

    mem_info = proc.memory_full_info()
    func_stats = yappi.get_func_stats(filter_callback=filter_callback)
    func_stats = func_stats.strip_dirs().sort('ttot', 'desc')
    name = name or func.__name__
    ttl_cpu_times = sum(cpu_times)
    ttl_wall_times = sum(wall_times)

    print('BENCHMARK\n---------')
    print(f'Name: {name}')
    print(f'Computer: {os.getenv("COMPUTERNAME")}')
    print(f'Date: {dt.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Loops: {loops}')

    print('\nPERFORMANCE')
    print(f'Mean (std) wall time: {np.mean(wall_times):.4f}s ({np.std(wall_times):.4f}s)')
    print(f'Mean (std) CPU time: {np.mean(cpu_times):.4f}s ({np.std(cpu_times):.4f}s)')
    print(f'CPU usage: {(100 / os.cpu_count()) * (ttl_cpu_times / ttl_wall_times):.2f}%')

    print('\nMEMORY')
    print(f'Peak RSS: {tqdm.format_sizeof(mem_info.peak_wset, suffix="B")}')
    print(f'Current RSS: {tqdm.format_sizeof(mem_info.rss, suffix="B")}')
    # TODO: report by major/minor page fault type with p.page_faults() when psutil updates to v8
    print(
        f'Page faults / sec: '
        f'{mem_info.num_page_faults / (1e-9 if ttl_cpu_times == 0 else ttl_cpu_times):.2f}'
    )

    print('\nPROFILE', end='')
    func_stats.print_all()
    print('', flush=True)

    if write_pstat:
        yappi.get_func_stats().save(out_path.joinpath(f'{name.lower()}.pstat'), type='pstat')
    yappi.clear_stats()


def bench_func(
    func: Generator,
    name: str | None = None,
    loops: int = 1,
    filter_callback: Callable[[yappi.YFuncStat], bool] | None = None,
    write_pstat: bool = False,
):
    """Report func() performance and memory usage.  func() should be a generator that performs
    any setup, yields, runs the code to benchmark, yields, then performs any teardown.
    Benchmarking is run in a separate process to separate memory usage from the calling process.
    """
    proc = Process(
        target=_bench_func,
        args=(func,),
        kwargs=dict(
            name=name,
            loops=loops,
            filter_callback=filter_callback,
            write_pstat=write_pstat,
        ),
    )
    proc.start()
    proc.join()


def ngi_bench_func():
    """Benchmarking func() to orthorectify an NGI image."""
    # setup
    ngi_path = src_path.joinpath('ngi')
    if untiled_src_files:
        im_file = Path(
            'V:/Data/NGI/Unrectified/3324C_2015_1004/RGBN/3324c_2015_1004_05_0182_RGBN.tif'
        )
    else:
        im_file = ngi_path.joinpath('3324c_2015_1004_05_0182_RGBN.tif')

    cli_str = (
        f'frame --dem {ngi_path.joinpath("x3324cb_2015_L3a.tif")} '
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
        *ortho_func_names,
        'frame',
        'FrameCamera.read',
        'FrameCamera.remap',
        'FrameCamera.read_remap',
        'cv2_remap',
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
        f'frame --dem {odm_path.joinpath("odm_dem/dsm.tif")} '
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
        *ortho_func_names,
        'frame',
        'BrownCamera.read',
        'BrownCamera.remap',
        'BrownCamera.read_remap',
        'cv2_remap',
    ]
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(
        func=odm_bench_func, name='ODM', loops=5, filter_callback=filter_callback, write_pstat=True
    )


def rpc_bench_func():
    """Benchmarking func() to orthorectify a satellite image using RPC tags."""
    if untiled_src_files:
        im_file = Path(
            "V:/Data/Digital Globe/056844553010_01/056844553010_01_P001_PAN/03NOV18082012-P1BS"
            "-056844553010_01_P001.TIF"
        )
    else:
        im_file = src_path.joinpath("rpc/03NOV18082012-P1BS-056844553010_01_P001.TIF")
    cli_str = (
        f'rpc --dem {src_path.joinpath("ngi/x3324cb_2015_L3a.tif")} --res 3e-5 -od {out_path} -o '
    )
    # purge cached disk reads etc
    _purge_ram()
    # delete any previous ortho file
    out_path.joinpath(f'{im_file.stem}_ORTHO.tif').unlink(missing_ok=True)
    yield
    _oty_cli(args=[*cli_str.split(), str(im_file)])
    yield


def rpc_bench_params() -> dict[str, Any]:
    """Return bench_func() parameters to orthorectify a satellite image using RPC tags."""
    func_names = [
        *ortho_func_names,
        'rpc',
        'RpcCamera.read',
        'RpcCamera.remap',
        'RpcCamera.read_remap',
        'cv2_remap',
    ]
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(
        func=rpc_bench_func, name='RPC', loops=5, filter_callback=filter_callback, write_pstat=True
    )


def pan_sharp_bench_func() -> dict[str, Any]:
    """Benchmarking func() to pan sharpen simulated pan / MS drone images."""
    src_file = src_path.joinpath('odm/images/100_0005_0140.JPG')
    ms_file = out_path.joinpath('ms.tif')
    pan_file = out_path.joinpath('pan.tif')

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
            with rio.open(pan_file, 'w', **profile) as pan_im:
                pan_im.write(pan_array)

            # downsample RGB source image to MS resolution
            downsample_image(src_file, ms_file, ds_fact=4, compress='deflate')

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
        'read_pan',
        'read_ms',
        'write_sharp',
    ]
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(
        func=pan_sharp_bench_func,
        name='Sharpen',
        loops=5,
        filter_callback=filter_callback,
        write_pstat=True,
    )


def run_benchmarks(params: Sequence[dict[str, Any]]) -> None:
    """Run benchmarks defined by a sequence of bench_func() parameter dictionaries."""
    print(f'Orthority version: {version.__version__}')
    # from https://stackoverflow.com/a/21901260
    git_rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    git_rev = git_rev.decode('utf8').strip()
    print(f'Current git commit: {git_rev}\n', flush=True)
    for param in params:
        bench_func(**param)


if __name__ == '__main__':
    params = [ngi_bench_params(), odm_bench_params(), rpc_bench_params(), pan_sharp_bench_params()]
    run_benchmarks(params)
