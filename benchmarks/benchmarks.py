"""Performance and memory benchmarking."""

import functools
import inspect
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from inspect import getsourcefile
from multiprocessing import Process
from pathlib import Path
from typing import Any

import psutil
import yappi
from tqdm import tqdm

from orthority import cli, version

yappi.LINESEP = '\n'  # prevent double newlines in yappi output
src_path = Path('D:/OneDrive/Data/Leftfield/test/orthority')
bench_path = Path(getsourcefile(lambda: None)).parent
ortho_func_names = [
    'cli.cli',
    'Ortho._get_init_dem',
    'Ortho.process',
    'Ortho._reproject_dem',
    'Ortho._mask_dem',
    'Ortho._remap',
    'Ortho._remap_tile',
    'common.build_overviews',
    'write_tile',
]


def _yappi_filter_by_names(x: yappi.YFuncStat, names: Sequence[str] = ()) -> bool:
    """Filtering callback for yappi.get_func_stats()."""
    return any(name in x.name for name in names)


def _oty_cli(**kwargs) -> None:
    """Orthority CLI wrapper."""
    # work around for PicklingError when target=cli.cli.main() is passed directly to
    # multiprocessing.Process()
    kwargs['standalone_mode'] = False
    cli.cli.main(**kwargs)


def _bench_func(
    func: Callable,
    name: str | None = None,
    loops: int = 1,
    filter_callback: Callable[[yappi.YFuncStat], bool] | None = None,
    write_pstat: bool = False,
):
    """Report func() performance and memory usage."""
    dt = datetime.now()
    yappi.set_clock_type('wall')
    yappi.start()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        for _ in range(loops):
            func()
        wall_end, cpu_end = time.perf_counter(), time.process_time()
    finally:
        yappi.stop()
    mem_info = psutil.Process().memory_full_info()

    wall_time = wall_end - wall_start
    cpu_time = cpu_end - cpu_start
    func_stats = yappi.get_func_stats(filter_callback=filter_callback)
    func_stats = func_stats.strip_dirs().sort('ttot', 'desc')

    name = name or func.__name__
    print('BENCHMARK\n---------')
    print(f'Name: {name}')
    print(f'Date: {dt.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Loops: {loops}')

    print('\nPERFORMANCE')
    print(f'Mean wall time: {wall_time / loops:.2f}s')
    print(f'Mean CPU time: {cpu_time / loops:.2f}s')
    print(f'CPU usage: {(100 / os.cpu_count()) * (cpu_time / wall_time):.2f}%', flush=True)

    print('\nMEMORY')
    print(f'Peak RSS: {tqdm.format_sizeof(mem_info.peak_wset, suffix="B")}')
    print(f'Current RSS: {tqdm.format_sizeof(mem_info.rss, suffix="B")}')
    # TODO: report by major/minor page fault type with p.page_faults() when psutil updates to v8
    print(f'Page faults / sec: {mem_info.num_page_faults / cpu_time:.2f}')

    print('\nPROFILE', end='')
    func_stats.print_all()
    print('', flush=True)

    if write_pstat:
        yappi.get_func_stats().save(bench_path.joinpath(f'{name.lower()}.pstat'), type='pstat')


def bench_func(
    func: Callable,
    name: str | None = None,
    loops: int = 1,
    filter_callback: Callable[[yappi.YFuncStat], bool] | None = None,
    write_pstat: bool = False,
):
    """Report func() performance and memory usage.  Runs in a separate process to separate peak
    memory usage from the calling process.
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


def ngi_bench_func_params() -> dict[str, Any]:
    """Return bench_func() parameters to orthorectify an NGI image."""
    ngi_path = src_path.joinpath('ngi')
    cli_str = (
        f'frame --dem {ngi_path.joinpath("x3324cb_2015_L3a.tif")} '
        f'--int-param {ngi_path.joinpath("int_param.yaml")} '
        f'--ext-param {ngi_path.joinpath("ext_param.csv")} -od {bench_path} -o '
        f'{ngi_path.joinpath("3324c_2015_1004_05_0182_RGBN.tif")}'
    )
    func = functools.partial(_oty_cli, args=cli_str.split())
    func_names = [*ortho_func_names, 'FrameCamera.read', 'FrameCamera.remap', 'cv2_remap']
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(func=func, name='NGI', loops=5, filter_callback=filter_callback, write_pstat=True)


def odm_bench_func_params() -> dict[str, Any]:
    """Return bench_func() parameters to orthorectify a drone image."""
    odm_path = src_path.joinpath('odm')
    cli_str = (
        f'frame --dem {odm_path.joinpath("odm_dem/dsm.tif")} '
        f'--int-param {odm_path.joinpath("opensfm/reconstruction.json")} '
        f'--ext-param {odm_path.joinpath("opensfm/reconstruction.json")} -od {bench_path} -o '
        f'{odm_path.joinpath("images/100_0005_0018.JPG")}'
    )
    func = functools.partial(_oty_cli, args=cli_str.split())
    func_names = [*ortho_func_names, 'BrownCamera.read', 'BrownCamera.remap', 'cv2_remap']
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(func=func, name='ODM', loops=5, filter_callback=filter_callback, write_pstat=True)


def rpc_bench_func_params() -> dict[str, Any]:
    """Return bench_func() parameters to orthorectify a satellite image using RPC tags."""
    cli_str = (
        f'rpc --dem {src_path.joinpath("ngi/x3324cb_2015_L3a.tif")} '
        f'--res 2e-5 -od {bench_path} -o '
        f'{src_path.joinpath("rpc/03NOV18082012-P1BS-056844553010_01_P001.TIF")}'
    )
    func = functools.partial(_oty_cli, args=cli_str.split())
    func_names = [*ortho_func_names, 'RpcCamera.read', 'RpcCamera.remap', 'cv2_remap']
    filter_callback = functools.partial(_yappi_filter_by_names, names=func_names)
    return dict(func=func, name='RPC', loops=1, filter_callback=filter_callback, write_pstat=True)


def run_benchmarks(param_funcs: Sequence[Callable]) -> None:
    """Run benchmarks defined by a sequence of callables that return or yield parameters for
    bench_func().
    """
    # param_funcs callables keep parameter creation / cleanup separate from the actual benchmark,
    # like a simple pytest fixture
    print(f'Orthority version: {version.__version__}')
    # from https://stackoverflow.com/a/21901260
    git_rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    git_rev = git_rev.decode('utf8').strip()
    print(f'Current git commit: {git_rev}\n', flush=True)

    for param_func in param_funcs:
        params = param_func()
        if inspect.isgenerator(params):
            for params_ in params:
                bench_func(**params_)
        else:
            bench_func(**params)


if __name__ == '__main__':
    param_funcs = [ngi_bench_func_params, odm_bench_func_params, rpc_bench_func_params]
    run_benchmarks(param_funcs)
