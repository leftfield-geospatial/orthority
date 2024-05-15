""" Create plot demonstrating --alpha for CLI getting started section."""

from contextlib import contextmanager
from pathlib import Path
import rasterio as rio
from rasterio.plot import show
import numpy as np
from matplotlib import pyplot as plt
import orthority as oty

if '__file__' in globals():
    root_path = Path(__file__).parents[2]
else:
    root_path = Path.cwd().parents[2]
odm_root_dir = root_path.joinpath('tests/data/odm')


def create_alpha_plot():
    """Create a plot showing the effect of --alpha with --no-full-remap."""

    def rgb_to_argb(rgb: np.ma.array) -> np.ndarray:
        """Return an RGBA array given an RGB masked array (both in rasterio ordering)"""
        alpha = np.all(~rgb.mask, axis=0).astype('uint8') * 255
        rgba = np.concatenate((rgb.data, alpha[np.newaxis]), axis=0)
        return rgba

    @contextmanager
    def temp_file(filename: Path) -> Path:
        """Context manager to yield a filename then delete when done."""
        filename.unlink(missing_ok=True)
        yield filename
        filename.unlink(missing_ok=True)

    # set up paths
    src_file = odm_root_dir.joinpath('images/100_0005_0142.tif')
    dem_file = odm_root_dir.joinpath('odm_dem/dsm.tif')

    # read interior and exterior params
    reader = oty.OsfmReader(odm_root_dir.joinpath('opensfm/reconstruction.json'))
    int_param_dict = reader.read_int_param()
    ext_param_dict = reader.read_ext_param()
    ext_param = ext_param_dict[src_file.stem]
    int_param = int_param_dict[ext_param.pop('camera')]

    # read source image
    with rio.open(src_file, 'r') as src_im:
        src_array: np.array = src_im.read()

    # find the width ratios for figure subplots
    ortho_file = oty.root_path.joinpath(f'docs/getting_started/ortho_alpha_1.tif')
    camera = oty.create_camera(**int_param, **ext_param, alpha=1)
    ortho = oty.Ortho(src_file, dem_file, camera, crs=reader.crs)
    ortho.process(ortho_file, full_remap=False, overwrite=True)
    with rio.open(ortho_file, 'r') as ortho_im:
        ortho_array = ortho_im.read(masked=True)
    ar_src = src_array.shape[-1] / src_array.shape[-2]
    ar_ortho = ortho_array.shape[-1] / ortho_array.shape[-2]
    width_ratios = [ar_src, ar_src, ar_ortho]

    fig, _axes = plt.subplots(
        nrows=2, ncols=3, figsize=(8, 4), width_ratios=width_ratios, tight_layout=True
    )
    axes = _axes.flatten()
    for alpha in [0, 1]:
        # create ortho object & undistorted image
        camera = oty.create_camera(**int_param, **ext_param, alpha=alpha)
        ortho = oty.Ortho(src_file, dem_file, camera, crs=reader.crs)
        und_array = ortho._undistort(src_array, nodata=0, interp=oty.Interp.cubic)
        und_array = np.ma.array(und_array, mask=und_array == 0)

        # orthorectify & read ortho image
        with temp_file(
            oty.root_path.joinpath(f'docs/scripts/ortho_alpha_{alpha}.tif')
        ) as ortho_file:
            ortho.process(ortho_file, full_remap=False, overwrite=True)
            with rio.open(ortho_file, 'r') as ortho_im:
                ortho_array = ortho_im.read(masked=True)

        # plot
        i = alpha * 3
        show(src_array, ax=axes[i])
        show(rgb_to_argb(und_array), ax=axes[i + 1])
        show(rgb_to_argb(ortho_array), transform=ortho_im.transform, ax=axes[i + 2])

    # make limits of ortho subplots the same
    _axes[0, 2].set_xlim(_axes[-1, -1].get_xlim())
    _axes[0, 2].set_ylim(_axes[-1, -1].get_ylim())

    # figure styling
    color = '#8f8f8f'
    for ax, col in zip(_axes[0], ['Source', 'Undistorted', 'Orthorectified']):
        ax.set_title(col, color=color)
    for ax, row in zip(_axes[:, 0], ['alpha=0', 'alpha=1']):
        ax.set_ylabel(row, size='large', color=color)
    for ax in _axes.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(color=color, labelcolor=color)
        ax.spines[:].set_color(color)

    fig.savefig(
        root_path.joinpath('docs/getting_started/cli/alpha_plot.webp'), transparent=True, dpi=128
    )


if __name__ == '__main__':
    create_alpha_plot()
