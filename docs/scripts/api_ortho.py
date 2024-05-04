# [create camera]
import orthority as oty

# URLs of required files
url_root = (
    'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/feature_docs/tests/data/'
)
src_file = url_root + 'ngi/3324c_2015_1004_05_0182_RGB.tif'  # NGI aerial image
dem_file = url_root + 'ngi/dem.tif'  # DEM covering imaged area
int_param_file = url_root + 'io/ngi_int_param.yaml'  # Orthority format interior parameters
ext_param_file = url_root + 'io/ngi_xyz_opk.csv'  # CSV format exterior parameters

# create a camera for src_file from interior / exterior parameters
cameras = oty.FrameCameras(int_param_file, ext_param_file)
camera = cameras.get(src_file)
# [end create camera]

# [orthorectify]
ortho = oty.Ortho(src_file, dem_file, camera=camera, crs=cameras.crs)
ortho.process('ortho.tif')
# [end orthorectify]

# [config ortho]
ortho.process(
    'ortho.tif',
    resolution=(5, 5),  # resolution in units of the world / ortho CRS
    compress=oty.Compress.jpeg,  # compression
    dtype='uint8',  # data type
    write_mask=True,  # write internal mask
    build_ovw=True,  # build internal overviews
)
# [end config ortho]

# [config algo]
ortho.process(
    'ortho.tif',
    interp=oty.Interp.cubic,  # source image interpolation
    dem_interp=oty.Interp.cubic,  # dem interpolation
    per_band=True,  # orthorectify band-by-band rather than all bands at once
    progress=True,  # display a progress bar
    overwrite=True,  # overwrite existing ortho
)
# [end config algo]

# [progress]
ortho.process('ortho.tif', progress=True)
# [end progress]

# [custom progress]
from tqdm.auto import tqdm

progress = tqdm(desc='Test')  # custom progress bar
ortho.process('ortho.tif', progress=progress)
# [end custom progress]
