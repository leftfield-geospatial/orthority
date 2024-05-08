import orthority as oty

# URLs of required files
url_root = (
    'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/feature_docs/tests/data/'
)
src_file = url_root + 'ngi/3324c_2015_1004_05_0182_RGB.tif'  # aerial image
dem_file = url_root + 'ngi/dem.tif'  # DEM covering imaged area
int_param_file = url_root + 'io/ngi_int_param.yaml'  # interior parameters
ext_param_file = url_root + 'io/ngi_xyz_opk.csv'  # exterior parameters

# create a camera model for src_file from interior & exterior parameters
cameras = oty.FrameCameras(int_param_file, ext_param_file)
camera = cameras.get(src_file)

# create Ortho object and orthorectify
ortho = oty.Ortho(src_file, dem_file, camera=camera, crs=cameras.crs)
ortho.process('ortho.tif')
