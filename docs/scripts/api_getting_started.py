# [import]
from orthority import param_io

url_root = 'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/main/tests/data/'
# [end import]

# [read]
# read interior parameters from an Orthority format YAML file
int_param_dict = param_io.read_oty_int_param(url_root + 'io/ngi_int_param.yaml')

# read exterior parameters from a CSV file with header
reader = param_io.CsvReader(url_root + 'io/ngi_xyz_opk.csv')
ext_param_dict = reader.read_ext_param()
# [end read]

# [read both]
# read both interior and exterior parameters from image EXIF / XMP tags
reader = param_io.ExifReader([url_root + 'odm/images/100_0005_0140.tif'])
int_param_dict = reader.read_int_param()
ext_param_dict = reader.read_ext_param()
# [end read both]

# [crs]
# find the CRS of an OpenSfm reconstruction.json file
reader = param_io.OsfmReader(url_root + 'odm/opensfm/reconstruction.json')
print(reader.crs)
# EPSG:32651
# [end crs]

# [export]
# read parameters from an OpenSfM reconstruction.json file
reader = param_io.OsfmReader(url_root + 'odm/opensfm/reconstruction.json')
int_param_dict = reader.read_int_param()
ext_param_dict = reader.read_ext_param()

# write parameters to Orthority format files
param_io.write_int_param('int_param.yaml', int_param_dict)
param_io.write_ext_param('ext_param.geojson', ext_param_dict, reader.crs)
# [end export]

# [create_camera]
import orthority as oty

# TODO: change to main branch
url_root = (
    'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/feature_docs/tests'
    '/data/'
)

# read interior parameters
int_param_dict = param_io.read_oty_int_param(url_root + 'io/ngi_int_param.yaml')

# create camera
int_params = int_param_dict['Integraph DMC']
camera = oty.create_camera(**int_params)

# read exterior parameters
reader = param_io.CsvReader(url_root + 'io/ngi_xyz_opk.csv')
ext_param_dict = reader.read_ext_param()

# initialise camera exterior component
ext_params = ext_param_dict['3324c_2015_1004_05_0182_RGB']
ext_params.pop('camera')
camera.update(**ext_params)

# get interior and exterior parameters for file 3324c_2015_1004_05_0182_RGB.tif
ext_params = ext_param_dict['3324c_2015_1004_05_0182_RGB']
cam_id = ext_params.pop('camera')
int_params = int_param_dict[cam_id]

# create camera
camera = oty.create_camera(**int_params, **ext_params)


# [end create_camera]

# [update]
# read exterior parameters
reader = param_io.CsvReader(url_root + 'io/ngi_xyz_opk.csv')
ext_param_dict = reader.read_ext_param()

# update the camera exterior component
ext_params = ext_param_dict['3324c_2015_1004_05_0182_RGB']
ext_params.pop('camera')
camera.update(**ext_params)

# [end update]

# [PinholeCamera]
from orthority import camera

camera = camera.PinholeCamera()
# [end PinholeCamera]

# TODO:
#  - Reading params section
#    - basic reading pattern with e.g. to read yaml and geojson
#    - discuss & show dict structure, common to all formats
#  - Converting params
#    - read osfm reconstruction & write
#    - crs provided by reader or user
#  - Orthorectification
#    - req dem + camera model
#    - create camera directly from int param
#    - create ortho object & process
#  - Ortho & algorithm configuration
#    - args to process
