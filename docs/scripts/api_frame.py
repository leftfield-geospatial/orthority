# code for getting started->api->camera models->frame cameras
# [create camera]
import orthority as oty

# URLs of required files
url_root = 'https://raw.githubusercontent.com/leftfield-geospatial/orthority/main/tests/data/'
src_file = url_root + 'ngi/3324c_2015_1004_05_0182_RGB.tif'  # NGI aerial image
int_param_file = url_root + 'io/ngi_int_param.yaml'  # Orthority format interior parameters
ext_param_file = url_root + 'io/ngi_xyz_opk.csv'  # CSV format exterior parameters

# create camera for src_file
cameras = oty.FrameCameras(int_param_file, ext_param_file)
camera = cameras.get(src_file)
# [end create camera]

# [create exif]
src_file = url_root + 'odm/images/100_0005_0140.tif'  # drone image with EXIF / XMP tags

cameras = oty.FrameCameras.from_images([src_file])
camera = cameras.get(src_file)
# [end create exif]

# [crs]
cameras = oty.FrameCameras.from_images([src_file])
print(cameras.crs)
# EPSG:32651
# [end crs]

# [io_kwargs]
io_kwargs = dict(crs='EPSG:32751')
cameras = oty.FrameCameras.from_images([src_file], io_kwargs=io_kwargs)

print(cameras.crs)
# EPSG:32751
# [end io_kwargs]

# [cam_kwargs]
cam_kwargs = dict(distort=False, alpha=0.0)
cameras = oty.FrameCameras.from_images([src_file], cam_kwargs=cam_kwargs)
camera = cameras.get(src_file)

print(camera.distort)
print(camera.alpha)
# False
# 0.0
# [end cam_kwargs]

# [export]
cameras = oty.FrameCameras.from_images([src_file])
cameras.write_param('.')
# [end export]
