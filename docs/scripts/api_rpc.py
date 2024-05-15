# TODO: change URL to main branch
# code for getting started->api->camera models->rpc cameras
# [create camera]
import orthority as oty

# URLs of required files
url_root = 'https://raw.githubusercontent.com/leftfield-geospatial/orthority/main/tests/data/'
src_file = url_root + 'rpc/qb2_basic1b.tif'  # satellite image with RPC tags
rpc_param_file = url_root + 'rpc/rpc_param.yaml'  # Orthority format RPC parameters

# create camera for src_file
cameras = oty.RpcCameras(rpc_param_file)
camera = cameras.get(src_file)
# [end create camera]

# [create tag]
cameras = oty.RpcCameras.from_images((src_file,))
camera = cameras.get(src_file)
# [end create tag]

# [io_kwargs]
io_kwargs = dict(progress=True)
cameras = oty.RpcCameras.from_images((src_file,), io_kwargs=io_kwargs)
# [end io_kwargs]

# [cam_kwargs]
cam_kwargs = dict(crs='EPSG:32735')
cameras = oty.RpcCameras.from_images((src_file,), cam_kwargs=cam_kwargs)
camera = cameras.get(src_file)

print(camera.crs)
# EPSG:32735
# [end cam_kwargs]

# [export]
cameras = oty.RpcCameras.from_images((src_file,))
cameras.write_param('.')
# [end export]
