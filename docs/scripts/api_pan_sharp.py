import orthority as oty

# URLs of required files
url_root = 'https://raw.githubusercontent.com/leftfield-geospatial/orthority/main/tests/data/'
pan_file = url_root + 'pan_sharp/pan.tif'  # panchromatic drone image
ms_file = url_root + 'pan_sharp/ms.tif'  # multispectral (RGB) drone image

# create PanSharpen object and pan sharpen
pan_sharp = oty.PanSharpen(pan_file, ms_file)
pan_sharp.process('pan_sharp.tif')
