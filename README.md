# Simple orthorectification
Fast and simple orthorectification of images with known DEM and camera model.  Designed and tested on [NGI](http://www.ngi.gov.za/index.php/what-we-do/aerial-photography-and-imagery) aerial imagery.  

## Installation
Using `conda` to manage packages and dependencies is recommended.  The [Minconda](https://docs.conda.io/en/latest/miniconda.html) installation includes a minimal `conda`.
1) Create a conda environment and install dependencies:
```shell
conda create -n <environment name> python=3.8 -c conda-forge 
conda activate <environment name> 
conda install -c conda-forge rasterio opencv pandas pyyaml
````
2) Clone the git repository and link into the conda environment:
``` shell
git clone https://github.com/dugalh/simple_ortho.git
pip install -e simple_ortho
```

### Requirements  
The following dependencies are installed in the process above.  The `rasterio` package has binary dependencies that are not directly available through `pip`, hence the recommendation for using `conda`.  
  
  - python >= 3.8
  - rasterio >= 1.2
  - opencv >= 4.5
  - pandas >= 1.2
  - pyyaml >= 5.4

## Usage
### Scripts
table or list here maybe
#### [ortho_im](scripts/ortho_im.py)
Orthorectification of an aerial image.  Requires specification of a DEM, and [text file containing camera position and orientation information](#camera position and orientation) in the command line arguments.  Run ```python simple_ortho/scripts/ortho_im.py -h``` to get help.
```
usage: ortho_im.py [-h] [-o ORTHO] [-rc READCONF] [-wc WRITECONF] [-v {1,2,3,4}] src_im_file dem_file pos_ori_file

positional arguments:
  src_im_file           path to the source image file
  dem_file              path to the DEM file
  pos_ori_file          path to the camera position and orientaion file

optional arguments:
  -h, --help            show this help message and exit
  -o ORTHO, --ortho ORTHO
                        write ortho image to this path (default: append '_ORTHO' to src_im_file)
  -rc READCONF, --readconf READCONF
                        read custom config from this path (default: use config.yaml in simple_ortho root)
  -wc WRITECONF, --writeconf WRITECONF
                        write default config to this path and exit
  -v {1,2,3,4}, --verbosity {1,2,3,4}
                        logging level: 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)
```
#### [batch_ortho_im](scripts/batch_ortho_im.py)
Batch orthorectification of images matching a wildcard pattern.  Required arguments are similar to those of [ortho_im](#ortho_im) i.e. a DEM and [camera position and orientation information in a text file](#camera position and orientation).  Run ```python simple_ortho/scripts/batch_ortho_im.py -h``` to get help.
```
usage: batch_ortho_im.py [-h] [-rc READCONF] [-v {1,2,3,4}] src_im_wildcard dem_file pos_ori_file

positional arguments:
  src_im_wildcard       source image wildcard pattern or directory (e.g. '.' or '*_CMP.TIF')
  dem_file              path to the DEM file
  pos_ori_file          path to the camera position and orientaion file

optional arguments:
  -h, --help            show this help message and exit
  -rc READCONF, --readconf READCONF
                        read custom config from this path (default: use config.yaml in simple_ortho root)
  -v {1,2,3,4}, --verbosity {1,2,3,4}
                        logging level: 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)
 ```
#### [batch_recompress](scripts/batch_recompress.bat)
A support script to get around the incompatibility of `conda`'s `gdal` package with 12bit jpeg compressed tiffs sometimes used by [NGI](http://www.ngi.gov.za/index.php/what-we-do/aerial-photography-and-imagery).   It uses [OSGeo4W's](https://trac.osgeo.org/osgeo4w/) `gdal` to batch recompress tiffs matching a wildcard pattern with DEFLATE.  These recompressed tiffs can then be used with [ortho_im](scripts/ortho_im.py) and [batch_ortho_im](scripts/batch_ortho_im.py).

### Camera position and orientation
Camera position and orientation for each image is specified in a space separated text file.  The format is the same as that used by PCI Geomatica's OrthoEgine i.e. each row specified the camera position and orientation for an image as follows:
```
<Image file stem> <Easting (m)> <Northing (m)> <Altitude (m)> <Phi (deg)> <Kappa (deg)> <Omega (deg)> 
```
The `<Image file stem>` is the file name excluding extension.  For example:
```
3323d_2015_1001_01_0001_RGBN 43333.970620 -3709166.407240 5672.686250 0.448258 -0.200394 -0.184258
3323d_2015_1001_01_0002_RGBN 44710.649080 -3709211.341900 5672.299410 -0.168341 0.013147 -0.380978
3323d_2015_1001_01_0003_RGBN 46091.888940 -3709233.718060 5676.132710 -1.493311 -0.004520 -0.158283
...
```
**Note** that the position information must be specified in the same CRS as the source images.  
### Configuration
Detailed configuration is specified in [config.yaml](config.yaml)
### Known limitations

### Example



## License
This project is licensed under the terms of the [MIT license](LICENSE).

## Author
**Dugal Harris** - [dugalh@gmail.com](mailto:dugalh@gmail.com)
