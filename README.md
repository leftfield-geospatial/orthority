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

## Scripts
simple_ortho functionality is accessed by calling [python scripts](scripts) on the command line.  

### [ortho_im](scripts/ortho_im.py)
Orthorectifies an image.  Run ```python simple_ortho/scripts/ortho_im.py -h``` to get help.

**Usage:** `ortho_im.py [-h] [-o <ortho_path>] [-rc <config_path>] [-wc <config_path>] [-v {1,2,3,4}] src_im_file dem_file pos_ori_file`

#### Positional (required) arguments
Argument  | Description
----------|--------------
`src_im_file` | Path to the source unrectified image file.
`dem_file` | Path to a DEM, that covers `src_im_file`.  
`pos_ori_file` | Path to a text file specifying the camera position and orientation file for `src_im_file`.  See [camera position and orientation section](#camera_position_and_orientation) for more detail. 

#### Optional arguments
Argument | Long form | Description
---------|-----------|------------
`-h` | `--help` | Print help and exit
`-o` `<ortho_path>` | `--ortho` `<ortho_path>` | Write the orthorectified file to the specified `<ortho_path>` filename.  (Default: name the orthorectified image '`<src_im_file>`_ORTHO.tif').
`-rc` `<config_path>` | `--readconf` `<config_path>` | Read a custom configuration from the specified `<config_path>`.  (Default read configuration from [config.yaml](config.yaml)).  See [configuration](#configuration) for more details.  
`-wc` `<config_path>` | `--writeconf` `<config_path>` | Write current configuration to  `<config_path>` and exit.  
`-v` `{1,2,3,4}` | `--verbosity {1,2,3,4}` | Set the logging level (lower means more logging).  1=debug, 2=info, 3=warning, 4=error (default: 2)

### [batch_ortho_im](scripts/batch_ortho_im.py)
Orthorectifies a group of images matching a wildcard.  Run ```python simple_ortho/scripts/batch_ortho_im.py -h``` to get help.

**Usage:** `batch_ortho_im.py [-h] [-rc <config_path>] [-v {1,2,3,4}] src_im_wildcard dem_file pos_ori_file`

#### Positional (required) arguments
Argument  | Description
----------|--------------
`src_im_wildcard` | Source image wildcard pattern or directory (e.g. '.' or './*_CMP.TIF')
`dem_file` | Path to a DEM, that covers the images matching `src_im_wildcard`.  
`pos_ori_file` | Path to a text file specifying the camera position and orientation for the images matching `src_im_wildcard`.  See [camera position and orientation section](#camera_position_and_orientation) for more detail. 

#### Optional arguments
Argument | Long form | Description
---------|-----------|------------
`-h` | `--help` | Print help and exit
`-rc` `<config_path>` | `--readconf` `<config_path>` | Read a custom configuration from the specified `<config_path>`.  (Default read configuration from [config.yaml](config.yaml)).  See [configuration](#configuration) for more details.  
`-v` `{1,2,3,4}` | `--verbosity {1,2,3,4}` | Set the logging level (lower means more logging).  1=debug, 2=info, 3=warning, 4=error (default: 2)

### [batch_recompress](scripts/batch_recompress.bat)
Recompress images matching a wildcard using DEFLATE compression.  If necessary, this script can be used to address the incompatibility of `conda`'s `gdal` package with 12bit jpeg compressed tiffs sometimes used by [NGI](http://www.ngi.gov.za/index.php/what-we-do/aerial-photography-and-imagery).   [OSGeo4W](https://trac.osgeo.org/osgeo4w/) with `gdal` is required.  DEFLATE compressed files can then be processed with [`ortho_im`](#ortho_im) or [`batch_ortho_im`](#batch_ortho_im).   Run ```simple_ortho/scripts/batch_recompress.bat``` without arguments to get help.
#### Positional (required) arguments
Argument  | Description
----------|--------------
`src_im_wildcard` | Process images matching this wildcard pattern (e.g. './*_RGB.TIF').  Recompressed files are written to new files named '\*_CMP.tif'.


## File formats
### Configuration

### Camera_position_and_orientation
Camera position and orientation correspondiong to an image is specified in a space separated text file.  The file format is the same as that used by PCI Geomatica's OrthoEgine i.e. each row specifies the camera position and orientation for an image as follows.    
```
<Image file stem> <Easting (m)> <Northing (m)> <Altitude (m)> <Omega (deg)> <Phi (deg)> <Kappa (deg)> 
```
Where `<Image file stem>` is the source file name excluding extension.  **Note** that the position information must be specified in the same co-ordinate reference system as the source images.

For example:
```
...
3323d_2015_1001_01_0001_RGBN 43333.970620 -3709166.407240 5672.686250 0.448258 -0.200394 -0.184258
3323d_2015_1001_01_0002_RGBN 44710.649080 -3709211.341900 5672.299410 -0.168341 0.013147 -0.380978
3323d_2015_1001_01_0003_RGBN 46091.888940 -3709233.718060 5676.132710 -1.493311 -0.004520 -0.158283
...
```
When running [ortho_im](#ortho_im) and [batch_ortho_im](), the source image filename (stem) must exist in the camera position and orientation file.     

## Known limitations

## Example



## License
This project is licensed under the terms of the [MIT license](LICENSE).

## Author
**Dugal Harris** - [dugalh@gmail.com](mailto:dugalh@gmail.com)
