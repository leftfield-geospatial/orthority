# Simple orthorectification
Orthorectify images with known DEM and camera model.  Designed and tested on [NGI](http://www.ngi.gov.za/index.php/what-we-do/aerial-photography-and-imagery) aerial imagery and triangulation.  

## Getting Started
### Installation
Using `conda`to manage packages and dependencies is recommended.  The [Minconda](https://docs.conda.io/en/latest/miniconda.html) installation includes a minimal `conda`.
1) Create a conda environment and install dependencies:
```shell
conda create -n <environment name> python=3.8 -c conda-forge 
conda activate <environment name> 
conda install -c conda-forge rasterio opencv 
```
2) Clone the git repository and link into the conda environment:
```shell
git clone https://github.com/dugalh/simple_ortho.git
pip install -e ortholite
```

### Requirements  
The following dependencies are installed in the process above.  The `rasterio` package has binary dependencies that are not directly available through `pip`, hence the recommendation for using `conda`.  
  
  - python >= 3.8
  - rasterio >= 1.2
  - opencv >= 4.5

## License
[MIT](LICENSE)

## Author
**Dugal Harris** - [dugalh@gmail.com](mailto:dugalh@gmail.com)
