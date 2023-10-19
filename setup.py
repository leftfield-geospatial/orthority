from setuptools import setup, find_packages

# To install local development version use:
#    pip install -e .
setup(
    name='orthority',
    version='0.2.0',
    description='Orthorectification with known camera model and DEM',
    author='Dugal Harris',
    author_email='dugalh@gmail.com',
    url='https://github.com/leftfield-geospatial/simple-ortho',
    license='Apache-2.0',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'rasterio>=1.2',
        # 'opencv>=4.5',     # pip does not see the conda installed opencv, so commented out for now
        'pyyaml>=5.4',
        'click>=8',
        'tqdm>=4.6',
    ],
    entry_points={
        'console_scripts': ['simple-ortho=orthority.cli:orthority', 'oty=orthority.cli:cli']
    },
    # scripts=['scripts/batch_recompress.bat']
)
# TODO: the EPSG:<horiz>+<vert> format is not supported in rio 1.3.3, gdal 3.5.3, proj 9.1.0, but is supported
#  in rio 1.3.6, gdal 3.6.2, proj 9.1.1.  The exact version where support begins (proj=9.1.1?) should be set in
#  setup.py (note that pyproj seems to support the EPSG:<>+<> format with PROJ<9.1.1 so something is exactly right
#  with thre prev statement).  Then the multithreaded overview building bug should be fixed in gdal 3.8.  Also note
#  that proj<9.1.0 auto promotes 2d CRSs to 3d.
# TODO: also, now that conda-forge rasterio supports 12 bit jpeg, change the readme warning reflect relevant version
#  / package, or remove entirely, if this version is pinned.
# TODO: change license (?)
