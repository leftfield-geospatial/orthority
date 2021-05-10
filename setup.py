from setuptools import setup, find_packages
import glob

# To install local development version use:
#    pip install -e .

setup(
    name='simple-ortho',
    version='0.1.0',
    description='Orthorectification with known camera model and DEM',
    author='Dugal Harris',
    author_email='dugalh@gmail.com',
    url='https://github.com/dugalh/simple_ortho/blob/develop/setup.py',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        # 'python>=3.6',
        'rasterio>=1.2',
        # 'opencv>=4.5',     # how to we get pip to see this?
        'pandas>=1.2',
        'pyyaml>=5.4'
    ],
)
