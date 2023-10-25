|Tests| |codecov| |License: MPL v2|

orthority
=========

.. image:: https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/main/docs/readme_banner.webp
   :alt: banner

``orthority`` provides a command line interface and Python API for orthorectifying drone and aerial
imagery, given a camera model and DEM. It supports common lens distortion types. Camera parameters
can be read from various file formats, or image EXIF / XMP tags.

Installation
------------

pip
~~~

.. code:: shell

   pip install orthority

conda
~~~~~

.. code:: shell

   conda install -c conda-forge orthority

Quick start
-----------

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

``orthority`` command line functionality is accessed with the ``oty`` command, and its sub-commands:

-  ``ortho``: Orthorectify with interior and exterior parameter file(s).
-  ``exif``: Orthorectify using EXIF / XMP tags.
-  ``odm``: Orthorectify OpenDroneMap outputs.

Get help on ``oty`` with:

.. code:: shell

   oty --help

and help on an ``oty`` sub-command with:

.. code:: shell

   oty <sub-command> --help

Options for the orthorectification algorithm and ortho image format are common to all sub-commands.

Examples
^^^^^^^^

Orthorectify *source.tif* using DEM *dem.tif*, and camera model defined by interior parameters in
*int_param.yaml* and exterior parameters in *ext_param.geojson*:

.. code:: shell

   oty ortho --dem dem.tif --int-param int_param.yaml --ext-param ext_param.geojson source.tif

Orthorectify *source.tif* using DEM *dem.tif*, and camera model defined by *source.tif* EXIF / XMP
tags:

.. code:: shell

   oty exif --dem dem.tif source.tif

As above, but use *bilinear* interpolation, a resolution of 0.5 m, and *deflate* compression to
create the ortho image:

.. code:: shell

   oty exif --dem dem.tif --interp bilinear --res 0.5 --compress deflate source.tif

Orthorectify the OpenDroneMap data set in *odm_root*, placing ortho images in *odm_root/orthority*:

.. code:: shell

   oty odm --proj-dir odm_root --out-dir odm_root/orthority

API
~~~

Orthorectify a drone image using its EXIF / XMP tags to form the camera model:

.. code:: python

   from pathlib import Path
   import orthority as oty

   # URLs of source image and DEM
   src_file = (
       'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/main/'
       'tests/data/odm/images/100_0005_0140.tif'
   )
   dem_file = (
       'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/main/'
       'tests/data/odm/odm_dem/dsm.tif'
   )

   # read interior and exterior parameters from src_file EXIF / XMP tags
   reader = oty.ExifReader((src_file,))
   int_param_dict = reader.read_int_param()
   ext_param_dict = reader.read_ext_param()

   # extract exterior parameters for src_file, and interior parameters for src_file's
   # camera
   ext_params = ext_param_dict[Path(src_file).name]
   int_params = int_param_dict[ext_params.pop('camera')]

   # create camera from interior & exterior parameters
   camera = oty.create_camera(**int_params, **ext_params)

   # orthorectify src_file with dem_file, camera & exterior parameter ('world') CRS
   ortho_file = 'ortho.tif'
   ortho = oty.Ortho(src_file, dem_file, camera, crs=reader.crs)
   ortho.process(ortho_file)

Documentation
-------------

There is usage and reference documentation at
`orthority.readthedocs.io <https://orthority.readthedocs.io/>`__.

License
-------

``orthority`` is licensed under the `Mozilla Public License 2.0 <LICENSE>`__.

Acknowledgements
----------------

Special thanks to `Yu-Huang
Wang <https://community.opendronemap.org/t/2019-04-11-tuniu-river-toufeng-miaoli-county-taiwan/3292>`__
& `OpenDroneMap <https://opendronemap.org/>`__, `National Geo-spatial
Information <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`__ and
the `Centre for Geographical Analysis <https://www0.sun.ac.za/cga/>`__ for sharing imagery, DEM and
aero-triangulation data that form part of the package test data.

.. |Tests| image:: https://github.com/leftfield-geospatial/simple-ortho/actions/workflows/run-unit-tests_pypi.yml/badge.svg
   :target: https://github.com/leftfield-geospatial/simple-ortho/actions/workflows/run-unit-tests_pypi.yml
.. |codecov| image:: https://codecov.io/gh/leftfield-geospatial/simple-ortho/branch/main/graph/badge.svg?token=YPZAQS4S15
   :target: https://codecov.io/gh/leftfield-geospatial/simple-ortho
.. |License: MPL v2| image:: https://img.shields.io/badge/License-MPL_v2-blue.svg
   :target: https://www.mozilla.org/en-US/MPL/2.0/
