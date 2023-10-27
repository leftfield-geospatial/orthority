|Tests| |codecov| |License: AGPL v3|

Orthority
=========

.. image:: https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/main/docs/readme_banner.webp
   :alt: banner

Orthority provides a command line interface and Python API for orthorectifying drone and aerial
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

Orthority command line functionality is accessed with the ``oty`` command, and its sub-commands:

-  ``ortho``: Orthorectify with camera model(s) defined by interior and exterior parameter files.
-  ``exif``: Orthorectify with camera model(s) defined by image EXIF / XMP tags.
-  ``odm``: Orthorectify with OpenDroneMap camera models and DSM.

Get help on ``oty`` with:

.. code:: shell

   oty --help

and help on an ``oty`` sub-command with:

.. code:: shell

   oty <sub-command> --help

Options for the orthorectification algorithm and ortho image format are common to all sub-commands.

Examples
^^^^^^^^

Orthorectify *source.tif* with the DEM in *dem.tif*, and camera model defined by *int_param.yaml*
and *ext_param.geojson* interior and exterior parameters:

.. code:: shell

   oty ortho --dem dem.tif --int-param int_param.yaml --ext-param ext_param.geojson source.tif

Orthorectify *source.tif* with the DEM in *dem.tif*, and camera model defined by *source.tif* EXIF /
XMP tags:

.. code:: shell

   oty exif --dem dem.tif source.tif

As above, but the create the ortho image with *bilinear* interpolation, a 0.5 m pixel size and
*deflate* compression:

.. code:: shell

   oty exif --dem dem.tif --interp bilinear --res 0.5 --compress deflate source.tif

Orthorectify images in the OpenDroneMap dataset *odm_data*, with the dataset DSM and camera models.
Ortho images are placed in *odm_data/orthority*.

.. code:: shell

   oty odm --dataset-dir odm_data --out-dir odm_data/orthority

API
~~~

Orthorectify an image with the camera model defined by its EXIF / XMP tags:

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

   # extract exterior parameters for src_file, and interior parameters for
   # src_file's camera
   ext_params = ext_param_dict[Path(src_file).name]
   int_params = int_param_dict[ext_params.pop('camera')]

   # create camera from interior & exterior parameters
   camera = oty.create_camera(**int_params, **ext_params)

   # orthorectify src_file with dem_file, the created camera & exterior parameter
   # ('world') CRS
   ortho = oty.Ortho(src_file, dem_file, camera, crs=reader.crs)
   ortho.process('ortho.tif')

Documentation
-------------

See `orthority.readthedocs.io <https://orthority.readthedocs.io/>`__ for usage and reference
documentation.

Contributing
------------

Contributions are welcome! There is a guide for developers in the `documentation
<https://orthority.readthedocs.io/contributing>`__. Please report bugs and make
feature requests with the `github issue tracker
<https://github.com/leftfield-geospatial/simple-ortho/issues>`__.

Licensing
---------

Orthority is licensed under the `AGPLv3 <LICENSE>`__.

Portions of the `AGPLv3 <https://github.com/OpenDroneMap/ODM/blob/master/LICENSE>`__ licensed
`OpenDroneMap software <https://github.com/OpenDroneMap/ODM>`__, and
`BSD-style <https://github.com/mapillary/OpenSfM/blob/main/LICENSE>`__ licensed `OpenSfM
library <https://github.com/mapillary/OpenSfM>`__ have been adapted and included in the Orthority
package.

Acknowledgements
----------------

Special thanks to `Yu-Huang
Wang <https://community.opendronemap.org/t/2019-04-11-tuniu-river-toufeng-miaoli-county-taiwan/3292>`__
& the `OpenDroneMap Community <https://community.opendronemap.org/>`__, `National Geo-spatial
Information <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`__ and
the `Centre for Geographical Analysis <https://www0.sun.ac.za/cga/>`__ for sharing imagery, DEM and
aero-triangulation data that form part of the package test data.

.. |Tests| image:: https://github.com/leftfield-geospatial/simple-ortho/actions/workflows/run-unit-tests_pypi.yml/badge.svg
   :target: https://github.com/leftfield-geospatial/simple-ortho/actions/workflows/run-unit-tests_pypi.yml
.. |codecov| image:: https://codecov.io/gh/leftfield-geospatial/simple-ortho/branch/main/graph/badge.svg?token=YPZAQS4S15
   :target: https://codecov.io/gh/leftfield-geospatial/simple-ortho
.. |License: AGPL v3| image:: https://img.shields.io/badge/License-AGPL_v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0
