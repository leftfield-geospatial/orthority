|Tests| |codecov| |License: AGPL v3|

Orthority
=========

.. image:: https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/main/docs/readme_banner.webp
   :alt: banner

.. description_start

Orthority provides a command line interface and Python API for orthorectifying drone and aerial imagery, given a camera model and DEM. It supports common frame camera models. Camera parameters can be read from various file formats, or image EXIF / XMP tags.

.. description_end

.. installation_start

Installation
------------

Orthority is a python 3 package that can be installed with `pip <https://pip.pypa.io/>`_ or `conda <https://docs.conda.io/projects/miniconda>`_.

pip
~~~

.. code-block:: bash

   pip install orthority

conda
~~~~~

.. code-block:: bash

   conda install -c conda-forge orthority

.. installation_end

Quick start
-----------

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

.. cli_start

Orthority command line functionality is accessed with the ``oty`` command, and its sub-commands:

-  ``frame``: Orthorectify images with frame camera model(s) defined by interior and exterior parameter files.
-  ``exif``: Orthorectify images with frame camera model(s) defined by image EXIF / XMP tags.
-  ``odm``: Orthorectify images in a processed OpenDroneMap dataset that includes a DSM.

Get help on ``oty`` with:

.. code-block:: bash

   oty --help

and help on an ``oty`` sub-command with:

.. code-block:: bash

   oty <sub-command> --help

.. cli_end

Options for the output files and orthorectification algorithm are common to all orthorectification sub-commands.

.. note::

    The ``simple-ortho`` command is deprecated and will be removed in future.  Please switch to ``oty`` and its sub-commands.

Examples
^^^^^^^^

Orthorectify *source.tif* with the DEM in *dem.tif*, and frame camera model defined by *int_param.yaml* and *ext_param.geojson* interior and exterior parameters:

.. code-block:: bash

   oty frame --dem dem.tif --int-param int_param.yaml --ext-param ext_param.geojson source.tif

Orthorectify *source.tif* with the DEM in *dem.tif*, and frame camera model defined by *source.tif* EXIF / XMP tags:

.. code-block:: bash

   oty exif --dem dem.tif source.tif

As above, but the create the ortho image with *bilinear* interpolation, a 0.5 m pixel size and *deflate* compression:

.. code-block:: bash

   oty exif --dem dem.tif --interp bilinear --res 0.5 --compress deflate source.tif

Orthorectify images in the OpenDroneMap dataset *odm*, with the dataset DSM and camera models.  Ortho images are placed in *odm/orthority*.

.. code-block:: bash

   oty odm --dataset-dir odm --out-dir odm/orthority

API
~~~

Orthorectify an image using interior and exterior parameter files to generate the camera model:

.. below copied from docs/scripts/overview.py

.. code-block:: python

    import orthority as oty

    # URLs of required files
    url_root = (
        'https://raw.githubusercontent.com/leftfield-geospatial/simple-ortho/feature_docs/tests/data/'
    )
    src_file = url_root + 'ngi/3324c_2015_1004_05_0182_RGB.tif'
    dem_file = url_root + 'ngi/dem.tif'
    int_param_file = url_root + 'io/ngi_int_param.yaml'
    ext_param_file = url_root + 'io/ngi_xyz_opk.csv'

    # create a camera for src_file from interior / exterior parameters
    cameras = oty.FrameCameras(int_param_file, ext_param_file)
    camera = cameras.get(src_file)

    # orthorectify src_file with dem_file, the created camera & world / ortho CRS
    ortho = oty.Ortho(src_file, dem_file, camera=camera, crs=cameras.crs)
    ortho.process('ortho.tif')

Documentation
-------------

See `orthority.readthedocs.io <https://orthority.readthedocs.io/>`__ for usage and reference documentation.

Contributing
------------

Contributions are welcome. There is a guide in the `documentation <https://orthority.readthedocs.io/contributing>`__. Please report bugs and make feature requests with the `github issue tracker <https://github.com/leftfield-geospatial/simple-ortho/issues>`__.

Licensing
---------

Orthority is licensed under the `GNU Affero General Public License v3.0 (AGPLv3) <LICENSE>`__.

Portions of the `AGPLv3 <https://github.com/OpenDroneMap/ODM/blob/master/LICENSE>`__ licensed `OpenDroneMap software <https://github.com/OpenDroneMap/ODM>`__, and `BSD-style <https://github.com/mapillary/OpenSfM/blob/main/LICENSE>`__ licensed `OpenSfM library <https://github.com/mapillary/OpenSfM>`__ have been adapted and included in the Orthority package.

Acknowledgements
----------------

Special thanks to `Yu-Huang Wang <https://community.opendronemap.org/t/2019-04-11-tuniu-river-toufeng-miaoli-county-taiwan/3292>`__ & the `OpenDroneMap Community <https://community.opendronemap.org/>`__, `National Geo-spatial Information <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`__ and the `Centre for Geographical Analysis <https://www0.sun.ac.za/cga/>`__ for sharing imagery, DEM and aero-triangulation data that form part of the package test data.

.. |Tests| image:: https://github.com/leftfield-geospatial/simple-ortho/actions/workflows/run-unit-tests_pypi.yml/badge.svg
   :target: https://github.com/leftfield-geospatial/simple-ortho/actions/workflows/run-unit-tests_pypi.yml
.. |codecov| image:: https://codecov.io/gh/leftfield-geospatial/simple-ortho/branch/main/graph/badge.svg?token=YPZAQS4S15
   :target: https://codecov.io/gh/leftfield-geospatial/simple-ortho
.. |License: AGPL v3| image:: https://img.shields.io/badge/License-AGPL_v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0
