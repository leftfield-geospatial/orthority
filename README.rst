|Tests| |codecov| |PyPI version| |conda-forge version| |docs| |License: AGPL v3|

Orthority
=========

.. image:: https://raw.githubusercontent.com/leftfield-geospatial/orthority/main/docs/readme_banner.webp
   :alt: banner

.. description_start

Orthority provides a command line toolkit and Python API for orthorectifying drone, aerial and satellite imagery, given a camera model and DEM. It supports common frame, and RPC camera models. Camera parameters can be read from various file formats, or image tags.  Related algorithms including RPC refinement and pan-sharpening, are also provided.

.. description_end

.. installation_start

Installation
------------

Orthority is a python 3 package that can be installed with `pip <https://pip.pypa.io/>`_ or `conda <https://docs.anaconda.com/free/miniconda/>`_.

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
-  ``rpc``: Orthorectify images with RPC camera models defined by image tags / sidecar files or parameter files.
-  ``sharpen``: Pan-sharpen an image using the Gram-Schmidt method.

Get help on ``oty`` with:

.. code-block:: bash

   oty --help

and help on an ``oty`` sub-command with:

.. code-block:: bash

   oty <sub-command> --help

.. cli_end

Options for configuring output images are common to all sub-commands.

.. note::

    The ``simple-ortho`` command is deprecated and will be removed in future.  Please switch to ``oty`` and its sub-commands.

Examples
^^^^^^^^

Orthorectify ``source.tif`` with the DEM in ``dem.tif``, and frame camera model defined by ``int_param.yaml`` and ``ext_param.geojson`` interior and exterior parameters:

.. code-block:: bash

   oty frame --dem dem.tif --int-param int_param.yaml --ext-param ext_param.geojson source.tif

Orthorectify ``source.tif`` with the DEM in ``dem.tif``, and frame camera model defined by ``source.tif`` EXIF / XMP tags:

.. code-block:: bash

   oty exif --dem dem.tif source.tif

As above, but the create the ortho image with ``bilinear`` interpolation, a 0.5 m pixel size and ``deflate`` compression:

.. code-block:: bash

   oty exif --dem dem.tif --interp bilinear --res 0.5 --compress deflate source.tif

Orthorectify images in the OpenDroneMap dataset ``odm``, with the dataset DSM and camera models.  Ortho images are placed in ``odm/orthority``.

.. code-block:: bash

   oty odm --dataset-dir odm --out-dir odm/orthority
   
Orthorectify ``source.tif`` with the DEM in ``dem.tif``, and RPC camera model defined by ``source.tif`` tags / sidecar files:
   
.. code-block:: bash

   oty rpc --dem dem.tif source.tif

As above, but refine the RPC camera model with GCPs in ``source.tif`` tags:

.. code-block:: bash

   oty rpc --dem dem.tif --gcp-refine tags source.tif

Pan-sharpen the multispectral image ``ms.tif`` with the panchromatic image ``pan.tif``:

.. code-block:: bash

   oty sharpen  --pan pan.tif --multispectral ms.tif --out-file pan_sharp.tif


API
~~~

Orthorectify an image using interior and exterior parameter files to generate the camera model:

.. below copied from docs/scripts/api_ortho.py

.. code-block:: python

    import orthority as oty

    # URLs of required files
    url_root = 'https://raw.githubusercontent.com/leftfield-geospatial/orthority/main/tests/data/'
    src_file = url_root + 'ngi/3324c_2015_1004_05_0182_RGB.tif'  # aerial image
    dem_file = url_root + 'ngi/dem.tif'  # DEM covering imaged area
    int_param_file = url_root + 'io/ngi_int_param.yaml'  # interior parameters
    ext_param_file = url_root + 'io/ngi_xyz_opk.csv'  # exterior parameters

    # create a camera model for src_file from interior & exterior parameters
    cameras = oty.FrameCameras(int_param_file, ext_param_file)
    camera = cameras.get(src_file)

    # create Ortho object and orthorectify
    ortho = oty.Ortho(src_file, dem_file, camera=camera, crs=cameras.crs)
    ortho.process('ortho.tif')


Pan-sharpen a multispectral image with the matching panchromatic image:

.. below copied from docs/scripts/api_pan_sharp.py

.. code-block:: python

    import orthority as oty

    # URLs of required files
    url_root = 'https://raw.githubusercontent.com/leftfield-geospatial/orthority/main/tests/data/'
    pan_file = url_root + 'pan_sharp/pan.tif'  # panchromatic drone image
    ms_file = url_root + 'pan_sharp/ms.tif'  # multispectral (RGB) drone image

    # create PanSharpen object and pan-sharpen
    pan_sharp = oty.PanSharpen(pan_file, ms_file)
    pan_sharp.process('pan_sharp.tif')

Documentation
-------------

See `orthority.readthedocs.io <https://orthority.readthedocs.io/>`__ for usage and reference documentation.

Contributing
------------

Contributions are welcome - the online documentation has a `guide <https://orthority.readthedocs.io/en/latest/contributing.html>`__.  Please report bugs and make feature requests with the `github issue tracker <https://github.com/leftfield-geospatial/orthority/issues>`__.

Licensing
---------

Orthority is licensed under the `GNU Affero General Public License v3.0 (AGPLv3) <LICENSE>`__.

Portions of the `AGPLv3 <https://github.com/OpenDroneMap/ODM/blob/master/LICENSE>`__ licensed `OpenDroneMap software <https://github.com/OpenDroneMap/ODM>`__, and `BSD-style <https://github.com/mapillary/OpenSfM/blob/main/LICENSE>`__ licensed `OpenSfM library <https://github.com/mapillary/OpenSfM>`__ have been adapted and included in the Orthority package.

Acknowledgements
----------------

Special thanks to `Yu-Huang Wang <https://community.opendronemap.org/t/2019-04-11-tuniu-river-toufeng-miaoli-county-taiwan/3292>`__ & the `OpenDroneMap Community <https://community.opendronemap.org/>`__, `National Geo-spatial Information <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`__ and the `Centre for Geographical Analysis <https://www0.sun.ac.za/cga/>`__ for sharing imagery, DEM and aero-triangulation data that form part of the package test data.

.. |Tests| image:: https://github.com/leftfield-geospatial/orthority/actions/workflows/run-unit-tests_pypi.yml/badge.svg
   :target: https://github.com/leftfield-geospatial/orthority/actions/workflows/run-unit-tests_pypi.yml
.. |codecov| image:: https://codecov.io/gh/leftfield-geospatial/orthority/branch/main/graph/badge.svg?token=YPZAQS4S15
   :target: https://codecov.io/gh/leftfield-geospatial/orthority
.. |PyPI version| image:: https://img.shields.io/pypi/v/orthority?color=blue
   :target: https://pypi.org/project/orthority/

.. |conda-forge version| image:: https://img.shields.io/conda/vn/conda-forge/orthority.svg?color=blue
   :alt: conda-forge
   :target: https://anaconda.org/conda-forge/orthority

.. |docs| image:: https://readthedocs.org/projects/orthority/badge/?version=latest
    :target: https://orthority.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |License: AGPL v3| image:: https://img.shields.io/badge/License-AGPL_v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0
