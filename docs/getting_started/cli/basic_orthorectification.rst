.. include:: ../../shared.txt

Basic orthorectification
========================

|oty|_ orthorectification sub-commands allow :doc:`camera models <../../background/camera_models>` to be specified in different ways.

``oty frame``
-------------

|oty frame|_ uses interior and exterior parameter files to specify :ref:`frame camera models <background/camera_models:frame cameras>`.  Here we orthorectify `NGI <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`__ aerial images using the associated DEM, :doc:`Orthority format <../../file_formats/oty_int>` interior parameters, and :doc:`CSV format <../../file_formats/csv>` exterior parameters.  Ortho images are placed in the current directory:

.. code-block:: bash

    oty frame --dem ngi/dem.tif --int-param io/ngi_int_param.yaml --ext-param io/ngi_xyz_opk.csv ngi/*RGB.tif

In order of priority, the world / ortho CRS is read from:

#. :option:`--crs <oty-frame --crs>` option, if supplied.
#. Exterior parameters, if in:
    * :doc:`Orthority format <../../file_formats/oty_ext>`.
    * :doc:`OpenDroneMap / OpenSfM <../../file_formats/opensfm>` :file:`reconstruction.json` format.
    * :doc:`CSV format <../../file_formats/csv>` with ``x``, ``y`` & ``z`` fields and a :file:`.prj` sidecar file.
    * :doc:`CSV format <../../file_formats/csv>` with ``latitude``, ``longitude``, ``altitude``, ``roll``, ``pitch`` & ``yaw`` fields.
#. Source image CRS metadata, if any.

In the first example, the world / ortho CRS is read from the CSV file's :file:`.prj` sidecar file.

The next example orthorectifies drone images with `OpenDroneMap <https://github.com/OpenDroneMap/ODM>`__ generated DEM and interior parameters.  Exterior parameters are in CSV format with ``x``, ``y`` & ``z`` fields, but no :file:`.prj` sidecar file.  The world / ortho CRS cannot be read from the input files, so it is specified with :option:`--crs <oty-frame --crs>`:

.. code-block:: bash

    oty frame --dem odm/odm_dem/dsm.tif --int-param odm/opensfm/reconstruction.json --ext-param io/odm_xyz_opk.csv --crs EPSG:32651 odm/images/*.tif

See the :doc:`file format documentation <../../file_formats/index>` for details on supported frame camera parameter formats.

``oty exif``
------------

Frame camera models can be derived from image EXIF / XMP tags with |oty exif|_.  The :doc:`EXIF / XMP documentation <../../file_formats/exif_xmp>` describes the required tags.  This example orthorectifies drone imagery using an `OpenDroneMap <https://github.com/OpenDroneMap/ODM>`__ generated DSM.  Ortho images are placed in the current directory:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif odm/images/*.tif

The world / ortho CRS defaults to a UTM CRS whose zone covers the camera positions.  This can be changed with :option:`--crs <oty-exif --crs>`:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --crs EPSG:32651 odm/images/*.tif

.. note::

    EXIF / XMP tag values typically contain inaccuracies that can result in distortion and positioning errors in the ortho images.  This is a general limitation of this approach.

``oty odm``
-----------

|oty odm|_ orthorectifies images in an `OpenDroneMap <https://github.com/OpenDroneMap/ODM>`__ generated dataset using the dataset DSM (:file:`{dataset}/odm_dem/dsm.tif`) and camera models (:file:`{dataset}/opensfm/reconstruction.json`).  Here we orthorectify images in the :file:`odm` dataset.  Ortho images are placed in the :file:`{dataset}/orthority` directory:

.. code-block:: bash

    oty odm --dataset-dir odm

Without the :option:`--crs <oty-odm --crs>` option, the world / ortho CRS is read from the OpenDroneMap DSM.  This is the recommended setting.

``oty rpc``
------------

RPC camera model(s) can be derived from :doc:`image tags / sidecar files <../../file_formats/image_rpc>` or an :doc:`Orthority RPC parameter file <../../file_formats/oty_rpc>` with |oty rpc|_.  This example orthorectifies a satellite image with RPC tags using the NGI DEM of the same area.  Ortho images are placed in the current directory:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif rpc/qb2_basic1b.tif

The same image can be orthorectified with the model defined by a RPC parameter file:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif --rpc-param rpc/rpc_param.yaml rpc/qb2_basic1b.tif

The world / ortho CRS defaults to the WGS84 geographic 3D CRS.  This can be changed with  :option:`--crs <oty-rpc --crs>`:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif --crs EPSG:32735 rpc/qb2_basic1b.tif


Output files
------------

Ortho images are named automatically based on the source image names.  The output directory for ortho images and :doc:`exported files <model_export>` can be changed from its default with the ``--out-dir`` option.  Passing ``--overwrite`` overwrites existing files.  These options are common to all |oty|_ orthorectification sub-commands.  E.g., this repeats the :ref:`getting_started/cli/basic_orthorectification:``oty odm``` example, creating and using :file:`ortho` as the output directory, and overwriting existing files:

.. code-block:: bash

    mkdir ortho
    oty odm --dataset-dir odm --out-dir ortho --overwrite
