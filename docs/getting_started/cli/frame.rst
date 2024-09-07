.. include:: ../../shared.txt
.. include:: shared.txt

``oty frame``
=============

|oty frame|_ uses interior and exterior parameter files to specify :ref:`frame camera models <background/camera_models:frame cameras>`.  Here we orthorectify NGI_ aerial images using the associated DEM, :doc:`Orthority format <../../file_formats/oty_int>` interior parameters, and :doc:`CSV format <../../file_formats/csv>` exterior parameters.  Ortho images are placed in the current directory:

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

In the first example, the world / ortho CRS is read from exterior parameters in CSV format with ``x``, ``y`` & ``z`` fields and a  :file:`.prj` sidecar file.

In the next example, exterior parameters are in CSV format with ``x``, ``y`` & ``z`` fields, but no :file:`.prj` sidecar file.  The world / ortho CRS cannot be read from the input files, so it must be specified with :option:`--crs <oty-frame --crs>`:

.. code-block:: bash

    oty frame --dem odm/odm_dem/dsm.tif --int-param odm/opensfm/reconstruction.json --ext-param io/odm_xyz_opk.csv --crs EPSG:32651 odm/images/*.tif

.. note::

    See the :doc:`file format documentation <../../file_formats/index>` for supported frame camera parameter formats.

