CSV exterior parameters
=======================

CSV files can be used to specify frame camera :ref:`exterior parameters <background/camera_models:exterior parameters>` with a row per source image file, and column fields for filename, camera position, camera orientation and camera ID.

Fields
------

Field contents can be specified with a file header, or the :meth:`~orthority.param_io.CsvReader` ``fieldnames`` argument if using the API. Recognised field names are:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Field name(s)
      - Description
    * - ``filename``
      - Image file name excluding parent path, with or without extension.  Case sensitive.
    * - ``x``, ``y``, ``z``
      - Camera position in :ref:`world / ortho coordinates <background/coordinates:world coordinates>`.
    * - ``latitude``, ``longitude``, ``altitude``
      - Camera position in geographic coordinates.
    * - ``omega``, ``phi``, ``kappa``
      - Camera orientation angles.
    * - ``roll``, ``pitch``, ``yaw``
      - Camera orientation angles.
    * - ``camera``
      - ID of camera :ref:`interior parameters <background/camera_models:interior parameters>` (optional).

Fields should include ``filename``, camera position as either ``x``, ``y`` & ``z`` or ``latitude``, ``longitude`` & ``altitude``, and camera orientation as either ``omega``, ``phi`` & ``kappa`` or ``roll``, ``pitch`` & ``yaw``.  The ``camera`` field should be included for multi-camera setups, but is otherwise optional. Other fields not in the table can be included, but will be ignored.

For example, this is a valid file with a header:

.. literalinclude:: ../../tests/data/io/ngi_xyz_opk.csv
   :language: text
   :lines: 1-3

When there is no file header (and ``fieldnames`` is not provided to :meth:`~orthority.param_io.CsvReader` if using the API), fields are assumed to be in the legacy ``simple-ortho`` format: ``filename``, ``x``, ``y``, ``z``, ``omega``, ``phi``, ``kappa`` (in that order).

.. note::

    Conversion from (*roll*, *pitch*, *yaw*) camera orientation is approximate and relies on the :ref:`world <background/coordinates:world coordinates>` CRS having minimal distortion in the imaged area.

Dialect
-------

By default the file dialect (i.e. delimiter, line terminator, quote character etc.) is automatically detected.  It can also be specified with the :meth:`~orthority.param_io.CsvReader` ``dialect`` argument if using the API.  Comma, space, semicolon, colon or tab delimiters; and windows or unix line terminators are supported.  Field values can optionally be enclosed in single or double quotes.  This is required with e.g. the ``filename`` or ``camera`` fields if values contain the CSV delimiter.

This example shows a valid space delimited file with a header, and camera ID values in quotes:

.. literalinclude:: ../../tests/data/io/odm_lla_rpy.csv
   :language: text
   :lines: 1-3

Angle units and ``.prj`` file
-----------------------------

Orientation angles are assumed to be in degrees by default.  The :option:`--radians <oty-frame --radians>` option on the command line, or the :meth:`~orthority.param_io.CsvReader` ``radians`` argument in the API should be supplied if angles are in radians.

Optionally, the CRS of ``x``, ``y``, ``z`` world coordinate positions can be supplied in a sidecar :file:`.prj` file (i.e. a text file containing a WKT, proj4 or EPSG string, and having the same path as the CSV file, but a :file:`.prj` extension).  In this case, the world / ortho CRS does not have to be supplied with the :option:`--crs <oty-frame --crs>` command line option, or :meth:`~orthority.param_io.CsvReader` ``crs`` API argument.

.. note::

    See the `test data <https://github.com/leftfield-geospatial/simple-ortho/tree/main/tests/data/io>`__ for other CSV file examples.

