.. include:: shared.txt
.. include:: ../../shared.txt

Ortho image configuration
=========================

Ortho image resolution and format can be configured.  Configuration options are common to all |oty|_ orthorectification sub-commands and default to sensible values when not supplied.

Resolution, data type and compression
-------------------------------------

Ortho resolution defaults to an estimate of the `ground sampling distance <https://en.wikipedia.org/wiki/Ground_sample_distance>`__.  This can be changed with ``--res``.  The ortho data type defaults to the source image data type, and can be changed with ``--dtype``.

Compression can be configured with ``--compress`` as either ``deflate`` or ``lzw`` (with any ortho data type), or ``jpeg`` (with the ``uint8`` or ``uint16`` ortho data types).  If ``--compress`` is not specified, compression defaults to ``jpeg`` when the ortho data type is ``uint8``, and to ``deflate`` otherwise.  When ``jpeg`` compression is used with the ``uint16`` data type, the ortho is 12 bit ``jpeg`` compressed.

.. note::

    Support for 12 bit JPEG compression is Rasterio_ build / package dependent.

The next example orthorectifies using EXIF / XMP tags, and configures the ortho image with a 0.2m resolution, the ``uint8`` data type, and ``deflate`` compression:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --res 0.2 --dtype uint8 --compress deflate odm/images/100_0005_0140.tif

Masking and overviews
---------------------

Valid ortho pixels are masked with either an internal mask band or a nodata value.  By default, an internal mask is used when the ortho image is ``jpeg`` compressed.  This avoids ``jpeg`` artefacts in invalid areas.  When the ortho is ``deflate`` or ``lzw`` compressed, the default is use to a nodata value based on the data type.  Masking behaviour can be changed with ``--write-mask`` to write an internal mask, or ``--no-write-mask`` to use a nodata value.

Internal overviews are added by default.  This can be changed with ``--no-build-ovw``.  In this example, we create an ortho image with ``deflate`` compression, internal masks, and no internal overviews:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --compress deflate --write-mask --no-build-ovw odm/images/100_0005_0140.tif

Custom creation options and driver
----------------------------------

For ortho image configurations not possible with the above options, custom creation options can be specified with ``--creation-option``.  The ``--compress`` option is ignored, and no other creation options are set by Orthority when this is supplied.

The ortho can be formatted as a ``gtiff`` (GeoTIFF - the default) or ``cog`` (Cloud Optimised GeoTIFF) with ``--driver``.

This example formats the ortho as a GeoTIFF with internal masks, and specifies custom creation options for tiled YCbCr JPEG compression with a quality of 90:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --write-mask --driver gtiff --creation-option tiled=yes --creation-option compress=jpeg --creation-option photometric=ycbcr --creation-option jpeg_quality=90 odm/images/100_0005_0140.tif

.. note::

    Each driver has its own creation options.  See the GDAL `GeoTIFF <https://gdal.org/en/latest/drivers/raster/gtiff.html#creation-options>`__ and `COG <https://gdal.org/en/latest/drivers/raster/cog.html#creation-options>`__ docs for details.
