.. include:: shared.txt
.. include:: ../../shared.txt

Ortho image configuration
=========================

Ortho images are created as GeoTIFFs.  Image resolution and format (data type, compression, nodata / internal mask and overviews) can be configured.  Configuration options are common to all |oty|_ orthorectification sub-commands and default to sensible values when not supplied.

Ortho resolution defaults to an estimate of the `ground sampling distance <https://en.wikipedia.org/wiki/Ground_sample_distance>`__.  This can be changed with ``--res``.  The ortho data type defaults to the source image data type, and can be changed with ``--dtype``.

Compression can be configured with ``--compress`` as either ``deflate`` (with any ortho data type) or ``jpeg`` (with the ``uint8`` or ``uint16`` ortho data types).  If ``--compress`` is not specified, compression defaults to ``jpeg`` when the ortho data type is ``uint8``, otherwise it defaults to ``deflate``.  When ``jpeg`` compression is used with the ``uint16`` data type, the ortho is 12 bit ``jpeg`` compressed.

.. note::

    Support for 12 bit JPEG compression is Rasterio_ build / package dependent.

The next example orthorectifies using EXIF / XMP tags, and configures the ortho image with a 0.2m resolution, the ``uint8`` data type, and ``deflate`` compression:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --res 0.2 --dtype uint8 --compress deflate odm/images/100_0005_0140.tif

Valid ortho pixels are masked with either an internal mask band or a nodata value.  By default, an internal mask is used when the ortho image is ``jpeg`` compressed.  This avoids ``jpeg`` artefacts in invalid areas.  When the ortho is ``deflate`` compressed, the default is use to a nodata value based on the data type.  Masking behaviour can be changed with ``--write-mask`` to write an internal mask, or ``--no-write-mask`` to use a nodata value.

Internal overviews are added by default.  This can be changed with ``--no-build-ovw``.  In this example, we create an ortho image with ``deflate`` compression, nodata rather than internal masking, and no internal overviews:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --compress deflate --no-write-mask --no-build-ovw odm/images/100_0005_0140.tif

