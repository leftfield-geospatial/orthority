Ortho image configuration
-------------------------

Ortho images are created as GeoTIFFs.  Options for configuring image resolution and format (data type, compression, nodata / internal mask and overviews) are common to all |oty|_ sub-commands.  These options default to sensible values when not supplied.

The ortho resolution defaults to an estimate of the `ground sampling distance <https://en.wikipedia.org/wiki/Ground_sample_distance>`__.  This can be changed with ``--res``.  The ortho data type defaults to the source image data type, and can be changed with ``--dtype``.  When the ortho data type is ``uint8``, compression defaults to ``jpeg``, otherwise it defaults to ``deflate``.  Compression can be changed from its default changed with ``--compress``.  In the next example, we orthorectify an image (using its EXIF / XMP tags), configuring the ortho image with a 0.2m resolution, the ``uint8`` data type, and ``deflate`` compression:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --res 0.2 --dtype uint8 --compress deflate --overwrite odm/images/100_0005_0140.tif

By default, valid ortho pixels are masked with an internal mask when the ortho image is ``jpeg`` compressed.  This avoids ``jpeg`` artefacts in invalid areas.  For ``deflate`` compression, the default is to mask valid pixels with a nodata value based on the data type.  Default internal masking / nodata behaviour can be changed with ``--write-mask`` / ``--no-write-mask``.  Internal overviews are added by default.  This can changed with ``--no-build-ovw``.  In this example, we create an ortho image with ``deflate`` compression, nodata rather than internal masking, and no internal overviews:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --compress deflate --no-write-mask --no-build-ovw --overwrite odm/images/100_0005_0140.tif

.. |oty| replace:: ``oty``
.. _oty: ../../cli/oty.html

