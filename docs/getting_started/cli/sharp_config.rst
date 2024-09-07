.. include:: shared.txt
.. include:: ../../shared.txt

Output image configuration
==========================

Pan sharpened images are created as GeoTIFFs.  Image format (data type, compression, nodata / internal mask and overviews) can be configured.  Configuration options are shared with the |oty|_ orthorectification sub-commands, and default to sensible values when not supplied.

The image data type defaults to the multispectral image data type, and can be changed with ``--dtype``.  Compression can be configured with ``--compress`` as either ``deflate`` (with any image data type) or ``jpeg`` (with the ``uint8`` or ``uint16`` image data types).  If ``--compress`` is not specified, compression defaults to ``jpeg`` when the image data type is ``uint8``, otherwise it defaults to ``deflate``.  When ``jpeg`` compression is used with the ``uint16`` data type, the image is 12 bit ``jpeg`` compressed.

.. note::

    Support for 12 bit JPEG compression is Rasterio_ build / package dependent.

The next example creates a pan sharpened image with the ``int16`` data type, and ``deflate`` compression:

.. code-block:: bash

    oty sharpen --pan pan_sharp/pan.tif --multispectral pan_sharp/ms.tif --out-file pan_sharp.tif --dtype int16 --compress deflate

Valid image pixels are masked with either an internal mask band or a nodata value.  By default, an internal mask is used when the image image is ``jpeg`` compressed.  This avoids ``jpeg`` artefacts in invalid areas.  When the image is ``deflate`` compressed, the default is use to a nodata value based on the data type.  Masking behaviour can be changed with ``--write-mask`` to write an internal mask, or ``--no-write-mask`` to use a nodata value.

Internal overviews are added by default.  This can be changed with ``--no-build-ovw``.  In this example, we create an pan sharpened image with ``deflate`` compression, internal masking rather than nodata, and no internal overviews:

.. code-block:: bash

    oty sharpen --pan pan_sharp/pan.tif --multispectral pan_sharp/ms.tif --out-file pan_sharp.tif --compress deflate --write-mask --no-build-ovw
