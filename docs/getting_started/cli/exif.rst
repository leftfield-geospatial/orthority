.. include:: shared.txt

``oty exif``
============

Frame camera models can be derived from image EXIF / XMP tags with |oty exif|_.  The :doc:`EXIF / XMP documentation <../../file_formats/exif_xmp>` describes the required tags.  This example orthorectifies drone imagery using an OpenDroneMap generated DSM.  Ortho images are placed in the current directory:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif odm/images/*.tif

The world / ortho CRS defaults to a UTM CRS whose zone covers the camera positions.  This can be changed with :option:`--crs <oty-exif --crs>`:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif --crs EPSG:32651 odm/images/*.tif

.. note::

    EXIF / XMP tag values typically contain inaccuracies that can result in distortion and positioning errors in the ortho images.  This is the case with the Orthority test data, and a general limitation of this approach.
