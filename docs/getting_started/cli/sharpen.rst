.. include:: shared.txt

``oty sharpen``
===============

|oty sharpen|_ implements the Gram-Schmidt pan sharpening method.  This example pan sharpens a multispectral (RGB) drone image with its matching panchromatic image:

.. code-block:: bash

    oty sharpen --pan pan_sharp/pan.tif --multispectral pan_sharp/ms.tif --out-file pan_sharp.tif

Multispectral band indexes for sharpening can be specified with :option:`--ms-index <oty-sharpen --ms-index>`, and multispectral to panchromatic weights with :option:`--weight <oty-sharpen --weight>`.  By default all non-alpha multispectral bands are used, and weights are estimated from the images.

.. code-block:: bash

    oty sharpen --pan pan_sharp/pan.tif --multispectral pan_sharp/ms.tif --out-file pan_sharp.tif --ms-index 1 --ms-index 2 --ms-index 3 --weight 1 --weight 1 --weight 1
