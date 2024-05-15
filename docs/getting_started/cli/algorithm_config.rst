.. include:: shared.txt

Algorithm configuration
=======================

.. note::

    An occlusion masking option will be added in a future release.

Options for configuring the orthorectification algorithm can be used with any of the |oty|_ orthorectification sub-commands.  Their default values will work for most use cases.

Interpolation methods for remapping the source image and reprojecting the DEM are configurable with ``--interp`` and ``--dem-interp`` respectively.  They default to :attr:`~orthority.enums.Inter.cubic`.  The ortho image is generated all bands at once by default.  Changing to band-by-band generation with ``--per-band`` is slower but reduces memory usage for some source image formats.  The next example orthorectifies the OpenDroneMap dataset using :attr:`~orthority.enums.Inter.lanczos` interpolation and band-by-band ortho generation:

.. code-block:: bash

    oty odm --dataset-dir odm --interp lanczos --dem-interp lanczos --per-band

