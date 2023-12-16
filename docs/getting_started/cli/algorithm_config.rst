Algorithm configuration
-----------------------

Various options allow configuration of the orthorectification algorithm.  Again, these are common to all |oty|_ sub-commands.  For most use cases, these options can be left on their defaults.

The interpolation methods for remapping the source image and reprojecting the DEM are configurable with ``--interp`` and ``--dem-interp`` respectively.  They default to ``cubic``.  The ortho image is generated all bands at once by default.  Changing to band-by-band generation with ``--per-band`` is slower but reduces memory usage for some source image formats.  The next example orthorectifies the OpenDroneMap dataset using ``lanczos`` interpolation and band-by-band ortho generation:

.. code-block:: bash

    oty odm --dataset-dir odm --interp lanczos --dem-interp lanczos --per-band --overwrite

The ``--full-remap`` option configures the orthorectification algorithm to remap the source to ortho image using the full :doc:`camera model <../../background/models>` that includes distortion.  This is the default behaviour.  With the ``--no-full-remap`` option, the source is undistorted and then remapped to the ortho image using a pinhole model that excludes distortion.  ``--no-full-remap`` is faster, but can reduce the ortho image quality as it interpolates source pixels twice.  The ``--alpha`` option can be supplied with ``--no-full-remap`` to specify the scaling of the undistorted image.  An ``--alpha`` value of 0 scales the undistorted image so that all its pixels are valid.  An ``--alpha`` value of 1 (the default) scales the undistorted image so that it includes all source pixels.  The plot below illustrates this with images generated at ``--alpha`` values of 0 and 1:

.. image:: alpha_plot.webp

To show the use of these options on the command line, we orthorectify the OpenDroneMap dataset, using the ``--no-full-remap`` option with an ``--alpha`` value of 0:

.. code-block:: bash

    oty odm --dataset-dir odm --no-full-remap --alpha 0 --overwrite

.. note::

    The ``--full-remap`` / ``--no-full-remap`` and ``--alpha`` options have no effect for pinhole camera models.

.. TODO:
 - link to background sections
 - notes on test data
 - vertical CRS and --crs with eg

.. |oty| replace:: ``oty``
.. _oty: ../../cli/oty.html

