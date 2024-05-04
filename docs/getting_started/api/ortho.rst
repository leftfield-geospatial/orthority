Orthorectification
==================

Orthorectification is performed by the :class:`~orthority.ortho.Ortho` class, which requires a source image, DEM, camera model and world / ortho CRS to instantiate.

To start, we create a frame camera model from interior and exterior parameter files:

.. literalinclude:: ../../scripts/ortho.py
    :language: python
    :start-after: [create camera]
    :end-before: [end create camera]

Then we can create the :class:`~orthority.ortho.Ortho` object and orthorectify:

.. literalinclude:: ../../scripts/ortho.py
    :language: python
    :start-after: [orthorectify]
    :end-before: [end orthorectify]

Configuration
-------------

The ortho image resolution and format are configurable:

.. literalinclude:: ../../scripts/ortho.py
    :language: python
    :start-after: [config ortho]
    :end-before: [end config ortho]

Some aspects of the algorithm can also be configured:

.. literalinclude:: ../../scripts/ortho.py
    :language: python
    :start-after: [config algo]
    :end-before: [end config algo]

.. TODO:
  - ortho options
  - algorithm options
  - progress bar


.. |tqdm| replace:: ``tqdm``
.. _tqdm: https://tqdm.github.io
