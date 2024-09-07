Orthorectification
==================

Orthorectification is performed by the :class:`~orthority.ortho.Ortho` class, which requires a source image, DEM, :doc:`camera model <camera>` and world / ortho CRS to instantiate.  The :meth:`~orthority.ortho.Ortho.process` method orthorectifies:

.. literalinclude:: ../../scripts/api_ortho.py
    :language: python

See the :meth:`~orthority.ortho.Ortho.process` documentation for details on configuration options.

