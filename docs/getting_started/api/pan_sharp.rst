Pan sharpening
==============

The :class:`~orthority.pan_sharp.PanSharpen` class implements Gram-Schmidt pan sharpening.  Panchromatic and multispectral images are required to instantiate.  The :meth:`~orthority.pan_sharp.PanSharpen.process` method pan sharpens:

.. literalinclude:: ../../scripts/api_pan_sharp.py
    :language: python
    :start-after: [pan_sharpen]
    :end-before: [end pan_sharpen]

See the :meth:`~orthority.pan_sharp.PanSharpen.process` documentation for details on other configuration options.
