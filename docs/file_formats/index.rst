File formats
============

This section describes camera model file formats supported by Orthority.

.. Orthority supports various file formats for specifying camera models.  For :ref:`frame cameras <background/camera_models:frame cameras>`, interior and exterior parameters may be split into separate files, or contained in the same file, depending on the format.  In the API, files are converted to/from standard format dictionaries in the :mod:`~orthority.param_io` module, and files or dictionaries converted to camera objects in the :mod:`~orthority.factory` module.  Reading is supported for all formats, and writing for :doc:`YAML interior <yaml>` and :doc:`GeoJSON exterior parameter <geojson>` formats only.

.. toctree::
    :maxdepth: 1

    yaml
    geojson
    csv
    exif_xmp
    opensfm
    image_rpc
    yaml_rpc
    legacy

.. TODO: note that files define one or both of frame camera interior and or exterior parameters. and make better sense of the statement above.
.. TODO: add reference to contribution guidelines
