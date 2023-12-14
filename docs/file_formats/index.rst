File formats
============

This section describes camera model file formats supported by Orthority.  Reading is supported for all formats, and writing for :doc:`YAML interior <yaml>` and :doc:`GeoJSON exterior parameter <geojson>` format files.  Files are read when orthorectifying, and optionally written with the ``oty`` :doc:`command line <../cli/index>`.  In the API, parameter file IO is implemented in the :mod:`~orthority.io` module.

.. toctree::
    :maxdepth: 1

    yaml
    geojson
    csv
    exif_xmp
    opensfm
    legacy
