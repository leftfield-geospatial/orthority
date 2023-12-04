File formats
============

This section describes camera model file formats supported by Orthority.  Reading is supported for all formats, and writing for :doc:`YAML interior <yaml>` and :doc:`GeoJSON exterior parameter <geojson>` format files.  See the ``oty`` :doc:`subcommand documentation <../cli/index>` for guidance on reading from different formats on the command line.  The ``--export-params`` option can be used with any subcommand to write parameters to file.  In the API, parameter file IO is implemented in the :mod:`~orthority.io` module.

.. toctree::
    :maxdepth: 1

    yaml
    geojson
    csv
    exif_xmp
    opensfm
    legacy
