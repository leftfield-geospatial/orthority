.. include:: shared.txt

File IO
=======

Files can be read from or written to local or remote :doc:`file paths / URIs <../../background/path_uri>`.  ``SOURCE`` images can be specified with local or remote wildcard patterns.  Ortho images are named automatically based on the source image names.

.. note::

    Some file systems, like HTTP, are read-only and don't support wildcards.

Camera parameters can be exported to Orthority format files by supplying the ``--export-params`` option to any of the |oty|_ orthorectification sub-commands.  When ``--export-params`` is supplied, the command exits after exporting, and no orthorectification is performed.  This example exports models derived from image EXIF / XMP tags:

.. code-block:: bash

    oty exif --export-params odm/images/*.tif

The default output directory for ortho images and exported files is specific to the sub-command.  It can be changed with ``--out-dir``.  Passing ``--overwrite`` overwrites existing files.  Again, these options are common to all |oty|_ orthorectification sub-commands.  E.g., this orthorectifies the OpenDroneMap  dataset, creating and using :file:`ortho` as the output directory, and overwriting existing files:

.. code-block:: bash

    mkdir ortho
    oty odm --dataset-dir odm --out-dir ortho --overwrite

