Camera model export
-------------------

:doc:`Camera models <../../background/camera_models>` can be exported to Orthority format files by supplying the ``--export-params`` option to any of the |oty|_ sub-commands.  When ``--export-params`` is supplied, the command exits after exporting, and no orthorectification is performed.  This example exports camera models derived from drone image EXIF / XMP tags.  Exported files are placed in the current directory:

.. code-block:: bash

    oty exif --export-params odm/images/*.tif

As with the :ref:`ortho images <getting_started/cli/basic_orthorectification:output files>`, ``--out-dir`` can be used to change the default output directory, and ``--overwrite`` to replace existing files.

.. |oty| replace:: ``oty``
.. _oty: ../../cli/oty.html

