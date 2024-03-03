Basic orthorectification
========================

The |oty|_ sub-commands allow :doc:`DEM <../../background/dem>` and :doc:`camera models <../../background/camera_models>` to be specified in different ways.

``oty ortho``
-------------

.. TODO: rephrase more like urllib3

|oty ortho|_ uses :ref:`interior <background/camera_models:interior parameters>` and :ref:`exterior <background/camera_models:interior parameters>` parameter files to specify camera models.  Here we orthorectify `NGI <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`__ aerial images using the associated DEM, :doc:`YAML format <../../file_formats/yaml>` interior parameters, and :doc:`CSV format <../../file_formats/csv>` exterior parameters.  Ortho images are placed in the current directory:

.. code-block:: bash

    oty ortho --dem ngi/dem.tif --int-param io/ngi_int_param.yaml --ext-param io/ngi_xyz_opk.csv ngi/*RGB.tif

See the :doc:`file format documentation <../../file_formats/index>` for other supported parameter formats.

``oty exif``
------------

Camera models can be derived from image EXIF / XMP tags with |oty exif|_.  The :doc:`EXIF / XMP documentation <../../file_formats/exif_xmp>` describes the required tags.  This example orthorectifies drone imagery using the DSM from an `OpenDroneMap <https://github.com/OpenDroneMap/ODM>`__ dataset.  Ortho images are placed in the current directory:

.. code-block:: bash

    oty exif --dem odm/odm_dem/dsm.tif odm/images/*RGB.tif

.. note::

    EXIF / XMP tag values contain inaccuracies which result in distortion and positioning errors in the ortho images.  This is a general problem with direct georeferencing from image tags, and a limitation of this approach.

``oty odm``
-----------

|oty odm|_ orthorectifies images in an `OpenDroneMap <https://github.com/OpenDroneMap/ODM>`__ generated dataset using the dataset DSM (:file:`{dataset}/odm_dem/dsm.tif`) and camera models (:file:`{dataset}/openfm/reconstruction.json`).  Here we orthorectify images in the :file:`odm` dataset.  Ortho images are placed in the :file:`{dataset}/orthority` directory:

.. code-block:: bash

    oty odm --dataset-dir odm

Output files
------------

Ortho images are named automatically based on the source image names.  The output directory for ortho images and :doc:`exported files <model_export>` can be changed from its default with the ``--out-dir`` option.  Passing ``--overwrite`` overwrites existing files.  These options are common to all |oty|_ sub-commands.  E.g., repeating the :ref:`getting_started/cli/basic_orthorectification:``oty odm``` example with these options:

.. code-block:: bash

    oty odm --dataset-dir odm --out-dir odm/orthority --overwrite

.. |oty| replace:: ``oty``
.. _oty: ../../cli/oty.html

.. |oty ortho| replace:: ``oty ortho``
.. _oty ortho: ../../cli/ortho.html

.. |oty exif| replace:: ``oty exif``
.. _oty exif: ../../cli/exif.html

.. |oty odm| replace:: ``oty odm``
.. _oty odm: ../../cli/odm.html
