.. include:: shared.txt

``oty odm``
===========

|oty odm|_ orthorectifies images in an `OpenDroneMap <https://github.com/OpenDroneMap/ODM>`__ generated dataset using the dataset DSM (:file:`{dataset}/odm_dem/dsm.tif`) and camera models (:file:`{dataset}/opensfm/reconstruction.json`).  Here we orthorectify images in the :file:`odm` dataset.  Ortho images are placed in the :file:`{dataset}/orthority` directory:

.. code-block:: bash

    oty odm --dataset-dir odm

Without the :option:`--crs <oty-odm --crs>` option, the world / ortho CRS is read from the OpenDroneMap DSM.  This is the recommended setting.
