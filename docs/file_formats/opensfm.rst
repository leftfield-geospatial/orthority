OpenDroneMap / OpenSfM parameters
=================================

Orthority can read frame camera :ref:`interior parameters <background/camera_models:interior parameters>` from OpenDroneMap :file:`cameras.json` files, and frame camera interior and :ref:`exterior parameters <background/camera_models:exterior parameters>` from OpenDroneMap / OpenSfM |reconstruction.json|_ files.  A subset of the OpenDroneMap / OpenSfM `camera models <https://opensfm.org/docs/geometry.html#camera-models>`__ are supported.  Supported models with their corresponding Orthority model are shown below:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - OpenDroneMap / OpenSfM model(s)
      - Orthority model
    * - ``perspective``, ``simple_radial``, ``radial``, ``brown``
      - :attr:`~orthority.enums.CameraType.brown`
    * - ``fisheye``
      - :attr:`~orthority.enums.CameraType.fisheye`

Similar to the :doc:`YAML format <yaml>`, interior parameters in :file:`cameras.json` or :file:`reconstruction.json` files are defined as dictionaries of camera ID keys, and nested parameter dictionary values.  E.g.:

.. literalinclude:: ../../tests/data/odm/opensfm/reconstruction.json
    :language: json
    :lines: 4-17

Exterior parameters in the :file:`reconstruction.json` file, consist of filename keys, and nested parameter dictionary values.  Only the ``rotation``, ``translation`` and ``camera`` parameters are used.  E.g.:

.. literalinclude:: ../../tests/data/odm/opensfm/reconstruction.json
    :language: json
    :lines: 20-45
    :emphasize-lines: 2-12


.. |reconstruction.json| replace:: :file:`reconstruction.json`
.. _reconstruction.json: https://opensfm.readthedocs.io/en/latest/dataset.html#reconstruction-file-format
