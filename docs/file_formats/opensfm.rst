OpenDroneMap / OpenSfM parameters
=================================

Orthority can read interior parameters from OpenDroneMap :file:`cameras.json` files, and interior and exterior parameters from OpenDroneMap / OpenSfM |reconstruction.json|_ files.  A subset of the OpenDroneMap / OpenSfM `camera models <https://opensfm.org/docs/geometry.html#camera-models>`__ are supported.  Supported models with their corresponding Orthority model are shown below:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - OpenDroneMap / OpenSfM model(s)
      - Orthority model
    * - ``perspective``, ``simple_radial``, ``radial``, ``brown``
      - :attr:`~orthority.enums.CameraType.brown`
    * - ``fisheye``
      - :attr:`~orthority.enums.CameraType.fisheye`

.. |reconstruction.json| replace:: :file:`reconstruction.json`
.. _reconstruction.json: https://opensfm.readthedocs.io/en/latest/dataset.html#reconstruction-file-format
