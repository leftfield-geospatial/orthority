EXIF / XMP tags
===============

Interior and exterior parameters can be read from image EXIF / XMP tags.  The tables below list required tag sets for different parameters.  If more than one set is present, the first complete set is used.

Interior parameters
-------------------

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Tag set
      - Tag type
      - XMP Namespace(s)
      - Camera type
    * - - ``DewarpData``
      - XMP
      - - ``http://www.dji.com/drone-dji/1.0/``
      - :attr:`~orthority.enums.CameraType.brown`
    * - - ``EXIF_FocalLength``
        - ``EXIF_FocalPlaneXResolution``
        - ``EXIF_FocalPlaneYResolution``
        - ``EXIF_FocalPlaneResolutionUnit``
      - EXIF
      -
      - :attr:`~orthority.enums.CameraType.pinhole`
    * - - ``EXIF_FocalLengthIn35mmFilm``
      - EXIF
      -
      - :attr:`~orthority.enums.CameraType.pinhole`

Exterior parameters
-------------------

Camera position
~~~~~~~~~~~~~~~

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Tag set
      - Tag type
      - XMP Namespace(s)
    * - - ``GpsLatitude``
        - ``GpsLongtitude``
        - ``AbsoluteAltitude``
      - XMP
      - - ``http://www.dji.com/drone-dji/1.0/``
    * - - ``EXIF_GPSLatitudeRef``
        - ``EXIF_GPSLongitudeRef``
        - ``EXIF_GPSLatitude``
        - ``EXIF_GPSLongitude``
        - ``EXIF_GPSAltitude``
      - EXIF
      -

Camera orientation
~~~~~~~~~~~~~~~~~~

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Tag set
      - Tag type
      - XMP Namespace(s)
    * - - ``GimbalRollDegree``
        - ``GimbalPitchDegree``
        - ``GimbalYawDegree``
      - XMP
      - - ``http://www.dji.com/drone-dji/1.0/``
    * - - ``Roll``
        - ``Pitch``
        - ``Yaw``
      - XMP
      - - ``http://ns.sensefly.com/Camera/1.0/``
        - ``http://pix4d.com/camera/1.0/``

.. TODO: add an oty info subcommand and refer to it here

.. TODO: add reference to contribution guidelines

.. note::

    Reading camera parameters from EXIF / XMP tags is currently experimental.  If you need support for tags not included above, please see the contribution guide.

