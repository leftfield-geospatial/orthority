Camera models
=============

A camera model describes the relationship between 3D :ref:`world <background/coordinates:world coordinates>` and 2D :ref:`pixel <background/coordinates:pixel coordinates>` coordinates.

Digital frame cameras
---------------------

Digital frame cameras (like those used for drone and aerial images) are described by a camera model that is split into *interior* and *exterior* components, each with its own set of parameters.

Interior parameters
~~~~~~~~~~~~~~~~~~~

The *interior* component describes the relationship between :ref:`camera <background/coordinates:camera coordinates>` and :ref:`pixel <background/coordinates:pixel coordinates>` coordinates.  It depends on the interior geometry and optical properties of the camera.  Interior parameters are: image size, focal length, sensor size, principal point and distortion coefficients.  These can be specified with supported :doc:`files <../file_formats/index>`, or directly with the :mod:`~orthority.camera` API.

Exterior parameters
~~~~~~~~~~~~~~~~~~~

The *exterior* component describes the relationship between :ref:`world <background/coordinates:world coordinates>` and :ref:`pixel <background/coordinates:pixel coordinates>` coordinates. It is an affine transform consisting of a rotation and translation.  Internally, Orthority represents external parameters as a :ref:`world <background/coordinates:world coordinates>` coordinate (*x*, *y*, *z*) camera position, and (*omega*, *phi*, *kappa*) camera orientation, where the (*omega*, *phi*, *kappa*) angles rotate from :ref:`camera <background/coordinates:camera coordinates>` to :ref:`world <background/coordinates:world coordinates>` coordinates.  Exterior parameters can be read from supported :doc:`files <../file_formats/index>`, or specified directly with the :mod:`~orthority.camera` API.  Note that conversion from (*roll*, *pitch*, *yaw*) camera orientation in e.g. :doc:`CSV files <../file_formats/csv>` or :doc:`EXIF / XMP tags <../file_formats/exif_xmp>` is approximate and relies on the world coordinate CRS having minimal distortion in the imaged area.
