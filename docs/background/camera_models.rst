Camera models
=============

A camera model describes the relationship between 3D :ref:`world <background/coordinates:world coordinates>` and 2D :ref:`pixel <background/coordinates:pixel coordinates>` coordinates.  Model parameters can be specified with supported :doc:`files <../file_formats/index>`, or directly with the :mod:`~orthority.camera` API.

Frame cameras
-------------

Orthority uses the term *frame camera* for area-scan cameras like those used in drone and aerial surveys.  These cameras capture a 2D image in a single instance of time.  The *frame camera* model is a physical model that splits the image formation process into *interior* and *exterior* components, each with its own set of parameters.  In a typical drone or aerial survey, *interior* parameters are fixed for a camera, and *exterior* parameters vary for each image.

Interior parameters
~~~~~~~~~~~~~~~~~~~

The *interior* component describes the relationship between :ref:`camera <background/coordinates:camera coordinates>` and :ref:`pixel <background/coordinates:pixel coordinates>` coordinates.  It depends on the interior geometry and optical properties of the camera.  Interior parameters are: image size, focal length, sensor size, principal point and distortion coefficients.

Exterior parameters
~~~~~~~~~~~~~~~~~~~

The *exterior* component describes the relationship between :ref:`world <background/coordinates:world coordinates>` and :ref:`camera <background/coordinates:camera coordinates>` coordinates. It is an affine transform consisting of a translation and rotation.  Orthority represents exterior parameters as a world coordinate (*x*, *y*, *z*) camera position, and (*omega*, *phi*, *kappa*) camera orientation, where the (*omega*, *phi*, *kappa*) angles rotate from camera to world coordinates.

RPC cameras
-----------

The RPC (rational polynomial coefficient) camera model represents the relationship between :ref:`world <background/coordinates:world coordinates>` and :ref:`pixel <background/coordinates:pixel coordinates>` coordinates as the ratio of two polynomials.  It is a general (non-physical) model typically used for describing narrow field of view and push-broom satellite cameras.
