YAML interior parameters
========================

This is the native Orthority format for interior parameters.  It is a YAML file containing a nested dictionary that is based on the OpenDroneMap / OpenSfM :file:`cameras.json` format.  The root level consists of one or more camera ID keys and the corresponding parameter dictionary values.  Parameter dictionaries define parameter name - value pairs for a camera.  The basic layout is:

.. code-block:: yaml

    camera 1 ID:
        # camera 1 parameters:
        name 1: value 1
        # ...
        name N: value N
    # ...
    camera N ID:
        # camera N parameters:
        name 1: value 1
        # ...
        name N: value N

Camera IDs can be used in exterior parameters to refer to a specific camera in multi-camera set ups (e.g. with the ``camera`` field in :doc:`CSV <csv>` and :doc:`GeoJSON <geojson>` files), and should be unique.  Parameter names and value descriptions are as follows:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Name
      - Value
    * - ``type``
      - Camera type (``pinhole``, ``brown``, ``fisheye``, ``opencv``).
    * - ``im_size``
      - Image ``[width, height]`` in pixels.
    * - ``focal_len``
      - Focal length(s) with the same units/scale as ``sensor_size``.  Can be a single value or ``[x, y]`` pair.
    * - ``sensor_size``
      - Optional sensor ``[width, height]`` with the same units/scale as ``focal_len``.  If omitted, pixels are assumed square and ``focal_len`` normalised and unitless (i.e. ``focal_len`` = focal length / max(sensor width & height)).
    * - ``cx``, ``cy``
      - Optional principal point offsets in `normalised image coordinates <https://opensfm.readthedocs.io/en/latest/geometry.html#normalized-image-coordinates>`__.  Values default to zero if not supplied.
    * - ``k1``, ``k2``, …
      - Optional distortion coefficients for the camera ``type``.  Values default to zero if not supplied.

.. From the API perspective, the ``type`` parameter specifies which :class:`~orthority.camera.Camera` subclass to use.  See the :mod:`camera module docs <orthority.camera>` for details of distortion coefficients for each camera ``type``.

Available camera types and their distortion coefficients are detailed below:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - ``type``
      - Coefficients
      - Description
    * - ``pinhole``
      -
      - Pinhole camera model with no distortion.
    * - ``brown``
      - ``k1``, ``k2``, ``p1``, ``p2``, ``k3``
      - Brown-Conrady lens distortion compatible with `OpenDroneMap / OpenSfM <https://opensfm.org/docs/geometry.html#camera-models>`__ ``perspective``, ``simple_radial``, ``radial`` and ``brown`` model parameters, and the 4- and 5-coefficient versions of the `OpenCV general model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__.
    * - ``fisheye``
      - ``k1``, ``k2``, ``k3``, ``k4``
      - Fisheye lens distortion Compatible with `OpenDroneMap / OpenSfM <https://opensfm.org/docs/geometry.html#fisheye-camera>`__, and `OpenCV <https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html>`__  ``fisheye`` model parameters. The OpenDroneMap / OpenSfM model is a special case of the OpenCV version with ``k3, k4 = 0``.
    * - ``opencv``
      - ``k1``, ``k2``, ``p1``, ``p2``, ``k3``, ``k4``, ``k5``, ``k6``, ``s1``, ``s2``, ``s3``, ``s4``, ``tx``, ``ty``
      - `OpenCV general camera model <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`__. Partial or special cases of the model can be specified by omitting some or all of the coefficients. E.g. if no distortion coefficients are specified, this model corresponds to ``pinhole``, or if the first 5 distortion coefficients are specified, this model corresponds to ``brown``.

From the API perspective, the ``type`` parameter specifies which :class:`~orthority.camera.Camera` subclass to use.  The remaining parameters correspond to the ``__init__()`` arguments for that class.

An example of a valid YAML file defining two cameras:

.. code-block:: yaml

    Pinhole camera:
        type: pinhole
        im_size: [150, 200]
        focal_len: 120.                 # focal length in mm
        sensor_size: [75., 100.]        # sensor size in mm
        cx: -0.01
        cy: 0.02
    Brown camera:
        type: brown
        im_size: [400, 300]
        focal_len: 0.8333               # normalised & unitless focal length
        cx: -0.01
        cy: 0.02
        k1: -0.25
        k2: 0.2
        p1: 0.01
        p2: 0.01
        k3: -0.1
