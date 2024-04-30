YAML RPC parameters
===================

This is the native Orthority format for :ref:`RPC camera <background/camera_models:rpc cameras>` parameters.  It is a YAML file containing a nested dictionary of one or more image filename keys and the corresponding RPC parameter dictionary values.  The basic layout is:

.. code-block:: yaml

    image filename 1:
        # camera 1 parameters:
        name 1: value 1
        # ...
        name N: value N
    # ...
    image filename N:
        # camera N parameters:
        name 1: value 1
        # ...
        name N: value N

Root parameter names and value descriptions are:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Name
      - Value
    * - ``im_size``
      - Image ``[width, height]`` in pixels.
    * - ``rpc``
      - RPC parameters names and values as a dictionary.

Nested ``rpc`` parameter names and value descriptions are the same as the `GeoTIFF definition <http://geotiff.maptools.org/rpc_prop.html>`__:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Name
      - Value
    * - ``lat_off``
      - Geodetic latitude offset (degrees).
    * - ``lat_scale``
      - Geodetic latitude scale (unitless).
    * - ``long_off``
      - Geodetic longitude offset (degrees).
    * - ``long_scale``
      - Geodetic longitude scale (unitless).
    * - ``height_off``
      - Geodetic height offset (meters).
    * - ``height_scale``
      - Geodetic height scale (unitless).
    * - ``line_off``
      - Line / row offset (pixels).
    * - ``line_scale``
      - Line / row scale (unitless).
    * - ``samp_off``
      - Sample / column offset (pixels).
    * - ``samp_scale``
      - Sample / column scale (unitless).
    * - ``line_num_coeff``
      - Line / row numerator polynomial coefficients (``[C1, C2, ..., C20]``).
    * - ``line_den_coeff``
      - Line / row denominator polynomial coefficients (``[C1, C2, ..., C20]``).
    * - ``samp_num_coeff``
      - Sample / column numerator polynomial coefficients (``[C1, C2, ..., C20]``).
    * - ``samp_den_coeff``
      - Sample / column denominator polynomial coefficients (``[C1, C2, ..., C20]``).

An example file with parameters for a single image:

.. literalinclude:: ../../tests/data/rpc/rpc_param.yaml
    :language: yaml
