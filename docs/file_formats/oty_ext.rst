Orthority exterior parameters
=============================

This is the native Orthority format for frame camera :ref:`exterior parameters <background/camera_models:exterior parameters>`.  It is a standard GeoJSON ``FeatureCollection`` of ``Point`` features, that can be visualised in a GIS.  The world / ortho CRS is stored in a ``world_crs`` member of the ``FeatureCollection`` as a WKT, proj4 or EPSG string.  E.g.

.. code-block:: json
    :emphasize-lines: 3

    {
        "type": "FeatureCollection",
        "world_crs": "EPSG:32651",
        "features": [
            "..."
        ]
    }

Each ``Feature`` corresponds to a source image file, where the Feature geometry is the camera position in geographic coordinates.  The ``Feature`` properties are as follows:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Property
      - Value
    * - ``filename``
      - Image file name excluding parent path, with or without extension.
    * - ``camera``
      - ID of camera :ref:`interior parameters <background/camera_models:interior parameters>` (can be ``null``).
    * - ``xyz``
      - Camera ``[x, y, z]`` position in :ref:`world / ortho coordinates <background/coordinates:world coordinates>`.
    * - ``opk``
      - Camera ``[omega, phi, kappa]`` orientation angles in radians.

An example file with exterior parameters for a single image:

.. code-block:: json

    {
        "type": "FeatureCollection",
        "world_crs": "EPSG:32634",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "filename": "src_image_0.jpg",
                    "camera": "Pinhole camera ID",
                    "xyz": [20000.0, 30000.0, 1000.0],
                    "opk": [-0.05, 0.03, 0.17]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [16.690410, 0.270646, 1000.0]
                }
            }
        ]
    }


