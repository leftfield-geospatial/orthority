Orthority GCPs
==============

The native Orthority GCP format is a standard GeoJSON ``FeatureCollection`` of ``Point`` features, that can be visualised in a GIS.  Each ``Feature`` is a GCP, where the geometry is its position in 3D WGS84 geographic coordinates.  ``Feature`` properties are as follows:

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Property
      - Value
    * - ``filename``
      - Image file name excluding parent path, with or without extension.
    * - ``ji``
      - Image ``[j, i]`` position in :ref:`pixel coordinates <background/coordinates:pixel coordinates>`.
    * - ``id``
      - ID string (optional).
    * - ``info``
      - Information string (optional).

An example file defining a single GCP:

.. code-block:: json

    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "filename": "qb2_basic1b.tif",
                    "ji": [
                        821.3001696660183,
                        62.303697728645055
                    ],
                    "id": "concrete-plinth-70",
                    "info": "concrete-plinth-70"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        24.41948061951812,
                        -33.65426900104435,
                        214.75143153141929
                    ]
                }
            }
        ]
    }
