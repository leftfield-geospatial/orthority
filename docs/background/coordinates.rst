Coordinate systems
==================

Orthority uses these coordinate systems:

Pixel coordinates
-----------------

*Pixel coordinates* are 2D *(j, i)* column and row indices of the center of image pixels.  Orthority uses the standard convention where the origin is on the top left pixel, the *j*-axis points right, and the *i*-axis points down.

Camera coordinates
------------------

*Camera coordinates* are 3D *(x, y, z)* cartesian coordinates aligned with the camera and centered on its optical point.  Internally, Orthority uses the OpenCV / OpenSfM convention for these axes, where the *x*-axis points right, the *y*-axis down and *z*-axis forwards (looking through the camera at the world scene).  For (*omega*, *phi*, *kappa*) angles supplied in :doc:`CSV <../file_formats/csv>` / :doc:`Orthority exterior parameter <../file_formats/oty_ext>` files or with the :mod:`~orthority.camera` API, Orthority uses the PATB axis convention.  In this convention, the *x*-axis points right, the *y*-axis up and the *z*-axis backwards.

World coordinates
------------------

*World coordinates* are 3D *(x, y, z)* coordinates where the origin and axis alignment is fixed relative to the Earth.  For orthorectification this system is represented by a geospatial CRS (coordinate reference system).  :ref:`Frame cameras <background/camera_models:frame cameras>` use a cartesian world system, approximated by a projected CRS.  :ref:`RPC cameras <background/camera_models:rpc cameras>` use the WGS84 geographic CRS, optionally transformed to a projected CRS.  The ortho image is georeferenced in this coordinate system.

