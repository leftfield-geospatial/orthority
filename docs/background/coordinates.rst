Coordinate systems
==================

Orthority uses these coordinate systems:

Pixel coordinates
-----------------

*Pixel coordinates* are 2D *(j, i)* column and row indices of image pixels.  Orthority uses the standard convention where the origin is on the center of the top left pixel, the *j*-axis points right, and the *i*-axis points down.

Camera coordinates
------------------

*Camera coordinates* are 3D *(x, y, z)* cartesian coordinates aligned with the camera and centered on its optical point.  Internally, Orthority uses the OpenCV / OpenSfM convention for these axes, where the *x*-axis points right, the *y*-axis down and *z*-axis forwards (looking through the camera at the world scene).  For (*omega*, *phi*, *kappa*) angles supplied in :doc:`CSV <../file_formats/csv>` / :doc:`GeoJSON <../file_formats/geojson>` files or with the :mod:`~orthority.camera` API, Orthority uses the PATB axis convention.  In this convention, the *x*-axis points right, the *y*-axis up and the *z*-axis backwards.

World coordinates
------------------

*World coordinates* are 3D *(x, y, z)* cartesian coordinates.  For orthorectification this system is represented by a projected CRS (coordinate reference system) where the origin and axis alignment is fixed relative to the earth surface.  Typically the *x*-axis points East, the *y*-axis points North, and the *z*-axis (height) points up away from the earth.  The ortho image is georeferenced in this coordinate system.
