Coordinate systems
==================

Orthority uses the following coordinate systems:

Pixel coordinates
-----------------

*Pixel coordinates* are 2D *(j, i)* column and row indices of image pixels.  Orthority uses the standard system where the origin is on the center of the top left pixel, the *j*-axis points right, and the *i*-axis points down.

Camera coordinates
------------------

*Camera coordinates* are 3D *(x, y, z)* cartesian coordinates.  The system is aligned with the camera and centered on its optical point.  For user-supplied parameters, Orthority uses the PATB convention where the *x*-axis points right, the *y*-axis points up and the *z*-axis backwards (looking through the camera at the world scene).  Internally it uses the OpenCV / OpenSfM convention where the *x*-axis points right, the *y*-axis points down and *z*-axis forwards.

World coordinates
------------------

*World coordinates* are 3D *(x, y, z)* cartesian coordinates.  For orthorectification this system is represented by a projected CRS (coordinate reference system) where the origin and axis alignment is fixed relative to the earth surface.  Typically the *x*-axis points East, the *y*-axis points North, and the *z*-axis points up away from the earth.  The ortho image is georeferenced in this coordinate system.
