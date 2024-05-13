Vertical CRS
============

DEM and :ref:`world coordinate <background/coordinates:world coordinates>` heights may have different vertical coordinate systems e.g. heights may be referenced to different datums (ellipsoid or geoid) and have different units.  Accurate orthorectification requires conversion of heights into a common system.  When both the DEM and world / ortho CRS have a defined vertical system (either as a 3D CRS, or a 2D+1D compound CRS), this conversion is performed automatically. Otherwise, DEM and world vertical systems are assumed the same, and no conversion is performed.

.. note::

    Public and commercial DEMs seldom have a defined vertical coordinate system.

As a brief introduction to defining 3D or 2D+1D CRSs with vertical coordinate systems, it is easiest to use `EPSG codes <https://epsg.io>`__  when they exist for the CRS.  The basic form for an EPSG CRS string is ``EPSG:<code>``.  Compound 2D+1D CRSs can be defined with ``EPSG:<horizontal code>+<vertical code>``.  CRSs can be specified like this with the Orthority command line and API, or with third party tools for editing a raster's metadata e.g. Rasterio's |rio edit-info|_.  Equivalent proj4 or WKT strings can also be used.

.. note::

    Orthority uses the network capabilities of the PROJ library to transform vertical CRSs by setting the ``PROJ_NETWORK`` environment variable to ``'ON'``.  See the `PROJ documentation <https://proj.org/en/latest/usage/network.html>`__ for details.

.. |rio edit-info| replace:: ``rio edit-info``
.. _rio edit-info: https://rasterio.readthedocs.io/en/stable/cli.html#edit-info
