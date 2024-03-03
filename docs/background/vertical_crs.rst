Vertical CRS
============

DEM and camera position heights may have different vertical coordinate systems e.g. heights may be referenced to the ellipsoid, or to a geoid.  Differences between vertical coordinate systems can be substantial, so it is important to convert heights into a common system for accurate orthorectification.  When both the DEM and world / ortho CRS have a defined vertical system (either as a 3D geographic CRS, or a 2D+1D compound CRS), Orthority performs this conversion automatically. Otherwise, the DEM and world / ortho vertical coordinate systems are assumed the same, and no conversion is performed.

.. note::

    Publicly available and commercially produced DEMs seldom have a defined vertical coordinate system, so users should ensure that DEM and camera position vertical systems are the same, and/or are properly defined.

As a brief introduction to defining 3D or 2D+1D CRSs with vertical coordinate systems, it is easiest to use `EPSG codes <https://epsg.io>`__  when they exist for the CRS.  The basic form for an EPSG CRS string is ``EPSG:<code>``.  Compound 2D+1D CRSs can be defined with ``EPSG:<horizontal code>+<vertical code>``.  CRSs can be specified in this way with the Orthority command line and API, or with third party tools for editing a raster's metadata e.g. Rasterio's |rio edit-info|_ command line utility.  Equivalent proj4 or WKT strings can also be used.

.. TODO: refer to path_uri with a note about CRSs requiring PAM files not being supported

.. |rio edit-info| replace:: ``rio edit-info``
.. _rio edit-info: https://rasterio.readthedocs.io/en/stable/cli.html#edit-info
