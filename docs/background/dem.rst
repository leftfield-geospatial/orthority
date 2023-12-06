DEM
===

The DEM used for orthorectification should be a raster in a `GDAL supported format <https://gdal.org/drivers/raster/index.html>`__, with height values in units of its :doc:`vertical CRS <vertical_crs>` (usually meters).  For best results, DEM heights should include vegetation and buildings i.e. the DEM should be a digital surface model (DSM).  While there are no requirements for the DEM resolution and CRS, higher DEM resolutions generally produce more accurate ortho images, and orthorectification is faster when DEM and world / ortho CRS are the same.
