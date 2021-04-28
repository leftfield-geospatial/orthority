REM Preprocess NGI unrectified imagery for orthorectification with simple_ortho
REM 
REM Conda gdal does not support 12 bit jpeg compression, so re-compress to DEFLATE
REM with osgeo4w gdal (not conda)
REM NOTE: Do not resample to be north up as this invalidates the camera extrinsic position & rotation

echo on
REM setup environment for osgeo4w
C:\OSGeo4W\osgeo4w.bat	
setlocal EnableDelayedExpansion
for %%i in (*_RGB.tif) do (
set jj=%%i
REM gdalwarp -r bilinear -srcnodata 0 -dstnodata 0 -co "TILED=YES" -co "COMPRESS=DEFLATE" -co "PREDICTOR=2" -co "NUM_THREADS=ALL_CPUS" -co "BLOCKXSIZE=512" -co "BLOCKYSIZE=512" -wm 4000 %%~ni.tif %%~ni_PRE.tif && echo SUCCESS
gdal_translate -r bilinear -a_nodata 0 -co "TILED=YES" -co "COMPRESS=DEFLATE" -co "PREDICTOR=2" -co "NUM_THREADS=ALL_CPUS" -co "BLOCKXSIZE=512" -co "BLOCKYSIZE=512" %%~ni.tif %%~ni_DFL.tif && echo SUCCESS
)
pause
