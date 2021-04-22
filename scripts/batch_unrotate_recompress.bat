REM Preprocess NGI unrectified imagery for orthorectification with simple_ortho
REM 
REM Resample to be north up, 
REM and recompress with deflate, to avoid 12 bit jpeg compatibility issues
REM Note that this must use osgeo4w gdal, not conda   

echo on
REM setup environment for osgeo4w
C:\OSGeo4W\osgeo4w.bat	
setlocal EnableDelayedExpansion
for %%i in (*_RGB.tif) do (
set jj=%%i
gdalwarp -r bilinear -srcnodata 0 -dstnodata 0 -co "TILED=YES" -co "COMPRESS=DEFLATE" -co "PREDICTOR=2" -co "NUM_THREADS=ALL_CPUS" -co "BLOCKXSIZE=512" -co "BLOCKYSIZE=512" -wm 4000 %%~ni.tif %%~ni_PRE.tif && echo SUCCESS
)
pause
