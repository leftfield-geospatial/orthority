Background
==========

This section covers terminology and background information relevant to command line and API users.

.. toctree::
    :maxdepth: 1

    coordinates
    camera_models
    dem
    vertical_crs
    path_uri

.. TODO: somewhere we can/should note that orthorectification is multithreaded and the ortho is generated in blocks.
.. TODO: we could add a section on orthorectification here that explains the basic process of remapping source to ortho, perhaps also reprojecting the DEM. This could be useful for understanding the some of the options like interpolation & full_remap.  This section can include the note above about concurrent block processing.  Perhaps it should include the plot that is currently in the cli getting started section.  And it could include another plot that describes full_remap=True or False, showing how the interior component is used to generate the undistorted image from the source, and the exterior component the ortho image from the undistorted image for full_remap=False etc.  This would show the 2 interpolations.
.. TODO: somewhere make a note about 12 bit JPEG compression
.. TODO: document fsspec backend allowing URIs etc to be passed.  and case sensitivity for wildcards?  and ext param matching?