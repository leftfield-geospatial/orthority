Changelog
=========

v0.7.0 - 2026-08-19
-------------------

Features
~~~~~~~~

- Limit OpenCV, BLAS and OpenMP operations to single threads inside Python pool threads to reduce contention.
- Allow for source and ortho images with dimensions greater than 32767 (`#26 <https://github.com/leftfield-geospatial/orthority/issues/26>`__).
- Improve pan-sharpening performance with an increased block size that results in multi-threaded ``WarpedVRT`` reads.
- Add a ``ztd`` compression option.
- Add an ``--aligned-pixels`` CLI option, and ``aligned_pixels`` ``Ortho.process()`` parameter to align ortho pixels so their world coordinates are multiples of the resolution.

Fixes
~~~~~

- Fix ``Camera.pixel_boundary()`` returning an incorrect number of points for some image size / ``num_pts`` combinations.

Packaging
~~~~~~~~~

- Limit the OpenCV dependency version to <5 to avoid https://github.com/opencv/opencv/issues/29412.
- Add a threadpoolctl dependency for limiting BLAS and OpenMP threads.
- Increase the minimum supported python version to 3.10.

Documentation
~~~~~~~~~~~~~

- Update for ``--aligned-pixels``.
- Add a changelog.
- Remove CLI help examples.

Internal changes
~~~~~~~~~~~~~~~~

- Simplify CLI logging and help formatting.
- Add benchmarks.
- Generate ortho / pan-sharpening block windows row-wise.
- Improve ``common.nan_equals()`` speed for special cases.
- Add ``cancel_futures=True`` to executor shutdown.
- Refactor ``exif.Exif`` to use cached properties.

Previous versions
-----------------

See the `GitHub releases page <https://github.com/leftfield-geospatial/orthority/releases>`__.