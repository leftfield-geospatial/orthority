Camera models
=============

Camera models are implemented as sub-classes of :class:`~orthority.camera.Camera` in the :mod:`~orthority.camera` module.   Each source image requires its own camera model instance.  Models can be created from :doc:`supported file formats <../../file_formats/index>` with a camera factory.  Camera factories are implemented as sub-classes of :class:`~orthority.factory.Cameras` in the :mod:`~orthority.factory` module.

Frame cameras
-------------

Factory creation
~~~~~~~~~~~~~~~~

The :class:`~orthority.factory.FrameCameras` factory creates :class:`~orthority.camera.FrameCamera` models, from interior and exterior parameter files:

.. literalinclude:: ../../scripts/api_frame.py
    :language: python
    :start-after: [create camera]
    :end-before: [end create camera]

The factory can also be created from :doc:`image EXIF / XMP tags <../../file_formats/exif_xmp>`:

.. literalinclude:: ../../scripts/api_frame.py
    :language: python
    :start-after: [create exif]
    :end-before: [end create exif]

World / ortho CRS
~~~~~~~~~~~~~~~~~

:class:`~orthority.factory.FrameCameras` will read or choose a world / ortho CRS from these exterior parameter formats:

* :doc:`Orthority format <../../file_formats/oty_ext>`.
* :doc:`OpenDroneMap / OpenSfM <../../file_formats/opensfm>` :file:`reconstruction.json` format.
* :doc:`CSV format <../../file_formats/csv>` with ``x``, ``y`` & ``z`` fields and a :file:`.prj` sidecar file.
* :doc:`CSV format <../../file_formats/csv>` with ``latitude``, ``longitude``, ``altitude``, ``roll``, ``pitch`` & ``yaw`` fields.
* :doc:`Image EXIF / XMP tags <../../file_formats/exif_xmp>`.

The world / ortho CRS is available via the :attr:`~orthority.factory.FrameCameras.crs` attribute:

.. literalinclude:: ../../scripts/api_frame.py
    :language: python
    :start-after: [crs]
    :end-before: [end crs]

An explicit world / ortho CRS can be specified with ``io_kwargs`` as explained in the next section.

Parameter IO
~~~~~~~~~~~~

Internally, camera factories use the :mod:`~orthority.param_io` module to read and interpret parameter files.

The ``io_kwargs`` argument can be used in :class:`~orthority.factory.FrameCameras` or :meth:`~orthority.factory.FrameCameras.from_images` to pass keyword arguments to the :class:`~orthority.param_io.FrameReader` sub-class corresponding to the exterior parameter file format.

E.g. to create a factory from image EXIF / XMP tags and pass an explicit world / ortho CRS to :class:`~orthority.param_io.ExifReader`:

.. literalinclude:: ../../scripts/api_frame.py
    :language: python
    :start-after: [io_kwargs]
    :end-before: [end io_kwargs]

Camera options
~~~~~~~~~~~~~~

The ``cam_kwargs`` argument can be used in :class:`~orthority.factory.FrameCameras` or :meth:`~orthority.factory.FrameCameras.from_images` to pass keyword arguments to the :class:`~orthority.camera.FrameCamera` class.  E.g. to pass the ``distort`` and ``alpha`` arguments:

.. literalinclude:: ../../scripts/api_frame.py
    :language: python
    :start-after: [cam_kwargs]
    :end-before: [end cam_kwargs]

Model export
~~~~~~~~~~~~

Camera model parameters can be exported to Orthority format :doc:`interior <../../file_formats/oty_int>` and :doc:`exterior <../../file_formats/oty_ext>` parameter files with :meth:`~orthority.factory.FrameCameras.write_param`:

.. literalinclude:: ../../scripts/api_frame.py
    :language: python
    :start-after: [export]
    :end-before: [end export]

RPC cameras
-------------

Factory creation
~~~~~~~~~~~~~~~~

The :class:`~orthority.factory.RpcCameras` factory creates :class:`~orthority.camera.RpcCamera` models from an :doc:`Orthority RPC parameter file <../../file_formats/oty_rpc>`:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [create camera]
    :end-before: [end create camera]

Camera models can also be created from :doc:`image RPC tags / sidecar files <../../file_formats/image_rpc>`:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [create tag]
    :end-before: [end create tag]

Parameter IO
~~~~~~~~~~~~

Internally, the :func:`~orthority.param_io.read_im_rpc_param` function is used to read image RPC tags / sidecar files.  The ``io_kwargs`` argument can be used in :meth:`~orthority.factory.RpcCameras.from_images` to pass keyword arguments to :func:`~orthority.param_io.read_im_rpc_param`.

E.g. to display a progress bar while reading image RPC tags / sidecar files:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [io_kwargs]
    :end-before: [end io_kwargs]

Camera options
~~~~~~~~~~~~~~

The ``cam_kwargs`` argument can be used in :class:`~orthority.factory.RpcCameras` or :meth:`~orthority.factory.RpcCameras.from_images` to pass keyword arguments to :class:`~orthority.camera.RpcCamera`.  E.g. to specify a world / ortho ``crs``:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [cam_kwargs]
    :end-before: [end cam_kwargs]

Model refinement
~~~~~~~~~~~~~~~~

RPC models can be refined with GCPs using the :meth:`~orthority.factory.RpcCameras.refine` method.  GCPs can be read from an :doc:`Orthority GCPs file <../../file_formats/oty_gcps>`:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [refine]
    :end-before: [end refine]

Or GCPs can be read from :doc:`image tags <../../file_formats/image_gcps>`:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [refine tag]
    :end-before: [end refine tag]

The :func:`~orthority.param_io.read_im_gcps` function is used internally to read GCPs from image tags.  ``io_kwargs`` can be used in :meth:`~orthority.factory.RpcCameras.refine` to pass keyword arguments to :func:`~orthority.param_io.read_im_gcps` when GCPs are supplied in image tags:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [refine io_kwargs]
    :end-before: [end refine io_kwargs]

The :func:`~orthority.fit.refine_rpc` function is used internally to perform RPC refinement.  ``fit_kwargs`` can be used in :meth:`~orthority.factory.RpcCameras.refine` to pass keyword arguments to :func:`~orthority.fit.refine_rpc`:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [refine fit_kwargs]
    :end-before: [end refine fit_kwargs]

Model export
~~~~~~~~~~~~

Camera model parameters can be exported to an :doc:`Orthority RPC parameter file <../../file_formats/oty_rpc>` with :meth:`~orthority.factory.RpcCameras.write_param`:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [export]
    :end-before: [end export]

When the models have been refined, :meth:`~orthority.factory.RpcCameras.write_param` exports the refined models together with the GCPs used to refine them.  GCPs are written to an :doc:`Orthority GCPs file <../../file_formats/oty_gcps>`:

.. literalinclude:: ../../scripts/api_rpc.py
    :language: python
    :start-after: [export gcps]
    :end-before: [end export gcps]
