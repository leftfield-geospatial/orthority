.. include:: shared.txt

``oty rpc``
===========

|oty rpc|_ orthorectifies with RPC camera model(s) derived from :doc:`image tags / sidecar files <../../file_formats/image_rpc>` or an :doc:`Orthority RPC parameter file <../../file_formats/oty_rpc>`.  This example orthorectifies a satellite image with RPC tags using the NGI DEM of the same area.  Ortho images are placed in the current directory:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif rpc/qb2_basic1b.tif

The model can also be defined by an RPC parameter file:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif --rpc-param rpc/rpc_param.yaml rpc/qb2_basic1b.tif

By default, a 3D WGS84 geographic CRS is used as the world / ortho CRS.  This can be changed with :option:`--crs <oty-rpc --crs>`.  The RPC world / ortho CRS is required to use ellipsoidal heights (m).  If DEM heights are not ellipsoidal, both world / ortho and DEM CRSs should define a :doc:`vertical CRS <../../background/vertical_crs>`.  In this example, the NGI DEM has a defined EGM2008 vertical CRS, so the world / ortho CRS is defined as a 2D+1D CRS with ellipsoidal heights:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif --crs EPSG:32735+4326 rpc/qb2_basic1b.tif

The RPC model can be refined with GCPs using :option:`--gcp-refine <oty-rpc --gcp-refine>`.  GCPs can be provided in ``SOURCE`` :doc:`image tags <../../file_formats/image_gcps>`:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif --gcp-refine tags rpc/qb2_basic1b.tif

Or GCPs can be provided with an :doc:`Orthority GCPs file <../../file_formats/oty_gcps>`.  The :option:`--refine-method <oty-rpc --refine-method>` option specifies the refinement method:

.. code-block:: bash

    oty rpc --dem ngi/dem.tif --gcp-refine rpc/gcps.geojson --refine-method shift rpc/qb2_basic1b.tif

