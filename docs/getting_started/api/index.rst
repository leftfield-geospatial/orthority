.. include:: ../../shared.txt

API
===

To run the examples in this section, you need the `Requests <https://requests.readthedocs.io>`__ and `AIOHTTP <https://docs.aiohttp.org>`__ packages installed.  With pip_:

.. code-block:: bash

    pip install requests aiohttp

Or, with conda_:

.. code-block:: bash

    conda install -c conda-forge requests aiohttp

.. note::

    If you're using the Orthority and Rasterio_ packages together, Orthority should be imported first to configure the PROJ setting for :doc:`vertical CRS transformations <../../background/vertical_crs>`.

.. toctree::
    :maxdepth: 1

    camera
    ortho
