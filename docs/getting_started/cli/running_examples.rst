Running examples
----------------

Examples that follow use Orthority's test data.  You can get this by downloading the repository directly:

.. code-block:: bash

    curl -LO# "https://github.com/leftfield-geospatial/orthority/archive/refs/heads/main.zip"
    tar -xf main.zip

Alternatively, you can clone the repository with `git <https://git-scm.com/downloads>`_:

.. code-block:: bash

    git clone https://github.com/leftfield-geospatial/orthority.git

Once you have the repository, navigate to ``<orthority root>/tests/data``, and create an ``ortho`` sub-directory to contain processed images:

.. code-block:: bash

    cd <orthority root>/tests/data
    mkdir ortho

Commands that follow use relative paths, and should be run from ``<orthority root>/tests/data`` (``<orthority root>`` will be one of ``orthority-main`` or ``orthority``, depending on your download method).
