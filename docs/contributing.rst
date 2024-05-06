Contributing
============

Contributions are welcome.

If Orthority is useful to you, please consider `making a donation <https://github.com/sponsors/leftfield-geospatial>`__ to fund its development.

Bug reports and feature requests can be made with the `github issue tracker <https://github.com/leftfield-geospatial/orthority/issues>`__.  Funded requests will be prioritised.

Development
-----------

To set up a development environment, start by cloning a fork of the repository:

.. code-block:: bash

    git clone https://github.com/<username>/orthority
    cd orthority


If installing with `pip <https://pip.pypa.io/>`_, you can install dependencies and link the repository into your environment with:

.. code-block:: bash

    pip install -e .[tests]

If installing into a `conda <https://docs.conda.io/projects/miniconda>`_ environment, it is best to install the `dependencies <https://github.com/leftfield-geospatial/orthority/blob/main/pyproject.yaml>`__ with ``conda`` first, before running the command above.

Please work on features in a new branch, and submit your changes as a `GitHub pull request <https://docs.github.com/articles/about-pull-requests>`__ for review.  It is recommended to discuss possible pull requests in an issue beforehand.

Orthority uses `black <https://black.readthedocs.io>`__ for formatting (with settings in |pyproject.yaml|_), and the RST docstring style.  Please include `pytest <https://docs.pytest.org>`__ unit tests with your code.

.. |pyproject.yaml| replace:: ``pyproject.yaml``
.. _pyproject.yaml: https://github.com/leftfield-geospatial/orthority/blob/main/pyproject.yaml
