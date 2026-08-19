.. include:: shared.txt

Contributing
============

Contributions are welcome.

If Orthority is useful to you, please consider supporting its development and maintenance with `a donation <https://github.com/sponsors/leftfield-geospatial>`__.

Bug reports and feature requests can be made with the `GitHub issue tracker <https://github.com/leftfield-geospatial/orthority/issues>`__.

Development
-----------

To set up a development environment, start by cloning a fork of the repository:

.. code-block:: bash

    git clone https://github.com/<username>/orthority
    cd orthority

If installing with pip_, you can install dependencies and link the repository into your environment with:

.. code-block:: bash

    pip install --group tests -e .

If installing into a conda_ environment, it is best to install the `dependencies <https://github.com/leftfield-geospatial/orthority/blob/main/pyproject.toml>`__ with ``conda`` first, before running:

.. code-block:: bash

    pip install --no-deps -e .

Please work on features in a new branch, and submit your changes as a `GitHub pull request <https://docs.github.com/articles/about-pull-requests>`__ for review.

Orthority uses `Ruff <https://docs.astral.sh/ruff>`__ for linting and formatting (with settings in |pyproject.toml|_), and the RST docstring style.  Please include `pytest <https://docs.pytest.org>`__ unit tests with your code.

.. |pyproject.toml| replace:: ``pyproject.toml``
.. _pyproject.toml: https://github.com/leftfield-geospatial/orthority/blob/main/pyproject.toml
