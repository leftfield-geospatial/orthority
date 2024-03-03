File paths and URIs
===================

Command line and API file parameters can be specified by local file paths or remote URIs.  Orthority uses `fsspec <https://github.com/fsspec/filesystem_spec>`__ to provide built-in support for `a number of remote file systems <https://filesystem-spec.readthedocs.io/en/stable/api.html#implementations>`__.  Support for other remote systems, including cloud storage, is available by installing `the relevant fsspec extension package <https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations>`__.  See the `fsspec documentation <https://filesystem-spec.readthedocs.io/en/stable/features.html#configuration>`__ if your file system requires credentials or other configuration.

On the :doc:`command line <../cli/index>`, ``SOURCE`` images can be specified with local or remote wildcard patterns, and the ``--out-dir`` option can refer to a local path or remote URI.  Note that some file systems, like HTTP, are read-only and don't support wildcards.

.. TODO: add a note about sidecar files (PAM and RPC) not being supported.
