.. include:: ../links.inc

Install
=======

Dependencies
------------

* ``mne`` (>=1.0)
* ``numpy`` (>=1.21)
* ``scipy``
* ``scikit-learn``
* ``matplotlib``
* ``pooch``
* ``decorator``
* ``jinja2``

We require that you use Python ``3.7`` or higher.

pycrostates works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, see `MNE installation instructions <mne install_>`_.

.. tab-set::

    .. tab-item:: Pypi

        ``pycrostates`` can be installed from `Pypi <project pypi_>`_:

        .. code-block:: bash

            pip install pycrostates

    .. tab-item:: Conda

        ``pycrostates`` can be installed from `conda-forge <project conda_>`_:

        .. code-block:: bash

            conda install -c conda-forge pycrostates

    .. tab-item:: MNE installers

        As of MNE-Python 1.1, ``pycrostates`` is distributed in the
        `MNE standalone installers <mne installers_>`_.

        The installers create a conda environment with the entire MNE-ecosystem
        setup, and more!

    .. tab-item:: Snapshot of the current version

        ``pycrostates`` can be installed from `GitHub <project github_>`_:

        .. code-block:: bash

            $ pip install git+https://github.com/vferat/pycrostates
