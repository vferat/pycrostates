Install
=======

Dependencies
------------

* ``mne`` (>=1.0)
* ``numpy`` (>=1.20)
* ``scipy``
* ``scikit-learn``
* ``matplotlib``
* ``pooch``
* ``decorator``
* ``jinja2``

We require that you use Python ``3.7`` or higher.

pycrostates works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, see their `installation instructions <https://mne.tools/stable/install/index.html>`_.

.. tab-set::

    .. tab-item:: Pypi

        .. code-block:: bash

            pip install stimuli

    .. tab-item:: Conda

        .. code-block:: bash

            conda install -c conda-forge stimuli

    .. tab-item:: MNE installers

        As of MNE-Python 1.1, ``pycrostates`` is distributed in the
        `MNE standalone installers <https://mne.tools/stable/install/installers.html>`_.

        The installers create a conda environment with the entire MNE-ecosystem
        setup, and more!

    .. tab-item:: Snapshot of the current version

        .. code-block:: bash

            $ pip install git+https://github.com/vferat/pycrostates
