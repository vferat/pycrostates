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

We require that you use Python 3.7 or higher.
You may choose to install ``pycrostates`` via conda or via pip.

pycrostates works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, see their `installation instructions <https://mne.tools/stable/install/index.html>`_.

As of MNE-Python 1.1, ``pycrostates`` is distributed in the
`MNE standalone installers <https://mne.tools/stable/install/installers.html>`_.

Installation via Conda
----------------------

To install the latest stable version, use ``conda`` in your terminal:

.. code-block:: bash

    $ conda install -c conda-forge pycrostates

Installation via Pip
--------------------

To install the latest stable version, use ``pip`` in your terminal:

.. code-block:: bash

    $ pip install pycrostates

To install the latest development version, run:

.. code-block:: bash

    $ pip install git+https://github.com/vferat/pycrostates
