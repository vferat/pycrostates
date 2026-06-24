.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below, so the changelog page will have a link to
   the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. include:: ./names.inc

.. _latest:

Version 0.7
-----------

Enhancements
~~~~~~~~~~~~

- Update ``pycrostates`` function to use different GFP computation functions based on data type (:pr:`197` by `Victor Férat`_).
- Update the distance function used in ``pycrostates.metrics`` from ``1 / |corrcoef|`` to ``1 - |corrcoef|`` (:pr:`220` by `Victor Férat`_).

Bugs
~~~~

- Update montage name from ``standard_1020`` to ``colin27_1005`` for mne.version >= 1.13 (:pr:`276` by `Victor Férat`_).

API and behavior changes
~~~~~~~~~~~~~~~~~~~~~~~~

- xxx
