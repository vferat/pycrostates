.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: changes from first-time contributors should be added to the TOP of
   the relevant section (Enhancements / Bugs / API changes), and should look
   like this (where xxxx is the pull request number):

       - description of enhancement/bugfix/API change (:pr:`xxxx` by `Firstname Lastname`_)

   Also add a corresponding entry for yourself in docs/dev/changes/names.inc

.. include:: ./names.inc

.. _latest:

Current 0.3.0.dev
-----------------

Enhancements
~~~~~~~~~~~~
- Improve changelog. (:pr:`86` by `Victor Férat`_)
- Add ``Microstate Segmentation`` tutorial. (:pr:`91` by `Victor Férat`_)
- Add :meth:`pycrostates.segmentation.RawSegmentation.get_transition_matrix` and :meth:`pycrostates.segmentation.RawSegmentation.get_expected_transition_matrix`. (:pr:`91` by `Victor Férat`_)

Bugs
~~~~


API changes
~~~~~~~~~~~

