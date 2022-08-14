Microstates
===========

.. include:: ../links.inc

What is EEG microstates ?
^^^^^^^^^^^^^^^^^^^^^^^^^

Microstates analysis is a method allowing investigation of spatiotemporal
characteristics of EEG recordings. It consists of breaking down the
multichannel EEG signal into a succession of quasi-stable state, each state
being characterized by a spatial distribution of its scalp potentials also
called microstate map or microstate topography.

.. sidebar:: Looking for more information?

    If you are looking to learn more about microstates we recommend you to read
    the following review: `michel 2018`_.

How to compute EEG microstates ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This decomposition is based on two consecutive steps: the clustering which
allows to define topographies that best represent the studied data and the
backfitting than consist on assigning one of the previously extracted
topographies to each timepoint of one or several EEG recordings.

The methods relies on assigning timepoints to the most similar microstate map,
which is why it is important to define how distance between two topographies is
computed. For microstate analysis, the inverse of the absolute value of the
spatial correlation is used as a measure of distance to carry out all
computations. The absolute value is used in order to ignore the topography
polarity.

Pycrostates implements a convenient class
:class:`pycrostates.cluster.ModKMeans` to perform clustering through
the :meth:`pycrostates.cluster.ModKMeans.fit` method and backfitting through
the :meth:`pycrostates.cluster.ModKMeans.predict` method. It also implements
other methods to facilitate the analysis and display of results.
