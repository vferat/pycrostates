Microstates
===========

.. include:: ../links.inc

What is EEG microstates ?
^^^^^^^^^^^^^^^^^^^^^^^^^

Microstates analysis is a method allowing investigation of spatiotemporal characteristics of EEG recordings. 
It consists in breaking down the multichannel EEG signal into a succession of quasi-stable state, each state being
characterized by a spatial distribution of its scalp potentials also called microstate map or microstate topography.

.. sidebar:: Looking for more information?

    If you are looking to learn more about microstates we recommend you to read the following review: `michel 2018`_.

How to compute EEG microstates ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This decomposition is based on two consecutive steps: the clustering which allows to define topographies that best represent the studied data
and the backfitting than consist on assigning each timepoint of one or several EEG recordings to the previously extracted topographies.

The methods relies on assigning timepoints to the most similar microstate map, that is why it is important to define how distance between two topographies is computed.
For microstate analysis, it is the inverse of the absolute value of the spatial correlation that is used as a measure of distance to carry out all computations.
The absolute value is used in order to ignore topography polarities of the topographies, as state can oscillate in a given configuration (i.e topography).

Pycrostates implements a conveniente class :class:`pycrostates.clustering.ModKMeans` 
allowing to perform clustering through the :meth:`pycrostates.clustering.ModKMeans.fit` method
and backfitting throught the :meth:`pycrostates.clustering.ModKMeans.predict` method.
It also implements other methods to facilitate the analysis and display of results.