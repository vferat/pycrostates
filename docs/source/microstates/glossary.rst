Glossary
========

.. glossary::
    :sorted:

    GFP
    global field power
        Global Field Power (GFP) is a measure of the (non-)uniformity of the
        electromagnetic field at the sensors. It is typically calculated as the
        standard deviation of the sensor values at each time point. Thus, it is
        a one-dimensional time series capturing the spatial variability of the
        signal across sensor locations. Local maxima of the global field power
        (GFP) are known to represent the portions of EEG data with highest
        signal-to-noise ratio (:cite:t:`KOENIG20161104`).

    GEV
    global explained variance
        Global explained variance.
        Total explained variance expressed by a given state.
        It is computed as the sum of the explained variance
        multiplied by the :term:`Global Field Power` (:term:`GFP`) of each
        sample assigned to a given state.

    cluster centers
        Arithmetic mean of all the points belonging to the cluster. In the
        context of microstate analysis, it corresponds to the microstates
        topographies.

    inter-cluster distance
        Distance between two datapoints belonging to two different clusters.
        Depending on the metric used, it can be computed in different ways, for
        example (but no limited to) as the average distance between all
        datapoint belonging to different clusters, the distance between two
        :term:`cluster centers`, the minimal distance between two datapoints
        belonging to different cluster.

        .. image:: ../../_static/img/inter_cluster_distance_dm.png
            :class: only-dark

        .. image:: ../../_static/img/inter_cluster_distance_lm.png
            :class: only-light

    intra-cluster distance
        Distance between two datapoint belonging to the same cluster. Depending
        on the metric used, it can be computed in different ways, for example
        (but no limited to) as the average distance between all datapoint
        belonging to the same cluster, the average distance between all
        datapoint at the cluster center or the maximal distance between two
        datapoints belonging to the same cluster.

        .. image:: ../../_static/img/intra_cluster_distance_dm.png
            :class: only-dark

        .. image:: ../../_static/img/intra_cluster_distance_lm.png
            :class: only-light
