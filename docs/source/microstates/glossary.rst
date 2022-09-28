Glossary
========

.. glossary::
    :sorted:

    GFP
        Global Field Power. For EEG data, it is computed as the standard
        deviation of the sensors at a given timepoint. Local maxima of the
        global field power (GFP) are known to represent the portions of EEG
        data with highest signal-to-noise ratio (:cite:t:`KOENIG20161104`).

    GEV
        Global explained variance.
        Total explained variance expressed by a given state.
        It is computed as the sum of the explained variance
        multiplied by the :term:`GFP` of each sample
        assigned to a given state.

    cluster centers
        Arithmetic mean of all the points belonging to the cluster. In the
        context of microstate analysis, it corresponds to the microstates
        topographies.

    inter-cluster distance
        Distance between two datapoints belonging to two different clusters.
        Depending on the metric used, it can be computed in different ways, for example
        (but no limited to) as the average distance between all datapoint belonging to
        different clusters, the distance between two cluster centers, the minimal distance
        between two datapoints belonging to different cluster.

        .. figure:: ../../_static/img/inter_cluster_distance.png
            :width: 600
            :alt: Inter-cluster distance illustration.

    intra-cluster distance
        Distance between two datapoint belonging to same cluster. Depending on
        the metric used, it can be computed in different ways, for example
        (but no limited to) as the average distance between all datapoint
        belonging to same cluster, the average distance between all datapoint
        at the cluster center or the maximal distance between two datapoints
        belonging to the same cluster.

        .. figure:: ../../_static/img/intra_cluster_distance.png
            :width: 600
            :alt: Intra-cluster distance illustration.
