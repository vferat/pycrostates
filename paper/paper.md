---
title: 'Pycrostates: a Python library to study EEG microstates'
tags:
  - Python
  - neuroscience
  - neuroimaging
  - EEG
  - microstates
  - brain
authors:
  - name: Victor Férat^[Co-first author]^[Corresponding author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0003-1952-7657
    affiliation: 1
  - name: Mathieu Scheltienne^[Co-first author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0001-8316-7436
    affiliation: 2
  - name: Denis Brunet
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
  - name: Tomas Ros
    orcid: 0000-0001-6952-0459
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
  - name: Christoph Michel
    orcid: 0000-0003-3426-5739
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
affiliations:
 - name: Functional Brain Mapping Laboratory, Department of Basic Neurosciences, Campus Biotech, University of Geneva, Geneva, Switzerland
   index: 1
 - name: Human Neuroscience Platform, Fondation Campus Biotech Geneva, Geneva, Switzerland
   index: 2
 - name: Centre for Biomedical Imaging (CIBM) Lausanne-Geneva, Geneva, Switzerland
   index: 3
date: 30 May 2022
bibliography: paper.bib
---

# Summary
Microstate analysis of the electroencephalogram (EEG), introduced in 1972 by Lehman [@lehmann1971multichannel], is a spatiotemporal analysis technique that takes advantage of the full spatial resolution of EEG recordings. Formalized further by Pascual Marqui and colleagues [@pascual-marqui_segmentation_1995], microstate analysis studies the distribution of the surface EEG potential maps over time. It transforms the EEG recordings into sequences of successive states of variable duration, called EEG microstates.

Pycrostates implements multiple modules that allow researchers to apply microstate analysis on their data:

- the cluster module supports the different clustering algorithms that find the optimal topographic maps that sparsely represent the EEG. 
- the segmentation module supports the study of microstates sequences and their summary measures.
- the metrics module quantifies the quality of the fitted clustering algorithms.

Additional modules help researchers to develop analysis pipelines:

- the dataset module provides direct access to preprocessed data that can be used to test pipelines. As of writing, it supports the LEMON dataset [@babayan_mind-brain-body_2019] comprising preprocessed EEG recordings of 227 healthy participants.
- the viz module provides visualization tools for performing microstate analyses.

By design, Pycrostates supports natively all data types from MNE-Python [@gramfort_meg_2013] and is not restricted to EEG data. It is designed to support future improvements and additions such as different clustering algorithms or new tools for sequence analysis such as Markov chains. Moreover, this work builds its API on top of the robust MNE-Python ecosystem, enabling a seamless integration of microstates analysis.

# Statement of need

Currently, several software and libraries are available to perform EEG microstates analysis: Cartool [@brunet_spatiotemporal_nodate], Thomas Koening's Matlab plugins (https://www.thomaskoenig.ch/index.php/software/microstates-in-eeglab), the Matlab microstates plugin [@poulsen_microstate_2018], or the python 2 library from Frederic von Wegner [@von_wegner_information-theoretical_2018]. In the last few years, the Python programming language ecosystem has expanded considerably, especially in scientific fields. The MNE-Python [@gramfort_meg_2013] library stands out for the analysis of human neurophysiological data. However, the current implementation of microstates analysis from Marijn van Vliet has a limited number of features and is not maintained on a regular basis. Pycrostates is based on the robust MNE-Python ecosystem and complements it with simple tools for developing integrated microstates analysis pipelines. 
In addition to providing a simple and complete API that can replicate most of the analyses proposed in the literature [@MICHEL2018577], Pycrostates is built with a modular and scalable design, following modern development methods and standards to facilitate its maintenance and evolution over time. Finally, new users will benefits from the exhaustive documentation and tutorials describing the microstate analysis modules.

With this contribution, the developer team is excited to provide the state of the art in microstates analysis and is looking forward to welcoming new contributors and users from the broader MNE, neuroscience, and electrophysiology community.

# Acknowledgements

Pycrostates development is partly supported by the Swiss National Science Foundation (grant No. 320030_184677) and by the Human Neuroscience Platform, Fondation Campus Biotech Geneva, Geneva, Switzerland.

# References