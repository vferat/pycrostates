---
title: 'Pycrostates: a Python library to study EEG microstates'
tags:
  - Python
  - neuroscience
  - neuroimaging
  - eeg
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
Microstate analysis of the electroencephalogram (EEG), introduced in 1972 by Lehman [citation] is a spatiotemporal analysis technique that takes advantage of the full spatial resolution of EEG recordings. Formalized by Pascual Marqui and colleagues [@pascual-marqui_segmentation_1995], it consists in studying the distribution of the surface EEG potential maps over time and thus transforming the EEG recordings into sequences of successive states of variable duration called EEG microstates.


Pycrostates implements multiple modules that allow researchers to apply this method to their data:

- the dataset module provides direct access to preprocessed EEG data from the LEMON dataset [@babayan_mind-brain-body_2019] and thus offers test and experimental data.
the preprocessing module completes the classical EEG preprocessing tools [@pernet2020issues] with those specific to the preparation of data for microstate analysis.

- a segmentation module allows the sequences of microstates obtained to be studied

- the cluster module gathers the different algorithms of transformation of the recordings in sequence of microstates. It takes example of the API of scikit-learn [@pedregosa_scikit-learn_nodate] [@buitinck_api_2013] for a simple and effective integration.

- a segmentation module that allows to study the sequences of microstates obtained from clustering prediction.

- a metrics module that allows to quantify the quality of the clustering results.

By construction, Pycrostates is evolutionary. Future improvements can easily be added, such as the addition of clustering algorithms or new tools for sequence analysis such as Markov chains. Authors welcome proposals for improvements and new contributions and look forward to promoting and advancing the analysis of EEG microstates.

# Statement of need

Today, there are several software suites and libraries that allow research to carry out EEG microstatate analysis such as Cartool [@brunet_spatiotemporal_nodate], Tomas Koening's Matlab plugins (https://www.thomaskoenig.ch/index.php/software/microstates-in-eeglab),the Matlab microstates plugin [@poulsen_microstate_2018] and some python toolboxes[@von_wegner_information-theoretical_2018]. In the last few years, the python programming language ecosystem has expanded widely, especially in scientific fields. In neuroscience, the MNE-python library [@gramfort_meg_2013] stands out for the analysis of EEG signals. However, it does not currently offer an implementation allowing the decomposition of EEG recordings and the analysis of microstates. Pycrostates is based on this existing system and completes it by offering researchers simple tools that can be easily integrated with existing tools for EEG microstate analysis.
In addition to providing a simple but complete API that can replicate most of the analyses proposed in the literature [@MICHEL2018577], it is built in a modular and scalable way and provided exhaustive documentation and tutorials. Pycrosatates follows modern development methods and uses a number of code review and testing tools, thus facilitating its maintenance and evolution over time.

# Acknowledgements

Acknowledgements

# References