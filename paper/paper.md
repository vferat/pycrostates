---
title: 'Pycrostates: : a python library to study EEG microstates'
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

Pycrostates is a free and open source project for electroencephalography (EEG) microstates analysis. Microstates analysis is a method allowing investigation of spatiotemporal characteristics of EEG recordings. It consists in breaking down the multichannel EEG signal into a succession of quasi-stable state, each state being characterized by a spatial distribution of its scalp potentials also called microstate map or microstate topography. Pycrostates implements core functions such as preprocessing tools (global field power peaks extraction, resampling), clustering algorithm (modified version of the kmeans algorithm), clustering quality evaluations tools (silhouette, Dunn) and backfitting (also called prediction) needed to perform such analysis. It provides researchers all the block needed to design their own microstate analysis. Pycrostates is built to fit as best as it can in the python scientific environment and more particularly scikit-learn [@pedregosa_scikit-learn_nodate] and MNE-python [@gramfort_meg_2013]  from which it is inspired in its philosophy and its implementation [@buitinck_api_2013]. The library comes with extensive documentation including descriptions of all its algorithms and functions as well as several tutorials to help researcher to get started. Finally, pycrostates is provided under the new BSD license allowing code reuse, even in commercial products.

# Statement of need

Statement of need

# Mathematics

Mathematics

# Citations

Citations

# Figures

Figures

# Acknowledgements

Acknowledgements

# References