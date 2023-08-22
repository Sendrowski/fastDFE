.. _introduction:

Introduction
============
fastDFE is a user-friendly software package designed for estimating the distribution of fitness effects (DFE) in evolutionary biology, ecology, and conservation. Building upon the foundation laid by polyDFE :cite:`polydfe`, fastDFE addresses the limitations of its predecessor by providing a faster, more flexible, and user-friendly approach to DFE inference.

fastDFE is implemented in Python but also offers compatibility with R through the reticulate package, ensuring a smooth integration with existing workflows. In the R wrapper, visualizations have been reimplemented for a consistent user experience. The package is thoroughly documented and tested, ensuring its reliability and ease of use for researchers in the field.

Motivation
----------
The DFE plays a crucial role in understanding how natural selection shapes genetic variation within and between species. Accurate estimation of the DFE is essential for a wide range of applications, including conservation management, understanding species' responses to changing environments, and the identification of functionally important genomic regions. fastDFE aims to meet the growing need for efficient and adaptable tools that can be easily adopted by researchers working with increasingly complex genomic data.

How it works
------------
As it precursors, fastDFE infers the DFE by contrasting the Site Frequency Spectrum (SFS) of putatively neutral (synonymous) and selected (non-synonymous) sites. The effect of demography is corrected for by introducing nuisance parameters which rescale the observed neutral SFS to the standard Kingman SFS. Assuming that demography affects the neutral and selected SFS alike, we apply the same rescaling to the observed selected SFS. To infer selection from the (demography-corrected) SFS, we make use of the expected allele frequency sojourn times obtained from Poisson Random Field (PRF) theory. In this step we need to integrate over the DFE to obtain the expected (selected) SFS. In fastDFE, this integral is linearized which results in significant speed improvements over polyDFE. To optimize for the best possible DFE, maximum likelihood optimization (MLE) is used.

Features
--------
fastDFE offers a range of features to facilitate DFE inference, including:

* Joint inference of DFE among different types with shared parameters
* Parametric bootstrapping
* Easy and flexible visualization of results
* Inclusion of covariates and likelihood ratio tests (LRTs) to assess their significance
* Nested model comparison with LRTs
* Customizable DFE parametrizations
* Estimation of α, the proportion of beneficial substitutions
* Serialization of objects and result summaries
* Support for YAML configuration files facilitating usage
* Built-in VCF parser for extracting site frequency spectra (SFS) with support for versatile stratifications and annotations

.. bibliography::
   :style: plain