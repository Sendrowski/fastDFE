.. _introduction:

Introduction
============
fastDFE is a user-friendly software package designed for estimating the distribution of fitness effects (DFE) in evolutionary biology, ecology, and conservation. Building upon the foundation laid by polyDFE (Tataru et al., 2017), fastDFE addresses the limitations of its predecessor by providing a faster, more flexible, and user-friendly approach to DFE inference.

fastDFE is implemented in Python and offers compatibility with R through the reticulate package, ensuring a smooth integration with existing workflows. The package is thoroughly documented and tested, ensuring its reliability and ease of use for researchers in the field.

Motivation
----------
The DFE plays a crucial role in understanding how natural selection shapes genetic variation within and between species. Accurate estimation of the DFE is essential for a wide range of applications, including conservation management, understanding species' responses to changing environments, and the identification of functionally important genomic regions. fastDFE aims to meet the growing need for efficient and adaptable tools that can be easily adopted by researchers working with increasingly complex genomic data.

Features
--------
fastDFE offers a range of features to facilitate DFE inference, including:

* Joint inference of DFE among different types with shared parameters
* Parametric bootstrapping
* Easy and flexible visualization of results
* Inclusion ofcovariates and likelihood ratio tests (LRTs) to assess their significance
* Nested model comparison with LRTs
* Customizable DFE parameterizations
* Estimation of α, the proportion of beneficial substitutions
* Serialization of objects and result summaries
* Support for YAML configuration files facilitating usage
