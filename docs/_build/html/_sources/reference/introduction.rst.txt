.. _introduction:

Introduction
============
fastDFE is a software package designed for estimating the distribution of fitness effects (DFE) from site-frequency spectra (SFS). Building upon the foundation laid by polyDFE :cite:`polydfe`, fastDFE addresses the limitations of its predecessor by providing a faster, more flexible, and user-friendly approach to DFE inference.

fastDFE is implemented in Python but also offers compatibility with R through the reticulate package, ensuring a smooth integration with existing workflows. In the R wrapper, visualizations have been reimplemented for a consistent user experience. The package is thoroughly documented and tested, ensuring its reliability and ease of use for researchers in the field.

Motivation
----------
The DFE is instrumental in understanding how natural selection shapes genetic variation. SFS-based methods for estimating the DFE condense population genetic variation by quantifying the number of alleles at specific frequencies, discarding details about which sites exhibited which frequencies. As a result, we obtain one DFE for all variants used. This approach provides limited information when considering a single DFE, but its utility significantly increases when comparing DFEs across multiple species, populations, or genomic regions. fastDFE improves upon the groundwork laid by polyDFE, overcoming its constraints such as extensive computational time and limited scalability :cite:`polydfe,polydfe2`. It is specifically designed to facilitate and encourage the joint inference of multiple DFEs at once. Given the importance of consistency when comparing DFEs from different datasets, it's essential to derive the SFS in a similar manner to ensure results are directly comparable. To aid this process, fastDFE includes a VCF parser, enabling the extraction of the necessary SFS input data from raw VCF files.

How it works
------------
As it precursors, fastDFE infers the DFE by contrasting the site-frequency spectrum (SFS) of putatively neutral (synonymous) and selected (nonsynonymous) sites :cite:`dfe-alpha,grapes,polydfe`. fastDFE builds upon the model of polyDFE, introducing several improvements: The impact of demography is corrected for by introducing nuisance parameters which are fit so as to re-scale the observed neutral SFS to the standard Kingman SFS. Assuming that demography affects the neutral and selected SFS alike, we apply the same re-scaling to the observed selected SFS. Unlike polyDFE, demography is inferred solely from the synonymous sites, which should provide sufficient information for reasonably large sample sizes, and has the benefit to speed up calculations. To infer selection from the (demography-corrected) SFS, we also make use of the expected allele frequency sojourn times obtained from Poisson random field (PRF) theory. Obtaining the modelled (selected) SFS requires integrating over the DFE in question. In fastDFE, this integral is discretized, precomputed and cached, leading to significant speed enhancements over polyDFE while producing very similar results. Discontinuities and asymptotes are furthermore handled properly by parametrizing the DFE by means of its cumulative distribution function, which provides more reliable estimates. Ultimately, the most likely DFE is obtained using maximum likelihood estimation (MLE).

Features
--------

fastDFE offers a range of features designed to facilitate DFE inference.

**Workflow**: streamlining the overall data preparation and inference process:

- Built-in VCF-to-SFS parser, with support for versatile stratification, site annotation, and filtering
- Site-degeneracy annotation
- Ancestral-allele annotation with outgroups
- Utilities to determine the number of mutational target sites when monomorphic sites are not present in the provided VCF file
- Support for configuration files, facilitating reproducibility
- Serialization of objects and result summaries
- Object-oriented and customizable design

**Modeling**: robust utilities for fitting models to data and estimating DFE parameters:

- Joint inference of DFE across different types which share parameters
- Inclusion of covariates and likelihood ratio tests (LRTs) to assess their significance
- Parametric bootstrapping
- Nested model comparison with LRTs
- Customizable DFE parametrizations
- Estimation of :math:`\alpha`, the proportion of beneficial substitutions

**Visualization**: encompassing a variety of visualization utilities:

- Visualization of DFE, nested model p-values, inferred parameters, and their confidence intervals and more
- Support for both Python and R

References
----------
.. bibliography::
   :style: plain