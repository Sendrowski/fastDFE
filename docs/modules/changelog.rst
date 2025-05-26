.. _modules.changelog:

Changelog
=========

[1.1.12] - 2025-05-26
^^^^^^^^^^^^^^^^^^^^^
- Made ``cyvcf2`` an optional dependency via ``fastdfe[vcf]``
- Improved x-axis labels for discretized DFE plots
- Added support for introducing ancestral misidentification in ``Spectrum``
- Added unnormalized Watterson’s theta property

[1.1.11] - 2025-03-30
^^^^^^^^^^^^^^^^^^^^^
- Support for passing alternative optimizer to ``BaseInference`` and ``JointInference``.

[1.1.10] - 2025-03-25
^^^^^^^^^^^^^^^^^^^^^
- Allow for probabilistic polarization when parsing SFS by looking at ancestral allele probability VCF info tag
- Allow the transition/transversion ratio in the ``K2SubstitutionModel`` to be fixed to the value observed in the data
- Adjust LRTs to account for parameters near boundaries. The resulting p-values are similar but tend to be somewhat lower
- Extend ``ExistingOutgroupFiltration`` so that number of missing outgroups can be specified
- Add ``RandomStratification`` and ``ContigFiltration`` classes

[1.1.9] - 2025-01-01
^^^^^^^^^^^^^^^^^^^^
- Add simulation class for simulating SFS data with known DFE

[1.1.8] - 2024-08-14
^^^^^^^^^^^^^^^^^^^^
- Update cyvcf2 dependency to fix broken wheel for Mac ARM (see `issue <https://github.com/brentp/cyvcf2/issues/305>`_)
- Fix minor problem with remote files when disabling file caching

[1.1.7] - 2024-05-31
^^^^^^^^^^^^^^^^^^^^
- Implement serialization of maximum likelihood ancestral annotation to allow for later inspection of results
- Improved ancestral allele info tag information for site where annotation was not possible

[1.1.6] - 2024-04-20
^^^^^^^^^^^^^^^^^^^^
- Lazy-load some modules to allow faster initial loading

[1.1.5] - 2024-03-12
^^^^^^^^^^^^^^^^^^^^
- Support for Python 3.12
- Exclude large files from distribution to speed up installation in R

[1.1.4] - 2024-03-05
^^^^^^^^^^^^^^^^^^^^
- Lazy-load some modules to allow faster initial loading

[1.1.3] - 2023-12-27
^^^^^^^^^^^^^^^^^^^^
- Implement probabilistic subsampling for ancestral allele annotation

[1.1.2] - 2023-11-27
^^^^^^^^^^^^^^^^^^^^
- Improved bootstrapping
- Set ``allow_divergence`` flag to false by default which as it has the potential to bias the SFS

[1.1.1] - 2023-11-21
^^^^^^^^^^^^^^^^^^^^
- Probabilistic parsing of SFS
- Functionality to subsample already existing SFS to lower sample size
- Support for number of target sites for ancestral allele annotation
- Improved parallelization when bootstrapping
- New plotting functionalities
- Improved logging

[1.1.0] - 2023-10-10
^^^^^^^^^^^^^^^^^^^^
- Improved parsing utilities
- Ancestral allele annotation with outgroups

[1.0.0] - 2023-08-12
^^^^^^^^^^^^^^^^^^^^
- First stable release

