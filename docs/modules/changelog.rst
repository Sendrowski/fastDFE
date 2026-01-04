.. _modules.changelog:

Changelog
=========

[1.2.1] - 2026-01-04
^^^^^^^^^^^^^^^^^^^^
- Don't average likelihood over bootstraps by default as it may mitigate optimization noise but is not statistically sound (see `update_likelihood <../search.html?q=update_likelihood>`_).
- Set ``do_bootstrap`` flag to false by default for methods performing nested model comparison.

[1.2.0] - 2025-12-24
^^^^^^^^^^^^^^^^^^^^
- Adding dominance coefficient ``h`` to DFE inference. ``h`` can either be fixed or inferred jointly with the DFE parameters. We can also introduce a relationship between ``h`` and ``S`` (see `h_callback <../search.html?q=h_callback>`_). When estimating ``h``, precomputation over a grid of dominance coefficients takes some time. Implementation was validated with ``SLiM``.
- By default, parameters are now fixed to infer a semidominant (``h=0.5``) deleterious DFE without correcting for ancestral misidentification (``eps=0``).
- Improved bootstrapping. By default, 2 runs are carried out per bootstrap sample and the most likely result is taken (see `n_bootstrap_retries <../search.html?q=n_bootstrap_retries>`_ which previously controlled the number of retries in case of optimization failure). Bootstrapping is now also carried out by default (`do_bootstrap <../search.html?q=do_bootstrap>`_), and mean and standard deviation across bootstraps are logged.
- Initial optimization runs are now recorded in :attr:`~fastdfe.base_inference.BaseInference.runs` dataframe.
- Added :class:`~fastdfe.parametrization.DFE` class representing a frozen :class:`~fastdfe.parametrization.Parametrization`.
- Expanded documentation on SFS parsing and DFE inference.
- Allow to specify how the point estimate is determined when plotting discretized DFE with confidence intervals (see `point_estimate <../search.html?q=point_estimate>`_).
- Refactored :class:`~fastdfe.base_inference.InferenceResult`.
- Refactored methods returning CIs. For example, removed ``get_cis_params_mle()``, use `get_errors_params_mle() <../search.html?q=get_errors_params_mle>`_ instead.

[1.1.13] - 2025-11-22
^^^^^^^^^^^^^^^^^^^^^
- Fixed bootstrap issue where seeding caused unwanted correlation between the resampled neutral and selected SFS, which could result in smaller confidence intervals.

[1.1.12] - 2025-05-26
^^^^^^^^^^^^^^^^^^^^^
- Made ``cyvcf2`` an optional dependency via ``fastdfe[vcf]``
- Added support for ``Conda``
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

