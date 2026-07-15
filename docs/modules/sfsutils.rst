.. _modules.sfsutils:

SFS & VCF handling
------------------

Site-frequency spectra objects, VCF-to-SFS parsing, ancestral-allele and
site-degeneracy annotation, and site filtration are provided by the standalone
``sfsutils`` package (`repository <https://github.com/Sendrowski/SFSUtils>`_). ``fastdfe`` depends on it
and re-exports these classes, so they remain importable directly from ``fastdfe``
(e.g. ``from fastdfe import Spectrum, Spectra, Parser, Annotator, Filterer``).

The full API reference is hosted in the ``sfsutils`` `documentation
<https://sfsutils.readthedocs.io>`_:

- `Spectrum <https://sfsutils.readthedocs.io/en/latest/modules/spectrum.html>`_ — a single site-frequency spectrum, with folding, polarising, resampling, and plotting.
- `Spectra <https://sfsutils.readthedocs.io/en/latest/modules/spectra.html>`_ — a named collection of spectra supporting grouped operations and joint visualisation.
- `Parser <https://sfsutils.readthedocs.io/en/latest/modules/parser.html>`_ — parsing spectra from VCF files, with support for versatile stratification.
- `Annotator <https://sfsutils.readthedocs.io/en/latest/modules/annotation.html>`_ — ancestral-allele and site-degeneracy / synonymy annotation of VCF sites.
- `Filterer <https://sfsutils.readthedocs.io/en/latest/modules/filtration.html>`_ — filtering VCF sites prior to parsing.
