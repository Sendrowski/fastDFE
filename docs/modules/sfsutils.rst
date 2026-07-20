.. _modules.sfsutils:

SFS & VCF handling
------------------

Site-frequency spectra objects, VCF-to-SFS parsing, ancestral-allele and
site-degeneracy annotation, and site filtration are provided by the standalone
`sfsutils <https://sfsutils.readthedocs.io>`_ package. ``fastdfe`` depends on it
and re-exports these classes, so they remain importable directly from ``fastdfe``
(e.g. ``from fastdfe import Spectrum, Spectra, Parser, Annotator, Filterer``).

The corresponding ``sfsutils`` API reference pages:

- `Spectrum classes <https://sfsutils.readthedocs.io/en/latest/modules/spectrum.html>`_ — the :class:`~sfsutils.spectrum.Spectrum` and :class:`~sfsutils.spectrum.Spectra` site-frequency spectra, with folding, polarising, resampling, and visualisation.
- `Parsing <https://sfsutils.readthedocs.io/en/latest/modules/parser.html>`_ — the :class:`~sfsutils.parser.Parser` for extracting spectra from VCF files.
- `Site stratification <https://sfsutils.readthedocs.io/en/latest/modules/stratification.html>`_ — stratifying sites into separate spectra, e.g. by degeneracy or base context.
- `Site annotation <https://sfsutils.readthedocs.io/en/latest/modules/annotation.html>`_ — the :class:`~sfsutils.annotation.Annotator` for ancestral-allele and site-degeneracy / synonymy annotation.
- `Site filtration <https://sfsutils.readthedocs.io/en/latest/modules/filtration.html>`_ — the :class:`~sfsutils.filtration.Filterer` for excluding sites prior to parsing.
- `Input and output <https://sfsutils.readthedocs.io/en/latest/modules/io.html>`_ — reading VCF, FASTA and GFF sources and serialising results.
