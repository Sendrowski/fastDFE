# VCF parsing
``fastdfe`` infers a DFE from site-frequency spectra. The classes that compute these spectra from a VCF file, which
are the parser and its stratifications, annotations and filtrations, are provided by
[``sfsutils``](https://sfsutils.readthedocs.io), a separate package installed with ``fastdfe``. ``fastdfe`` re-exports
them, so that ``fastdfe.Parser`` and ``sfsutils.Parser`` are the same class.

```{note}
These classes were part of ``fastdfe`` itself before version 1.4.0. Existing imports such as ``fastdfe.Parser`` and
``fastdfe.parser.Parser`` continue to work.
```

```{code-cell} python
import fastdfe as fd
import sfsutils as su

fd.Parser.__module__, su.Parser.__module__
```

```{code-cell} python
:tags: [remove-cell]
assert fd.Parser is su.Parser
```

```{code-cell} r
library(fastdfe)

fd <- load_fastdfe()
su <- reticulate::import("sfsutils")

c(fd$Parser$`__module__`, su$Parser$`__module__`)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(reticulate::py_id(fd$Parser) == reticulate::py_id(su$Parser))
```

{class}`~sfsutils.parser.Parser` reads a VCF and returns the {class}`~sfsutils.spectrum.Spectra` that the inference
pages take as input. It determines polarization from the ``AA`` tag by default (see
{attr}`~sfsutils.parser.Parser.skip_non_polarized`), and takes three kinds of component:

- Stratifications split the spectrum by site property, most importantly
  {class}`~sfsutils.parser.DegeneracyStratification`, which gives the neutral and selected spectra a DFE inference
  needs.
- Annotations add the site information a stratification reads.
  {class}`~sfsutils.annotation.DegeneracyAnnotation` determines each site's degeneracy from a reference genome and a
  GFF, which {class}`~sfsutils.parser.DegeneracyStratification` then splits on, and an ancestral allele annotation
  provides the polarization an unfolded spectrum requires.
- Filtrations drop sites before they reach the spectrum, such as
  {class}`~sfsutils.filtration.BiasedGCConversionFiltration`, and are applied to a VCF directly with the
  {class}`~sfsutils.filtration.Filterer`.

For worked examples in Python and R, see [``sfsutils``](https://sfsutils.readthedocs.io)' guides on
[parsing](https://sfsutils.readthedocs.io/en/latest/reference/parser.html),
[annotations](https://sfsutils.readthedocs.io/en/latest/reference/annotations.html),
[filtrations](https://sfsutils.readthedocs.io/en/latest/reference/filtrations.html) and
[manipulating the SFS](https://sfsutils.readthedocs.io/en/latest/reference/spectra.html).
