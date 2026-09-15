# VCF parsing
## Introduction
``fastdfe`` provides parser utilities that enable convenient parsing of frequency spectra from VCF files. By default, {class}`~sfsutils.parser.Parser` looks at the ``AA`` tag in the VCF file's info field to retrieve the correct polarization. Sites for which this tag is not well-defined are by default included (see {attr}`~sfsutils.parser.Parser.skip_non_polarized`). Non-polarized frequency spectra provide little information on the distribution of beneficial mutations, however.

We might also want to stratify the SFS by some site properties, such as site-degeneracy. This is done by passing stratifications to the parser. In this example, we stratify the SFS by 0-fold and 4-fold degenerate sites using a VCF file for ``Betula spp.``

+++
```{seealso}
VCF-to-SFS parsing, annotation, and filtration are provided by the standalone [``sfsutils``](https://sfsutils.readthedocs.io) package.
```

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.8, 3.3]
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.8, repr.plot.height = 3.3)
```

```{code-cell} python
import fastdfe as fd

# URL of the fastDFE GitHub repository
url = "https://github.com/Sendrowski/fastDFE/blob/dev/"

parser = fd.Parser(
    n=8,
    source=url + "resources/genome/betula/biallelic.polarized.subset.50000.vcf.gz?raw=true",
    fasta=url + "resources/genome/betula/genome.subset.1000.fasta.gz?raw=true",
    gff=url + "resources/genome/betula/genome.gff.gz?raw=true",
    annotations=[
        fd.DegeneracyAnnotation()
    ],
    stratifications=[fd.DegeneracyStratification()]
)

spectra: fd.Spectra = parser.parse()
```

```{code-cell} python
:tags: [remove-cell]
assert sorted(spectra.types) == ['neutral', 'selected']
```

```{code-cell} python
spectra.plot();
```

```{code-cell} r
library(fastdfe)
fd <- load_fastdfe()

# URL of the fastDFE GitHub repository
url <- "https://github.com/Sendrowski/fastDFE/blob/dev/"

parser <- fd$Parser(
  n = 8,
  source = paste0(url, "resources/genome/betula/biallelic.polarized.subset.50000.vcf.gz?raw=true"),
  fasta = paste0(url, "resources/genome/betula/genome.subset.1000.fasta.gz?raw=true"),
  gff = paste0(url, "resources/genome/betula/genome.gff.gz?raw=true"),
  annotations = list(
    fd$DegeneracyAnnotation()
  ),
  stratifications = list(fd$DegeneracyStratification())
)

spectra <- parser$parse()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(sort(unlist(spectra$types)), c("neutral", "selected")))
```

```{code-cell} r
p <- spectra$plot()
```

+++
``fastdfe`` relies here on VCF info tags to determine the degeneracy of a site, but this behaviour can be customized (cf. {class}`~sfsutils.parser.DegeneracyStratification`).

+++
## Stratifications
Several stratifications can be used in tandem by specifying a list of stratifications. In this example, we stratify the SFS by degeneracy as well as ancestral base. The resulting spectra can be fed directly into ``fastdfe``'s inference routines. See the ``sfsutils`` [stratifications reference](https://sfsutils.readthedocs.io/en/latest/modules/stratification.html) for a complete list of available stratifications.

```{code-cell} python
parser = fd.Parser(
    n=10,
    source=url + "resources/genome/betula/biallelic.polarized.subset.50000.vcf.gz?raw=true",
    fasta=url + "resources/genome/betula/genome.subset.1000.fasta.gz?raw=true",
    gff=url + "resources/genome/betula/genome.gff.gz?raw=true",
    annotations=[
        fd.DegeneracyAnnotation()
    ],
    stratifications=[
        fd.DegeneracyStratification(),
        fd.AncestralBaseStratification()
    ]
)

spectra: fd.Spectra = parser.parse()
```

```{code-cell} python
:tags: [remove-cell]
assert sorted(spectra.types) == [f'{d}.{b}' for d in ['neutral', 'selected'] for b in 'ACGT']
```

```{code-cell} python
spectra.plot();
```

```{code-cell} r
parser <- fd$Parser(
  n = 10,
  source = paste0(url, "resources/genome/betula/biallelic.polarized.subset.50000.vcf.gz?raw=true"),
  fasta = paste0(url, "resources/genome/betula/genome.subset.1000.fasta.gz?raw=true"),
  gff = paste0(url, "resources/genome/betula/genome.gff.gz?raw=true"),
  annotations = list(
    fd$DegeneracyAnnotation()
  ),
  stratifications = list(
    fd$DegeneracyStratification(),
    fd$AncestralBaseStratification()
  )
)

spectra <- parser$parse()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(sort(unlist(spectra$types)), sort(outer(c("neutral", "selected"), c("A", "C", "G", "T"), paste, sep = "."))))
```

```{code-cell} r
p <- spectra$plot()
```

+++
``fastdfe`` requires the ancestral state of sites to be determined. The {class}`~sfsutils.parser.Parser` achieves this by examining the `AA` field, although this behaviour can be customized.

## Annotations
``fastdfe`` provides a number of annotations accessible directly during the parsing process. To annotate a VCF file directly, the {class}`~sfsutils.annotation.Annotator` class can be used.

### Degeneracy Annotation
{class}`~sfsutils.annotation.DegeneracyAnnotation` annotates the SFS by the degeneracy of the site. This annotation requires information from a FASTA and GFF file and is useful for stratifying the SFS by 0-fold and 4-fold degenerate sites, which is commonly done when inferring the DFE (see {class}`~sfsutils.parser.DegeneracyStratification`).

```{code-cell} python
ann = fd.Annotator(
    source=url + "resources/genome/betula/biallelic.subset.10000.vcf.gz?raw=true",
    fasta=url + "resources/genome/betula/genome.subset.1000.fasta.gz?raw=true",
    gff=url + "resources/genome/betula/genome.gff.gz?raw=true",
    annotations=[fd.DegeneracyAnnotation()],
    output="genome.deg.vcf.gz"
)

ann.annotate()
```

```{code-cell} python
:tags: [remove-cell]
assert ann.annotations[0].n_annotated > 0
```

```{code-cell} r
ann <- fd$Annotator(
  source = paste0(url, "resources/genome/betula/biallelic.subset.10000.vcf.gz?raw=true"),
  fasta = paste0(url, "resources/genome/betula/genome.subset.1000.fasta.gz?raw=true"),
  gff = paste0(url, "resources/genome/betula/genome.gff.gz?raw=true"),
  annotations = list(fd$DegeneracyAnnotation()),
  output = "genome.deg.vcf.gz"
)

ann$annotate()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(ann$annotations[[1]]$n_annotated > 0)
```

+++
### Ancestral Allele Annotation
Currently, two ancestral allele annotations are available: {class}`~sfsutils.annotation.MaximumParsimonyAncestralAnnotation` and {class}`~sfsutils.annotation.MaximumLikelihoodAncestralAnnotation`. The former is straightforward but susceptible to errors, and only appropriate if no outgroup information is available. Alternatively, if outgroups are missing, DFE inference can also be performed on folded spectra, although this yields less precise estimates. Ideally, we would like to use {class}`~sfsutils.annotation.MaximumLikelihoodAncestralAnnotation`, which is more sophisticated and requires one or several outgroups to be specified. Its underlying model is very similar to [EST-SFS](https://doi.org/10.1534/genetics.118.301120).

```{code-cell} python
ann = fd.Annotator(
    source=url + "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz?raw=true",
    annotations=[fd.MaximumLikelihoodAncestralAnnotation(
        outgroups=["ERR2103730"],
        n_ingroups=10
    )],
    output="genome.aa.vcf.gz"
)

ann.annotate()
```

```{code-cell} python
:tags: [remove-cell]
assert ann.annotations[0].n_annotated > 0
```

```{code-cell} r
ann <- fd$Annotator(
  source = paste0(url, "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz?raw=true"),
  annotations = list(fd$MaximumLikelihoodAncestralAnnotation(
    outgroups = list("ERR2103730"),
    n_ingroups = 10
  )),
  output = "genome.aa.vcf.gz"
)

ann$annotate()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(ann$annotations[[1]]$n_annotated > 0)
```

+++
## Filtrations
``fastdfe`` also offers a number of filtrations which can be applied while parsing. Alternatively, to filter a VCF file directly, the {class}`~sfsutils.filtration.Filterer` class can be used. Some useful filtrations include {class}`~sfsutils.filtration.DeviantOutgroupFiltration`, {class}`~sfsutils.filtration.CodingSequenceFiltration`, and {class}`~sfsutils.filtration.BiasedGCConversionFiltration`. For a complete list of available filtrations, refer to the API reference.

```{code-cell} python
f = fd.Filterer(
    source=url + "resources/genome/betula/biallelic.subset.10000.vcf.gz?raw=true",
    filtrations=[fd.BiasedGCConversionFiltration()],
    output="genome.gc.vcf.gz"
)

f.filter()
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < f.n_filtered < f.n_sites
```

```{code-cell} r
f <- fd$Filterer(
  source = paste0(url, "resources/genome/betula/biallelic.subset.10000.vcf.gz?raw=true"),
  filtrations = list(fd$BiasedGCConversionFiltration()),
  output = "genome.gc.vcf.gz"
)

f$filter()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(f$n_filtered > 0, f$n_filtered < f$n_sites)
```

+++
All components can be customized by extending the corresponding base class.

+++
## Manipulating spectra

For the full set of operations on the resulting {class}`~sfsutils.spectrum.Spectrum` and {class}`~sfsutils.spectrum.Spectra` objects (folding, polarising, resampling, plotting, and serialisation), see the [Manipulating the SFS](https://sfsutils.readthedocs.io/en/latest/reference/spectra.html) guide in the [``sfsutils``](https://sfsutils.readthedocs.io) documentation.
