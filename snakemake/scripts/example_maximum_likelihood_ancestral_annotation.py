# example for ancestral allele annotation
import fastdfe as fd

ann = fd.Annotator(
    vcf="https://github.com/Sendrowski/fastDFE/"
        "blob/dev/resources/genome/betula/biallelic."
        "with_outgroups.subset.50000.vcf.gz?raw=true",
    annotations=[fd.MaximumLikelihoodAncestralAnnotation(
        outgroups=["ERR2103730"],
        n_ingroups=15,
        max_sites=10000
    )],
    output="genome.polarized.vcf.gz"
)

ann.annotate()

pass
