# example for ancestral allele annotation
import fastdfe as fd

ann = fd.Annotator(
    vcf="https://github.com/Sendrowski/fastDFE/"
        "blob/dev/resources/genome/betula/all."
        "with_outgroups.subset.10000.vcf.gz?raw=true",
    annotations=[fd.MaximumLikelihoodAncestralAnnotation(
        outgroups=["ERR2103730"],
        n_ingroups=15
    )],
    output="genome.polarized.vcf.gz"
)

ann.annotate()

pass
