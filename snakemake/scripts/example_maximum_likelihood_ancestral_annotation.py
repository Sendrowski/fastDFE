# example for ancestral allele annotation
ann = fd.Annotator(
    vcf=url + "resources/genome/betula/biallelic.with_outgroups.subset.50000.vcf.gz?raw=true",
    annotations=[fd.MaximumLikelihoodAncestralAnnotation(
        outgroups=["ERR2103730"],
        n_ingroups=15,
        max_sites=10000
    )],
    output="out/genome.polarized.vcf.gz"
)

ann.annotate()
