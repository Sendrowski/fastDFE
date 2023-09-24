# example for ancestral allele annotation
ann = fd.Annotator(
    vcf=url + "resources/genome/betula/biallelic.with_outgroups.subset.50000.vcf.gz?raw=true",
    annotations=[fd.MaximumLikelihoodAncestralAnnotation(
        outgroups=["ERR2103730", "ERR2103731"],
        n_ingroups=15
    )],
    output="out/genome.aa.vcf.gz"
)

ann.annotate()