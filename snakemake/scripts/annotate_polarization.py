"""
Annotate degeneracy of variants in a VCF file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-09-26"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    vcf_file = snakemake.input.vcf
    n_ingroups = snakemake.params.n_ingroups
    outgroups = snakemake.params.outgroups
    exclude = snakemake.params.exclude
    max_sites = snakemake.params.get("max_sites", None)
    confidence_threshold = snakemake.params.get("confidence_threshold", 0.5)
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    vcf_file = "resources/genome/betula/biallelic.with_outgroups.vcf.gz"
    outgroups = ["ERR2103730"]
    exclude = ["ERR2103731"]
    n_ingroups = 20
    max_sites = 10000
    confidence_threshold = 0
    out = 'scratch/biallelic.polarized.vcf.gz'

import fastdfe as fd

# initialize annotator
ann = fd.Annotator(
    vcf=vcf_file,
    output=out,
    annotations=[
        fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=outgroups,
            exclude=exclude,
            n_ingroups=n_ingroups,
            max_sites=max_sites,
            confidence_threshold=confidence_threshold
        )
    ]
)

# run annotation
ann.annotate()

pass