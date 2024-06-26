"""
Example presentation parser
"""

import fastdfe as fd

basepath = "https://github.com/Sendrowski/fastDFE/blob/dev/resources/genome/betula/"

p = fd.Parser(
    n=8,
    vcf=basepath + "biallelic.with_outgroups.subset.50000.vcf.gz?raw=true",
    fasta=basepath + "genome.subset.1000.fasta.gz?raw=true",
    gff=basepath + "genome.gff.gz?raw=true",
    annotations=[
        fd.MaximumLikelihoodAncestralAnnotation(outgroups=["ERR2103730"]),
        fd.DegeneracyAnnotation()
    ],
    stratifications=[fd.DegeneracyStratification()],
    target_site_counter=fd.TargetSiteCounter(n_target_sites=350000)
)

spectra: fd.Spectra = p.parse()

spectra.plot()

pass
