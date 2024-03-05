import fastdfe as fd

basepath = ("https://github.com/Sendrowski/fastDFE/"
            "blob/dev/resources/genome/betula/")

# instantiate parser
p = fd.Parser(
    n=8,  # SFS sample size
    vcf=(basepath + "biallelic.with_outgroups."
                    "subset.50000.vcf.gz?raw=true"),
    fasta=basepath + "genome.subset.1000.fasta.gz?raw=true",
    gff=basepath + "genome.gff.gz?raw=true",
    target_site_counter=fd.TargetSiteCounter(
        n_target_sites=350000  # total number of target sites
    ),
    annotations=[
        fd.DegeneracyAnnotation(),  # determine degeneracy
        fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730"]  # use one outgroup
        )
    ],
    stratifications=[fd.DegeneracyStratification()]
)

# obtain SFS
spectra: fd.Spectra = p.parse()

spectra.plot()

pass
