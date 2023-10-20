import fastdfe as fd

basepath = ("https://github.com/Sendrowski/fastDFE/"
            "blob/dev/resources/genome/betula/")

# instantiate parser
p = fd.Parser(
    n=8,  # SFS sample size
    vcf=(basepath + "all.with_outgroups."
                    "subset.200000.vcf.gz?raw=true"),
    fasta=basepath + "genome.subset.1000.fasta.gz?raw=true",
    gff=basepath + "genome.gff.gz?raw=true",
    annotations=[
        fd.DegeneracyAnnotation(),  # determine degeneracy
        fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730"],  # use one outgroup
            n_ingroups=20,  # subsample size
            max_sites=50000  # number of sites for inference
        )
    ],
    stratifications=[fd.DegeneracyStratification()]
)

# obtain SFS
spectra: fd.Spectra = p.parse()

spectra.plot(title="SFS")

pass
