import fastdfe as fd

basepath = "../resources/genome/betula/"

# instantiate parser
p = fd.Parser(
    n=8,  # SFS sample size
    vcf=(basepath + "all.with_outgroups.vcf.gz"),
    fasta=basepath + "genome.subset.1000.fasta.gz",
    gff=basepath + "genome.gff.gz",
    annotations=[
        fd.DegeneracyAnnotation(),  # determine degeneracy
        fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730"],  # use one outgroup
            n_ingroups=20,  # subsample size
            max_sites=50000
        )
    ],
    stratifications=[fd.DegeneracyStratification()]
)

# obtain SFS
spectra: fd.Spectra = p.parse()

spectra.plot(title="SFS")

pass
