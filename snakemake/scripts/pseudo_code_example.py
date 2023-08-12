import fastdfe as fd

fd.JointInference(
    ...,  # additional arguments
    # mixture of a gamma and exponential distribution
    model=fd.GammaExpParametrization(),
    # introduce covariate for S_d, the mean deleterious DFE
    covariates=[fd.Covariate(
        param='S_d',
        values=dict(type1=-4, type2=2, type3=0, type4=1)
    )]
)

# examples of parametrizations
[
    # mixture of a gamma and exponential distribution
    fd.GammaExpParametrization(),
    # mixture of a gamma and discrete distribution
    fd.GammaDiscreteParametrization(),
    # reflected displaced gamma distribution
    fd.DisplacedGammaParametrization(),
    # discrete distribution with explicit intervals
    fd.DiscreteParametrization([-100000, -100, -10, -1, 0, 1, 1000])
]

fd.Parser(
    vcf="example.vcf.gz",
    n=10,
    # annotate degeneracy and ancestral base
    annotations=[
        fd.DegeneracyAnnotation(
            fasta_file="example.fasta",
            gff_file="example.gff"
        ),
        fd.MaximumParsimonyAnnotation()
    ],
    # filter out non-coding sequences
    filtrations=[
        fd.CodingSequenceFiltration(
            gff_file="example.gff",
        )
    ],
    # stratify by degeneracy and ancestral base
    stratifications=[
        fd.DegeneracyStratification(),
        fd.AncestralBaseStratification()
    ],
)

pass
