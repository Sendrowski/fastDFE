import fastdfe as fd

# parse selected and neutral SFS from human chromosome 1
p = fd.Parser(
    vcf="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
        "hgdp_wgs.20190516.full.chr1.vcf.gz",
    fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
          "dna/Homo_sapiens.GRCh38.dna.chromosome.1.fa.gz",
    gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
        "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz",
    aliases=dict(chr1=['1']),
    n=10,
    target_site_counter=fd.TargetSiteCounter(
        n_samples=1000000,
        n_target_sites=fd.Annotation.count_target_sites(
            "http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
            "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz"
        )['1']
    ),
    annotations=[
        fd.DegeneracyAnnotation()
    ],
    filtrations=[
        fd.CodingSequenceFiltration()
    ],
    stratifications=[fd.DegeneracyStratification()],
    info_ancestral='AA_ensembl'
)

sfs = p.parse()

sfs.plot()

pass
