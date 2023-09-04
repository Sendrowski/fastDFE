import fastdfe as fd

# only keep variants in coding sequences
f = fd.Filterer(
    vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
        "1000_genomes_project/release/20181203_biallelic_SNV/"
        "ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
    gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
        "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
    output='scratch/sapiens.chr21.coding.vcf',
    filtrations=[fd.CodingSequenceFiltration()],
    aliases=dict(chr21=['21']),
)

f.filter()
