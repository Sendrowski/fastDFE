import fastdfe as fd

ann = fd.Annotator(
    vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
        "1000_genomes_project/release/20181203_biallelic_SNV/"
        "ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
    fasta_file="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
               "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
    gff_file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
             "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
    output='scratch/sapiens.chr21.degeneracy.vcf.gz',
    annotations=[fd.DegeneracyAnnotation()],
    aliases=dict(chr21=['21']),
)

ann.annotate()
