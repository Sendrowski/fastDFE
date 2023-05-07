# extract first 5 contigs from a fasta file
rule extract_first_n_contigs:
    input:
        "{path}.fasta"
    output:
        "{path}.subset.{n}.fasta"
    params:
        n_contigs=lambda w: int(w.n)
    shell:
        "grep -m {params.n_contigs} -A 1 '^>' {input} | grep -v '^--' > {output}"
