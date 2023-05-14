# subset a VCF file by selecting the n file rows
# this is useful for creating test sets
rule subset_vcf:
    input:
        "{path}.vcf.gz"
    output:
        "{path}.subset.{n}.vcf.gz"
    params:
        n=lambda w: int(w.n)
    conda:
        "../envs/tabix.yaml"
    shell:
        """
        (zcat < {input} | grep "^#" ; zcat < {input} | grep -v "^#" | head -n {params.n} || true) | bgzip -c > {output}
        """
