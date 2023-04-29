n_chunks = config['n_chunks']
ref = config['ref']
vcf = config['vcf']
dict = ref.replace('.fasta','.dict')
out_prefix = config['out_prefix']

"""rule all:
    input: (
        expand("{out_prefix}/vcf/{i}.vcf.gz",i=range(n_chunks))
    )"""

# create a tbi index for a vcf file
rule create_tbi:
    input:
        "{path}.vcf.gz"
    output:
        "{path}.vcf.gz.tbi"
    conda:
        "../envs/gatk.yaml"
    shell:
        "gatk IndexFeatureFile -I {input}"

# index a fasta file using samtools
rule create_fai:
    input:
        "{path}.fasta"
    output:
        "{path}.fasta.fai"
    conda:
        "../envs/samtools.yaml"
    shell:
        "samtools faidx {input}"

# create a dict file for a fasta file
rule create_dict:
    input:
        "{path}.fasta"
    output:
        "{path}.dict"
    conda:
        "../envs/gatk.yaml"
    shell:
        "gatk CreateSequenceDictionary R={input} O={output}"

# get the interval lists of the chunks to be created
rule split_intervals:
    input:
        ref=ref,
        index=f"{ref}.fai",
        vcf=f"{vcf}",
        tbi=f"{vcf}.tbi"
    output:
        [f"{out_prefix}/interval_lists/{str(i).zfill(4)}-scattered.interval_list" for i in range(n_chunks)]
    params:
        n=n_chunks,
        out_prefix=out_prefix + '/interval_lists'
    conda:
        "../envs/gatk.yaml"
    shell:
        "gatk SplitIntervals -R {input.ref} -L {input.vcf} --scatter-count {params.n} -O {params.out_prefix}"

# split the VCF file into chunks
rule split_vcf:
    input:
        vcf=f"{vcf}",
        tbi=f"{vcf}.tbi",
        ref=f"{ref}",
        index=f"{ref}.fai",
        dict=dict,
        list=lambda w: f"{out_prefix}/interval_lists/{str(w.i).zfill(4)}-scattered.interval_list"
    output:
        "{out_prefix}/vcf/{i}.vcf.gz"
    conda:
        "../envs/gatk.yaml"
    shell:
        "gatk SelectVariants -R {input.ref} -V {input.vcf} -L {input.list} -O {output}"
