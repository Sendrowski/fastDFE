# compress file
rule compress_bgzip:
    input:
        "{path}"
    output:
        "{path}.gz"
    conda:
        "../envs/tabix.yaml"
    shell:
        "bgzip {input} -c > {output}"

# decompress file
rule decompress_bgzip:
    input:
        "{path}.gz"
    output:
        r"{path,.*(?<!(\.gz))$}"
    conda:
        "../envs/tabix.yaml"
    shell:
        "bgzip {input} -cd > {output}"
