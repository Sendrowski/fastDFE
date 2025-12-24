"""
Rules for copying files to and from a remote server.
"""

# copy files from remote server to local machine
# snakemake -c8 --snakefile workflow/rules/remote.smk -R copy_from_remote -k {path}
rule copy_from_remote:
    output:
        "{path}"
    params:
        output_dir=lambda w: os.path.dirname(w.path),
    shell:
        """
        scp -r sendrowskij@login.genome.au.dk:/Users/au732936/PycharmProjects/fastDFE/snakemake/{output} {params.output_dir}
        """
