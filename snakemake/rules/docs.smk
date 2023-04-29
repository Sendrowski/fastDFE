
rule all:
    input:
        "../docs/_build"

# run sphinx
rule run_sphinx:
    output:
        directory("../docs/_build")
    conda:
        "../envs/dev.yaml"
    shell:
        """
            cd ../docs
            make html
        """
