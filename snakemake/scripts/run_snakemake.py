from snakemake import snakemake

snakemake(
    snakefile='Snakefile',
    cores=3,
    printshellcmds=True,
    targets=['results/graphs/sfs/hgdp/2/opts.n.10/Japanese.png'],
    force_incomplete=True,
    use_conda=True,
    #dryrun=True,
)
