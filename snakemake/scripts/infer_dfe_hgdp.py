"""
Infer the DFE from the SFS for the HGDP dataset.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-28"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    config_file = snakemake.input.config
    spectra_file = snakemake.input.spectra
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
    out_dfe = snakemake.output.dfe
    out_spectra = snakemake.output.spectra
    out_params = snakemake.output.params
except NameError:
    # testing
    testing = True
    config_file = '../resources/configs/HGDP/test.yaml'
    spectra_file = "results/sfs/hgdp/21/opts.subset.50000.n.20/all.csv"
    out_summary = "scratch/summary.json"
    out_serialized = "scratch/serialized.json"
    out_dfe = "scratch/dfe.json"
    out_spectra = "scratch/spectra.csv"
    out_params = "scratch/params.json"

from fastdfe import Config, BaseInference, Spectra

# load config from file
config = Config.from_file(config_file)

# load spectra from file
spectra = Spectra.from_file(spectra_file)

# update config with spectra
config.update(
    sfs_neut=spectra['neutral'],
    sfs_sel=spectra['selected']
)

# create from config
inference = BaseInference.from_config(config)

# perform inference
inference.run()

# save object in serialized form
inference.to_file(out_serialized)

# save summary
inference.get_summary().to_file(out_summary)

inference.plot_inferred_parameters(file=out_params, show=testing)
inference.plot_sfs_comparison(file=out_spectra, show=testing)
inference.plot_discretized(file=out_dfe, show=testing)

pass
