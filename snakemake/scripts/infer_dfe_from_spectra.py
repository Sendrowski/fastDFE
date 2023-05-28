"""
Infer the DFE from the SFS.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-25"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    spectra_file = snakemake.input.spectra
    out_serialized = snakemake.output.serialized
except NameError:
    # testing
    testing = True
    spectra_file = "scratch/parse_spectra_from_url.spectra.csv"
    out_serialized = "scratch/infer_dfe_from_url.serialized.json"

from fastdfe import BaseInference, Spectra

# parse spectra and fold
spectra = Spectra.from_file(spectra_file)

# fold spectra
spectra = spectra.fold()

# plot spectra
spectra.plot()

# create inference
inf = BaseInference(
    sfs_neut=spectra['neutral'],
    sfs_sel=spectra['selected'],
    n_bootstraps=100,
    do_bootstrap=True,
    n_runs=30
)

# perform inference
inf.run()

# save object in serialized form
inf.to_file(out_serialized)

# plot data
if testing:
    inf.plot_discretized()
    inf.plot_inferred_parameters()
    inf.plot_sfs_comparison()

pass
