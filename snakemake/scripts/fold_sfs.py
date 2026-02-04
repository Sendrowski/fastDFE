"""
Fold SFS
"""


try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('.')

    testing = False
    file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = "results/sfs/hg38/chr/chr22.degeneracy/ingroup_pan_troglodytes/outgroups_ref_gorilla_gorilla_gorilla.ref_pongo_abelii/sfs.11.csv"
    out = f"scratch/folded_sfs.csv"

import fastdfe as fd

fd.Spectra.from_file(file).fold().to_file(out)
