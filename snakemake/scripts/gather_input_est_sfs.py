"""
Gather the scattered input for est-sfs.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

try:
    data_files = snakemake.input
    out = snakemake.output[0]
except NameError:
    # testing
    data_files = [f"output/default/est-sfs/data/{n}.txt" for n in range(1, 10)]
    out = "scratch/data.combined.txt"

# combine contents of all data files
with open(out, 'w') as f:
    for file in data_files:
        f.writelines(open(file, 'r').readlines())
