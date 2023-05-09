"""
Scatter the output of est-sfs.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

try:
    probs_in = snakemake.input.probs
    data_in = snakemake.input.data
    probs_out = snakemake.output.probs
except NameError:
    # testing
    probs_in = "results/test/remote/est-sfs/passed.biallelic.90.probs.all.txt"
    data_in = [f"output/default/est-sfs/data/{n}.txt" for n in range(1, 11)]
    probs_out = [f"scratch/est-sfs/probs/{n}.txt" for n in range(1, 11)]

with open(probs_in, 'r') as probs_all:
    # iterate over files
    for data_file, probs_file in zip(data_in, probs_out):

        # obtain number of lines to read and write
        line_count = sum(1 for line in open(data_file))

        with open(probs_file, 'w') as out:

            # iterate over lines
            for i in range(line_count):

                line = probs_all.readline()

                # write to out file
                out.write(line)

    # raise error if there are lines left from the EST-SFS output
    if next(probs_all, None) is not None:
        raise AssertionError("Unassigned sites left from EST-SFS output.")
