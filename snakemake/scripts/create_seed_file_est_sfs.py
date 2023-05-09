"""
Create a seed file for est-sfs.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

try:
    out = snakemake.output[0]
except NameError:
    # testing
    out = "scratch/seed.txt"

# create seed file
open(out, 'w').write('0')
