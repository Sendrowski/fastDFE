"""
Parse polyDFE output.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-06"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    input = snakemake.input[0]
    postprocessing_source = '../resources/polydfe/postprocessing/script.R'
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    input = 'scratch/polydfe_out.txt'
    postprocessing_source = '../resources/polydfe/postprocessing/script.R'
    out = 'scratch/polydfe_out.json'

import json
import fastdfe

with open(out, 'w') as fh:
    json.dump(fastdfe.PolyDFE.parse_output(input, postprocessing_source), fh, indent=4)
