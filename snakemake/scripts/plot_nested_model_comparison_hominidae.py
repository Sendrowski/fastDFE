"""
Plot nested model comparison for the hominidae dataset.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-31"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    config = snakemake.params.config
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    config = "results/fastdfe/hominidae/cov/eps.yaml"
    out = "scratch/joint_inference.yaml"

import fastdfe as fd
import matplotlib.pyplot as plt

inf = fd.JointInference.from_config_file(config)

fig, axs = plt.subplots(len(inf.types), figsize=(10, 10))

for i, t in enumerate(inf.types):
    inf.marginal_inferences[t].plot_nested_likelihoods(ax=axs[i], show=False)

plt.show()
