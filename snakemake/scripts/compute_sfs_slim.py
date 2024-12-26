"""
Compute the site frequency spectrum (SFS) for selected and neutral sites in a tree sequence.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2024-02-26"

import random

import matplotlib.pyplot as plt
import msprime as ms
import numpy as np
import pandas as pd
import tskit
from scipy.stats import hypergeom
from tqdm import tqdm

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    file_in = snakemake.input[0]
    n = snakemake.params.n
    mu = snakemake.params.mu
    L = snakemake.params.L
    N = snakemake.params.N
    out = snakemake.output.sfs
    out_ds_continuous = snakemake.output.ds_continuous
    out_ds_discretized = snakemake.output.ds_discretized
except NameError:
    # testing
    testing = True
    file_in = "results/slim/n_replicate=1/g=10000/L=10000000000/mu=1e-09/r=1e-09/N=1000/s_b=0.05/b=1/s_d=0.1/p_b=0.2/sequence.trees"
    n = 10
    mu = 1e-9
    L = 10000000000
    N = 1000
    out = "scratch/slim_sfs.csv"
    out_ds_continuous = "scratch/slim_dfe.png"
    out_ds_discretized = "scratch/slim_discretized.png"

# load the tree sequence
ts = tskit.load(file_in)

# sample 20 individuals
ids = random.sample([ind.id for ind in ts.individuals()], n)

# get the nodes for the selected individuals
nodes = []
for i in ids:
    individual = ts.individual(i)

    # randomly select one node from each individual
    nodes += [random.sample(list(individual.nodes), 1)[0]]

j = 0
sfs_sel = np.zeros(n + 1)
n_repeat_sel = 0
# selection coefficients
s = np.zeros(ts.num_mutations)

for i, var in tqdm(enumerate(ts.variants()), desc='Computing selected SFS'):
    for mut in var.site.mutations:
        s[j] = mut.metadata['mutation_list'][0]['selection_coeff']
        j += 1

    n_repeat_sel += int(len(var.site.mutations) > 1)

    # add hypergeometric counts
    sfs_sel += hypergeom.pmf(k=range(n + 1), M=ts.sample_size, n=np.sum(var.genotypes > 0), N=n)

print(f"Repeat mutations (selected): {n_repeat_sel}")

# add monomorphic sites
sfs_sel[0] = L - sfs_sel.sum()

ts = ms.sim_mutations(ts, rate=mu, keep=False)

sfs_neut = np.zeros(n + 1)
n_repeat_neut = 0

for var in tqdm(ts.variants(), desc='Computing neutral SFS'):
    n_repeat_neut += int(len(var.site.mutations) > 1)
    sfs_neut += hypergeom.pmf(k=range(n + 1), M=ts.sample_size, n=np.sum(var.genotypes > 0), N=n)

print(f"Repeat mutations (neutral): {n_repeat_neut}")

# add monomorphic sites
sfs_neut[0] = L - sfs_neut.sum()

# save as csv using pandas
spectra = pd.DataFrame(dict(neutral=sfs_neut, selected=sfs_sel))

spectra.to_csv(out, index=False)

# determine Ne from sfs_neut using Watterson's estimator
theta = sum(sfs_neut[1:-1]) / np.sum(1 / np.arange(1, n)) / sum(sfs_neut)
# theta = 4 * Ne * mu -> Ne = theta / (4 * mu)
Ne = theta / (4 * mu)

S = 4 * Ne * s

# plot histogram of selection coefficients
plt.hist(S, bins=400, density=True)

# set upper and lower bounds to 1 and 99th percentiles
plt.xlim(np.percentile(S, 1), np.percentile(S, 99))
plt.xlabel('S')
plt.title(f'Observed S, $N_e$ = {Ne:.2f}')
plt.margins(x=0)

plt.savefig(out_ds_continuous)

if testing:
    plt.show()

plt.clf()

# bin the selection coefficients
values, _ = np.histogram(S, bins=[-np.inf, -100, -10, -1, 0, 1, np.inf])
values = values / np.sum(values)

plt.bar(['<-100', '-100:-10', '-10:-1', '-1:0', '0:1', '>1'], values)
plt.xlabel('S')
plt.title(f'Observed S, $N_e$ = {Ne:.2f}')
plt.margins(x=0)

plt.savefig(out_ds_discretized)

if testing:
    plt.show()

pass
