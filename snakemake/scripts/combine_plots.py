"""
Combine a bunch of images in a single plot.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2022-30-03"

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import re
import itertools
from typing import List

try:
    testing = False
    files = snakemake.input
    n_cols = snakemake.params.get('n_cols', None)
    n_rows = snakemake.params.get('n_rows', None)
    titles = snakemake.params.get('titles', None)
    title_size_rel = snakemake.params.get('title_size_rel', 20)
    figsize = snakemake.params.get('figsize', None)
    dpi = snakemake.params.get('dpi', 1000)
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    files = [
        "results/graphs/slim/n_replicate=1/n_chunks=2/g=1e2/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/constant/unfolded/dfe.inferred.png",
        "results/graphs/slim/n_replicate=1/n_chunks=2/g=1e2/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/constant/unfolded/dfe.inferred.png",
        "results/graphs/slim/n_replicate=1/n_chunks=2/g=1e2/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/constant/unfolded/dfe.inferred.png",
        "results/graphs/slim/n_replicate=1/n_chunks=2/g=1e2/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/constant/unfolded/dfe.inferred.png",
    ]
    n_cols = None
    n_rows = None
    titles = None
    title_size_rel = 20
    figsize = None
    dpi = 1000
    out = "scratch/combined.png"


def get_index_common_start(strs: List[str]):
    """
    Return first position where given strings differ.

    :param strs: List of strings
    :return: Index
    """
    n = min(len(f) for f in strs)
    for i in range(n):
        if not all([strs[0][i] == strs[j][i] for j in range(1, len(strs))]):
            return i

    return n


def determine_names_dep(files: List[str]) -> List[str]:
    """
    Determine names by removing common start and end substrings.

    :return: List of names
    """
    i = get_index_common_start(files)
    j = get_index_common_start([f[::-1] for f in files])
    return [s[i:-j] for s in files]


def determine_names(out: str) -> List[str]:
    """
    Determine names from output file path.

    :param out: Output file path
    :return: List of names
    """
    matches = re.finditer("\[(.*?)\]", out)

    elements = [match.groups()[0].split(',') for match in matches]
    prod = itertools.product(*elements)

    return [','.join(p) for p in prod]


if titles is None:
    titles = determine_names(out)

    # infer names from input file names if names
    # inferred from output file name are too long
    if len(titles[0]) > 30:
        titles = determine_names_dep(files)

    if len(titles) != len(files):
        titles = [''] * len(files)

# determine number of rows and columns if not specified
n_files = len(files)
if n_cols is None and n_rows is None:
    # display up to three files in one row only
    if n_files < 4:
        n_rows, n_cols = 1, n_files

    # take the number of rows and columns to
    # be the same for more than three files
    else:
        n = int(np.ceil(np.sqrt(n_files)))
        n_rows, n_cols = n, n

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=figsize)
axs = axs.flatten()

for file, title, ax in zip(files, titles, axs):
    ax.imshow(mpimg.imread(file))
    ax.set_title(title, fontdict=dict(fontsize=title_size_rel / n_cols), pad=0)

# turn off axes
[ax.axis("off") for ax in axs]

fig.tight_layout(pad=0)

plt.savefig(out, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()

pass