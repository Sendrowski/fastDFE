"""
Plot SFS from different demographic scenarios.
"""
import sys

# necessary to import fastdfe locally
sys.path.append('.')

from matplotlib import pyplot as plt
import fastdfe as fd

from matplotlib.ticker import MultipleLocator

try:
    testing = False
    files_fd = snakemake.input.fastdfe
    files_slim = snakemake.input.slim
    labels = snakemake.params.labels
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    files_fd = [
        "results/sfs/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/constant/unfolded/sfs.csv",
        "results/sfs/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/expansion_4/unfolded/sfs.csv",
        "results/sfs/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/reduction_4/unfolded/sfs.csv",
        "results/sfs/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/bottleneck_20/unfolded/sfs.csv",
        "results/sfs/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/substructure_0.0001/unfolded/sfs.csv",
        "results/sfs/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/dominance_0.2/unfolded/sfs.semidominant.csv"
    ]
    files_slim = [
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/constant/unfolded/sfs.csv",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/expansion_4/unfolded/sfs.csv",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/reduction_4/unfolded/sfs.csv",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/bottleneck_20/unfolded/sfs.csv",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/substructure_0.0001/unfolded/sfs.csv",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/dominance_0.2/unfolded/sfs.csv"
    ]
    labels = ["constant", "expansion", "reduction", "bottleneck", "substructure", "recessiveness"]
    out = "scratch/sfs_demographies.png"

fig, ax = plt.subplots(3, 2, figsize=(7, 4.5), sharex=True)
ax = ax.flatten()

for i in range(len(files_fd)):
    sfs = fd.Spectra.from_spectra(dict(
        SLiM=fd.Spectra.from_file(files_slim[i])['selected'],
        fastDFE=fd.Spectrum.from_file(files_fd[i]),
    ))
    sfs.plot(ax=ax[i], show=False)
    ax[i].set_xlabel("", fontsize=6)
    ax[i].set_title(labels[i], fontsize=14)
    if labels[i] == "expansion":
        ax[i].yaxis.set_major_locator(MultipleLocator(20000))
    ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.tight_layout(pad=0.7)

if testing:
    plt.show()

fig.savefig(out, dpi=300)
pass
