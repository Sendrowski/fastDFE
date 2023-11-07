import fastdfe as fd
import matplotlib.pyplot as plt

# example for joint inference with covariates
inf = fd.JointInference.from_file("scratch/example_manuscript_covariates.json")

_, axs = plt.subplots(1, 2, figsize=(10.5, 4))

inf.plot_covariate(ax=axs[0], xlabel='RSA', show=False)
inf.plot_discretized(
    ax=axs[1], show_marginals=False,
    kwargs_legend=dict(prop=dict(size=9)),
    show=True
)

pass
