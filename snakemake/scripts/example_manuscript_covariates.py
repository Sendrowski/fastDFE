import fastdfe as fd
import matplotlib.pyplot as plt

# example for joint inference with covariates
inf = fd.JointInference.from_config_file(
    "https://github.com/Sendrowski/fastDFE/"
    "blob/dev/resources/configs/arabidopsis/"
    "covariates_example.yaml?raw=true"
)

inf.run()

p = inf.perform_lrt_covariates()

_, axs = plt.subplots(1, 2, figsize=(10.5, 3.5))

inf.plot_covariate(ax=axs[0], xlabel='RSA', show=False)
inf.plot_discretized(
    title=f"DFE comparison, p={p:.1e}",
    ax=axs[1], show_marginals=False,
    show=False
)

pass
