import matplotlib.pyplot as plt

import fastdfe as fd

# example for joint inference with covariates
inf = fd.JointInference.from_config_file(
    "https://github.com/Sendrowski/fastDFE/"
    "blob/dev/resources/configs/arabidopsis/"
    "covariates_example.yaml?raw=true"
)

inf.run()

# get p-value for covariate significance
p = inf.perform_lrt_covariates()
p_str = f"p = {p:.1e}" if p >= 1e-100 else "p < 1e-100"

# plot results
_, axs = plt.subplots(1, 2, figsize=(10.5, 3.5))

inf.plot_covariate(ax=axs[0], xlabel='RSA', show=False)
inf.plot_discretized(
    title=f"DFE comparison, " + p_str, ax=axs[1],
    show_marginals=False, show=False
)

pass
# plt.savefig("scratch/joint_inference_covariates.png")
