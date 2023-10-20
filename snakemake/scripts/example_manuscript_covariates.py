import fastdfe as fd

# example for joint inference with covariates
inf = fd.JointInference.from_config_file(
    "https://github.com/Sendrowski/fastDFE/"
    "blob/dev/resources/configs/arabidopsis/"
    "covariates_example.yaml?raw=true"
)

inf.run()

p = inf.perform_lrt_covariates()

inf.plot_discretized(
    title=f"DFE comparison, p={p:.1e}",
    kwargs_legend=dict(ncols=2, prop=dict(size=6))
)

pass
