
source("R/fastdfe.R")

# import classes
base_inference <- fastdfe$BaseInference
spectrum <- fastdfe$Spectrum

# configure inference
inf <- base_inference(
  sfs_neut=spectrum(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652)),
  sfs_sel=spectrum(c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794))
)

# Run inference
base_inference$run(inf)

base_inference$plot_discretized(inf)

base_inference$bootstrap(inf, n_samples = as.integer(10))

base_inference$plot_discretized(inf)
