
source("R/fastdfe.R")

# import classes
joint_inference <- fastdfe$JointInference
spectrum <- fastdfe$Spectrum
spectra <- fastdfe$Spectra

# neutral SFS for two types
sfs_neut <- spectra(list(
  pendula=c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652),
  pubescens=c(172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31)
))

# selected SFS for two types
sfs_sel <- spectra(list(
  pendula=c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794),
  pubescens=c(791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41)
))

# configure inference
inf <- joint_inference(
  sfs_neut=sfs_neut,
  sfs_sel=sfs_sel
)

# Run inference
joint_inference$run(inf)

joint_inference$plot_discretized(inf)

joint_inference$bootstrap(inf, n_samples = as.integer(10))

joint_inference$plot_discretized(inf)

