library(fastdfe)

if (fastdfe_is_installed()) {

  # load python package
  fd <- load_fastdfe()
  
  # import classes
  JointInference <- fd$JointInference
  Spectrum <- fd$Spectrum
  Spectra <- fd$Spectra
  
  # neutral SFS for two types
  sfs_neut <- Spectra(list(
    pendula=c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652),
    pubescens=c(172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31)
  ))
  
  # selected SFS for two types
  sfs_sel <- Spectra(list(
    pendula=c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794),
    pubescens=c(791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41)
  ))
  
  # configure inference
  inf <- JointInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel
  )
  
  # Run inference
  JointInference$run(inf)
  
  JointInference$plot_likelihoods(inf)
  JointInference$plot_inferred_parameters(inf)
  JointInference$plot_discretized(inf)
  
  JointInference$bootstrap(inf, n_samples = 10)
  
  JointInference$plot_inferred_parameters(inf)
  JointInference$plot_discretized(inf)
}

