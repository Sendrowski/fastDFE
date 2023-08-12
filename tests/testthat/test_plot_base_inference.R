library(fastdfe)

if (fastdfe_is_installed()) {

  # load python package
  fd <- load_fastdfe()
  
  # import classes
  Spectrum <- fd$Spectrum
  BaseInference <- fd$BaseInference
  
  sfs_neut <- Spectrum(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652))
  sfs_sel <- Spectrum(c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794))
  
  Spectrum$plot(sfs_neut)
  
  # configure inference
  inf <- BaseInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel
  )
  
  # Run inference
  BaseInference$run(inf)
  
  BaseInference$plot_likelihoods(inf)
  BaseInference$plot_inferred_parameters(inf)
  BaseInference$plot_discretized(inf)
  
  BaseInference$bootstrap(inf, n_samples = 10)
  
  BaseInference$plot_inferred_parameters(inf)
  BaseInference$plot_discretized(inf)
  
  BaseInference$plot_nested_models(inf)
}
