library(fastdfe)

if (fastdfe_is_installed()) {

  fd <- load_fastdfe()
  viz <- fd$visualization$Visualization

  test_that("log-scale spectrum bars have a positive height for a smallest count that is a power of ten", {
    p <- viz$plot_spectra(list(c(0, 10, 100, 1000, 0)), log_scale = TRUE, show = FALSE)
    bars <- ggplot2::layer_data(p)

    expect_equal(nrow(bars), 3)
    expect_true(all(bars$ymax > bars$ymin))
  })
}
