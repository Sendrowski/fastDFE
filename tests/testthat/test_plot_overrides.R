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

  test_that("the scatter plot uses a symmetric log axis for 'symlog' as for 'log'", {
    scale_of <- function(scale) {
      p <- viz$plot_scatter(values = c(-1, 2, 3), scale = scale, show = FALSE)
      p$scales$get_scales("y")$trans$name
    }

    expect_equal(scale_of("symlog"), scale_of("log"))
    expect_false(scale_of("symlog") == scale_of("lin"))
  })

  test_that("spectra can be passed as the rows of a matrix", {
    spectra <- rbind(c(100, 5, 3, 2, 0), c(100, 4, 2, 1, 0))

    from_matrix <- ggplot2::layer_data(viz$plot_spectra(spectra, show = FALSE))
    from_list <- ggplot2::layer_data(viz$plot_spectra(list(spectra[1, ], spectra[2, ]), show = FALSE))

    expect_equal(nrow(from_matrix), 6)
    expect_equal(from_matrix$ymax, from_list$ymax)
  })
}
