library(fastdfe)

fd <- load_fastdfe()
viz <- fd$visualization$Visualization

test_that("log-scale spectrum bars have a positive height for a smallest count that is a power of ten", {
  p <- viz$plot_spectra(list(c(0, 10, 100, 1000, 0)), log_scale = TRUE, show = FALSE)
  bars <- ggplot2::layer_data(p)

  expect_equal(nrow(bars), 3)
  expect_true(all(bars$ymax > bars$ymin))
})

test_that("a log-scale spectrum plot without a positive count is an empty plot", {
  p <- viz$plot_spectra(list(c(100, 0, 0, 0, 10)), log_scale = TRUE, show = FALSE)

  expect_equal(nrow(ggplot2::layer_data(p)), 0)
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

test_that("discretized DFEs with the same label are drawn side by side", {
  values <- list(c(0.5, 0.5), c(0.3, 0.7))

  p <- viz$plot_discretized(values, labels = c("a", "a"), intervals = c(-Inf, 0, Inf), show = FALSE)
  bars <- ggplot2::layer_data(p)

  expect_equal(nrow(bars), 4)
  expect_equal(length(unique(bars$xmin)), 4)
})

test_that("the parameter plot accepts a parameter that is not a number", {
  values <- list(a = list(S_d = -10, b = NaN), b = list(S_d = -20, b = 0.3))

  expect_no_error(viz$plot_inferred_parameters(values, labels = c("a", "b"), param_names = c("S_d", "b"),
                                               show = FALSE))
})

test_that("the nested-model plot accepts the name of a colour map or a vector of colours", {
  P <- matrix(c(0.2, 0.5, 1e-5, 0.9), nrow = 2)

  for (cmap in list("viridis", "inferno", c("white", "red"))) {
    # the colours are resolved when the plot is built
    expect_no_error(ggplot2::ggplot_build(
      viz$plot_nested_models(P, labels_x = c("a", "b"), labels_y = c("a", "b"), cmap = cmap, show = FALSE)
    ))
  }
})
