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

test_that("bars of the same group differ in opacity by the prefix of their label", {
  labels <- c("marginal.a", "marginal.b", "joint.a", "joint.b")

  params <- stats::setNames(list(list(S_d = -10), list(S_d = -20), list(S_d = -12), list(S_d = -18)), labels)
  bars <- ggplot2::layer_data(viz$plot_inferred_parameters(params, labels = labels, param_names = "S_d", show = FALSE))

  # the marginal and the joint bar of a type share the colour, the joint one being opaque
  expect_equal(bars$fill[c(1, 2)], bars$fill[c(3, 4)])
  expect_equal(bars$alpha[c(3, 4)], c(1, 1))
  expect_true(all(bars$alpha[c(1, 2)] < 1))

  values <- list(c(0.5, 0.5), c(0.3, 0.7), c(0.4, 0.6), c(0.2, 0.8))
  bars <- ggplot2::layer_data(viz$plot_discretized(values, labels = labels, intervals = c(-Inf, 0, Inf), show = FALSE))

  expect_equal(nrow(unique(bars[, c("fill", "alpha")])), 4)

  plain <- ggplot2::layer_data(viz$plot_discretized(values[1:2], labels = c("a", "b"), intervals = c(-Inf, 0, Inf),
                                                    show = FALSE))

  expect_true(all(plain$alpha == 1))

  # a dot within a label that has no single-word prefix does not count
  dotted <- ggplot2::layer_data(viz$plot_discretized(values[1:2], labels = c("True DFE (h=0.3)", "Inferred DFE (h=0.5)"),
                                                     intervals = c(-Inf, 0, Inf), show = FALSE))

  expect_true(all(dotted$alpha == 1))
})

test_that("error bars are opaque and stay out of the legend", {
  values <- list(c(0.5, 0.5), c(0.3, 0.7))
  errors <- list(matrix(0.05, nrow = 2, ncol = 2), matrix(0.05, nrow = 2, ncol = 2))

  p <- viz$plot_discretized(values, errors = errors, labels = c("marginal.a", "joint.a"),
                            intervals = c(-Inf, 0, Inf), show = FALSE)

  expect_false(p$layers[[2]]$show.legend)
  expect_true(all(ggplot2::layer_data(p, 2)$alpha == 1))

  # a single DFE has no legend
  single <- viz$plot_discretized(values[1], errors = errors[1], intervals = c(-Inf, 0, Inf), show = FALSE)

  expect_false(any(vapply(single$layers, function(layer) isTRUE(layer$show.legend), logical(1))))
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
