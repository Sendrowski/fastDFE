library(fastdfe)

test_that("install_fastdfe replaces the declared requirement and Python version", {
  skip_if(reticulate::py_available(initialize = FALSE), "Python is initialised, so requirements are fixed")

  requirement <- function() unclass(reticulate::py_require())[c("packages", "python_version")]
  declared <- function() grep("^fastdfe", requirement()$packages, value = TRUE)

  install_fastdfe(extras = NULL, python_version = "3.12", silent = TRUE)

  expect_equal(declared(), "fastdfe")
  expect_equal(requirement()$python_version, "3.12")
  expect_true("numpy" %in% requirement()$packages)

  install_fastdfe(version = "1.4.1", extras = c("vcf", "zarr"), silent = TRUE)

  expect_equal(declared(), "fastdfe[vcf,zarr]==1.4.1")
  expect_equal(requirement()$python_version, "3.11")

  # back to the requirement declared on load, for the tests that follow
  install_fastdfe(silent = TRUE)

  expect_equal(declared(), "fastdfe[vcf]")
})
