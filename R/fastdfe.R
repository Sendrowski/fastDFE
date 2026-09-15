if (getRversion() >= "2.15.1") utils::globalVariables(c(".data"))

# ggplot2 theme placing the legend inside the panel (top-right) over a semi-transparent
# background, so it stays within the figure on narrow plots instead of being pushed off the side
.legend_inside <- function() {
  ggplot2::theme(
    legend.position = "inside",
    legend.position.inside = c(0.97, 0.97),
    legend.justification = c(1, 1),
    legend.title = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(size = 8),
    legend.key.size = ggplot2::unit(0.8, "lines"),
    legend.margin = ggplot2::margin(4, 4, 4, 4),
    legend.background = ggplot2::element_rect(
      fill = grDevices::adjustcolor("white", alpha.f = 0.8), colour = "grey80"
    ),
    legend.key = ggplot2::element_rect(fill = NA, colour = NA)
  )
}

# matplotlib's default colour cycle ('C0', 'C1', ...)
.tab10 <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

# ggplot2 fill scale colouring each label with the colour of matplotlib's cycle at its index
.scale_fill_tab10 <- function(labels, index) {
  values <- .tab10[(index - 1) %% length(.tab10) + 1]
  names(values) <- labels
  ggplot2::scale_fill_manual(values = values)
}

# index of each label's group (the part after the last '.') among the sorted unique groups,
# as in Visualization.get_color of the Python package
.group_index <- function(labels) {
  groups <- sub(".*\\.", "", labels)
  match(groups, sort(unique(groups), method = "radix"))
}

# ggplot2 theme of the plots: no grid lines and a centred title
.plot_theme <- function() {
  ggplot2::theme(panel.grid = ggplot2::element_blank(), plot.title = ggplot2::element_text(hjust = 0.5))
}

# error bars from the 'ymin' and 'ymax' columns, placed over dodged bars
.error_bars <- function() {
  ggplot2::geom_errorbar(ggplot2::aes(ymin = .data$ymin, ymax = .data$ymax), width = 0.2,
                         position = ggplot2::position_dodge(0.9))
}

# print the plot if 'show' is TRUE, save it to 'file' if one is given, and return it
.show_and_save <- function(p, show, file) {
  if (show) print(p)
  if (!is.null(file)) ggplot2::ggsave(file, plot = p)
  p
}

# plotmath axis labels of parameter names, such as omega[a] for omega_a and -S[d] for a negative S_d,
# as in Visualization.name_to_label of the Python package
.param_labels <- function(param_names) {
  mapping <- c(alpha = "alpha", omega = "omega", omega_a = "omega[a]", eps = "epsilon")
  labels <- vapply(param_names, function(name) {
    key <- sub("^-", "", name)
    label <- if (key %in% names(mapping)) mapping[[key]] else sub("^([^_]+)_(.+)$", "\\1[\\2]", key)
    label <- paste0(if (startsWith(name, "-")) "-" else "", label)
    if (inherits(try(str2lang(label), silent = TRUE), "try-error")) deparse(name) else label
  }, character(1))
  parse(text = labels)
}

# scales transformation of matplotlib's symmetric log scale (base 10, linear scale 1), which is linear within
# [-linthresh, linthresh] and logarithmic outside
.symlog_trans <- function(linthresh) {
  adj <- 1 / (1 - 1 / 10)

  scales::trans_new(
    "symlog",
    transform = function(x) {
      ifelse(abs(x) <= linthresh, x * adj, sign(x) * linthresh * (adj + log10(abs(x) / linthresh)))
    },
    inverse = function(y) {
      ifelse(abs(y) <= linthresh * adj, y / adj, sign(y) * linthresh * 10^(abs(y) / linthresh - adj))
    }
  )
}

# breaks at the powers of ten within the limits of a log scale, and at their 1-3 or 1-2-5 multiples where the limits
# contain fewer than two powers
.log_breaks <- function(limits) {
  powers <- 10^(floor(log10(limits[1])):ceiling(log10(limits[2])))
  powers <- powers[powers >= limits[1] & powers <= limits[2]]

  if (length(powers) >= 2) powers else scales::breaks_log()(limits)
}

# breaks at 0 and the signed powers of ten within the limits of a symmetric log scale with linear threshold
# 'linthresh', and pretty breaks where the limits contain none of them
.symlog_breaks <- function(linthresh) {
  function(limits) {
    powers <- 10^(floor(log10(linthresh)):ceiling(log10(max(abs(limits), linthresh))))
    breaks <- c(-rev(powers), 0, powers)
    breaks <- breaks[breaks >= limits[1] & breaks <= limits[2]]

    if (length(breaks) > 0) breaks else scales::extended_breaks()(limits)
  }
}

# plotmath labels writing breaks that are all 0 or signed single-digit multiples of powers of ten as m %*% 10^k,
# and plain numbers otherwise
.power_labels <- function(breaks) {
  k <- floor(log10(abs(breaks)) + 1e-9)
  m <- round(abs(breaks) / 10^k, 8)
  is_power <- is.na(breaks) | breaks == 0 | m == round(m)

  if (!all(is_power)) {
    return(format(breaks, trim = TRUE, drop0trailing = TRUE))
  }

  text <- paste0(ifelse(breaks < 0, "-", ""), ifelse(m == 1, "", paste0(m, " %*% ")), "10^", k)
  parse(text = ifelse(is.na(breaks), "''", ifelse(breaks == 0, "0", text)))
}

# ggplot2 y scale that is linear ('lin'), log10 ('log') or matplotlib's symmetric log with linear threshold
# 'linthresh' ('symlog', matplotlib's default threshold 2), labelled at the powers of ten on the logarithmic scales
.scale_y <- function(scale, expand = ggplot2::waiver(), linthresh = 2) {
  switch(
    scale,
    lin = ggplot2::scale_y_continuous(expand = expand),
    log = ggplot2::scale_y_continuous(trans = "log10", breaks = .log_breaks, labels = .power_labels,
                                      expand = expand),
    symlog = ggplot2::scale_y_continuous(trans = .symlog_trans(linthresh), breaks = .symlog_breaks(linthresh),
                                         labels = .power_labels, expand = expand)
  )
}

#' Check if the `fastdfe` Python module is installed
#'
#' This function uses the reticulate package to verify if the `fastdfe` Python 
#' module is currently installed. 
#'
#' @return Logical `TRUE` if the `fastdfe` Python module is installed, otherwise `FALSE`.
#'
#' @examples
#' \dontrun{
#' fastdfe_is_installed()  # Returns TRUE or FALSE based on the installation status of fastdfe
#' }
#' 
#' @export
fastdfe_is_installed <- function() {

  # An unbound session reports FALSE without touching Python, leaving the interpreter
  # for the declared requirements to select at the version they ask for
  if (!reticulate::py_available(initialize = FALSE)) {
    return(FALSE)
  }

  # Check if fastdfe is installed
  installed <- reticulate::py_module_available("fastdfe")

  return(installed)
}


# Requirement string for the Python distribution, carrying the optional input backends
# and a pinned version where one is given
py_requirement <- function(version = NULL, extras = c("vcf")) {

  spec <- "fastdfe"

  if (length(extras) > 0) {
    spec <- paste0(spec, "[", paste(extras, collapse = ","), "]")
  }

  if (!is.null(version)) {
    spec <- paste0(spec, "==", version)
  }

  spec
}


.onLoad <- function(libname, pkgname) {
  reticulate::py_require(py_requirement(), python_version = "3.11")
}


#' Declare the `fastdfe` Python module requirement
#'
#' Loading the package declares `fastdfe` with the `vcf` backend. This function declares
#' a different set of backends, or a pinned version.
#' The requirement is resolved when Python is first initialised, at which point
#' reticulate provisions an environment satisfying it.
#'
#' @param version A character string specifying the version of the `fastdfe` module
#'        to require. Default is `NULL` which resolves to the latest version.
#' @param extras A character vector of optional input backends to require alongside the module:
#'        `'vcf'` for VCF files, `'zarr'` for VCF-Zarr stores and `'arg'` for tree sequences.
#'        Default is `c("vcf")`; pass `NULL` to require none of them.
#' @param force Logical, has no effect. Default is `FALSE`.
#' @param silent Logical, if `TRUE` it will suppress the message naming the declared
#'        requirement. Default is `FALSE`.
#' @param python_version A character string specifying the Python version reticulate
#'        should provision the environment with. Default is `'3.11'`.
#'
#' @return Invisible `NULL`.
#'
#' @examples
#' \dontrun{
#' install_fastdfe()  # Requires the latest version of fastdfe with the vcf backend
#' install_fastdfe(extras = c("vcf", "zarr", "arg"))  # Requires all input backends
#' install_fastdfe(extras = NULL)  # Requires none of the optional backends
#' }
#'
#' @export
install_fastdfe <- function(version = NULL, extras = c("vcf"), force = FALSE, silent = FALSE, python_version = '3.11') {

  if (force) {
    warning("'force' has no effect.", call. = FALSE)
  }

  spec <- py_requirement(version, extras)

  reticulate::py_require(spec, python_version = python_version)

  if (!silent) {
    message("Declared Python requirement '", spec, "' on Python ", python_version, ".")
  }

  invisible(NULL)
}

#' Load the fastdfe library and associated visualization functions
#'
#' This function imports the Python package 'fastdfe' using the reticulate package
#' and then configures it to work seamlessly with R, overriding some of the default
#' visualization functions with custom R-based ones.
#'
#' @param install A logical. If TRUE, the function will attempt to run install_fastdfe().
#'
#' @return A reference to the 'fastdfe' Python library loaded through reticulate.
#'         This reference can be used to access 'fastdfe' functionalities.
#'
#' @examples
#' \dontrun{
#' load_fastdfe(install = TRUE)
#' # now you can use fastdfe functionalities as per its API
#' }
#'
#' @seealso \link[reticulate]{import} for importing Python modules in R.
#'
#' @export
load_fastdfe <- function(install = FALSE) {
  
  # install if install flag is true
  if (install) {
    install_fastdfe(silent = TRUE)
  }
  
  forward_python_output()

  fd <- reticulate::import("fastdfe")
  
  # override python visualization functions
  viz <- fd$visualization$Visualization
  
  # 
  # Plot discretized DFEs using a bar plot
  #
  # @param values List or numeric vector. Array of values of 
  #               size `length(intervals) - 1`, containing the 
  #               discretized DFE for each type.
  # @param errors List or numeric vector. Array of errors of 
  #               size `length(intervals) - 1`, containing the 
  #               discretized DFE for each type. Default is `NULL`.
  # @param labels List or character vector. Labels for the different types 
  #               of DFEs. Default is `NULL`.
  # @param file Character. File path to save plot to. Default is `NULL`.
  # @param show Logical. Whether to show plot. Default is `TRUE`.
  # @param intervals Numeric vector. Array of interval boundaries yielding 
  #                  `length(intervals) - 1` bars. Default 
  #                  is `c(-Inf, -100, -10, -1, 0, 1, Inf)`.
  # @param title Character. Title of the plot. Default is 'discretized DFE'.
  # @param interval_labels List of character. Labels for the intervals, 
  #                        which are the same for all types. Default is `NULL`.
  # @param ... Additional arguments which are ignored
  # 
  # @return A ggplot object.
  viz$plot_discretized <- function(
    values,
    errors = NULL,
    labels = NULL,
    file = NULL,
    show = TRUE,
    intervals = c(-Inf, -100, -10, -1, 0, 1, Inf),
    title = 'discretized DFE',
    interval_labels = NULL,
    ...
  ) {
    # number of intervals and DFEs
    n_intervals <- length(intervals) - 1
    n_dfes <- length(values)

    # label the DFEs by number if no labels are given
    labels <- if (is.null(labels)) as.character(seq_len(n_dfes)) else as.character(unlist(labels))

    df <- data.frame(
      x = rep(seq_len(n_intervals), n_dfes),
      y = unlist(values),
      group = factor(rep(labels, each = n_intervals), levels = labels)
    )

    # error bars of the DFEs that have errors
    if (!is.null(errors)) {
      has_err <- vapply(errors, function(e) !is.null(e), logical(1))

      if (any(has_err)) {
        df$ymin <- unlist(lapply(seq_len(n_dfes), function(i)
          if (has_err[i]) values[[i]] - errors[[i]][1, ] else rep(NA_real_, n_intervals)
        ))
        df$ymax <- unlist(lapply(seq_len(n_dfes), function(i)
          if (has_err[i]) values[[i]] + errors[[i]][2, ] else rep(NA_real_, n_intervals)
        ))
      }
    }

    # labels of the intervals on the x-axis
    if (is.null(interval_labels)) {
      interval_labels <- vapply(seq_len(n_intervals), function(i)
        viz$interval_to_string(intervals[i], intervals[i + 1]), character(1))
    }

    p <- ggplot2::ggplot(df, ggplot2::aes(x = factor(.data$x), y = .data$y, fill = .data$group)) +
      ggplot2::geom_bar(stat = "identity", position = ggplot2::position_dodge(), show.legend = n_dfes > 1) +
      .scale_fill_tab10(labels, .group_index(labels)) +
      ggplot2::scale_x_discrete(labels = interval_labels, expand = ggplot2::expansion(mult = c(0, 0))) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.05))) +
      ggplot2::labs(x = "S", y = "fraction", title = title) +
      .plot_theme() +
      .legend_inside()

    if ("ymin" %in% names(df)) p <- p + .error_bars()

    .show_and_save(p, show, file)
  }


  # Visualize the inferred parameters and their confidence intervals
  # using a bar plot. Note that there problems with parameters that span 0 (which is usually not the case).
  #
  # @param values List of numeric vectors. Dictionary of parameter values with the parameter in the same order as `labels`.
  # @param labels List or character vector. Unique labels for the DFEs.
  # @param param_names List or character vector. Labels for the parameters.
  # @param errors List of numeric vectors. Dictionary of errors with the parameter in the same order as `labels`.
  # @param file Character. File path to save plot to. Default is `NULL`.
  # @param show Logical. Whether to show plot. Default is `TRUE`.
  # @param title Character. Title of the plot. Default is 'parameter estimates'.
  # @param legend Logical. Whether to show the legend. Default is `TRUE`.
  # @param scale Character. Whether to use a linear or log scale. Default is 'log'.
  # @param ... Additional arguments which are ignored
  #
  # @return A ggplot object.
  viz$plot_inferred_parameters <- function(
    values,
    labels,
    param_names,
    errors = NULL,
    file = NULL,
    show = TRUE,
    title = 'parameter estimates',
    legend = TRUE,
    scale = 'log',
    ...
  ) {
    # number of types and parameters
    n_types <- length(values)
    param_names <- as.character(unlist(param_names))
    n_params <- length(param_names)

    # label the types by number if no labels are given
    labels <- if (is.null(labels)) as.character(seq_len(n_types)) else as.character(unlist(labels))

    # absolute values are plotted, with a minus sign before the names of parameters that are negative for any type
    negative <- Reduce(`|`, lapply(values, function(x) unlist(x) < 0))
    param_names <- ifelse(negative, paste0("-", param_names), param_names)

    df <- data.frame(
      x = rep(seq_len(n_params), n_types),
      y = unlist(lapply(values, function(x) abs(unlist(x)))),
      group = factor(rep(labels, each = n_params), levels = unique(labels))
    )

    # error bars of the absolute values, not extending below zero
    if (!is.null(errors) && !is.null(errors[[1]])) {
      df$ymin <- pmax(unlist(lapply(seq_len(n_types), function(i) abs(values[[i]]) - errors[[i]][1, ])), 0)
      df$ymax <- unlist(lapply(seq_len(n_types), function(i) abs(values[[i]]) + errors[[i]][2, ]))
    }

    p <- ggplot2::ggplot(df, ggplot2::aes(x = factor(.data$x), y = .data$y, fill = .data$group)) +
      ggplot2::geom_bar(stat = "identity", position = ggplot2::position_dodge(), show.legend = n_types > 1) +
      .scale_fill_tab10(labels, .group_index(labels)) +
      ggplot2::scale_x_discrete(labels = .param_labels(param_names), expand = ggplot2::expansion(mult = c(0, 0))) +
      ggplot2::labs(x = "Parameters", y = "Values", title = title) +
      .plot_theme()

    if ("ymin" %in% names(df)) p <- p + .error_bars()

    if (legend) p <- p + .legend_inside()

    # symmetric log scale with linear threshold 1e-3, so that bars start at zero
    p <- p + .scale_y(if (scale == "log") "symlog" else "lin", expand = ggplot2::expansion(mult = c(0, 0.05)),
                      linthresh = 1e-3)

    suppressWarnings(.show_and_save(p, show, file))
  }


  # Create a scatter plot.
  #
  # @param values List or numeric vector. Values to plot.
  # @param file Character. File path to save plot to. Default is NULL.
  # @param show Logical. Whether to show plot. Default is TRUE.
  # @param title Character. Title of plot.
  # @param scale Character. Scale of y-axis. One of 'lin', 'log', where 'log' is a symmetric log scale.
  #              Default is 'lin'.
  # @param ylabel Character. Label of the y-axis. Default is 'lnl'.
  # @param ... Additional arguments which are ignored.
  #
  # @return A ggplot object.
  viz$plot_scatter <- function(
    values,
    file = NULL,
    show = TRUE,
    title = NULL,
    scale = 'lin',
    ylabel = 'lnl',
    ...
  ) {
    df <- data.frame(x = seq_along(values) - 1, y = unlist(values))

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y)) +
      ggplot2::geom_point(colour = .tab10[1], size = 2) +
      .scale_y(if (scale == 'log') 'symlog' else 'lin') +
      ggplot2::labs(x = NULL, y = ylabel, title = title) +
      .plot_theme()

    .show_and_save(p, show, file)
  }


  # Plot the given 1D spectra as bars, dodged within each allele count and coloured by matplotlib's colour cycle
  # in the order of the spectra.
  #
  # @param spectra List of lists of spectra or a 2D array in which each row
  #                is a spectrum in the same order as labels
  # @param labels Character vector. Labels for each spectrum
  # @param log_scale Logical. Whether to use logarithmic y-scale
  # @param use_subplots Logical. Whether to use subplots
  # @param show_monomorphic Logical. Whether to show monomorphic site counts
  # @param title Character. Title of plot
  # @param n_ticks Numeric. Number of x-ticks to use
  # @param file Character. File to save plot to
  # @param show Logical. Whether to show the plot
  # @param ... Additional arguments which are ignored.
  #
  # @return ggplot object
  plot_spectra <- function(
    spectra,
    labels = character(0),
    log_scale = FALSE,
    use_subplots = FALSE,
    show_monomorphic = FALSE,
    title = NULL,
    n_ticks = 10,
    file = NULL,
    show = TRUE,
    ...
  ) {
    if (length(spectra) == 0) {
      warning('No spectra to plot.')
      return(NULL)
    }

    labels <- as.character(unlist(labels))

    if (use_subplots) {
      # one plot per spectrum on a square grid, titled by its label
      n_cols <- ceiling(sqrt(length(spectra)))

      plot_list <- lapply(seq_along(spectra), function(i) {
        label <- if (length(labels) >= i) labels[i] else character(0)

        plot_spectra(
          spectra = list(spectra[[i]]),
          labels = label,
          log_scale = log_scale,
          show_monomorphic = show_monomorphic,
          title = if (length(label)) label else NULL,
          n_ticks = 15 %/% min(2, n_cols),
          show = FALSE
        )
      })

      return(.show_and_save(cowplot::plot_grid(plotlist = plot_list, nrow = n_cols, ncol = n_cols), show, file))
    }

    if (length(labels) == 0) {
      labels <- as.character(seq_along(spectra))
    }

    # allele counts of the bars, each spectrum taking an equal share of the width 0.9 per allele count
    n <- length(spectra[[1]]) - 1
    x <- if (show_monomorphic) 0:n else seq_len(n - 1)
    width <- 0.9 / length(spectra)

    df <- data.frame(
      xmin = unlist(lapply(seq_along(spectra), function(i) x - 0.45 + (i - 1) * width)),
      y = unlist(lapply(spectra, function(sfs) unlist(sfs)[x + 1])),
      group = factor(rep(labels, each = length(x)), levels = unique(labels))
    )
    df$xmax <- df$xmin + width

    # on the log scale, bars rise from the power of ten below the smallest positive count, which is the bottom of the axis
    if (log_scale) {
      df <- df[df$y > 0, ]
      df$ymin <- 10^floor(log10(min(df$y)))
    } else {
      df$ymin <- 0
    }

    # label every allele count, or every k-th starting at 1 where there are more than 'n_ticks'
    breaks <- if (n > n_ticks) x[x %% ceiling(n / n_ticks) == 1] else x

    p <- ggplot2::ggplot(df, ggplot2::aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin, ymax = .data$y,
                                          fill = .data$group)) +
      ggplot2::geom_rect(show.legend = length(spectra) > 1) +
      .scale_fill_tab10(levels(df$group), seq_len(nlevels(df$group))) +
      ggplot2::scale_x_continuous(breaks = breaks, expand = c(0, 0)) +
      .scale_y(if (log_scale) 'log' else 'lin', expand = ggplot2::expansion(mult = c(0, 0.05))) +
      ggplot2::labs(x = "allele count", y = NULL, title = title) +
      .plot_theme() +
      .legend_inside()

    .show_and_save(p, show, file)
  }

  viz$plot_spectra <- plot_spectra


  # Plot p-values of nested likelihoods.
  #
  # @param P Matrix of p-values
  # @param labels_x Labels for x-axis
  # @param labels_y Labels for y-axis
  # @param file File to save plot to
  # @param show Whether to show plot
  # @param cmap Colormap to use
  # @param title Title of plot
  # @param vmin Minimum value for colorbar
  # @param vmax Maximum value for colorbar
  # 
  # @return A ggplot object.
  viz$plot_nested_models <- function(
    P,
    labels_x,
    labels_y,
    file = NULL,
    show = TRUE,
    cmap = NULL,
    title = NULL,
    vmin = 1e-10,
    vmax = 1,
    ...
  ) {
    # Format number to be displayed.
    format_number <- function(x) {
      if (x == 0) return("0")
      if (x < 0.0001) return(sprintf("%.1e", x))
      as.character(round(x, 4))
    }

    # numeric matrix of the same shape as the list-matrix P, with NA where the models are not nested
    P_mat <- matrix(sapply(P, function(x) ifelse(is.null(x), NA, x)), nrow = nrow(P), ncol = ncol(P))

    # one row per cell, with the first row of P at the top
    df <- data.frame(
      x = factor(rep(seq_len(ncol(P_mat)), each = nrow(P_mat))),
      y = factor(rep(seq_len(nrow(P_mat)), times = ncol(P_mat)), levels = rev(seq_len(nrow(P_mat)))),
      p = as.vector(P_mat)
    )
    df$label <- vapply(df$p, function(x) if (is.na(x)) "-" else format_number(x), character(1))

    # colour cells of models that are not nested as p = 1 and keep values within the colour bar bounds
    df$fill <- pmin(pmax(ifelse(is.na(df$p), 1, df$p), vmin), vmax)

    fill_scale <- if (is.null(cmap)) {
      ggplot2::scale_fill_viridis_c(option = "inferno", begin = 0.3, trans = "log10",
                                    limits = c(vmin, vmax), name = "p-value")
    } else {
      ggplot2::scale_fill_gradientn(colours = cmap, trans = "log10", limits = c(vmin, vmax), name = "p-value")
    }

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$fill)) +
      ggplot2::geom_tile(colour = "#cccccc", linewidth = 0.5) +
      ggplot2::geom_text(ggplot2::aes(label = .data$label, colour = .data$fill < 1e-4), show.legend = FALSE) +
      ggplot2::scale_colour_manual(values = c(`FALSE` = "black", `TRUE` = "white")) +
      fill_scale +
      ggplot2::scale_x_discrete(labels = gsub("_", " ", unlist(labels_x)), expand = c(0, 0)) +
      ggplot2::scale_y_discrete(labels = rev(gsub("_", " ", unlist(labels_y))), expand = c(0, 0)) +
      ggplot2::coord_fixed() +
      ggplot2::labs(x = NULL, y = NULL, title = title) +
      .plot_theme() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

    .show_and_save(p, show, file)
  }

  # fastDFE re-exports sfsutils' spectra and annotations, which plot through sfsutils'
  # Visualization rather than fastdfe's. Route its two plotting methods through the same ggplot
  # overrides so they render as ggplot in R instead of falling through to matplotlib.
  viz_sfs <- reticulate::import("sfsutils")$visualization$Visualization
  viz_sfs$plot_spectra <- plot_spectra
  viz_sfs$plot_scatter <- viz$plot_scatter

  return(fd)
}


# In a Jupyter kernel, write Python's standard output and error through R's output and message streams, which the kernel
# captures, so log messages and progress bars reach the cell output.
forward_python_output <- function() {

  if (!isTRUE(getOption("jupyter.in_kernel"))) {
    return(invisible(NULL))
  }

  # the kernel ends every message with a line break, so the error stream is passed on as complete lines without one
  streams <- reticulate::py_run_string("
import io

class RStream(io.TextIOBase):
    encoding = 'utf-8'

    def __init__(self, write, lines=False):
        super().__init__()
        self._write = write
        self._lines = lines
        self._buffer = ''

    def writable(self):
        return True

    def write(self, text):
        if not self._lines:
            self._write(text)
            return len(text)

        *complete, partial = (self._buffer + text).split('\\n')
        for line in complete:
            self._write(line.split('\\r')[-1])

        # a carriage return starts the line over, as a progress bar redraws itself
        self._buffer = partial.split('\\r')[-1]

        return len(text)
", local = TRUE, convert = FALSE)

  sys <- reticulate::import("sys", convert = FALSE)
  sys$stdout <- streams$RStream(function(text) cat(reticulate::py_to_r(text)))
  sys$stderr <- streams$RStream(
    function(line) message(reticulate::py_to_r(line), appendLF = FALSE),
    lines = TRUE
  )

  invisible(NULL)
}
