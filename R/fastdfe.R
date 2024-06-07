if (getRversion() >= "2.15.1") utils::globalVariables(c(".data"))

# vector of required packages
required_packages <- c("reticulate", "ggplot2", "cowplot", "pheatmap", "RColorBrewer", "scales")

# install required R packages
for(package in required_packages){
  if(!package %in% installed.packages()[,"Package"]){
    install.packages(package)
  }
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
#' is_installed()  # Returns TRUE or FALSE based on the installation status of fastdfe
#' }
#' 
#' @export
fastdfe_is_installed <- function() {
  
  # Check if fastdfe is installed
  installed <- reticulate::py_module_available("fastdfe")
  
  return(installed)
}


#' Install the `fastdfe` Python module
#'
#' This function checks if the `fastdfe` Python module is available.
#' If not, or if the `force` argument is TRUE, it installs it via pip.
#' If the `silent` argument is set to TRUE, the function will not output a 
#' message when the module is already installed.
#'
#' @param version A character string specifying the version of the `fastdfe` module
#'        to install. Default is `NULL` which will install the latest version.
#' @param force Logical, if `TRUE` it will force the reinstallation of the `fastdfe` module 
#'        even if it's already available. Default is `FALSE`.
#' @param silent Logical, if `TRUE` it will suppress the message about `fastdfe` being 
#'        already installed. Default is `FALSE`.
#'
#' @return Invisible `NULL`.
#' 
#' @examples
#' \dontrun{
#' install_fastdfe()  # Installs the latest version of fastdfe
#' install_fastdfe("1.1.7")  # Installs version 1.1.7 of fastdfe
#' install_fastdfe(force = TRUE)  # Reinstalls the fastdfe module
#' }
#' 
#' @export
install_fastdfe <- function(version = NULL, force = FALSE, silent = FALSE, python_version = '3.11') {
  
  # Create the package string with the version if specified
  package_name <- "fastdfe"
  if (!is.null(version)) {
    package_name <- paste0(package_name, "==", version)
  }
  
  # Check if fastdfe is installed or if force is TRUE
  if (force || !fastdfe_is_installed()) {
    reticulate::py_install(
      package_name, 
      method = "conda",
      pip = TRUE,
      python_version = python_version,
      version = version, 
      ignore_installed = TRUE
   )
  } else {
    if (!silent) {
      message("The 'fastdfe' Python module is already installed.")
    }
  }
  
  invisible(NULL)
}

#' Load the fastdfe library and associated visualization functions
#'
#' This function imports the Python package 'fastdfe' using the reticulate package
#' and then configures it to work seamlessly with R, overriding some of the default
#' visualization functions with custom R-based ones. This function also ensures
#' that required R libraries are loaded for visualization.
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
#' @importFrom grDevices colorRampPalette dev.off pdf
#' @export
load_fastdfe <- function(install = FALSE) {
  
  # install if install flag is true
  if (install) {
    install_fastdfe(silent = TRUE)
  }
  
  # configure plot
  options(repr.plot.width = 4.6, repr.plot.height = 3.2)
  
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
    
    # create data frame with x and y values
    df <- data.frame(x = rep(1:n_intervals, n_dfes), 
                     y = unlist(values))
    
    # if labels provided, add as factor to data frame
    if (is.null(labels)) {
      labels <- as.character(1:n_dfes)
    }
    df$group <- as.factor(rep(unlist(labels), each = n_intervals))
    
    # if errors provided, calculate ymin and ymax
    if (!is.null(errors) && !is.null(errors[[1]])) {
      df$ymin <- unlist(lapply(1:n_dfes, function(i) values[[i]] - errors[[i]][1, ]))
      df$ymax <- unlist(lapply(1:n_dfes, function(i) values[[i]] + errors[[i]][2, ]))
    }
    
    # create labels for x-axis
    if (is.null(interval_labels)) {
      xlabels <- c()
      for (i in 2:length(intervals)) {
        xlabels <- c(xlabels, viz$interval_to_string(intervals[i - 1], 
                                                     intervals[i]))
      }
    } else {
      xlabels <- interval_labels
    }
    
    # base plot with bars
    p <- ggplot2::ggplot(df, ggplot2::aes(x = factor(.data$x), y = .data$y)) +
      ggplot2::geom_bar(stat = "identity", position = ggplot2::position_dodge(),
                        show.legend = n_dfes > 1) +
      ggplot2::scale_x_discrete(labels = xlabels, expand = ggplot2::expansion(mult = c(0, 0))) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.05))) +
      ggplot2::labs(x = "S", y = "fraction", title = title) +
      ggplot2::theme_bw() +
      ggplot2::theme(panel.grid.major = ggplot2::element_blank(), 
                     panel.grid.minor = ggplot2::element_blank())
    
    # add group aesthetic if 'group' column present
    if ('group' %in% colnames(df)) p <- p + ggplot2::aes(fill = .data$group)
    
    # add error bars if 'ymin' and 'ymax' columns present
    if (all(c('ymin', 'ymax') %in% colnames(df))) {
      p <- p + ggplot2::geom_errorbar(ggplot2::aes(ymin = .data$ymin, ymax = .data$ymax), 
                                      width = 0.2, 
                                      position = ggplot2::position_dodge(0.9))
    }
    
    # add legend on the right if labels were provided
    if (!is.null(labels)) p <- p + ggplot2::theme(legend.position = "right")
    
    # display plot if 'show' is TRUE
    if (show) print(p)
    
    # save plot to file if 'file' is provided
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)
    
    return(p)
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
    n_params <- length(param_names)
    
    # create data frame with x, y values and track the negative values
    negative_flags <- unlist(lapply(values, function(x) sapply(x, function(y) ifelse(y < 0, TRUE, FALSE))))
    df <- data.frame(x = rep(1:n_params, n_types), 
                     y = unlist(lapply(values, function(x) sapply(x, abs))))
    
    # Adjust the parameter names based on negative flags
    updated_param_names <- param_names
    for (i in 1:length(negative_flags)) {
      if (negative_flags[i]) {
        idx <- (i - 1) %% n_params + 1
        updated_param_names[idx] <- paste0("-", param_names[idx])
      }
    }
    
    # if labels provided, add as factor to data frame
    if (is.null(labels)) {
      labels <- as.character(1:n_types) # create numeric labels if labels are NULL
    }
    df$group <- as.factor(rep(unlist(labels), each = n_params))
    
    # if errors provided, add as y-error bars to data frame
    if (!is.null(errors) && !is.null(errors[[1]])) {
      df$ymin <- unlist(lapply(1:n_types, function(i) abs(values[[i]]) - errors[[i]][1, ]))
      df$ymax <- unlist(lapply(1:n_types, function(i) abs(values[[i]]) + errors[[i]][2, ]))
    }
    
    # base plot with bars
    p <- ggplot2::ggplot(df, ggplot2::aes(x = factor(.data$x), y = .data$y)) +
      ggplot2::geom_bar(stat = "identity", position = ggplot2::position_dodge(),
                        show.legend = n_types > 1) +
      ggplot2::scale_x_discrete(labels = updated_param_names, expand = ggplot2::expansion(mult = c(0, 0))) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.05))) +
      ggplot2::labs(x = "Parameters", y = "Values", title = title) +
      ggplot2::theme_bw() +
      ggplot2::theme(panel.grid.major = ggplot2::element_blank(), 
                     panel.grid.minor = ggplot2::element_blank())
    
    # add group aesthetic if 'group' column present
    if ('group' %in% colnames(df)) p <- p + ggplot2::aes(fill = .data$group)
    
    # add error bars if 'ymin' and 'ymax' columns present
    if (all(c('ymin', 'ymax') %in% colnames(df))) {
      p <- p + ggplot2::geom_errorbar(ggplot2::aes(ymin = .data$ymin, ymax = .data$ymax),
                                      width = 0.2, 
                                      position = ggplot2::position_dodge(0.9))
    }
    
    # add legend on the right if labels were provided
    if (legend) p <- p + ggplot2::theme(legend.position = "right")
    
    # scale y axis if specified
    if (scale == 'log') {
      suppressWarnings(p <- p + ggplot2::scale_y_continuous(trans = "log10"))
    }
    
    # display plot if 'show' is TRUE
    if (show) suppressWarnings(print(p))
    
    # save plot to file if 'file' is provided
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)
    
    return(p)
  }
  
  
  # Create a scatter plot.
  #
  # @param values List or matrix. Values to plot.
  # @param file Character. File path to save plot to. Default is NULL.
  # @param show Logical. Whether to show plot. Default is TRUE.
  # @param title Character. Title of plot.
  # @param scale Character. Scale of y-axis. One of 'lin', 'log'. Default is 'lin'.
  #
  # @return A ggplot object.
  viz$plot_scatter <- function(
    values,
    file = NULL,
    show = TRUE,
    title = NULL,
    scale = 'lin',
    ...
  ) {
    # Create data frame
    data <- data.frame(x = seq_along(values), y = unlist(values))
    
    # Create plot
    p <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y)) +
      ggplot2::geom_point() +
      ggplot2::labs(title = title, y = 'lnl')
    
    # Set y scale
    if (scale == 'log') {
      p <- p + ggplot2::scale_y_continuous(trans = 'log10')
    }
    
    # Display plot if 'show' is TRUE
    if (show) print(p)
    
    # Save plot to file if 'file' is provided
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)
    
    return(p)
  }
  
  
  # Plot the given 1D spectra
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
  #
  # @return ggplot object
  viz$plot_spectra <- function(
    spectra,
    labels = character(0),
    log_scale = FALSE,
    use_subplots = FALSE,
    show_monomorphic = FALSE,
    title = NULL,
    file = NULL,
    show = TRUE,
    ...
  ) {
    
    if (length(spectra) == 0) {
      warning('No spectra to plot.')
      return(NULL)
    }
    
    if (use_subplots) {
      # Creating a grid of plots
      plot_list <- lapply(1:length(spectra), function(i) {
        viz$plot_spectra(
          spectra = list(spectra[[i]]),
          labels = if (length(labels)) labels[i] else character(0),
          log_scale = log_scale,
          show_monomorphic = show_monomorphic,
          show = FALSE
        ) +
          ggplot2::labs(title = if (length(labels) >= i) labels[i] else '')
      })
      
      plot_grid <- cowplot::plot_grid(plotlist = plot_list)
      
      if (show) print(plot_grid)
      if (!is.null(file)) ggplot2::ggsave(file, plot = plot_grid)
      
      return(plot_grid)
    }
    
    if (length(labels) == 0) {
      labels <- as.character(1:length(spectra))
    }
    
    df <- data.frame()
    for (i in seq_along(spectra)) {
      indices <- if (show_monomorphic) seq_along(spectra[[i]]) else seq_along(spectra[[i]])[-c(1, length(spectra[[i]]))]
      heights <- if (show_monomorphic) unlist(spectra[[i]]) else unlist(spectra[[i]][-c(1, length(spectra[[i]]))])
      df_temp <- data.frame(indices = indices, 
                            heights = heights, 
                            group = rep(labels[i], length(indices)))
      df <- rbind(df, df_temp)
    }
    
    # Create a ggplot object
    p <- ggplot2::ggplot(df, ggplot2::aes(x = indices, y = heights, fill = .data$group)) +
      ggplot2::geom_bar(stat = "identity", position = "dodge",
                        width = 0.7, show.legend = length(spectra) > 1) +
      ggplot2::labs(x = "frequency", y = "", title = title) +
      ggplot2::theme_bw() +
      ggplot2::theme(panel.grid.major = ggplot2::element_blank(), 
                     panel.grid.minor = ggplot2::element_blank()) + 
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, .1)))
    
    if (log_scale) {
      p <- p + ggplot2::scale_y_log10()
    }
    
    # Adjust x-axis labels based on show_monomorphic
    if (show_monomorphic) {
      p <- p + ggplot2::scale_x_continuous(breaks = 0:(length(spectra[[1]]) + 1),
                                           labels = 0:(length(spectra[[1]]) + 1) - 1, 
                                           expand = c(0, 0))
    } else {
      p <- p + ggplot2::scale_x_continuous(breaks = 1:length(spectra[[1]]),
                                           labels = 1:length(spectra[[1]]) - 1,
                                           expand = c(0, 0))
    }
    
    # Display or save the plot
    if (show) print(p)
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)
    
    return(p)
  }
  
  
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
  # @return NULL
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
      if (is.null(x) || is.na(x) || x == 0) {
        return(0)
      }
      
      if (x < 0.0001) {
        return(format(x, scientific = TRUE))
      }
      
      return(round(x, 4))
    }
    
    # Convert list to matrix and replace NULLs with NAs
    P <- sapply(P, function(x) ifelse(is.null(x), NA, x))
    P_mat <- matrix(P, nrow = sqrt(length(P)), ncol = sqrt(length(P)), byrow = TRUE)
    
    # Determine values to display
    annotation <- apply(P_mat, c(1, 2), function(x) as.character(format_number(x)))
    annotation[is.na(P_mat)] <- '-'
    
    # Change NAs to 1 to get a nicer color
    P_mat[is.na(P_mat)] <- NA
    
    # Keep within color bar bounds
    P_mat[P_mat < vmin] <- vmin
    P_mat[P_mat > vmax] <- vmax
    
    # Default color map
    if (is.null(cmap)) {
      cmap <- rev(colorRampPalette(RColorBrewer::brewer.pal(9, "YlOrRd"))(100))
    }
    
    # Plot heatmap
    p = pheatmap::pheatmap(mat = P_mat, color = cmap, 
                           cluster_rows = FALSE, cluster_cols = FALSE, 
                           display_numbers = TRUE, legend = TRUE,
                           breaks = seq(vmin, 0.1, length.out = 101),
                           labels_row = gsub("_", " ", labels_y),
                           labels_col = gsub("_", " ", labels_x),
                           angle_col = 45,
                           cellheight = 40,
                           cellwidth = 40,
                           main = title
    )
    
    if (!is.null(file)) {
      pdf(file)
      print(p)
      dev.off()
    }
    
    if (!show) {
      dev.off()
    }
  }
  
  return(fd)
}
