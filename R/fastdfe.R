library(ggplot2)
library(reticulate)

# name of conda environment
envname <- "base-r-fastdfe"

# list the existing conda environments
envs <- conda_list()

# check if "base-r-fastdfe" is in the list of environments
if (!(envname %in% envs$name)) {
  # if "base-r-fastdfe" does not exist, create it
  conda_create(envname, environment = "snakemake/envs/base-r.yaml")
}

# activate conda environment
use_condaenv(envname)

fastdfe <- import("fastdfe")
# print(fastdfe$`__version__`)

viz <- fastdfe$visualization$Visualization

# configure plot
options(repr.plot.width = 4.6, repr.plot.height = 3.2)

#' Plot discretized DFEs using a bar plot
#'
#' @param values List or numeric vector. Array of values of 
#'               size `length(intervals) - 1`, containing the 
#'               discretized DFE for each type.
#' @param errors List or numeric vector. Array of errors of 
#'               size `length(intervals) - 1`, containing the 
#'               discretized DFE for each type. Default is `NULL`.
#' @param labels List or character vector. Labels for the different types 
#'               of DFEs. Default is `NULL`.
#' @param file Character. File path to save plot to. Default is `NULL`.
#' @param show Logical. Whether to show plot. Default is `TRUE`.
#' @param intervals Numeric vector. Array of interval boundaries yielding 
#'                  `length(intervals) - 1` bars. Default 
#'                  is `c(-Inf, -100, -10, -1, 0, 1, Inf)`.
#' @param title Character. Title of the plot. Default is 'discretized DFE'.
#' @param interval_labels List of character. Labels for the intervals, 
#'                        which are the same for all types. Default is `NULL`.
#' @param ... Additional arguments which are ignored
#' 
#' @return A ggplot object.
#' 
#' @export
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
  if (!is.null(labels)) {
    df$group <- as.factor(rep(unlist(labels), each = n_intervals))
  }
  
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
  p <- ggplot(df, aes(x = factor(x), y = y)) +
    geom_bar(stat = "identity", position = position_dodge(), colour = "black") +
    scale_x_discrete(labels = xlabels) +
    labs(x = "S", y = "fraction", title = title) +
    theme_bw() +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank())
  
  # add group aesthetic if 'group' column present
  if ('group' %in% colnames(df)) p <- p + aes(fill = group)
  
  # add error bars if 'ymin' and 'ymax' columns present
  if (all(c('ymin', 'ymax') %in% colnames(df))) {
    p <- p + geom_errorbar(aes(ymin = ymin, ymax = ymax), width = 0.2, 
                           position = position_dodge(0.9))
  }
  
  # add legend on the right if labels were provided
  if (!is.null(labels)) p <- p + theme(legend.position = "right")
  
  # display plot if 'show' is TRUE
  if (show) print(p)
  
  # save plot to file if 'file' is provided
  if (!is.null(file)) ggsave(file, plot = p)
  
  return(p)
}



