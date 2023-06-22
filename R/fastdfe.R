library(ggplot2)
library(reticulate)

# name of conda environment
envname = "base-r-fastdfe"

# list the existing conda environments
envs <- conda_list()

# check if "base-r-fastdfe" is in the list of environments
if (!(envname %in% envs$name)) {
  # if "base-r-fastdfe" does not exist, create it
  conda_create(envname, environment="snakemake/envs/base-r.yaml")
}

# activate conda environment
use_condaenv(envname)

fastdfe <- import("fastdfe")
print(fastdfe$`__version__`)

# redefine plot_discretized using native R plotting
fastdfe$visualization$Visualization$plot_discretized <- function(
    values,
    errors,
    labels,
    intervals,
    file = NULL,
    show = TRUE,
    title = 'discretized DFE',
    interval_labels = NULL,
    legend_text_size = 8,
    ax = NULL,
    kwargs_legend = NULL
) {
  
  # create dataframe
  df <- data.frame(values = unlist(values), 
                   errors = unlist(errors), 
                   labels = unlist(labels))
  
  # create interval factor variable
  df$interval <- cut(df$values, breaks = intervals, labels = interval_labels)
  
  # create the plot
  p <- ggplot(df, aes(x = interval, y = values, fill = labels)) +
    geom_bar(stat = 'identity', position = 'dodge') +
    geom_errorbar(aes(ymin = values - errors, ymax = values + errors), 
                  width = 0.2, position = position_dodge(0.9)) + 
    labs(title = title, x = 'S', y = 'fraction') + 
    theme(legend.position='bottom', 
          legend.text = element_text(size = legend_text_size), 
          legend.title=element_blank())
  
  # display the plot if show is TRUE
  if(show) {
    print(p)
  }
  
  # save the plot to a file if a filename is provided
  if(!is.null(file)) {
    ggsave(filename = file, plot = p, width = 7, height = 7, units = "in")
  }
  
  return(p)
}
