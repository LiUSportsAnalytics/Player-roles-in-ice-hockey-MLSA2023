library(relimp, pos = 4)
library(paran)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)
library(stringr)
library(patchwork)
source("utils.R")

#' Fit PCA to the processed data
#'
#' Fit a principal component analysis (PCA) to the processed data for the given
#' position. Also visualizes the proportion of variance explained by each component.
#'
#' @param processed_data data frame of normalized and standardized statistics for a position group
#'
#' @return prcomp object, i.e., pca object for the position-specific data
#' @export
#'
fit_pca <- function(processed_data){
  # Fit the principal component analysis
  pca <- prcomp(processed_data)
  
  # Compute variance explained for all components
  var_explained <- (pca$sdev)^2 / sum((pca$sdev)^2)
  
  # Capture the name of the input data
  input_data <- deparse(substitute(processed_data))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Visualize the results
  data.frame(PC = 1:length(pca$sdev), var_explained) %>% 
    ggplot(aes(x=PC, y=cumsum(var_explained))) +
    geom_col(fill="darkorange") + 
    geom_text(label=round(var_explained, 2), nudge_y = 0.01, size=1) + 
    xlab("Principal component") + ylab("Cumulative % of variance explained") + 
    scale_y_continuous(expand=c(0, 0.01), breaks=seq(0, 1, 0.2)) + 
    scale_x_continuous(expand=c(0.01, 0), breaks=seq(1, length(pca$sdev), 5)) +
    #ggtitle(paste("Principal component analysis for", position_name)) +
    theme_custom() + theme(axis.text = element_text(size=16),
                           axis.title = element_text(size=17)) -> g
  
  # Print the plot  
  print(g)
  
  # Save the plot
  ggsave(paste("../Plots/PCA", position_name, ".png"), height=4, width=6)
  ggsave(paste("../Plots/PCA", position_name, ".pdf"), height=4, width=6)
  
  return(pca)
}


#' Visualize proportion of variance explained.
#'
#' @param pca_forwards pca object for forwards
#' @param pca_defenders pca object for defenders
#'
#' @return plot
#' @export
#'
visualize_var_explained <- function(pca_forwards, pca_defenders){
  # Compute variance explained for all components
  var_explained_forwards <- (pca_forwards$sdev)^2 / sum((pca_forwards$sdev)^2)
  var_explained_defenders <- (pca_defenders$sdev)^2 / sum((pca_defenders$sdev)^2)
  
  # Visualize variance explained in the same graph
  data.frame(PC = 1:length(pca_forwards$sdev), forwards=var_explained_forwards, 
             defenders=var_explained_defenders) %>% 
    pivot_longer(-PC, names_to="Position", values_to="Variance") %>% 
    mutate(Position = str_to_title(Position)) %>% 
    group_by(Position) %>% 
    mutate(VarianceCum = cumsum(Variance)) %>% 
    ungroup() %>% 
    ggplot(aes(x=PC, y=VarianceCum, color=Position)) +
    geom_point(size=1.5, shape=19, alpha=1) + geom_line(linewidth=1.2, alpha=1) +  
    xlab("Principal component") + ylab("Cumulative proportion of \nvariance explained") + 
    scale_y_continuous(expand=c(0, 0.01), limits=c(0, 1), breaks=seq(0, 1, 0.2)) + 
    scale_x_continuous(expand=c(0.01, 0), breaks=seq(0, length(pca_forwards$sdev), 5)) +
    scale_color_manual(name="", values=c('#EE7733', '#0077BB')) + 
    theme_custom() + theme(axis.text = element_text(size=14),
                           axis.title = element_text(size=16), 
                           legend.text = element_text(size=14),
                           legend.position=c(0.85, 0.5))
  
  # Save the plot
  ggsave("../Plots/PCA.png", height=4, width=6)
  ggsave("../Plots/PCA.pdf", height=4, width=6)
}

#' Visualize loadings for a given PCA object.
#'
#' @param pca pca object for the position-specific data
#'
#' @return a plot, saved to disk
#' @export
#'
visualize_loadings <- function(pca){
  # Capture the name of the input data
  input_data <- deparse(substitute(pca))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Get all factor loadings
  loadings <- pca$rotation
  
  # Create a plot to visualize loadings
  loadings %>% 
    as.data.frame() %>% 
    dplyr::select(1:20) %>% 
    rownames_to_column(var="Variable") %>% 
    pivot_longer(-Variable, names_to = "PC") %>% 
    arrange(Variable) %>% 
    mutate(PC = factor(PC, levels=unique(PC)),
           Variable = factor(Variable, levels=rev(unique(Variable)))) %>% 
    drop_na() %>% 
    ggplot(aes(PC, Variable, fill=value)) + geom_tile() + 
    scale_fill_gradient2("Loading", low="#CC3311", mid="white", high="#117733",
                         limits=c(-0.5, 0.5)) +
    xlab("") + ylab("") + theme_custom() + 
    theme(legend.key.height = unit(1.2, 'cm'), 
          axis.text.x = element_text(angle = 45, vjust = 0.5))
  
  # Save the plot
  ggsave(paste("../Plots/PCA_loadings", position_name, ".png"), height=5, width=7)
  ggsave(paste("../Plots/PCA_loadings", position_name, ".pdf"), height=5, width=7)
}


#' Choose the number of principal component
#'
#' Perform parallel analysis (Horn, 1965) to select the number of principal 
#' components for the subsequent clustering. Also visualizes the results.
#'
#' @param processed_data data frame of normalized and standardized statistics for a position group
#'
#' @return list of results related to the parallel analysis
#' @export
#'
choose_nr_of_components <- function(processed_data, n_iter=100000, centile=99){ 
  # Fit Parallel analysis (Horn, 1965) to select number of PC 
  pa <- paran(processed_data, iterations = n_iter, centile = centile, quietly = FALSE,
              status = TRUE, all = TRUE, cfa = FALSE, graph = TRUE, color = TRUE, 
              col = c("black", "red", "blue"), lty = c(1, 2, 3), lwd = 1, legend = TRUE, 
              file = "", width = 640, height = 640, grdevice = "png", seed = 0)
  return(pa)
}

