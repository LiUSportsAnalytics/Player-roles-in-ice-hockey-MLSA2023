library(readr)
library(tidyr)
library(dplyr)
library(pheatmap)
library(ggplot2)
library(patchwork)
library(stringr)
library(ggplotify)

# Change font for pheatmap
windowsFonts(serif = windowsFont("Times New Roman"))
par(family = "sans")

# Specify the working directory
if(length(str_split(getwd(), "/Scripts")[[1]]) != 2){
  setwd("Scripts")
}
# Get the top n players (w.r.t. TOI) from each team
top_n <- read_csv("../Data/top_n.csv", show_col_types = FALSE)

# Get the fuzzy clustering
fuzzy_cluster <- read_csv("../Data/fuzzy_cluster.csv", show_col_types = FALSE)

# Add team players with their cluster probabilities
top_n_probs <- inner_join(top_n, fuzzy_cluster, by="PlayerId")  

# Compute the number of players per team and cluster
top_n_sums <- top_n_probs %>% 
  group_by(TeamName, Points, GF, GA, Rank) %>% 
  summarise(across(matches("F\\d|D\\d"), ~ sum(.x, na.rm=TRUE))) %>% 
  ungroup()

# Compute the number of forwards and defenders per team
top_n_sums <- top_n_sums %>% 
  mutate(n_forwards = rowSums(dplyr::select(., matches("F\\d"))),
         n_defenders = rowSums(dplyr::select(., matches("F\\d"))))

# Extract forward roles
top_n_sums %>% 
  dplyr::select(matches("F\\d")) %>% 
  rowwise() %>% 
  mutate(row_sum = rowSums(across(F1:F4)), 
         F1 = F1/row_sum, F2 = F2/row_sum, F3 = F3/row_sum, F4 = F4/row_sum) %>% 
  dplyr::select(-row_sum) %>% 
  as.matrix() -> forwards

# Extract defender roles
top_n_sums %>% 
  dplyr::select(matches("D\\d")) %>% 
  rowwise() %>% 
  mutate(row_sum = rowSums(across(D1:D3)), 
         D1 = D1/row_sum, D2 = D2/row_sum, D3 = D3/row_sum) %>% 
  dplyr::select(-row_sum) %>% 
  as.matrix() -> defenders

# Combine forwards and defenders into one matrix
plot_matrix <- cbind(forwards, defenders)

# Specify row names
rownames(plot_matrix) <- paste0(top_n_sums$TeamName, " (", top_n_sums$Points, ") ")

# Create row annotations
row_annotations <- data.frame(MadePlayoffs = factor(c(0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
                                                      1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0,
                                                      1, 0, 0, 1, 1, 1, 0, 0, 1, 0), 
                                                    labels = c("No", "Yes")))
# Specify row names for annotations
rownames(row_annotations) <- paste0(top_n_sums$TeamName, " (", top_n_sums$Points, ") ")

# Specify colors for playoffs
ann_colors = list(
  MadePlayoffs = c(No = "#E94B3CFF", Yes = "#2BAE66FF")
)

# Create initial plot with heatmap
p1 <- as.ggplot(pheatmap(plot_matrix, color = scico::scico(10, direction=-1, palette = "acton"), 
                         border_color=NA, legend_breaks = c(seq(0, 0.5, 0.1), max(plot_matrix)), 
                         main = "", legend_labels = c(paste(seq(0, 0.5, 0.1)), "Proportion"), 
                         annotation_row = row_annotations, annotation_colors = ann_colors, 
                         clustering_method = "ward.D2", cutree_rows = 2, angle_col = 0, 
                         cluster_cols = FALSE, annotation_names_row = FALSE, 
                         legend = TRUE, annotation_legend = FALSE))

# Create plot with just annotation legend
p2 <- as.ggplot(pheatmap(plot_matrix, color = scico::scico(10, direction=-1, palette = "acton"), 
                         border_color=NA, legend_breaks = c(seq(0, 0.5, 0.1), max(plot_matrix)), 
                         main = "", legend_labels = c(paste(seq(0, 0.5, 0.1)), "Proportion"), 
                         annotation_row = row_annotations, annotation_colors = ann_colors, 
                         cluster_rows = FALSE, cluster_cols = FALSE,
                         annotation_names_row = FALSE, cellwidth = 0, cellheight = 0,
                         legend = FALSE, annotation_legend = TRUE, 
                         show_colnames = FALSE, show_rownames = FALSE))

# Combine both plots into one and add extra features
p1 + inset_element(p2, left = 0, bottom = 0, right = 1.85, top = 1) +
  # Hide previous proportion text
  annotate("rect", xmin=0.475, xmax=0.6, ymin=0.93, ymax=0.99, fill="white") +
  # Add bold proportion text
  annotate("text", x=0.488, y=0.97, label="Proportion", fontface="bold", size=3.5) +
  # Add information regarding row
  annotate("text", x=0.348, y=0.97, label="Team (Points)", fontface="bold", size=3.5) +
  # Add text information regarding cells
  annotate("text", x=0.145, y=0.98, label="Forwards", fontface="bold", size=3.5) +
  annotate("text", x=0.261, y=0.98, label="Defensemen", fontface="bold", size=3.5) +
  # Add text information regarding forward cells
  annotate("segment", x=0.095, xend=0.095, y=0.95, yend=0.96) +
  annotate("segment", x=0.192, xend=0.192, y=0.95, yend=0.96) +
  annotate("segment", x=0.0944, xend=0.1924, y=0.96, yend=0.96) +
  # Add text information regarding forward cells
  annotate("segment", x=0.225, xend=0.225, y=0.95, yend=0.96) +
  annotate("segment", x=0.295, xend=0.295, y=0.95, yend=0.96) +
  annotate("segment", x=0.2244, xend=0.2944, y=0.96, yend=0.96) +
  # Hide unknown graphical bug
  annotate("rect", xmin=0.438, xmax=0.458, ymin=0.46, ymax=0.5, fill="white")

# Save plots
ggsave("../Plots/team_composition_playoffs.png", width=7, height=5)
ggsave("../Plots/team_composition_playoffs.pdf", width=7, height=5)



# Create a function for creating team composition plots
team_comp <- function(top_n_sums, position){
  
  # Extract the plot data
  plot_data <- top_n_sums %>%
    pivot_longer(-c(TeamName, Points, GF, GA, Rank)) %>% 
    filter(grepl(position, name)) %>% 
    rename(`Team points` = Points, `Goals for` = GF, `Goals against` = GA)
  
  # Empty list to be filled
  plot_list <- list()
  
  # The variables to plot
  plot_var <- c("Team points", "Goals for", "Goals against")  
  
  # Loop over all variables
  for(plot_idx in 1:3){
    # Convert the string to symbol to use in ggplot
    xvar <- rlang::sym(plot_var[plot_idx])
    
    # Create and save plot
    plot_list[[plot_idx]] <- plot_data %>% 
      ggplot(aes(x=!!xvar, value)) + 
      geom_smooth(color="#EE7733", alpha=0.2, 
                  method="loess", formula = y ~ x) +
      geom_smooth(color="#EE7733", se=FALSE,
                  method="loess", formula = y ~ x) +
      geom_point(color="#0077BB", fill="black", size=1) + 
      facet_grid(~name) + 
      ylab("Number of team players") +
      theme_custom()
  }
  
  # Add names to the list
  names(plot_list) <- c("Points", "GF", "GA")
  
  return(plot_list)
}

# Create plots for defenders and forwards
team_comp_d <- team_comp(top_n_sums, "D\\d")
team_comp_f <- team_comp(top_n_sums, "F\\d")

# Visualize the team composition of defensemen
(team_comp_d$Points + scale_x_reverse(limits=c(125, 50), breaks=seq(125, 50, -25)) + 
    team_comp_d$GF + scale_x_reverse(limits=c(340, 200), breaks=seq(340, 200, -35)) + 
    team_comp_d$GA + scale_x_continuous(limits=c(200, 320), breaks=seq(200, 320, 30)) +
    plot_layout(ncol=1, byrow=TRUE))

# Save figure
ggsave("../Plots/team_composition_defenders.png", height=6, width=8)

# Visualize the team composition of forwards
(team_comp_f$Points + scale_x_reverse(limits=c(125, 50), breaks=seq(125, 50, -25)) + 
    team_comp_f$GF + scale_x_reverse(limits=c(340, 200), breaks=seq(340, 200, -35)) +
    team_comp_f$GA + scale_x_continuous(limits=c(200, 320), breaks=seq(200, 320, 30)) +
    plot_layout(ncol=1, byrow=TRUE))

# Save figure
ggsave("../Plots/team_composition_forwards.png", height=6, width=8)

