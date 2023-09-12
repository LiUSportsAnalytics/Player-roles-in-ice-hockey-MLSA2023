library(readr)
library(dplyr)
library(ggplot2)
library(plotly)
library(tidyr)
library(tibble)
library(stringr)
library(patchwork)

# Specify the working directory
if(length(str_split(getwd(), "/Scripts")[[1]]) != 2){
  setwd("Scripts")
}

# Load all functions
source("readData.R")
source("PCAfunctions.R")
source("crispClusteringFunctions.R")


##### MAIN ####

# Read and combine all data that is to be used for the analysis
pca_data <- load_data()

# Get the players from a specific position group
defenders <- select_position(pca_data, defenders=TRUE)
forwards  <- select_position(pca_data, defenders=FALSE)

# Preprocess the data for each position group
processed_defenders <- preprocess_data(defenders)
processed_forwards  <- preprocess_data(forwards)

# Fit the PCA on each position group
# Also perform a visual examination of variance explained by each principal component
pca_defenders <- fit_pca(processed_defenders)
pca_forwards  <- fit_pca(processed_forwards)

# Perform Hopkin's test of cluster tendency on the original data
factoextra::get_clust_tendency(processed_defenders, n = nrow(processed_defenders)-1, 
                               graph = FALSE)$hopkins_stat
factoextra::get_clust_tendency(processed_forwards, n = nrow(processed_forwards)-1,
                               graph = FALSE)$hopkins_stat

# Perform parallel analysis to select the number of components
pa_defender <- choose_nr_of_components(processed_defenders)
pa_forward <- choose_nr_of_components(processed_forwards)

# Perform Hopkin's test of cluster tendency on the PCA rotated data
factoextra::get_clust_tendency(pca_defenders$x[, 1:pa_defender$Retained], 
                               n = nrow(pca_defenders$x[, 1:pa_defender$Retained])-1, 
                               graph = FALSE)$hopkins_stat
factoextra::get_clust_tendency(pca_forwards$x[, 1:pa_forward$Retained], 
                               n = nrow(pca_forwards$x[, 1:pa_forward$Retained])-1, 
                               graph = FALSE)$hopkins_stat

# Choose the number of clusters with the guiding by some cluster quality metrics
choose_nr_of_clusters(pca_defenders, nr_components=pa_defender$Retained)
choose_nr_of_clusters(pca_forwards , nr_components=pa_forward$Retained)

# Perform the clustering for each position group
processed_data_with_cluster_defenders <- fit_cluster(processed_defenders,
                                                     defenders, pca_defenders,
                                                     nr_components=pa_defender$Retained,
                                                     nr_clusters=3)
processed_data_with_cluster_forwards  <- fit_cluster(processed_forwards,
                                                     forwards, pca_forwards,
                                                     nr_components=pa_forward$Retained,
                                                     nr_clusters=5)

# Evaluate the quality of the clusters (visually and numerically)
cluster_evaluation_defenders <- evaluate_cluster(processed_data_with_cluster_defenders,
                                                 pca_defenders)
cluster_evaluation_forwards  <- evaluate_cluster(processed_data_with_cluster_forwards,
                                                 pca_forwards)

# Find the most common archetypes within each cluster
archetypes_in_clusters_defenders <- cluster_evaluation_defenders$archetypes
archetypes_in_clusters_forwards  <- cluster_evaluation_forwards$archetypes

# Plot the archetypes within each cluster
plot_archetypes_in_cluster(archetypes_in_clusters_defenders)
plot_archetypes_in_cluster(archetypes_in_clusters_forwards)

# Investigate what each cluster excels/is bad at
cluster_attributes_defenders <- cluster_evaluation_defenders$cluster_attributes
cluster_attributes_forwards <- cluster_evaluation_forwards$cluster_attributes

# Player who represent the medoid in each cluster
processed_data_with_cluster_defenders %>%
  filter(Medoid == 1) %>%
  dplyr::select(fullName, Cluster) %>%
  arrange(Cluster) %>%
  print()

processed_data_with_cluster_forwards %>%
  filter(Medoid == 1) %>%
  dplyr::select(fullName, Cluster) %>%
  arrange(Cluster) %>%
  print()

# Count the sizes per cluster
table(processed_data_with_cluster_defenders$Cluster)
table(processed_data_with_cluster_forwards$Cluster)

# Summarize cluster and player data
defenders_clusters <- processed_data_with_cluster_defenders %>%
  dplyr::select(PlayerId, fullName, Type, Points, TOI, Cluster)

forwards_clusters <- processed_data_with_cluster_forwards %>%
  dplyr::select(PlayerId, fullName, Type, Points, TOI, Cluster)

# Print cluster means
for(i in 2:ncol(cluster_attributes_defenders)){
  cat(" \\\\ \n")
  cat("&", colnames(cluster_attributes_defenders[, i]), "& ")
  cat(gsub("c|\\(|\\)", "",
           gsub(",", " &", paste(round(cluster_attributes_defenders[, i], 3), collapse=""))))
}

for(i in 2:ncol(cluster_attributes_forwards)){
  cat(" \\\\ \n")
  cat("&", colnames(cluster_attributes_forwards[, i]), "& ")
  cat(gsub("c|\\(|\\)", "",
           gsub(",", " &", paste(round(cluster_attributes_forwards[, i], 3), collapse=""))))
}

# Get the cap information
defenders_cap <- nearest_neigbors_cap(pca_defenders, defenders_clusters,
                                      n_components = pa_defender$Retained, p = 2,
                                      # dist_boundary = 6,
                                      neighbors = 10,
                                      log_cap = TRUE, same_cluster = FALSE)
forwards_cap <- nearest_neigbors_cap(pca_forwards, forwards_clusters,
                                     n_components = pa_forward$Retained, p = 2,
                                     #dist_boundary = 6,
                                     neighbors = 10,
                                     log_cap = TRUE, same_cluster = FALSE)

# Compute the average cap difference for both perspectives
avg_cap_diff_defenders <- avg_cap_diff(defenders_cap)
avg_cap_diff_forwards <- avg_cap_diff(forwards_cap)

# Compute cost per point etc.
cost_per_stat_defenders <- cost_per_stat(defenders_cap, defenders)
cost_per_stat_forwards <- cost_per_stat(forwards_cap, forwards)

# Find overpaid/bargains (defensemen)
avg_cap_diff_defenders %>% head(10) %>% dplyr::select(PlayerName, AvgDiffDist, n)
avg_cap_diff_defenders %>% tail(10) %>% dplyr::select(PlayerName, AvgDiffDist, n) %>% purrr::map_df(rev)

# Find overpaid/bargains (forwards)
avg_cap_diff_forwards %>% head(10) %>% dplyr::select(PlayerName, AvgDiffDist, n)
avg_cap_diff_forwards %>% tail(10) %>% dplyr::select(PlayerName, AvgDiffDist, n) %>% purrr::map_df(rev)

# Find the worst contracts per team
worst_defender_contracts <- avg_cap_diff_defenders %>%
  group_by(PlayerTeam) %>%
  slice_max(n=3, order_by=AvgDiffDist)

worst_forward_contracts <- avg_cap_diff_forwards %>%
  group_by(PlayerTeam) %>%
  slice_max(n=3, order_by=AvgDiffDist)

# Team-based overpaying or underpaying
team_defender_mean <- avg_cap_diff_defenders %>%
  group_by(PlayerTeam) %>%
  summarise(TeamMean = mean(AvgDiffDist)) %>%
  arrange(-TeamMean)

team_forward_mean <- avg_cap_diff_forwards %>%
  group_by(PlayerTeam) %>%
  summarise(TeamMean = mean(AvgDiffDist)) %>%
  arrange(-TeamMean)

