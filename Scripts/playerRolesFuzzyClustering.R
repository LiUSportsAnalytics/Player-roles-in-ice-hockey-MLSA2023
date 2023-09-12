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
source("fuzzyClusteringFunctions.R")
source("utils.R")

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

# Visualize proportion of variance explained in one figure
visualize_var_explained(pca_forwards, pca_defenders)

# Visualize PCA loadings
visualize_loadings(pca_defenders)
visualize_loadings(pca_forwards)

# Perform Hopkin's test of cluster tendency on the original data
factoextra::get_clust_tendency(processed_defenders, n = nrow(processed_defenders)-1, 
                               graph = FALSE)$hopkins_stat
factoextra::get_clust_tendency(processed_forwards, n = nrow(processed_forwards)-1,
                               graph = FALSE)$hopkins_stat

# Perform parallel analysis to select the number of components
pa_defender <- choose_nr_of_components(processed_defenders, n_iter=100000, centile=95)
pa_forward  <- choose_nr_of_components(processed_forwards,  n_iter=100000, centile=95)

# Perform Hopkin's test of cluster tendency on the PCA rotated data
factoextra::get_clust_tendency(pca_defenders$x[, 1:pa_defender$Retained], 
                               n = nrow(pca_defenders$x[, 1:pa_defender$Retained])-1, 
                               graph = FALSE)$hopkins_stat
factoextra::get_clust_tendency(pca_forwards$x[, 1:pa_forward$Retained], 
                               n = nrow(pca_forwards$x[, 1:pa_forward$Retained])-1, 
                               graph = FALSE)$hopkins_stat

# Choose the number of clusters
choose_nr_of_clusters_fuzzy(pca_defenders, nr_components=pa_defender$Retained,
                            m=fuzzifier(pa_defender$Retained, nrow(defenders)))

choose_nr_of_clusters_fuzzy(pca_forwards, nr_components=pa_forward$Retained,
                            m=fuzzifier(pa_forward$Retained, nrow(forwards)))

# Fit clustering
fuzzy_defenders_list <- fit_fuzzy_cluster(defenders, pca_defenders, 
                                          nr_components=pa_defender$Retained, 
                                          nr_clusters=3, 
                                          m=fuzzifier(pa_defender$Retained, nrow(defenders)))

fuzzy_forwards_list <- fit_fuzzy_cluster(forwards, pca_forwards, 
                                         nr_components=pa_forward$Retained, 
                                         nr_clusters=4, 
                                         m=fuzzifier(pa_forward$Retained, nrow(forwards)))

# Extract player data and cluster centroids
fuzzy_defenders <- fuzzy_defenders_list$processed_data_with_cluster
fuzzy_forwards <- fuzzy_forwards_list$processed_data_with_cluster

fuzzy_defenders_centroid <- fuzzy_defenders_list$centroids
fuzzy_forwards_centroid <- fuzzy_forwards_list$centroids

# Evaluate the quality of the clusters (visually and numerically)
fuzzy_clusters_defenders <- evaluate_fuzzy_cluster(fuzzy_defenders %>% 
                                                     mutate(across(starts_with("YMedian"), function(x) abs(x))),
                                                   pca_defenders)
fuzzy_clusters_forwards  <- evaluate_fuzzy_cluster(fuzzy_forwards %>% 
                                                     mutate(across(starts_with("YMedian"), function(x) abs(x))),
                                                   pca_forwards)

# Find the most common archetypes within each fuzzy cluster
archetypes_in_fuzzy_clusters_defenders <- fuzzy_clusters_defenders$archetypes
archetypes_in_fuzzy_clusters_forwards  <- fuzzy_clusters_forwards$archetypes

# Plot the archetypes within each fuzzy cluster
plot_archetypes_in_fuzzy_cluster(archetypes_in_fuzzy_clusters_defenders)
plot_archetypes_in_fuzzy_cluster(archetypes_in_fuzzy_clusters_forwards)

# Investigate what each fuzzy cluster excels/is bad at
fuzzy_cluster_attributes_defenders <- fuzzy_clusters_defenders$cluster_attributes
fuzzy_cluster_attributes_forwards <- fuzzy_clusters_forwards$cluster_attributes

# Summarize cluster and player data
fuzzy_defenders_clusters <- fuzzy_defenders %>% 
  dplyr::select(PlayerId, fullName, Type, Points, TOI, `5v5%`, `PP%`, `SH%`, `EV%`, `OT%`, 
                Cluster, starts_with("PrCluster"))

fuzzy_forwards_clusters <- fuzzy_forwards %>% 
  dplyr::select(PlayerId, fullName, Type, Points, TOI, `5v5%`, `PP%`, `SH%`, `EV%`, `OT%`, 
                Cluster, starts_with("PrCluster"))

# Get the cap information
fuzzy_defenders_cap <- nearest_fuzzy_neigbors_cap(fuzzy_defenders_clusters, p = 2, 
                                                  # dist_boundary = 0.2,
                                                  neighbors = 10,
                                                  log_cap = TRUE, same_cluster = FALSE)

fuzzy_forwards_cap <- nearest_fuzzy_neigbors_cap(fuzzy_forwards_clusters, p = 2, 
                                                 # dist_boundary = 0.2,
                                                 neighbors = 10,
                                                 log_cap = TRUE, same_cluster = FALSE) 

# Compute the average cap difference for both perspectives
avg_fuzzy_cap_diff_defenders <- avg_fuzzy_cap_diff(fuzzy_defenders_cap)
avg_fuzzy_cap_diff_forwards  <- avg_fuzzy_cap_diff(fuzzy_forwards_cap) 

# Find overpaid/bargains (defensemen)
avg_fuzzy_cap_diff_defenders %>% tail(15) %>% dplyr::select(PlayerName, AvgDiffDist, ExpiryYear, n) %>% purrr::map_df(rev)
avg_fuzzy_cap_diff_defenders %>% head(15) %>% dplyr::select(PlayerName, AvgDiffDist, ExpiryYear, n)

# Find overpaid/bargains (forwards)
avg_fuzzy_cap_diff_forwards %>% tail(15) %>% dplyr::select(PlayerName, AvgDiffDist, ExpiryYear, n) %>% purrr::map_df(rev)
avg_fuzzy_cap_diff_forwards %>% head(15) %>% dplyr::select(PlayerName, AvgDiffDist, ExpiryYear, n)

# Find the worst contracts per team
worst_defender_contracts <- avg_fuzzy_cap_diff_defenders %>% 
  group_by(PlayerTeam) %>% 
  slice_max(n=3, order_by=AvgDiffDist)

worst_forward_contracts <- avg_fuzzy_cap_diff_forwards %>% 
  group_by(PlayerTeam) %>% 
  slice_max(n=3, order_by=AvgDiffDist)

# Add cap information to cluster probabilities
fuzzy_defenders_cluster_caps <- fuzzy_defenders_clusters %>% 
  left_join(fuzzy_defenders_cap %>% dplyr::select(PlayerId, PlayerCapHit) %>% distinct(), by="PlayerId") %>% 
  mutate(PlayerCapHitOriginal = exp(PlayerCapHit)*1E5)

fuzzy_forwards_cluster_caps <- fuzzy_forwards_clusters %>% 
  left_join(fuzzy_forwards_cap %>% dplyr::select(PlayerId, PlayerCapHit) %>% distinct(), by="PlayerId") %>% 
  mutate(PlayerCapHitOriginal = exp(PlayerCapHit)*1E5)

# Plot cluster membership probability
plot_cluster_probabilities(fuzzy_defenders_cluster_caps, labels=TRUE)
plot_cluster_probabilities(fuzzy_forwards_cluster_caps, labels=TRUE)
plot_cluster_probabilities(fuzzy_defenders_cluster_caps, labels=FALSE)
plot_cluster_probabilities(fuzzy_forwards_cluster_caps, labels=FALSE)

# Plot cluster probability densities
plot_cluster_densities(fuzzy_defenders_cluster_caps)
plot_cluster_densities(fuzzy_forwards_cluster_caps)

# Compute approximate cluster centroids of original variables for each position
defender_centroids <- compute_original_centroids(fuzzy_defenders_centroid, pca_defenders, processed_defenders,
                                                nr_components = pa_defender$Retained) %>% 
  rename(D1 = 2, D2 = 3, D3 = 4) %>% 
  dplyr::select(-name)

forward_centroids <- compute_original_centroids(fuzzy_forwards_centroid, pca_forwards, processed_forwards, 
                                                 nr_components = pa_forward$Retained) %>% 
  rename(F1 = 2, F2 = 3, F3 = 4, F4 = 5)

# Print the centroids
bind_cols(forward_centroids, defender_centroids) %>%  
  mutate(across(ncol(.), function(x) paste(x, "\\") )) %>% 
  data.frame() %>% 
  print(n=50, row.names=FALSE, na.print="", right=FALSE, justify="center")

# Examine some players to see how cluster probabilities look
selected_d <- fuzzy_defenders_cluster_caps %>% 
  filter(fullName %in% c("Roman Josi", "Cale Makar", "Victor Hedman",
                         "Ivan Provorov", "Christopher Tanev", "Brian Dumoulin",
                         "Adam Larsson", "Rasmus Ristolainen", "Radko Gudas"
                         )) %>% 
  arrange(Cluster)

selected_f <- fuzzy_forwards_cluster_caps %>% 
  filter(fullName %in% c("Sidney Crosby", "Auston Matthews", "Connor McDavid",
                         "Anze Kopitar", "Jamie Benn", "Tyler Seguin",
                         "Tanner Jeannot", "Ryan Reaves", "Pat Maroon",
                         "Nick Bonino", "Colton Sissons", "Barclay Goodrow")) %>% 
  arrange(Cluster)

# Plot cluster centroids
plot_cluster_centroids(fuzzy_defenders_centroid)
plot_cluster_centroids(fuzzy_forwards_centroid)


# Get the top n players (w.r.t. TOI) from each team
top_n <- read_csv("../Data/top_n.csv", show_col_types = FALSE)

# Create a data frame containing all cluster information in one
fuzzy_cluster <- bind_rows(
  fuzzy_forwards_clusters %>% 
    dplyr::select(PlayerId, Cluster, starts_with("PrCluster")) %>% 
    rename("ForwardCluster" = "Cluster", 
           "F1" = "PrCluster1",
           "F2" = "PrCluster2",
           "F3" = "PrCluster3", 
           "F4" = "PrCluster4"),
  fuzzy_defenders_clusters %>% 
    dplyr::select(PlayerId, Cluster, starts_with("PrCluster")) %>% 
    rename("DefenderCluster" = "Cluster", 
           "D1" = "PrCluster1",
           "D2" = "PrCluster2",
           "D3" = "PrCluster3")
)

# Save as csv to save time
write_csv(fuzzy_cluster, "../Data/fuzzy_cluster.csv")
