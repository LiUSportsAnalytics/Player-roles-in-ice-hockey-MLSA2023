library(dplyr)
library(ggplot2)
library(plotly)
library(tidyr)
library(tibble)
library(stringr)
library(patchwork)
library(ggrepel)
source("utils.R")


#' Choose the number of fuzzy clusters
#'
#' Visualize the clustering tightness by 
#' silhouette score.
#' 
#' @param pca pca object for the position-specific data
#' @param nr_components the number of components to retain
#' @param m fuzziness parameter
#'
#' @return plots
#' @export
#'
choose_nr_of_clusters_fuzzy <- function(pca, nr_components, m){
  # Capture the name of the input data
  input_data <- deparse(substitute(pca))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Specify the minimum and maximum number of clusters to consider
  min_nc <- 2
  max_nc <- 8
  
  # Create an array to store results
  res <- array(0, c(max_nc-min_nc + 1, 3))
  
  # Save the values of k in the first column
  res[, 1] <- min_nc:max_nc
  
  # Loop over all candidates k
  for(k in min_nc:max_nc) {
    print(k)
    # Perform the clustering for the value of k on the PCA data
    clustering <- fclust::FKM(pca$x[, 1:nr_components], k=k, m=m, RS=10, seed=0, alpha=1, maxit=1e+05)
    
    # Save the results
    res[k-2+1, 2] <- fclust::MPC(clustering$U)
    res[k-2+1, 3] <- fclust::SIL.F(clustering$Xca, clustering$U, alpha=1)
  }
  # Get the summarized data
  index_data <- res %>% 
    as.data.frame() %>% 
    as_tibble() %>% 
    rename(k=1, MPC=2, SIL=3) 
  
  # Print summarized data
  print(index_data)
  
  # Visualize the validity indices
  res %>% 
    as.data.frame() %>% 
    as_tibble() %>% 
    rename(k=1, MPC=2, SIL.F=3) %>% 
    pivot_longer(-k, names_to="index") %>% 
    mutate(index=factor(index, levels=c("MPC", "SIL.F"), 
                          labels=c("Modified partition coefficient",
                                   "Fuzzy silhouette index"))) %>% 
    ggplot(aes(k, value, color=index)) + 
    geom_line(linewidth=0.9) + geom_point(size =1.2) +  
    xlab("Number of clusters") + ylab("Index value") + 
    ggtitle(paste("Optimal number of fuzzy clusters for", position_name)) + 
    scale_color_manual("Validity index", values=c('#EE7733', '#0077BB')) +
    scale_x_continuous(expand=c(0.01, 0)) +
    scale_y_continuous(expand=c(0, 0), limits=c(0, 0.5), breaks=seq(0, 0.5, 0.1)) +
    theme_custom() +
    theme(legend.direction="vertical", legend.position=c(0.8, 0.9)) -> index
  
  # Print the plot
  print(index)
  
  # Save the plot
  ggsave(paste("../Plots/fuzzy_index", position_name, ".png"), height=6, width=8)
  ggsave(paste("../Plots/fuzzy_index", position_name, ".pdf"), height=6, width=8)
}


#' Fit the fuzzy cluster
#'
#' Perform clustering using fuzzy c-means on the data.
#'
#' @param position_data data frame of statistics for a position group
#' @param pca pca object for the position-specific data
#' @param nr_components the number of components to retain
#' @param nr_clusters the number of clusters to form
#' @param m fuzziness parameter
#'
#' @return processed_data with clusters and medoid column added
#' @export
#'
fit_fuzzy_cluster <- function(position_data, pca, nr_components, nr_clusters, m){
  
  # Perform clustering on a set amount of principal components and clusters
  clustering <- fclust::FKM(pca$x[, 1:nr_components], k=nr_clusters, m=m, RS=10, seed=0, alpha=1)
  
  # Copy to avoid changing in-place
  processed_data_with_cluster <- position_data
  
  # Save the clustering in the original data frame
  processed_data_with_cluster["Cluster"] <- clustering$clus[, 1]
  
  # Save membership degree matrix
  processed_data_with_cluster %>% 
    bind_cols(clustering$U) %>% 
    setNames(gsub("Clus ", "PrCluster", names(.))) -> processed_data_with_cluster

  return(list(processed_data_with_cluster=processed_data_with_cluster,
              centroids=clustering$H))
}


#' Evaluate the clustering
#'
#' Evaluate the quality of the clustering visually (interactive) and via 
#' cluster means and counting archetypes within each cluster. 
#'
#' @param processed_data_with_cluster processed_data with clusters
#' @param pca pca object for the position-specific data
#'
#' @return list of count of archetypes within each cluster and cluster means
#' @export
#'
evaluate_fuzzy_cluster <- function(processed_data_with_cluster, pca){
  # Capture the name of the input data
  input_data <- deparse(substitute(pca))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Specify a colorblind friendly palette
  cbPalette <- c('#EE7733', '#0077BB', '#009988', '#EE3377', '#CC3311', '#33BBEE', '#BBBBBB')
  
  # Visualize the clustering for the first principal components
  data.frame(PC1=pca$x[, 1], PC2=pca$x[, 2], 
             cluster=processed_data_with_cluster["Cluster"], 
             name=processed_data_with_cluster["fullName"], 
             position=processed_data_with_cluster["centralRegistryPosition"]) %>% 
    bind_cols(processed_data_with_cluster %>% dplyr::select(starts_with("PrCluster"))) %>% 
    mutate(cluster=factor(Cluster)) %>%
    rename(name=fullName, position=centralRegistryPosition) %>%
    # Format probabilities
    mutate(across(starts_with("PrCluster"), ~format(round(.x, 3), nsmall=3))) %>% 
    ggplot() +
    suppressWarnings(geom_point(aes(PC1, PC2, color=cluster, name=name, position=position,
                                    text=paste("Name:", name, "\nPosition:", position, 
                                               "\nProb Cluster 1:", PrCluster1,
                                               "\nProb Cluster 2:", PrCluster2,
                                               "\nProb Cluster 3:", PrCluster3,
                                               {if ("PrCluster4" %in% colnames(processed_data_with_cluster)) 
                                                 paste("\nProb Cluster 4:", PrCluster4)}
                                               )), 
                                shape=16)) + 
    suppressWarnings(geom_text_repel(aes(PC1, PC2, label=name), max.overlaps = 5)) +
    scale_color_manual(values=cbPalette) +
    theme_custom() + theme(legend.key.height = unit(1.2, 'cm')) -> L2
  
  # Save the static plot
  ggsave(paste("../Plots/fuzzy_clusterPCA", position_name, ".png"), L2, height=12, width=16)
  ggsave(paste("../Plots/fuzzy_clusterPCA", position_name, ".pdf"), L2, height=12, width=16)
  
  # Print the plot
  print(suppressWarnings(ggplotly(L2, tooltip="text") %>% layout(hoverlabel = list(align = "left"))))
  
  # Compute the number of archetypes per cluster
  nhl_archetypes <- processed_data_with_cluster %>% group_by(Cluster, Type) %>% count() %>% arrange(Cluster, -n)
  
  # Compute the total number of players from each archetype
  archetype_count <- processed_data_with_cluster %>% group_by(Type) %>% count() %>% rename(ArchetypeN=n)
  
  # Combine cluster counts and total counts
  nhl_archetypes <- inner_join(nhl_archetypes, archetype_count, by="Type")
  
  # Compute the proportion of players from a specific archetype belonging to a cluster
  nhl_archetypes <- nhl_archetypes %>% 
    mutate(PropType=n/ArchetypeN)
  
  # Compute numerical summarized values for all clusters
  cluster_attributes <- processed_data_with_cluster %>% 
    dplyr::select(names(pca$center), Cluster) %>% 
    group_by(Cluster) %>% 
    summarise(across(where(is.numeric), list(mean = mean, sd = sd)))
  
  return(list(archetypes=nhl_archetypes, cluster_attributes=cluster_attributes))
}


#' Visualize the archetypes in each cluster
#'
#' Create a visualization to investigate the distribution of archetypes within
#' each cluster.
#'
#' @param archetypes_in_cluster data frame with counts of all archetypes in a cluster
#'
#' @return plot
#' @export
#'
plot_archetypes_in_fuzzy_cluster <- function(archetypes_in_cluster){
  # Capture the name of the input data
  input_data <- deparse(substitute(archetypes_in_cluster))
  
  # The position name
  position_name <- str_split(input_data, "_")[[1]][-c(1,2,3,4)]
  
  # If a player is in the wrong position group
  wrong_position <- ifelse(position_name == "defenders", "Forward", "Defenseman")
  
  # Specify a colorblind friendly palette
  cbPalette <- c('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB')
  
  if(position_name == "defenders"){
    cbPalette <- c(cbPalette[1:3], cbPalette[length(cbPalette)])
  }
  
  # Plot the distribution
  archetypes_in_cluster %>% 
    filter(!grepl(wrong_position, Type)) %>%
    mutate(Type=ifelse(is.na(Type), "Unknown", Type)) %>% 
    group_by(Cluster) %>% 
    mutate(Position=rank(-PropType)) %>% 
    ggplot(aes(Cluster, PropType, group=Position, fill=Type)) + 
    geom_bar(stat="identity", position="dodge") + 
    ggtitle(paste("NHL 22 archetypes belonging to each cluster for", position_name)) +
    xlab("Cluster") + ylab("Proportion")+ 
    scale_x_continuous(expand=c(0.01, 0), breaks=seq(1, 4, 1)) +
    scale_y_continuous(expand=c(0, 0)) +
    theme_custom() + theme(legend.position = "top", legend.direction = "horizontal") +
    scale_fill_manual("", values=cbPalette) -> p
  
  print(p)
  # Save the plot
  ggsave(paste("../Plots/fuzzy_archetypes", position_name, ".png"), height=6, width=8)
  ggsave(paste("../Plots/fuzzy_archetypes", position_name, ".pdf"), height=6, width=8)
  
}


#' Find nearest neighbors and cap hit
#'
#' For each player, find the n nearest neighbors and their cap hit information. 
#'
#' @param position_data data frame of statistics for a position group
#' @param neighbors the number of neighbors to consider
#' @param p the power of the Minkowski distance
#' @param dist_boundary the maximum distance from the player allowed
#' @param log_cap if the cap hit should be log-transformed
#' @param same_cluster if only players from the same cluster should be considered
#' 
#' @return data frame of player in position with each player's n nearest neighbors
#' @export
#'
nearest_fuzzy_neigbors_cap <- function(position_data,
                                       neighbors=NA, p=2, dist_boundary=NA,
                                       log_cap=TRUE, same_cluster=TRUE){
  if((is.na(neighbors) & is.na(dist_boundary)) |
     (!is.na(neighbors) & !is.na(dist_boundary))){
    stop( "Please specify only one of neighbors or dist_boundary.")
  }
  
  # Compute the distance between each pair of players
  d <- as.matrix(philentropy::distance(position_data %>% dplyr::select(starts_with("PrCluster")), 
                                       method="minkowski", p=p)) %>% as_tibble() %>% 
    bind_cols(position_data %>% dplyr::select(PlayerId, Cluster))
  
  # Specify the column names as player ids
  colnames(d)[1:(length(d)-2)] <- d$PlayerId
  
  # Find the n closest players with respect to distance
  d %>% 
    pivot_longer(-c(PlayerId, Cluster), names_to="OtherPlayerId", values_to="Dist") %>% 
    mutate(OtherPlayerId=as.integer(OtherPlayerId)) %>% 
    filter(PlayerId != OtherPlayerId) %>% 
    inner_join(d %>% 
                 pivot_longer(-c(PlayerId, Cluster), names_to="name", values_to="value") %>% 
                 dplyr::select(-c(name, value)) %>% 
                 distinct() %>% 
                 rename(OtherPlayerId=PlayerId, OtherPlayerCluster=Cluster),
               by="OtherPlayerId") %>% 
    {if (same_cluster) filter(., Cluster == OtherPlayerCluster) else filter(.)} %>% 
    group_by(PlayerId) %>% 
    {if (!is.na(neighbors) & is.na(dist_boundary)) slice_min(., n=neighbors, order_by=Dist) 
      else filter(., Dist < dist_boundary)} %>% 
    mutate(DistRank=rank(Dist)) -> nearest_neighbors
  
  # Read cap data
  cap <- read.csv("../Data/cap20212022.csv") %>% 
    add_row(PLAYER = "Sasha Chmelevski", TEAM = "SJS", PlayerId = 8480053,
            CAP.HIT = 778333, TYPE = "Entry-Level", EXP..YEAR = 2022) %>% 
    add_row(PLAYER = "Janis Moser", TEAM = "ARI", PlayerId = 8482655,
            CAP.HIT = 886667, TYPE = "Entry-Level", EXP..YEAR = 2024)
  
  # Add cap information for all players
  nearest_neighbors %>% 
    ungroup() %>% 
    inner_join(cap %>% dplyr::select(PLAYER, TEAM, PlayerId, CAP.HIT, TYPE, EXP..YEAR) %>% 
                 rowwise() %>% 
                 mutate(CAP.HIT.VAL=ifelse(log_cap, log(CAP.HIT / 1E5), CAP.HIT / 1E5)), by="PlayerId") %>% 
    rename(PlayerCapHit=CAP.HIT.VAL, PlayerName=PLAYER, PlayerTeam=TEAM, 
           ContractType=TYPE, ExpiryYear=EXP..YEAR) %>% 
    mutate(OtherPlayerId=as.integer(OtherPlayerId)) %>% 
    inner_join(cap %>% dplyr::select(PLAYER, TEAM, PlayerId, CAP.HIT, TYPE) %>%
                 rowwise() %>% 
                 mutate(CAP.HIT.VAL=ifelse(log_cap, log(CAP.HIT / 1E5), CAP.HIT / 1E5)), 
               by=c("OtherPlayerId"="PlayerId")) %>% 
    rename(OtherPlayerCapHit=CAP.HIT.VAL, OtherPlayerName=PLAYER, OtherPlayerTeam=TEAM) %>% 
    dplyr::select(PlayerId, PlayerName, PlayerTeam, PlayerCapHit, ContractType, ExpiryYear,
                  OtherPlayerId, OtherPlayerName, OtherPlayerTeam, OtherPlayerCapHit,
                  Dist, DistRank) %>% 
    rowwise() %>% 
    mutate(CapDiff=PlayerCapHit - OtherPlayerCapHit) -> nearest_neighbors_with_cap
  
  return(nearest_neighbors_with_cap)
}

#' Compute the average cap difference among neighbors
#'
#' For each player, compute the average cap difference among their neighbors to
#' determine if a player is paid more (overpaid) or less (bargain) than the neighbors.
#' A high "bargain" value indicate the player is paid far less than their neighbors.
#' A high "overpaid" value indicate the player is paid far more than their
#'
#' @param position_cap data frame of player in position with each player's n nearest neighbors
#'
#' @return data frame with average cap difference per player
#' @export
#'
avg_fuzzy_cap_diff <- function(position_cap){
  # Compute the average difference and distance for each player
  position_cap %>% 
    group_by(PlayerId, PlayerName, PlayerTeam, ContractType, ExpiryYear, PlayerCapHit) %>% 
    summarise(AvgDiff=mean(CapDiff), AvgDist=mean(Dist), 
              AvgDiffDist=sum(CapDiff) / sum(PlayerCapHit),   
              n=n(), .groups="keep") %>% 
    arrange(-AvgDiffDist) %>% 
    ungroup()
}



#' Select the fuzzifier parameter m
#'
#' @param D the dimension of data
#' @param N the number of observations in the data
#'
#' @return a suggested value for m
#' @export
#'
fuzzifier <- function(D, N){
  1 + (1418/N + 22.05) * D^(-2) + (12.33/N + 0.243) * D^(-0.0406 * log(N) - 0.1134)
}


#' Created CDF for each cluster probability column
#'
#' @param fuzzy_clusters data frame containing cluster membership probabilities
#'
#' @return data frame with transformed probabilities, based on the corresponding ECDF
#' @export
#'
create_cdf <- function(fuzzy_clusters){
  # Select all cluster probability columns
  fuzzy_clusters %>% 
    dplyr::select(starts_with("PrCluster")) %>% 
    colnames() -> variables
  
  # Create empty lists
  e_cdf_list <- list()
  plot_data_list <- list()
  
  # Loop over all cluster variables
  for(variable in variables){
    # Create an empirical CDF
    e_cdf <- ecdf(fuzzy_clusters %>% dplyr::select(variable) %>% pull())
    # Save the ECDF evaluated at the data
    e_cdf_list[[variable]] <- e_cdf(fuzzy_clusters %>% dplyr::select(variable) %>% pull())
    # Save an evaluated ECDF for plotting
    plot_data_list[[variable]] <- e_cdf(seq(0, 1, 0.001))
  }
  
  # Plot the ECDF
  bind_cols(plot_data_list) %>% 
    mutate(x=seq(0, 1, 0.001)) %>% 
    pivot_longer(-x) %>% 
    ggplot(aes(x, value, color=name)) + geom_step() + theme_classic() +
    scale_color_brewer(palette = "Dark2")
  
  # Create a data frame with the CDF for all variables
  cdf <- data.frame(PlayerId = fuzzy_clusters$PlayerId,
                    Cluster = fuzzy_clusters$Cluster) %>% 
    bind_cols(e_cdf_list)
  
  return(cdf)
}


#' Visualize the cluster membership degrees for different cap hit groups
#'
#' @param fuzzy_position_cluster_cap data frame with player name, cluster probabilities 
#' and cap hit
#' @param labels bool, if the player names should be added
#'
#' @return plot
#' @export
#'
plot_cluster_probabilities <- function(fuzzy_position_cluster_cap, labels){
  
  # Specify a colorblind friendly palette
  cbPalette <- c('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB')
  
  # Capture the name of the input data
  input_data <- deparse(substitute(fuzzy_position_cluster_cap))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Define a position of text and jittered points
  pos <- position_jitter(height=0, width = 0.2, seed = 2)
  
  # Plot the cluster membership probability as a function of cap hit group 
  fuzzy_position_cluster_cap %>% 
    dplyr::select(fullName, starts_with("PrCluster"), PlayerCapHitOriginal) %>% 
    pivot_longer(-c(fullName, PlayerCapHitOriginal), names_to = "Cluster",
                 values_to = "ClusterProbability") %>% 
    mutate(Cluster = str_replace(Cluster, "Pr", ""),
           Cluster = str_replace(Cluster, "Cluster", ifelse(position_name == "forwards", "F", "D"))) %>% 
    drop_na() %>% 
    mutate(CapHitGroup = case_when(
      between(PlayerCapHitOriginal, 0, 1E6-1) ~ "<1",
      between(PlayerCapHitOriginal, 1E6, 2E6) ~ "1-2",
      between(PlayerCapHitOriginal, 2E6+1, 4E6) ~ "2-4",
      between(PlayerCapHitOriginal, 4E6+1, 6E6) ~ "4-6",
      between(PlayerCapHitOriginal, 6E6+1, 8E6) ~ "6-8",
      between(PlayerCapHitOriginal, 8E6+1, 15E6) ~ ">8"), 
      CapHitGroup = factor(CapHitGroup, levels=c("<1", "1-2", "2-4", "4-6",
                                                 "6-8", ">8"))) %>% 
    ggplot(aes(CapHitGroup, ClusterProbability, color=Cluster)) + 
    geom_boxplot(outlier.shape = NA, width=0.5,
                 color="gray", alpha=1) + 
    geom_jitter(position=pos, alpha=0.5, size=0.8) + 
    {if(labels) geom_text_repel(aes(label=fullName), position=pos, max.overlaps=2, size=1.5)} +
    facet_wrap(~Cluster) + 
    xlab("Cap hit (Million USD)") + ylab("Cluster membership degree") +
    scale_y_continuous(expand=c(0.01, 0), limits=c(0, 1)) +
    scale_color_manual(name="", values=cbPalette) +
    theme_custom() + theme(legend.position = "none", 
                           strip.text = element_text(size=18), 
                           axis.text = element_text(size=14),
                           axis.title = element_text(size=16))
  
  # Specify name for plot
  labels_title <- ifelse(labels, "_labels", "")
  
  # Save the plot
  ggsave(paste0("../Plots/cluster_probability_distribution_", position_name, labels_title, ".png"), 
         height=6, width=8, dpi=300)
  ggsave(paste0("../Plots/cluster_probability_distribution_", position_name, labels_title, ".pdf"), 
         height=6, width=8, dpi=300)
}

#' Visualize the cluster membership degrees densities
#'
#' @param fuzzy_position_cluster_cap data frame with player name, cluster probabilities 
#' and cap hit
#'
#' @return plot
#' @export
#'
plot_cluster_densities <- function(fuzzy_position_cluster_cap){
  
  # Capture the name of the input data
  input_data <- deparse(substitute(fuzzy_position_cluster_cap))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Define a colorblind friendly palette
  cbPalette <- c('#EE7733', '#0077BB', '#009988', '#EE3377', '#CC3311', '#33BBEE', '#BBBBBB')
  
  # Visualize density of cluster membership per cluster
  fuzzy_position_cluster_cap %>% 
    dplyr::select(fullName, starts_with("PrCluster")) %>% 
    pivot_longer(-c(fullName), names_to = "Cluster",
                 values_to = "ClusterProbability") %>% 
    mutate(Cluster = str_replace(Cluster, "PrCluster", 
                                 ifelse(position_name == "forwards", "F", "D"))) %>% 
    ggplot(aes(ClusterProbability, fill=Cluster)) +
    xlab("Cluster membership degree") + ylab("Density") +
    geom_density(kernel="gaussian", alpha=0.7) +
    scale_fill_manual("", values=cbPalette) + 
    scale_y_continuous(expand=c(0, 0)) +
    scale_x_continuous(expand=c(0, 0), limits=c(0, 1), labels = c(0, 0.25, 0.5, 0.75, 1)) +
    theme_custom() + theme(legend.position = c(0.9, 0.8),
                           legend.key.width = unit(1, "cm"),
                           legend.direction = "vertical",
                           legend.text = element_text(size=20),
                           axis.text = element_text(size=20),
                           axis.title = element_text(size=22))

  # Save the plot
  ggsave(paste0("../Plots/cluster_density_", position_name, ".png"), 
         height=4, width=6, dpi=300)
  ggsave(paste0("../Plots/cluster_density_", position_name, ".pdf"), 
         height=4, width=6, dpi=300)
}

#' Plot cluster centroids 
#'
#' @param fuzzy_centroids matrix with cluster centroids
#'
#' @return plot
#' @export
#'
plot_cluster_centroids <- function(fuzzy_centroid){
  # Capture the name of the input data
  input_data <- deparse(substitute(fuzzy_centroid))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Define a colorblind friendly palette
  cbPalette <- c('#EE7733', '#0077BB', '#009988', '#EE3377', '#CC3311', '#33BBEE', '#BBBBBB', '#332288')
  
  # Create the plot
  fuzzy_centroid %>% 
    as.data.frame() %>% 
    rownames_to_column("Cluster") %>% 
    pivot_longer(-Cluster, names_to="PC") %>% 
    mutate(Cluster = gsub("Clus", "Cluster", Cluster), 
           ClusterNr = gsub("Cluster", "", Cluster)) %>% 
    ggplot(aes(PC, value, fill=Cluster, group=Cluster)) + facet_wrap(~PC, scales="free_x") +
    geom_hline(aes(yintercept=0), color="gray55", linetype="dashed") + 
    geom_bar(stat="identity", position = position_dodge2(width=0.9), color="black") + 
    xlab("") + ylab("Centroid value") + theme_custom() +
    scale_fill_manual(values=cbPalette) + theme(legend.position = "top",
                                                legend.direction = "horizontal",
                                                axis.text.x = element_blank(),
                                                axis.ticks.x = element_blank())
  
  # Save the plot
  ggsave(paste0("../Plots/cluster_centroid_", position_name, ".png"), 
         height=6, width=8, dpi=300)
  ggsave(paste0("../Plots/cluster_centroid_", position_name, ".pdf"), 
         height=6, width=8, dpi=300)
  
}



#' Compute approximate original centroids
#'
#' @param fuzzy_centroid the cluster centroids on in terms of PCA
#' @param pca pca object for a given position
#' @param processed_position pre-processed position data
#'
#' @return print
#' @export
#'
compute_original_centroids <- function(fuzzy_centroid, pca, processed_position, nr_components){
  # Compute approximate centroid values of the original variables
  cluster_centroids_original <- fuzzy_centroid %*% t(pca$rotation[, 1:nr_components]) 
  
  # The number of rows
  n_rows <- nrow(fuzzy_centroid)
  
  # Create a matrix of the center and scale for standardization
  scaled_mu <- matrix(data=rep(attr(processed_position, "scaled:center"), n_rows), 
                      nrow=n_rows, ncol=ncol(cluster_centroids_original), byrow=TRUE,
                      dimnames=list(rownames(cluster_centroids_original), colnames(cluster_centroids_original)))
  scaled_sd <- matrix(data=rep(attr(processed_position, "scaled:scale" ), n_rows),
                      nrow=n_rows, ncol=ncol(cluster_centroids_original), byrow=TRUE,
                      dimnames=list(rownames(cluster_centroids_original), colnames(cluster_centroids_original)))
  
  # Invert the standardization
  cluster_centroids_original_unscaled <- cluster_centroids_original * scaled_sd + scaled_mu
  
  # Compute the (approximate) centroid values per 60 minutes
  cluster_centroids_original_unscaled %>% 
    as.data.frame() %>% 
    rownames_to_column("Cluster") %>% 
    # If the values should be transformed
    mutate(across(!c(starts_with("X"), starts_with("Y"), contains("%"), "Cluster", "height", "weight"), 
                  function(x) x)) %>% 
    pivot_longer(-Cluster) %>% 
    pivot_wider(names_from="Cluster", values_from="value") %>% 
    mutate(name = case_when(name == "xGDiff" ~ "xG difference",
                            name == "xG%Rel" ~ "xGF%Rel",
                            name == "xG%" ~ "xGF%",
                            name == "xGOn" ~ "xGF",
                            name == "NumStickPenaltyOn" ~ "Stick penalties on",
                            name == "NumStickDrewBy" ~ "Stick penalties drawn",
                            name == "NumRestrainingPenaltyOn" ~ "Restraining penalties on",
                            name == "NumRestrainingDrewBy" ~ "Restraining penalties drawn",
                            name == "NumPhysicalPenaltyOn" ~ "Physical penalties on",
                            name == "NumPhysicalDrewBy" ~ "Physical penalties drawn",
                            name == "weight" ~ "Weight (pounds)",
                            name == "height" ~ "Height (inches)",
                            name == "PlusMinus" ~ "$+/-$",
                            name == "YmedianTakeaway" ~ "Median Y Takeaway",
                            name == "YmedianShooter" ~ "Median Y Shooter",
                            name == "YmedianPenaltyOn" ~ "Median Y Penalty",
                            name == "YmedianHitter" ~ "Median Y Hitter",
                            name == "YmedianHittee" ~ "Median Y Hit taken",
                            name == "YmedianGiveaway" ~ "Median Y Giveaway",
                            name == "YmedianDrewBy" ~ "Median Y Penalty drawn",
                            name == "YmedianBlocker" ~ "Median Y Blocker",
                            name == "XmedianTakeaway" ~ "Median X Takeaway",
                            name == "XmedianShooter" ~ "Median X Shooter",
                            name == "XmedianPenaltyOn" ~ "Median X Penalty",
                            name == "XmedianHitter" ~ "Median X Hitter",
                            name == "XmedianHittee" ~ "Median X Hit taken",
                            name == "XmedianGiveaway" ~ "Median X Giveaway",
                            name == "XmedianDrewBy" ~ "Median X Penalty drawn",
                            name == "XmedianBlocker" ~ "Median X Blocker",
                            name == "PenaltyMinutes" ~ "Penalty minutes",
                            name == "PenaltyNet" ~ "Net penalties",
                            name == "HitNet" ~ "Net hits", 
                            TRUE ~ name),
           name = factor(name, levels=c("Goals", "Assists", "xG", "xG difference",
                                        "S%", "$+/-$", "xGF", "xGF%", "xGF%Rel", 
                                        "Giveaways", "Takeaways", "Blocks", "Hits", 
                                        "Net hits", "Fights", "Penalties", 
                                        "Net penalties", "Penalty minutes", 
                                        "Physical penalties drawn", "Physical penalties on", 
                                        "Restraining penalties drawn", "Restraining penalties on", 
                                        "Stick penalties drawn", "Stick penalties on",
                                        "OZS%", "PP%", "SH%", "OT%", 
                                        "Median X Blocker", "Median X Giveaway", 
                                        "Median X Hit taken", "Median X Hitter", 
                                        "Median X Penalty drawn", "Median X Penalty", 
                                        "Median X Shooter", "Median X Takeaway",
                                        "Median Y Blocker", "Median Y Giveaway", 
                                        "Median Y Hit taken", "Median Y Hitter", 
                                        "Median Y Penalty drawn", "Median Y Penalty", 
                                        "Median Y Shooter", "Median Y Takeaway",
                                        "Height (inches)", "Weight (pounds)"))) %>% 
    arrange(name) %>% 
    mutate(across(where(is.numeric), round, digits=3)) %>% 
    mutate(across(everything(), function(x) paste("&", x)))
}

