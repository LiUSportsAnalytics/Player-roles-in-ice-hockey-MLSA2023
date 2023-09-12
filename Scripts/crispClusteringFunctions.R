library(dplyr)
library(ggplot2)
library(plotly)
library(tidyr)
library(tibble)
library(stringr)
library(patchwork)
library(ggrepel)


#' Choose the number of clusters
#'
#' Visualize the clustering tightness by rgw Calinski-Harabasz pseudo-F statistic and 
#' silhouette score.
#' 
#' @param pca pca object for the position-specific data
#' @param nr_components the number of components to retain
#'
#' @return plots
#' @export
#'
choose_nr_of_clusters <- function(pca, nr_components){
  # Capture the name of the input data
  input_data <- deparse(substitute(pca))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Specify the minimum and maximum number of clusters to consider
  min_nc <- 2
  max_nc <- 10
  
  # Create an array to store results
  res <- array(0, c(max_nc-min_nc + 1 , 2))
  
  # Save the values of k in the first column
  res[, 1] <- min_nc:max_nc
  
  # Loop over all candidates k
  for(k in min_nc:max_nc) {
    # Perform the clustering for the value of k on the PCA data
    clustering <- cluster::pam(pca$x[, 1:nr_components], k, keep.diss=TRUE)
    
    # Compute the dissimilarity matrix to align with the medoids method
    d <- clustering$diss
    
    # Save the results
    res[k-2+1, 2] <- clusterSim::index.G1(x = pca$x[, 1:nr_components], 
                                          cl = clustering$cluster, 
                                          d = d, centrotypes="medoids")
  }
  # Get the summarized data
  CH_data <- res %>% 
    as.data.frame() %>% 
    as_tibble() %>% 
    rename(k = 1, G1 = 2) 
  
  # Print summarized
  print(CH_data)
  
  # Visualize the Calinski-Harabasz pseudo-F statistic
  res %>% 
    as.data.frame() %>% 
    as_tibble() %>% 
    rename(k = 1, G1 = 2) %>% 
    ggplot(aes(k, G1)) + geom_line(color="#4477AA") + geom_point(color="#4477AA") +
    theme_classic() + 
    xlab("Number of clusters (k)") + 
    ggtitle(paste("Optimal number of clusters (CH) for", position_name)) + 
    theme(plot.title = element_text(hjust=0.5), 
          axis.text = element_text(colour="black", size=10)) -> CH
  
  # Print the plot
  print(CH)
  
  # Save the plot
  ggsave(paste("../Plots/CH", position_name, ".png"), height=6, width=8)
  
  # Create the plot for silhouette score visualizations
  silhouette_plot <- factoextra::fviz_nbclust(pca$x[, 1:nr_components], FUNcluster = cluster::pam)
  
  silhouette_plot$data %>% 
    ggplot(aes(clusters, y, group=1)) + geom_line(color="#4477AA") + geom_point(color="#4477AA") +
    theme_classic() + 
    xlab("Number of clusters (k)") + ylab("Average silhouette width") +
    ggtitle(paste("Optimal number of clusters (silhouette score) for", position_name)) + 
    theme(plot.title = element_text(hjust=0.5), 
          axis.text = element_text(colour="black", size=10)) -> silhouette
  
  # Print the summary values
  print(silhouette_plot$data %>% as_tibble() %>% rename(k = 1, Silhouette = 2))
  
  # Visualize the silhouette score
  print(silhouette)
  
  # Save the plot
  ggsave(paste("../Plots/silhouette", position_name, ".png"), height=6, width=8)
}


#' Fit the cluster
#'
#' Perform clustering using Partitioning Around Medoids (PAM) on the data.
#'
#' @param position_data data frame of statistics for a position group
#' @param pca pca object for the position-specific data
#' @param nr_components the number of components to retain
#' @param nr_clusters the number of clusters to form
#'
#' @return processed_data with clusters and medoid column added
#' @export
#'
fit_cluster <- function(position_data, pca, nr_components, nr_clusters){
  
  #  Perform clustering on a set amount of principal components and clusters
  PAM <- cluster::pam(pca$x[, 1:nr_components], nr_clusters)
  
  # Copy to avoid changing in-place
  processed_data_with_cluster <- position_data
  
  # Save the clustering in the original data frame
  processed_data_with_cluster["Cluster"] <- PAM$cluster
  
  # Save the cluster medoids
  processed_data_with_cluster["Medoid"] <- 0
  processed_data_with_cluster[PAM$id.med, "Medoid"] <- 1
  
  return(processed_data_with_cluster)
}


#' Evaluate the clustering
#'
#' Evaluate the quality of the clustering visually (interactive) and via 
#' cluster means and counting archetypes within each cluster. 
#'
#' @param processed_data_with_cluster processed_data with clusters and medoid column added
#' @param pca pca object for the position-specific data
#'
#' @return list of count of archetypes within each cluster and cluster means
#' @export
#'
evaluate_cluster <- function(processed_data_with_cluster, pca){
  # Capture the name of the input data
  input_data <- deparse(substitute(pca))
  
  # Get the name of the position
  position_name <- str_split(input_data, "_")[[1]][2]
  
  # Specify a colorblind friendly palette
  cbPalette <- c('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB')
  
  # Visualize the clustering for the first principal components
  data.frame(PC1=pca$x[, 1], PC2=pca$x[, 2], cluster=processed_data_with_cluster["Cluster"], 
             name=processed_data_with_cluster["fullName"], 
             position=processed_data_with_cluster["centralRegistryPosition"],
             medoid=processed_data_with_cluster["Medoid"]) %>% 
    mutate(cluster=factor(Cluster), medoid=factor(Medoid)) %>%
    rename(name = fullName, position = centralRegistryPosition, medoid=medoid) %>% 
    ggplot() + theme_classic() +
    suppressWarnings(geom_point(data = . %>% filter(medoid == 0), 
                                aes(PC1, PC2, color=cluster, name=name, position=position), 
                                shape=16))  + 
    suppressWarnings(geom_point(data = . %>% filter(medoid == 1), 
                                aes(PC1, PC2, color=cluster, name=name, position=position), 
                                shape=18, fill="black")) +
    suppressWarnings(geom_text_repel(aes(PC1, PC2, label=name))) + 
    scale_color_manual(values = cbPalette) +
    theme(plot.title = element_text(hjust=0.5), 
          axis.text = element_text(colour="black", size=10)) -> L2
  
  # Save the static plot
  ggsave(paste("../Plots/clusterPCA", position_name, ".png"), L2, height=9, width=16)
  
  # Print the plot
  print(suppressWarnings(ggplotly(L2)))
  
  # Compute the number of archetypes per cluster
  nhl_archetypes <- processed_data_with_cluster %>% group_by(Cluster, Type) %>% count() %>% arrange(Cluster, -n)
  
  # Compute the total number of players from each archetype
  archetype_count <- processed_data_with_cluster %>% group_by(Type) %>% count() %>% rename(ArchetypeN = n)
  
  # Combine cluster counts and total counts
  nhl_archetypes <- inner_join(nhl_archetypes, archetype_count, by="Type")
  
  # Compute the proportion of players from a specific archetype belonging to a cluster
  nhl_archetypes <- nhl_archetypes %>% 
    mutate(PropType = n/ArchetypeN)
  
  # Compute numerical summarized values for all clusters
  cluster_attributes <- processed_data_with_cluster %>% 
    dplyr::select(names(pca$center), Cluster) %>% 
    group_by(Cluster) %>% 
    summarise(across(where(is.numeric), mean))
  
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
plot_archetypes_in_cluster <- function(archetypes_in_cluster){
  # Capture the name of the input data
  input_data <- deparse(substitute(archetypes_in_cluster))
  
  # The position name
  position_name <- str_split(input_data, "_")[[1]][-c(1,2,3)]
  
  # If a player is in the wrong position group
  wrong_position <- ifelse(position_name == "defenders", "Forward", "Defenseman")
  
  # Specify a colorblind friendly palette
  cbPalette <- c('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB')
  
  # Plot the distribution
  archetypes_in_cluster %>% 
    filter(!grepl(wrong_position, Type)) %>% 
    mutate(Type = ifelse(is.na(Type), "Unknown", Type)) %>% 
    group_by(Cluster) %>% 
    mutate(Position = rank(-PropType)) %>% 
    ggplot(aes(Cluster, PropType, group=Position, fill=Type)) + 
    geom_bar(stat="identity", position = "dodge") + 
    ggtitle(paste("NHL 22 archetypes belonging to each cluster", position_name)) +
    theme_classic() + 
    theme(legend.position = "top", plot.title = element_text(hjust=0.5),
          axis.text = element_text(colour="black", size=10)) + 
    scale_fill_manual(values = cbPalette) -> p
  
  print(p)
  
  ggsave(paste("../Plots/archetypes", position_name, ".png"), height=6, width=8)
}


#' Find nearest neighbors and cap hit
#'
#' For each player, find the n nearest neighbors and their cap hit information. 
#'
#' @param pca pca object for the position-specific data
#' @param position_data data frame of statistics for a position group
#' @param n_components the number of components to retain
#' @param neighbors the number of neighbors to consider
#' @param p the power of the Minkowski distance
#' @param dist_boundary the maximum distance from the player allowed
#' @param log_cap if the cap hit should be log-transformed
#' @param same_cluster if only players from the same cluster should be considered
#' 
#' @return data frame of player in position with each player's n nearest neighbors
#' @export
#'
nearest_neigbors_cap <- function(pca, position_data, n_components, 
                                 neighbors = NA, p = 2, dist_boundary = NA,
                                 log_cap = TRUE, same_cluster = TRUE){
  if((is.na(neighbors) & is.na(dist_boundary)) |
     (!is.na(neighbors) & !is.na(dist_boundary))){
    stop( "Please specify only one of neighbors or dist_boundary.")
  }
  
  # Compute the distance between each pair of players
  d <- as.matrix(dist(pca$x[, 1:n_components], method="minkowski", p=p)) %>% as_tibble() %>% 
    bind_cols(position_data %>% dplyr::select(PlayerId, Cluster))
  
  # Specify the column names as player ids
  colnames(d)[1:(length(d)-2)] <- d$PlayerId
  
  # Find the n closest players with respect to distance
  d %>% 
    pivot_longer(-c(PlayerId, Cluster), names_to = "OtherPlayerId", values_to = "Dist") %>% 
    mutate(OtherPlayerId = as.integer(OtherPlayerId)) %>% 
    filter(Dist != 0) %>% 
    inner_join(d %>% 
                 pivot_longer(-c(PlayerId, Cluster), names_to = "name", values_to = "value") %>% 
                 dplyr::select(-c(name, value)) %>% 
                 distinct() %>% 
                 rename(OtherPlayerId = PlayerId, OtherPlayerCluster = Cluster),
               by="OtherPlayerId") %>% 
    {if (same_cluster) filter(., Cluster == OtherPlayerCluster) else filter(.)} %>% 
    group_by(PlayerId) %>% 
    {if (!is.na(neighbors) & is.na(dist_boundary)) slice_min(., n=neighbors, order_by = Dist) 
      else filter(., Dist < dist_boundary)} %>% 
    mutate(DistRank = rank(Dist)) -> nearest_neighbors
  
  # Read cap data
  cap <- read.csv("../Data/cap20212022.csv") %>% 
    add_row(PLAYER = "Sasha Chmelevski", TEAM = "SJS", PlayerId = 8480053,
            CAP.HIT = 778333, TYPE = "Entry-Level", EXP..YEAR = 2022)
  
  # Add cap information for all players
  nearest_neighbors %>% 
    ungroup() %>% 
    inner_join(cap %>% dplyr::select(PLAYER, TEAM, PlayerId, CAP.HIT, TYPE, EXP..YEAR) %>% 
                 rowwise() %>% 
                 mutate(CAP.HIT.VAL = ifelse(log_cap, log(CAP.HIT / 1E5), CAP.HIT / 1E5)), by="PlayerId") %>% 
    rename(PlayerCapHit = CAP.HIT.VAL, PlayerName = PLAYER, PlayerTeam = TEAM, 
           ContractType = TYPE, ExpiryYear = EXP..YEAR) %>% 
    mutate(OtherPlayerId = as.integer(OtherPlayerId)) %>% 
    inner_join(cap %>% dplyr::select(PLAYER, TEAM, PlayerId, CAP.HIT, TYPE) %>%
                 rowwise() %>% 
                 mutate(CAP.HIT.VAL = ifelse(log_cap, log(CAP.HIT / 1E5), CAP.HIT / 1E5)), 
               by=c("OtherPlayerId" = "PlayerId")) %>% 
    rename(OtherPlayerCapHit = CAP.HIT.VAL, OtherPlayerName = PLAYER, OtherPlayerTeam=TEAM) %>% 
    dplyr::select(PlayerId, PlayerName, PlayerTeam, PlayerCapHit, ContractType, ExpiryYear,
                  OtherPlayerId, OtherPlayerName, OtherPlayerTeam, OtherPlayerCapHit,
                  Dist, DistRank) %>% 
    rowwise() %>% 
    mutate(CapDiff = PlayerCapHit - OtherPlayerCapHit) -> nearest_neighbors_with_cap
  
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
avg_cap_diff <- function(position_cap){
  # Compute the average difference and distance for each player
  position_cap %>% 
    group_by(PlayerId, PlayerName, PlayerTeam, ContractType, PlayerCapHit) %>% 
    summarise(AvgDiff = mean(CapDiff), AvgDist = mean(Dist), AvgDiffDist = mean(CapDiff/Dist), 
              n=n(), .groups = "keep") %>% 
    arrange(-AvgDiffDist) %>% 
    ungroup()
}

#' Compute the cost per stat
#'
#' For each player, compute the $ equivalent of each point/goal/assist/minute on 
#' the ice by dividing their cap hit with each statistic
#'
#' @param position_cap data frame of player in position with each player's n nearest neighbors
#' @param position_data data frame of statistics for a position group
#'
#' @return data frame of cost per statistic
#' @export
#'
cost_per_stat <- function(position_cap, position_data){
  # Compute the cost per point, goal etc.
  position_cap %>% 
    dplyr::select(PlayerId, PlayerName, PlayerCapHit) %>% 
    distinct() %>% 
    inner_join(position_data %>% dplyr::select(PlayerId, Points, Goals, Assists, TOI), by="PlayerId") %>% 
    mutate(CostPerPoint = PlayerCapHit/Points, CostPerGoal = PlayerCapHit/Goals,
           CostPerAssist = PlayerCapHit/Assists, CostPerMin = PlayerCapHit/(TOI/60)) %>% 
    arrange(CostPerPoint)
}
