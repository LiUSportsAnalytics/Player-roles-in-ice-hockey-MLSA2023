library(readr)
library(dplyr)
library(tidyr)
library(tibble)
library(stringr)


#' Load data from disk
#'
#' Loads different data files from disk and merges them into the data used for the
#' analysis.
#'
#' @return pca_data
#' @export
#'
load_data <- function(){
  # Load player archetypes from NHL 22
  archetypes <- read_csv("../Data/PlayerArchetypes.csv", show_col_types = FALSE) %>% 
    dplyr::select(PlayerId, Type)
  
  # Load player metadata from NHL.com
  players <- read_csv("../Data/players.csv", show_col_types = FALSE)
  
  # Load features computed in Python
  pca_data <- read_csv("../Data/PCA.csv", show_col_types = FALSE) %>% 
    distinct(PlayerId, .keep_all = TRUE)  
  
  # Fill all NA values with 0 (should be fine in this scenario)
  pca_data[is.na(pca_data)] <- 0
  
  # Combine features with player names and positions
  pca_data <- merge(players[,c("id", "fullName", "centralRegistryPosition")], pca_data,
                    by.x="id", by.y="PlayerId", how="right") %>% 
    rename(PlayerId = id)
  
  # Combine features and player data with archetypes
  pca_data <- left_join(pca_data, archetypes, by=c("PlayerId")) %>% 
    distinct() %>% 
    mutate(xGDiff = Goals - xG) 
  
  return(pca_data)
}


#' Select position group
#'
#' Select the position group (defenders/forwards) to return data for.
#'
#' @param raw_data the raw data from load_data()
#' @param defenders if the position should be defenders (TRUE) or forwards (FALSE)
#'
#' @return position-specific statistics
#' @export
#'
select_position <- function(raw_data, defenders=TRUE){
  # Select defenders
  if(defenders){
    position_data <- raw_data %>% 
      filter(centralRegistryPosition == "D") 
  } else {
    # Select forwards
    position_data <- raw_data %>% 
      filter(centralRegistryPosition != "D") 
  }
  
  return(position_data)
}


#' Preprocess the data
#'
#' Process the data by first normalizing all (applicable) columns by time on ice 
#' and then multiply to be full season data. Afterwards remove undesired columns 
#' and scale by subtracting the mean and assuring each column has unit variance.
#'
#' @param position_data data frame of statistics for a position group
#'
#' @return data frame of normalized and standardized statistics for a position group
#' @export
#'
preprocess_data <- function(position_data){
  # Standardize by time on ice (multiply to be per game)
  position_data %>% 
    # Absolute Y coordinates
    mutate(across(starts_with("YMedian"), function(x) abs(x))) %>%
    mutate(across(Goals:Takeaways, function(x) 60 * 60 * x / TOI)) %>% 
    mutate(across(Hits:PenaltyMinutes, function(x) 60 * 60 * x / TOI))  %>% 
    mutate(across(c(Fights, GoalsCreated, xGDiff, FaceoffsWon, GoalsCreated), function(x) 60 * 60 * x / TOI)) %>% 
    mutate(across(Backhand:PlusMinus, function(x) 60 * 60 * x / TOI)) %>%
    mutate(across(NumOtherPenaltyOn:NumOtherDrewBy, function(x) 60 * 60 * x / TOI)) %>%
    mutate(across(Goals_xg:EN_xG, function(x) 60 * 60 * x / TOI)) %>%
    # Standardize Corsi/FenwickOn/xGOn as they are computed during 5 on 5
    mutate(across(c(CorsiOn, FenwickOn, xGOn), function(x) 60 * 60 * x / `5v5 TOI`)) -> standardized_data
  
  # Remove the features that will not be used in the analysis and then normalize the data
  processed_data <- standardized_data %>% 
    dplyr::select(
      -c(# Remove player meta data
         "PlayerId", "fullName", "centralRegistryPosition", "Type", 
         # Remove DZS% and Thru%
         "Thru%", "DZS%", 
         # Remove points as it is linearly related to goals and assists. Remove Goals created 
         "Points", "GoalsCreated",
         # Remove xG variables not used
         "Goals_xg", EV_Goals:EN_xG, "xGOn",
         # Remove "other" penalties
         starts_with("NumOther"),
         # Remove TOI % in various situations
         "EV%", "5v5%", 
         # Remove Corsi
         starts_with("CF"), starts_with("Corsi"),
         # Remove Fenwick
         starts_with("FF"), starts_with("Fenwick"),
         # Remove faceoffs and time on ice 
         starts_with("Face"), ends_with("TOI"), 
         # Remove shot types
         Backhand:Wrist
      )) %>% scale() 
  
  return(processed_data)
}

