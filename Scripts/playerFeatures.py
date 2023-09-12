#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from typing import Tuple
from pandera.typing import DataFrame

def compute_fights(events_long: DataFrame) -> DataFrame:
    """
    Compute the number of fights each player was involved in.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    fighting : DataFrame
        Information over the number of fights per player.

    """
    # Get all the fighting events
    fighting_events = events_long.loc[events_long.PenaltyType.eq("Fighting") & 
                                      events_long.PlayerType.eq("PenaltyOn")]

    # Count the number of fights for each player
    fighting = fighting_events.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Fights"})
    
    return fighting


def compute_goals_and_assists_and_points(events_long: DataFrame) -> Tuple[DataFrame,  
                                                                          DataFrame,  
                                                                          DataFrame]:
    """
    Compute the number of goals, assists and points per player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    goals : DataFrame
        Information over the number of goals by each player.
    assists : DataFrame
        Information over the number of assists by each player.
    points : DataFrame
        Information over the number of points by each player.

    """
    # Get all events where the player role is goalscorer
    goalscorers = events_long.loc[events_long.PlayerType.eq("Scorer")]
    
    # Compute the number of goals scored by each player
    goals = goalscorers.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Goals"})
    
    # Get all events where the player role is assister
    assister = events_long.loc[events_long.PlayerType.eq("Assist")]
    
    # Compute the number of assists made by each player
    assists = assister.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Assists"})
    
    # Combine goals and assists, filling NA with 0
    points = goals.merge(assists, how="outer").fillna(0)
    
    # Compute the number of points per player
    points["Points"] = points[["Goals", "Assists"]].sum(axis=1)
    
    return goals, assists, points
    

def compute_corsi_and_fenwick(events_long: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the number of shot attempts (Corsi) and unblocked shot attempts (Fenwick)
    per player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    corsi : DataFrame
        Information over the number of shot attempts for each player.
    fenwick : DataFrame
        Information over the number of unblocked shot attempts for each player.

    """
    # Get all shooting events, i.e., shooter or scorer
    shots = events_long.loc[events_long.PlayerType.isin(["Scorer", "Shooter"])]
        
    # Compute Corsi (SAT)
    corsi = shots.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Corsi"})
    
    # Find all shots that were not blocked
    unblocked_shots = shots.loc[shots.EventType.ne("BLOCKED SHOT")]

    # Compute Fenwick (USAT)
    fenwick = unblocked_shots.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Fenwick"})
    
    return corsi, fenwick
    

def compute_shot_types(events_long: DataFrame) -> DataFrame:
    """
    Count the number of shots from each shot type and player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    shot_types : DataFrame
        Information over the various shot types and their counts, by player.

    """
    # Get all shots and their type
    shots = events_long.loc[events_long.PlayerType.isin(["Shooter", "Scorer"])].groupby(
        ["PlayerId", "ShotType"]).size().reset_index().rename(columns={0: "Shots"})
    
    # Counts of different shot types
    shot_types = shots.pivot(index="PlayerId", columns="ShotType", values="Shots").fillna(0).reset_index()

    # Remove spaces and dashes from column names
    shot_types.columns = shot_types.columns.str.replace("\s|-", "", regex=True)
    
    return shot_types


def compute_shot_pct(events_long: DataFrame) -> DataFrame:
    """
    Compute S% (goals/shots on goal) and Thru% (shots on goal/all shots).

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    shot_counts_wide : DataFrame
        Data frame with S% and Thru%.

    """
    # Get all shots
    shots = events_long.loc[events_long.EventType.isin(["GOAL", "SHOT", "MISSED SHOT",
                                                        "BLOCKED SHOT"]) &
                            events_long.PlayerType.isin(["Shooter", "Scorer"])]
    
    # Compute the number of shots per player and type
    shot_counts = shots.groupby(["PlayerId", "EventType"], as_index=False).size()
    
    # Convert into a wide format
    shot_counts_wide = pd.pivot(shot_counts, index="PlayerId", columns="EventType", 
                                values="size").reset_index().fillna(0)
    
    # Compute shooting percentage
    shot_counts_wide["S%"] = (shot_counts_wide["GOAL"] / 
                              shot_counts_wide[["GOAL", "SHOT"]].sum(axis=1)).fillna(0)
    
    # Compute percentage of shots that go on goal
    shot_counts_wide["Thru%"] = (shot_counts_wide[["GOAL", "SHOT"]].sum(axis=1) / 
                                 shot_counts_wide.iloc[:, 1:5].sum(axis=1)).fillna(0)
    
    return shot_counts_wide[["PlayerId", "S%", "Thru%"]]


def compute_takeaways(events_long: DataFrame) -> DataFrame:
    """
    Count the number of takeaways by each player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    takeaways : DataFrame
        Information over the number of takeaways per player.

    """
    # Get all takeaway events
    takeaway_events = events_long.loc[events_long.EventType.eq("TAKEAWAY")]
    
    # Compute the number of takeaways per player
    takeaways = takeaway_events.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Takeaways"})
    
    return takeaways


def compute_giveaways(events_long: DataFrame) -> DataFrame:
    """
    Compute the number of giveaways per player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    giveaways : DataFrame
        Information over the number of giveaways per player.

    """
    # Get all giveaway events
    giveaway_events = events_long.loc[events_long.EventType.eq("GIVEAWAY")]
    
    # Compute the number of giveaways per player
    giveaways = giveaway_events.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Giveaways"})
    
    return giveaways


def compute_hits(events_long: DataFrame) -> DataFrame:
    """
    Compute the number of hits and net hits for each player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    hits : DataFrame
        Information over the number of hits per player.

    """
    # Get all hit events
    hit_events = events_long.loc[events_long.EventType.eq("HIT")]
    
    # Compute the number of hits delievered by a player
    hits_for = hit_events.loc[hit_events.PlayerType.eq("Hitter")].PlayerId.value_counts().reset_index(
        ).rename(columns={"index": "PlayerId", "PlayerId": "Hits"})
    
    # Compute the number of hits received by a player
    hits_against = hit_events.loc[hit_events.PlayerType.eq("Hittee")].PlayerId.value_counts().reset_index(
        ).rename(columns={"index": "PlayerId", "PlayerId": "HitsTaken"})
    
    # Combine hits given and taken
    hits = hits_for.merge(hits_against, how="outer").fillna(0)
    
    # Compute the net number of hits
    hits["HitNet"] = hits["Hits"] - hits["HitsTaken"]
    
    return hits
    
    
def compute_penalties_and_pim(events_long: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the number of penalties, net penalties, and penalty minutes per player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    penalties : DataFrame
        Information over the number of penalties per player.
    pim : DataFrame
        Information over the amount of penalty minutes per player.

    """
    # Get all penalty events
    penalty_events = events_long.loc[events_long.EventType.eq("PENALTY")]
    
    # Compute the number of penalties on a player
    penalties_on = penalty_events.loc[penalty_events.PlayerType.eq("PenaltyOn")].PlayerId.value_counts().reset_index(
        ).rename(columns={"index": "PlayerId", "PlayerId": "Penalties"})
    
    # Compute the number of penalties drawn by a player
    penalties_drawn = penalty_events.loc[penalty_events.PlayerType.eq("DrewBy")].PlayerId.value_counts().reset_index(
        ).rename(columns={"index": "PlayerId", "PlayerId": "PenaltiesDrawn"})
    
    # Combine the penalties on and drawn by a player
    penalties = penalties_on.merge(penalties_drawn, how="outer").fillna(0)
    
    # Compute the net penalties
    penalties["PenaltyNet"] = penalties["Penalties"] - penalties["PenaltiesDrawn"]
    
    # Compute penalties in minutes (PIM)
    pim = penalty_events.groupby("PlayerId", as_index=False)["PenaltyMinutes"].sum()
    
    return penalties, pim


def compute_blocked_shots(events_long: DataFrame) -> DataFrame:
    """
    Compute the number of blocked shots per player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    blocks : DataFrame
        Information over blocked shots for each player.

    """
    # Get all blocked shot events, from the perspective of the blocker
    blocked_shots = events_long.loc[events_long.PlayerType.isin(["Blocker"])]

    # Compute the number of blocked shots per player
    blocks = blocked_shots.PlayerId.value_counts().reset_index().rename(
        columns={"index": "PlayerId", "PlayerId": "Blocks"})
    
    return blocks


def compute_penalty_types(events_long: DataFrame) -> DataFrame:
    """
    Compute the count of the different penalty types.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    group_size_wide : DataFrame
        The count of all penalty types in wide format.

    """
    
    # Copy to avoid changing in-place
    events_long_copy = events_long.copy()
    
    # Determine the penalty type
    events_long_copy["PenaltyGroup"] = np.select(
        [events_long_copy.PenaltyType.isin(["Aggressor", "Boarding", "Charging", 
                                            "Checking from Behind", "Clipping", "Elbowing", 
                                            "Fighting", "Head-butting", 
                                            "Illegal check to head", "Instigator",
                                            "Instigator - Misconduct",
                                            "Kicking", "Kneeing", "Roughing", "Slew-footing",
                                            "Throwing stick"]), 
         events_long_copy.PenaltyType.isin(["Holding", "Holding the stick", "Hooking", 
                                            "Interference", "Tripping"]),
         events_long_copy.PenaltyType.isin(["Butt-ending", "Butt ending - double minor", 
                                            "Cross checking", "Hi-sticking", "Hi stick - double minor",
                                            "Slashing", "Spearing", "Spearing - double minor"]),
         events_long_copy.PenaltyType.isin(["Abuse of Officials", "Abusive language", 
                                            "Bench", "Broken stick", "Closing hand on puck",
                                            "Delay of game", "Delay GM - Face-off Violation",
                                            "Delaying Game - Illegal play by goalie", 
                                            "Delaying Game - Puck over glass",
                                            "Delaying Game - Smothering Puck",
                                            "Embellishment", "Face-off violation", 
                                            "Game misconduct", "Goalie leave crease",
                                            "Interference - Goalkeeper", "Match penaltiy",
                                            "Misconduct", "Playing without a helmet",
                                            "Too many men on the ice", "Unsportsmanlike conduct"])],   
    ["Physical", "Restraining", "Stick", "Other"], 
    "Other")
    
    # If the event is not a penalty, set it to NaN
    events_long_copy.loc[events_long_copy.EventType.ne("PENALTY") |
                         events_long_copy.PlayerType.eq("ServedBy"), "PenaltyGroup"] = np.nan
    
    # Compute the count of penalties per type and who drew/took the penalty
    group_size = events_long_copy.groupby(["PlayerId", "PenaltyGroup", "PlayerType"], 
                                          as_index=False).size().rename(
                                              columns={"size": "Num"})

    # Convert from long to wide
    group_size_wide = group_size.pivot(index="PlayerId", 
                                       columns=["PenaltyGroup", "PlayerType"], 
                                       values=["Num"]).reset_index()

    # Rename columns
    group_size_wide.columns = list(map(''.join, group_size_wide.columns.values))
    
    return group_size_wide


def compute_median_coordinates(events_long: DataFrame) -> DataFrame:
    """
    Compute the median coordinates (X and Y) for defensive, shooting, and other 
    actions for each player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    avg_xy_wide : DataFrame
        Information over the median coordinates (X and Y) for action groups and
        by player.

    """
    # Copy to avoid changing in-place
    events_long_copy = events_long.copy()
    
    # Create a group for shooters, assisters and other
    events_long_copy["EventGroup"] = np.select(
        [events_long_copy.PlayerType.isin(["Shooter", "Scorer"]), 
         # events_long_copy.PlayerType.isin(["Assist"]),  
         events_long_copy.EventType.eq("FACEOFF"),
         events_long_copy.EventType.eq("TAKEAWAY"),
         events_long_copy.EventType.eq("GIVEAWAY"),
         events_long_copy.PlayerType.eq("Blocker"),
         events_long_copy.PlayerType.eq("Hitter"),
         events_long_copy.PlayerType.eq("Hittee"),
         events_long_copy.PlayerType.eq("PenaltyOn"),
         events_long_copy.PlayerType.eq("DrewBy")],
        ["Shooter", "Faceoff", "Takeaway", "Giveaway", "Blocker",
         "Hitter", "Hittee", "PenaltyOn", "DrewBy"], 
        "Other")

    # Compute average X/Y coordinates and their sizes
    avg_xy = events_long_copy.groupby(["PlayerId", "EventGroup"], as_index=False).agg(
        {"X": ["median", "mean", "std", "size"],
         "Y": ["median", "mean", "std", "size"]})

    # Combine columns to remove multiindex
    avg_xy.columns = list(map(''.join, avg_xy.columns.values))

    # Convert from long to wide
    avg_xy_wide = avg_xy.pivot(index="PlayerId", columns="EventGroup", 
                               values=["Xmedian", "Ymedian"]).reset_index()

    # Rename columns
    avg_xy_wide.columns = list(map(''.join, avg_xy_wide.columns.values))

    return avg_xy_wide


def compute_faceoff_win_percentage(events_long: DataFrame) -> DataFrame:
    """
    Compute faceoff statistics for each player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.

    Returns
    -------
    faceoffs : DataFrame
        Information over faceoffs won, lost and faceoff win % for each player.

    """
    # Get all faceoff events
    faceoff_events = events_long.loc[events_long.EventType.eq("FACEOFF")]
    
    # Compute the number of won faceoffs
    faceoffs_won = faceoff_events.loc[faceoff_events.PlayerType.eq("Winner")].PlayerId.value_counts().reset_index(
        ).rename(columns={"index": "PlayerId", "PlayerId": "FaceoffsWon"})
    
    # Compute the number of lost faceoffs
    faceoffs_lost = faceoff_events.loc[faceoff_events.PlayerType.eq("Loser")].PlayerId.value_counts().reset_index(
        ).rename(columns={"index": "PlayerId", "PlayerId": "FaceoffsLost"})
    
    # Combine lost and won faceoffs to get total faceoffs
    faceoffs = faceoffs_won.merge(faceoffs_lost, how="outer").fillna(0)
    
    # Compute the percentage of faceoffs won
    faceoffs["FaceoffWonPct"] = faceoffs["FaceoffsWon"] / (faceoffs["FaceoffsWon"] + faceoffs["FaceoffsLost"])

    return faceoffs


def compute_zone_starts(df: DataFrame) -> DataFrame:
    """
    Compute the percentage of zone starts for each player.

    Parameters
    ----------
    df : DataFrame
        Original play by play data in wide format.

    Returns
    -------
    zone_start_wide : DataFrame
        Information over the percentage of starts in the offensive and defensive zones
        for each player.

    """
    # Create a copy to avoid changing in-place
    faceoffs = df.loc[df.EventType.eq("FACEOFF")].copy()
    
    # Find the number of players per side
    home_player_columns = faceoffs.columns[faceoffs.columns.str.startswith("HomePlayer")]
    away_player_columns = faceoffs.columns[faceoffs.columns.str.startswith("AwayPlayer")]
    
    # Find the maximum player number per side
    home_player_max = int(home_player_columns[-1][-1])
    away_player_max = int(away_player_columns[-1][-1])

    # Specify the amount of players for each team
    home_players = ['HomePlayer{}'.format(n) for n in range(1, home_player_max+1)]
    away_players = ['AwayPlayer{}'.format(n) for n in range(1, away_player_max+1)]
    
    # Compute manpower
    faceoffs["ManpowerHome"] = faceoffs[home_players].notna().sum(axis=1)
    faceoffs["ManpowerAway"] = faceoffs[away_players].notna().sum(axis=1)
    
    # Keep only faceoffs during 5v5
    faceoffs = faceoffs.loc[faceoffs.ManpowerHome.eq(faceoffs.ManpowerAway) & 
                            faceoffs.ManpowerHome.eq(6)]
    
    # Home players on faceoffs when the home team wins
    home_faceoffs_home = faceoffs.loc[faceoffs.Team.eq(faceoffs.HomeTeamName),
                                      ["HomeTeamName", "HomePlayerId1", "HomePlayerId2", "HomePlayerId3", 
                                       "HomePlayerId4", "HomePlayerId5", "HomePlayerId6", "X", "Y"]].rename(
                                      columns={"HomeTeamName": "TeamName"})
    
    # Away players on faceoffs when the home team wins
    home_faceoffs_away = faceoffs.loc[faceoffs.Team.eq(faceoffs.HomeTeamName),
                                      ["AwayTeamName", "AwayPlayerId1", "AwayPlayerId2", "AwayPlayerId3", 
                                       "AwayPlayerId4", "AwayPlayerId5", "AwayPlayerId6", "X", "Y"]].rename(
                                       columns={"AwayTeamName": "TeamName"})
    
    # Mirror coordinates for the losing team
    home_faceoffs_away["X"] *= -1
    home_faceoffs_away["Y"] *= -1

    # Away players on faceoffs when the away team wins
    away_faceoffs_away = faceoffs.loc[faceoffs.Team.ne(faceoffs.HomeTeamName),
                                      ["AwayTeamName", "AwayPlayerId1", "AwayPlayerId2", "AwayPlayerId3", 
                                       "AwayPlayerId4", "AwayPlayerId5", "AwayPlayerId6", "X", "Y"]].rename(
                                       columns={"AwayTeamName": "TeamName"})

    # Home players on faceoffs when the away team wins
    away_faceoffs_home = faceoffs.loc[faceoffs.Team.ne(faceoffs.HomeTeamName),
                                      ["HomeTeamName", "HomePlayerId1", "HomePlayerId2", "HomePlayerId3", 
                                       "HomePlayerId4", "HomePlayerId5", "HomePlayerId6", "X", "Y"]].rename(
                                       columns={"HomeTeamName": "TeamName"})
                                     
    # Mirror coordinates for the losing team
    away_faceoffs_home["X"] *= -1
    away_faceoffs_home["Y"] *= -1

    # All combinations of faceoff perspectives
    all_faceoffs = pd.concat([home_faceoffs_home.melt(id_vars=["TeamName", "X", "Y"]),
                              home_faceoffs_away.melt(id_vars=["TeamName", "X", "Y"]),
                              away_faceoffs_away.melt(id_vars=["TeamName", "X", "Y"]),
                              away_faceoffs_home.melt(id_vars=["TeamName", "X", "Y"]),
                              ])

    # Determine in which zone the faceoff took place: offensive, defensive or neutral
    all_faceoffs["Zone"] = np.select(
        [all_faceoffs["X"].ge(30), all_faceoffs["X"].le(-30)],
        ["Offensive", "Defensive"], "Neutral")

    # Compute number of starts in each zone
    zone_start = all_faceoffs.groupby(["value", "Zone"]).size().reset_index().merge(
        all_faceoffs.groupby(["value"]).size().reset_index(), on="value").merge(
        all_faceoffs.loc[all_faceoffs.Zone.ne("Neutral")].groupby(["value"]).size().reset_index(), 
        on="value")

    # Rename columns
    zone_start.rename(columns={"0_x": "ZoneStarts", "0_y": "TotalStartsNeutral",
                               0: "TotalStarts", "value": "PlayerId"},
                      inplace=True)

    # Compute % of zone starts
    zone_start["ZoneStartNeutral%"] = zone_start["ZoneStarts"] / zone_start["TotalStartsNeutral"]
    zone_start["ZoneStart%"] = zone_start["ZoneStarts"] / zone_start["TotalStarts"]

    # Convert from long to wide
    zone_start_wide = zone_start.pivot(index="PlayerId", 
                                       values=["ZoneStart%", "ZoneStartNeutral%"],
                                       columns="Zone").reset_index()
    
    # Flatten hierarchical index
    zone_start_wide.columns = ['_'.join(col).strip() for col in
                               zone_start_wide.columns.values]
    
    # Rename columns
    zone_start_wide.rename(columns={"PlayerId_": "PlayerId",
                                    "ZoneStart%_Defensive": "DZS%",
                                    "ZoneStart%_Neutral": "NZS%",
                                    "ZoneStart%_Offensive": "OZS%",
                                    "ZoneStartNeutral%_Defensive": "nDZS%",
                                    "ZoneStartNeutral%_Neutral": "nNZS%",
                                    "ZoneStartNeutral%_Offensive": "nOZS%"},
                           inplace=True)
    
    # Remove neutral zone starts 
    zone_start_wide.drop("NZS%", axis=1, inplace=True)
    
    return zone_start_wide
    

def compute_plus_minus(df: DataFrame) -> DataFrame:
    """
    Compute plus-minus for each player.

    Parameters
    ----------
    df : DataFrame
        Original play by play data in wide format.

    Returns
    -------
    all_plusminus : DataFrame
        Information over plus-minus for each player.

    """
    # Select all goals
    pm_df = df.loc[df.EventType.eq("GOAL") &
                   df.PeriodNumber.le(4)].copy()
        
    # Find the number of players per side
    home_player_columns = pm_df.columns[pm_df.columns.str.startswith("HomePlayerId")]
    away_player_columns = pm_df.columns[pm_df.columns.str.startswith("AwayPlayerId")]
    
    # Find the maximum player number per side
    home_player_max = int(home_player_columns[-1][-1])
    away_player_max = int(away_player_columns[-1][-1])
 
    # Specify the amount of players for each team
    home_players = ['HomePlayerId{}'.format(n) for n in range(1, home_player_max+1)]
    away_players = ['AwayPlayerId{}'.format(n) for n in range(1, away_player_max+1)]
    
    # Compute manpower
    pm_df["ManpowerHome"] = pm_df[home_player_columns].notna().sum(axis=1)
    pm_df["ManpowerAway"] = pm_df[away_player_columns].notna().sum(axis=1)
    
    # Indicator variable for whether the home team scored
    pm_df["ScoringTeam"] = np.where(pm_df["Team"] == pm_df["HomeTeamName"], 
                                    "Home", "Away")
    
    # Compute manpower difference
    pm_df.loc[pm_df.ScoringTeam.eq("Home"), "MD"] = (
        pm_df.loc[pm_df.ScoringTeam.eq("Home"), "ManpowerHome"] -
        pm_df.loc[pm_df.ScoringTeam.eq("Home"), "ManpowerAway"])
    
    pm_df.loc[pm_df.ScoringTeam.eq("Away"), "MD"] = (
        pm_df.loc[pm_df.ScoringTeam.eq("Away"), "ManpowerAway"] -
        pm_df.loc[pm_df.ScoringTeam.eq("Away"), "ManpowerHome"])
    
    # Calculate plus-minus from the away team perspective
    pm_df["PlusMinusAway"] = np.select(
        [
            # Traditional
            pm_df["ScoringTeam"].ne("Home") & pm_df.MD.le(0),
            pm_df["ScoringTeam"].ne("Home") & pm_df.MD.gt(0),
            pm_df["ScoringTeam"].eq("Home") & pm_df.MD.gt(0),
            pm_df["ScoringTeam"].eq("Home") & pm_df.MD.le(0),
        ], 
        [
            1, 
            0, 
            0, 
            -1,
        ], 
        default=999
    )
    
    # Calculate plus-minus from the home team perspective
    pm_df["PlusMinusHome"] = np.select(
        [     
            # Traditional
            pm_df["ScoringTeam"].eq("Home") & pm_df.MD.le(0),
            pm_df["ScoringTeam"].eq("Home") & pm_df.MD.gt(0),
            pm_df["ScoringTeam"].ne("Home") & pm_df.MD.gt(0),
            pm_df["ScoringTeam"].ne("Home") & pm_df.MD.le(0),
        ], 
        [
            1, 
            0, 
            0, 
            -1,
        ], 
        default=999
    )

    
    # Get players for both team and plusminus values
    plusminus = pd.concat([pm_df[home_players + away_players], 
                           pm_df[["GameId", "TotalElapsedTime", 
                                  "PlusMinusHome", "PlusMinusAway",
                                  "ScoringTeam"]]], 
                          axis=1)
    
    # Rename the columns and convert from wide to long
    plusminus = plusminus.rename(columns={**{player_nr: "Home" for player_nr in home_players}, 
                                          **{player_nr: "Away" for player_nr in away_players}
                                          }).melt(id_vars=("GameId", "TotalElapsedTime", 
                                                           "PlusMinusHome", "PlusMinusAway",
                                                           "ScoringTeam"),
                                                  var_name="Side",
                                                  value_name="PlayerId")
    
    # Drop NA values for PlayerId
    plusminus = plusminus.loc[~plusminus["PlayerId"].isna()]
                                  
    # Calculate plus-minus while playing home and away respectively                            
    home_pm = plusminus[plusminus["Side"] == "Home"].groupby(["PlayerId"])\
        [["PlusMinusHome"]].sum().reset_index()\
            .rename(columns={"PlusMinusHome": "PlusMinus"})
            
    away_pm = plusminus[plusminus["Side"] == "Away"].groupby(["PlayerId"])\
        [["PlusMinusAway"]].sum().reset_index()\
            .rename(columns={"PlusMinusAway": "PlusMinus"})    
            
    # Compute total plusminus per player
    all_plusminus = home_pm.append(away_pm).groupby("PlayerId").sum().\
       reset_index()
       
    return all_plusminus


def compute_possession_metrics(df: DataFrame, players: DataFrame) -> DataFrame:
    """
    Compute proxies for puck possesion, e.g., Corsi/Fenwick differential
    while on the ice.

    Parameters
    ----------
    df : DataFrame
        Original play by play data in wide format.
    players : DataFrame
        Data frame of all players and their metadata.

    Returns
    -------
    possession : DataFrame
        Information over puck possession while on and off the ice for each player.

    """
    # If the file already exists: read and return
    if os.path.isfile("../Data/possession.csv"):
        possession = pd.read_csv("../Data/possession.csv")
        return possession
    
    # Read xG data (if available)
    try: 
        df = pd.read_csv("../Data/xg_pbp.csv")
        xg_data = True
    except FileNotFoundError:
        xg_data = False
    
    # Get all shots in regulation or overtime
    df_shot = df.loc[df.EventType.isin(["SHOT", "BLOCKED SHOT", "MISSED SHOT", "GOAL"]) & 
                     df.PeriodNumber.le(4)].copy()
    
    # Find all blocked shots
    BS = df_shot.EventType.eq("BLOCKED SHOT")
    
    # Check if the blocking team was the home team
    home_team = df_shot.Team.eq(df_shot.HomeTeamName)
    
    # If the home team blocked the shot; set team name to away
    df_shot.loc[BS & home_team, "Team"] = df_shot.loc[BS & home_team, "AwayTeamName"]
    
    # If the away team blocked the shot; set team name to home
    df_shot.loc[BS & ~home_team, "Team"] = df_shot.loc[BS & ~home_team, "HomeTeamName"]
    
    # Find all goalkeepers
    goalkeepers = players.loc[players.Position.eq("G")]
    
    # Dictionary to store results
    corsi_player_dict = {}
    
    # All players who participated in the latest season
    season_players = players.loc[players.groupsEarnedThruSeason.eq(20212022) &
                                 ~players.PlayerId.isin(goalkeepers.PlayerId)]
    
    # Loop over all players who partcipated in the given season
    for player_id in tqdm(season_players.PlayerId.unique()):    
        # Check if the player was on the ice as an away player
        on_ice_away = (df_shot["AwayPlayerId1"].eq(player_id) | df_shot["AwayPlayerId2"].eq(player_id) | 
                       df_shot["AwayPlayerId3"].eq(player_id) | df_shot["AwayPlayerId4"].eq(player_id) |
                       df_shot["AwayPlayerId5"].eq(player_id) | df_shot["AwayPlayerId6"].eq(player_id)
                       )
        # Check if the player was on the ice as a home player
        on_ice_home = (df_shot["HomePlayerId1"].eq(player_id) | df_shot["HomePlayerId2"].eq(player_id) |
                       df_shot["HomePlayerId3"].eq(player_id) | df_shot["HomePlayerId4"].eq(player_id) | 
                       df_shot["HomePlayerId5"].eq(player_id) | df_shot["HomePlayerId6"].eq(player_id) 
                       )
    
        # If there are no on-ice events
        if all(~on_ice_away) & all(~on_ice_home):
            continue
    
        # All home games the player has participated in
        home_games_played_in = df_shot.loc[on_ice_home, ["HomeTeamName", "GameId"]].drop_duplicates(
            ).rename(columns={"HomeTeamName": "Team"})
        
        # All away games the player has participated in
        away_games_played_in = df_shot.loc[on_ice_away, ["AwayTeamName", "GameId"]].drop_duplicates(
            ).rename(columns={"AwayTeamName": "Team"})
        
        # All teams and games the player has played in
        games_played_in = pd.concat([home_games_played_in, away_games_played_in])
        
        # Get all games the player played in
        plays_in = df_shot.GameId.isin(games_played_in.GameId)
        
        # A series of all False
        team_events = df_shot.Team.eq("")
        opposition_events = df_shot.Team.eq("")
        
        # Go over all pairs of teams and game ids    
        for row in games_played_in.itertuples():
            # All events from the same game
            same_game = df_shot.GameId.eq(row.GameId)
            
            # Same team as the player belongs to
            same_team = df_shot.Team.eq(row.Team)
            
            # All events from the player's team
            same_team_events = df_shot.loc[same_game & same_team]    
            team_events[same_team_events.index] = True
            
            # All events from the opposition's team
            other_team_events = df_shot.loc[same_game & ~same_team]    
            opposition_events[other_team_events.index] = True   
        
        # Check if the home team performed the event
        home_event = df_shot["HomeTeamName"] == df_shot["Team"]
        
        # Check if there are six players for each team
        five_away = sum([~df_shot[f"AwayPlayerId{i}"].isna() for i in range(1, 7)]) == 6
        five_home = sum([~df_shot[f"HomePlayerId{i}"].isna() for i in range(1, 7)]) == 6

        # If the event occurred during five on five
        five_on_five = five_away & five_home

        # If the away goalie was present on the ice 
        away_goalie_on_ice = (df_shot["AwayPlayerId1"].isin(goalkeepers.PlayerId) | 
                              df_shot["AwayPlayerId2"].isin(goalkeepers.PlayerId) | 
                              df_shot["AwayPlayerId3"].isin(goalkeepers.PlayerId) | 
                              df_shot["AwayPlayerId4"].isin(goalkeepers.PlayerId) |
                              df_shot["AwayPlayerId5"].isin(goalkeepers.PlayerId) | 
                              df_shot["AwayPlayerId6"].isin(goalkeepers.PlayerId)
                              )
        
        # If the home goalie was present on the ice 
        home_goalie_on_ice = (df_shot["HomePlayerId1"].isin(goalkeepers.PlayerId) | 
                              df_shot["HomePlayerId2"].isin(goalkeepers.PlayerId) | 
                              df_shot["HomePlayerId3"].isin(goalkeepers.PlayerId) | 
                              df_shot["HomePlayerId4"].isin(goalkeepers.PlayerId) |
                              df_shot["HomePlayerId5"].isin(goalkeepers.PlayerId) | 
                              df_shot["HomePlayerId6"].isin(goalkeepers.PlayerId)
                              )
        
        # Shots for the team with the player on the ice
        shots_for_on = (
            home_goalie_on_ice & away_goalie_on_ice &
            ((on_ice_away & ~home_event) | (on_ice_home &  home_event)) & 
            plays_in & five_on_five
            )
        
        # Shots against the team with the player on the ice
        shots_against_on = (
            home_goalie_on_ice & away_goalie_on_ice &
            ((on_ice_away &  home_event) | (on_ice_home & ~home_event)) & 
            plays_in & five_on_five
            )
        
        # Shots for the team with the player off the ice
        shots_for_off = (
            ~on_ice_away & ~on_ice_home & home_goalie_on_ice & away_goalie_on_ice &
            ((team_events & ~home_event) | (team_events &  home_event)) &
            plays_in & five_on_five
            )
        
        # Shots against the team with the player off the ice
        shots_against_off = (
            ~on_ice_away & ~on_ice_home & home_goalie_on_ice & away_goalie_on_ice &
            ((opposition_events &  home_event) | (opposition_events & ~home_event)) & 
            plays_in & five_on_five
            )
        
        if xg_data:
            # xG for while on and off the ice, respectively
            xg_for_on_ice = df_shot.loc[shots_for_on, "xG"].sum() 
            xg_for_off_ice = df_shot.loc[shots_for_off, "xG"].sum()
            
            # xG against while on and off the ice, respectively
            xg_against_on_ice = df_shot.loc[shots_against_on, "xG"].sum() 
            xg_against_off_ice = df_shot.loc[shots_against_off, "xG"].sum()
            
            # Compute xG differential while on and off the ice, respectively
            xg_on_differential = xg_for_on_ice - xg_against_on_ice   
            xg_off_differential = xg_for_off_ice - xg_against_off_ice
            
            try: 
                # Compute xG% while on and off the ice, respectively
                xg_on_percent = xg_for_on_ice / (xg_for_on_ice + xg_against_on_ice)
                xg_off_percent = xg_for_off_ice / (xg_for_off_ice + xg_against_off_ice)
                
            except ZeroDivisionError:
                xg_on_percent = xg_off_percent = 0
            
            # Compute relative xG%                
            xg_relative = xg_on_percent - xg_off_percent
            
        # Corsi for while on and off the ice, respectively
        corsi_for_on_ice = len(df_shot.loc[shots_for_on])
        corsi_for_off_ice = len(df_shot.loc[shots_for_off])
    
        # Corsi against while on and off the ice, respectively
        corsi_against_on_ice = len(df_shot.loc[shots_against_on])
        corsi_against_off_ice = len(df_shot.loc[shots_against_off])
    
        # Fenwick for while on and off the ice, respectively
        fenwick_for_on_ice = len(df_shot.loc[shots_for_on & ~BS])
        fenwick_for_off_ice = len(df_shot.loc[shots_for_off & ~BS])
        
        # Fenwick against while on and off the ice, respectively
        fenwick_against_on_ice = len(df_shot.loc[shots_against_on & ~BS])
        fenwick_against_off_ice = len(df_shot.loc[shots_against_off & ~BS])
        
        # Compute Corsi and Fenwick differentials for when the player is on the ice
        corsi_on_differential = corsi_for_on_ice - corsi_against_on_ice
        fenwick_on_differential = fenwick_for_on_ice - fenwick_against_on_ice
        
        # Compute Corsi and Fenwick differentials for when the player is off the ice
        corsi_off_differential = corsi_for_off_ice - corsi_against_off_ice
        fenwick_off_differential = fenwick_for_off_ice - fenwick_against_off_ice
        
        try:
            # Compute Corsi% while on the ice
            corsi_on_percent = corsi_for_on_ice / (corsi_for_on_ice + corsi_against_on_ice)
            
            # Compute Fenwick% while on the ice
            fenwick_on_percent = fenwick_for_on_ice / (fenwick_for_on_ice + fenwick_against_on_ice)
        
            # Compute Corsi% while off the ice
            corsi_off_percent = corsi_for_off_ice / (corsi_for_off_ice + corsi_against_off_ice)
            
            # Compute Fenwick% while off the ice
            fenwick_off_percent = fenwick_for_off_ice / (fenwick_for_off_ice + fenwick_against_off_ice)
        
        except ZeroDivisionError:
            corsi_on_percent = fenwick_on_percent = corsi_off_percent = fenwick_off_percent = 0
    
        # Compute relative % for Corsi and Fenwick
        corsi_relative = corsi_on_percent - corsi_off_percent
        fenwick_relative = fenwick_on_percent - fenwick_off_percent
    
        # Specify column names
        columns = ["CFOnIce", "CAOnIce", "CorsiOn",
                   "CFOffIce", "CAOffIce", "CorsiOff",
                   "CFOn%", "CFOff%", "CF%",
                   "FFOnIce", "FAOnIce", "FenwickOn",
                   "FFOffIce", "FAOffIce", "FenwickOff",
                   "FFOn%", "FFOff%", "FF%"]
        
        # Combine all values into a data frame
        player_data = pd.DataFrame([corsi_for_on_ice, corsi_against_on_ice, corsi_on_differential,
                                    corsi_for_off_ice, corsi_against_off_ice, corsi_off_differential,
                                    corsi_on_percent, corsi_off_percent, corsi_relative,
                                    fenwick_for_on_ice, fenwick_against_on_ice, fenwick_on_differential,
                                    fenwick_for_off_ice, fenwick_against_off_ice, fenwick_off_differential,
                                    fenwick_on_percent, fenwick_off_percent, fenwick_relative]).T
        
        if xg_data: 
            # Create a data frame of all xG variables
            xg_variables = pd.DataFrame([xg_for_on_ice, xg_against_on_ice, xg_on_differential, 
                                         xg_for_off_ice, xg_against_off_ice, xg_off_differential, 
                                         xg_on_percent, xg_off_percent, xg_relative]).T
            
            # Append to the data frame
            player_data = pd.concat([player_data, xg_variables], axis=1)
            
        
        # Rename columns
        player_data.columns = columns if not xg_data else columns + ["xGFOnIce", "xGAOnIce", "xGOn",
                                                                     "xGFOffIce", "xGAOffIce", "xGOff",
                                                                     "xGOn%", "xGOff%", "xG%"]
        
        # Save in dictionary
        corsi_player_dict[player_id] = player_data
    
    # Combine all players into one data frame
    possession = pd.concat(corsi_player_dict).reset_index(level=0).rename(
        columns={"level_0": "PlayerId"}).reset_index(drop=True)
    
    # Save as a csv file
    possession.to_csv("../Data/possession.csv", index=False)
    
    return possession


def compute_goals_created(events_long: DataFrame, players: DataFrame) -> DataFrame:
    """
    Compute goals created (GC) for each player, according to the formula
    (Player Goals + 0.5 * Player Assists) * (Team Goals / (Team Goals + 0.5 * Team Assists)).
    Originally from hockey-reference.com.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format.
    players : DataFrame
        Data frame of all players and their metadata.

    Returns
    -------
    goals_created : DataFrame
        Information over goals created for each player.

    """
    # Find all goal events (including both Scorer, Assist & Goalie)
    goal_events = events_long.loc[events_long.EventType.eq("GOAL")].copy()
    
    # Dictionary to store goals created per player
    player_dict = {}
    
    # Find all goalkeepers
    goalkeepers = players.loc[players.Position.eq("G")]
    
    # All players who participated in the latest season
    season_players = players.loc[players.groupsEarnedThruSeason.eq(20212022) &
                                 ~players.PlayerId.isin(goalkeepers.PlayerId)]
    
    # Loop over all players who partcipated in the given season
    for player_id in tqdm(season_players.PlayerId.unique()):
        
        # Get all goal events the player was involved with
        player_events = goal_events.loc[goal_events.PlayerId.eq(player_id)].copy()
        
        # Get all games the player did an action in, and which team he played for
        team_events = events_long.loc[events_long.PlayerId.eq(player_id),
                                      ["GameId", "Team"]].drop_duplicates(["GameId", "Team"])
    
        # Dictionary to store goals created per player and team
        game_dict = {}
        
        # Loop over all games 
        for games in team_events.itertuples():
            
            # The game id of the current game
            game_id = games.GameId
            
            # The team name of the current game
            team = games.Team
    
            # Find all assists the player made in the game
            player_assists = player_events.loc[player_events.PlayerType.eq("Assist") &
                                               player_events.Team.eq(team) &
                                               player_events.GameId.eq(game_id)]
            
            # Find all goals the player scored in the game
            player_goals = player_events.loc[player_events.PlayerType.eq("Scorer") &
                                             player_events.Team.eq(team) &
                                             player_events.GameId.eq(game_id)]
    
            # Compute the player goal creation component
            player_creation = len(player_goals) + 0.5 * len(player_assists)
            
            # Find all assists the player made in the game 
            team_assists = goal_events.loc[goal_events.PlayerType.eq("Assist") &
                                           goal_events.Team.eq(team) &
                                           goal_events.GameId.eq(game_id)]
            
            # Find all goals the player scored in the game
            team_goals = goal_events.loc[goal_events.PlayerType.eq("Scorer") &
                                         goal_events.Team.eq(team) &
                                         goal_events.GameId.eq(game_id)]
            
            # Compute the team goal creation component
            team_creation = len(team_goals) + 0.5 * len(team_assists)
            
            # Create a data frame containing all needed information
            creation = pd.DataFrame([[player_creation, team_creation, len(team_goals)]], 
                                    columns=["PlayerCreation", "TeamCreation", "TeamGoals"])
            
            # Save the result in the dictionary
            game_dict[game_id] = creation
            
        # If the player did not player
        if len(game_dict) == 0:
            continue
        
        # Combine all games into one data frame
        player_data = pd.concat(game_dict).reset_index(level=1, drop=True).reset_index().rename(
            columns={"index": "GameId"})
        
        # Compute goals created (GC) for each player
        goals_created = (player_data.PlayerCreation * player_data.TeamGoals / player_data.TeamCreation).sum()
        
        # Save the result in the dictionary
        player_dict[player_id] = goals_created
        
    # Create a dataframe which contains all players and their goal creation score
    goals_created = pd.DataFrame.from_dict(player_dict, orient="index").reset_index().rename(
        columns={"index": "PlayerId", 0: "GoalsCreated"}).sort_values("GoalsCreated", ascending=False)
    
    return goals_created

