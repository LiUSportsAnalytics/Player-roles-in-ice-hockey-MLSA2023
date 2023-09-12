#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
from playerFeatures import compute_blocked_shots, compute_corsi_and_fenwick, \
    compute_fights, compute_giveaways, compute_goals_and_assists_and_points, \
    compute_hits, compute_penalties_and_pim, compute_shot_types, compute_takeaways, \
    compute_zone_starts, compute_median_coordinates, compute_faceoff_win_percentage, \
    compute_plus_minus, compute_possession_metrics, compute_goals_created, \
    compute_penalty_types, compute_shot_pct
from combinePlayByPlayAndShifts import remove_duplicated_shifts, \
    remove_penalties_that_does_not_make_shorthanded, adjust_time_for_stacked_penalties, \
    compute_manpower, compute_toi_player, compute_toi_team
from typing import Tuple
from pandera.typing import DataFrame


def read_data(season: int=2021) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Read play-by-play and shift data from a given season.

    Parameters
    ----------
    season : int, optional
        The season to consider. 
        The default is 2021.

    Returns
    -------
    df : DataFrame
        Original play by play data in wide format.
    shifts : DataFrame
        ALl shifts throughout the season for all players.
    players : DataFrame
        Data frame of all players and their metadata.

    """
    # Read data
    df = pd.read_csv(f"../Data/PBP/pbp_{season}.csv")
    
    # Read shift data
    shifts = pd.read_csv(f"../Data/Shifts/shifts_{season}.csv")

    # Get all player positions and their ids
    players = pd.read_csv("../Data/players.csv", low_memory=False)[["id", "position", "fullName", 
                                                                    "groupsEarnedThruSeason",
                                                                    "height", "weight"]].rename(
                                                                        columns={"id": "PlayerId", 
                                                                                 "position": "Position"})
    # Remove penalty shootouts
    df = df.loc[df["PeriodNumber"] < 5].copy()
    
    return df, shifts, players


def compute_minutes_played(shifts: DataFrame) -> DataFrame:
    """
    Compute minutes played for an entire season.

    Parameters
    ----------
    shifts : DataFrame
        ALl shifts throughout the season for all players.

    Returns
    -------
    minutes_played : DataFrame
        Data frame of where each row represents a player and their season total 
        of minutes played.

    """
    # Compute seconds played
    minutes_played = shifts.groupby(["PlayerId"])["Duration"].sum().reset_index()
    
    # Compute minutes played
    minutes_played["Minutes"] = minutes_played["Duration"] / 60
    
    return minutes_played


def transform_wide_to_long(df: DataFrame) -> DataFrame:
    """
    Transform the play-by-play data from wide to long, while also only keeping
    the events a player is responsible for.

    Parameters
    ----------
    df : DataFrame
        Original play by play data in wide format.

    Returns
    -------
    events_long : DataFrame
        The play-by-play data in long format.

    """
    # Events performed by a player
    player_events = df.EventType.isin(['FACEOFF', 'HIT', 'SHOT', 'TAKEAWAY', 
                                       'PENALTY', 'BLOCKED SHOT', 'MISSED SHOT', 
                                       'GIVEAWAY', 'GOAL'])
    
    # Specify columns
    cols = ["GameId", "AwayTeamName", "HomeTeamName", "EventNumber", 
            "TotalElapsedTime", "EventType", "Team", "GoalsAgainst", "GoalsFor", 
            "X_adj", "Y_adj", "EmptyNet", "ShotType", "PenaltyType", "PenaltyMinutes", 
            "PlayerType1", "PlayerType2", "PlayerType3",
            "PlayerId1", "PlayerId2", "PlayerId3"]
    
    # Keep only actual player events
    events = df.loc[player_events, cols]
    
    # Convert wide to long
    events_long = events.melt(id_vars=cols[:-3], var_name="Player", value_name="PlayerId")
    
    # Remove rows where played id is NA, i.e., there are no players taking on role 2, 3, or 4
    events_long = events_long.loc[events_long["PlayerId"].notna()]
    
    # Reorder rows
    events_long.sort_values(["GameId", "EventNumber"], inplace=True)
    
    # Reset index
    events_long.reset_index(drop=True, inplace=True)
    
    # Save player type/role in separate column
    for i in range(3, 0, -1):
        events_long.loc[events_long.Player.eq(f"PlayerId{i}"), 
                        "PlayerType"] = events_long.loc[events_long.Player.eq(f"PlayerId{i}"), f"PlayerType{i}"]
    
    # Remove redundant columns
    events_long.drop(["PlayerType1", "PlayerType2", "PlayerType3"], axis=1, inplace=True)
    
    # Remove missed shots with unknown results (= goalkeeper events)
    events_long = events_long.loc[~((events_long["EventType"].eq("MISSED SHOT")) &
                                    (events_long["PlayerType"].eq("Unknown")))]

    # Rename X and Y columns
    events_long.rename(columns={"X_adj": "X", "Y_adj": "Y"}, inplace=True)

    return events_long


def adjust_coordinates(events_long: DataFrame) -> DataFrame:
    """
    Adjust coordinates for events that are of the opposite perspective. That is,
    the coordinates are mirrored for the receiving player.

    Parameters
    ----------
    events_long : DataFrame
        The play-by-play data in long format..

    Returns
    -------
    events_long : DataFrame
        The play-by-play data in long format with adjusted coordinates.

    """
    # Create a copy to avoid changing in-place
    events_long = events_long.copy()
    
    # Coordinates to be flipped 
    flip_coords = (
        (events_long["EventType"].eq("BLOCKED SHOT") & events_long["PlayerType"].eq("Shooter")) |
        (events_long["EventType"].eq("HIT") & events_long["PlayerType"].eq("Hittee")) |
        (events_long["EventType"].eq("PENALTY") & events_long["PlayerType"].eq("DrewBy")) |
        (events_long["EventType"].eq("FACEOFF") & events_long["PlayerType"].eq("Loser")) 
        )
    
    # Flip coordinates to other perspective for the other team
    events_long.loc[flip_coords, "X"] *= -1
    events_long.loc[flip_coords, "Y"] *= -1
    
    # For events that were from the other team perspective
    flip_team = flip_coords | (events_long["PlayerType"].isin(["Goalie"]))
    
    # If the home team did the event
    home_team = events_long["HomeTeamName"].eq(events_long["Team"])
    
    # Correct the team names such that the corresponding player is tagged with the right team
    events_long.loc[flip_team & home_team, "Team"]  = events_long.loc[flip_team & home_team, "AwayTeamName"]
    events_long.loc[flip_team & ~home_team, "Team"] = events_long.loc[flip_team & ~home_team, "HomeTeamName"]
    
    # Convert player id to integer
    events_long["PlayerId"] = events_long["PlayerId"].astype("int64")
    
    return events_long


def team_players(shifts: DataFrame, players: DataFrame, n: int=18, 
                 filter_minutes: bool=True, filter_duplicates: bool=False) -> DataFrame:
    """
    Get all the players per team and keep the top n players w.r.t. TOI.

    Parameters
    ----------
    shifts : DataFrame
        ALl shifts throughout the season for all players.
    players : DataFrame
        Data frame of all players and their metadata.
    n : int, default is 18.
        Number of players to keep.
    filter_minutes : bool, default is True
        If a filter should be placed for minimum 200 minutes played
    filter_duplicates : bool, default is False
        If duplicates (traded players) should be removed and only represent one team.

    Returns
    -------
    top_n_standings : DataFrame
        The n most played players from each team and the team points/GF/GA.

    """
    # Compute seconds played
    team_players_time = shifts.groupby(["TeamName", "PlayerId"])["Duration"].sum().reset_index()
    
    # Compute minutes played
    team_players_time["Minutes"] = team_players_time["Duration"] / 60
    
    # Add player names and positions
    team_players = team_players_time.merge(players.drop(["height", "weight", "groupsEarnedThruSeason"], axis=1), 
                                           left_on="PlayerId", right_on="PlayerId")
    # Select columns    
    team_players = team_players[["PlayerId", "fullName", "Position", "TeamName", "Minutes"]].copy()
    
    # Remove goalkeepers and sort by minutes played per team
    team_skaters = team_players.loc[team_players.Position.ne("G")].copy().sort_values(
        ["TeamName", "Minutes"], ascending=[True, False])
    
    if filter_minutes:
        # Keep only players with minimum 200 minutes played
        team_skaters = team_skaters.loc[team_skaters.Minutes.ge(200)].copy()
    
    if filter_duplicates:
        # Remove duplicates for player ids
        team_skaters = team_skaters.sort_values(["PlayerId", "Minutes"]).drop_duplicates(
            subset=["PlayerId"], keep="last").sort_values(["TeamName", "Minutes"], ascending=[True, False]).copy()
    
    # Keep only the top 18 players w.r.t. TOI per team
    top_n = team_skaters.groupby("TeamName", as_index=False).head(n)
    
    # Create a data frame containing the final standings
    team_standings = pd.DataFrame(
        data=[['Anaheim Ducks', 'Arizona Coyotes', 'Boston Bruins',
               'Buffalo Sabres', 'Calgary Flames', 'Carolina Hurricanes',
               'Chicago Blackhawks', 'Colorado Avalanche',
               'Columbus Blue Jackets', 'Dallas Stars', 'Detroit Red Wings',
               'Edmonton Oilers', 'Florida Panthers', 'Los Angeles Kings',
               'Minnesota Wild', 'Montréal Canadiens', 'Nashville Predators',
               'New Jersey Devils', 'New York Islanders', 'New York Rangers',
               'Ottawa Senators', 'Philadelphia Flyers', 'Pittsburgh Penguins',
               'San Jose Sharks', 'Seattle Kraken', 'St. Louis Blues',
               'Tampa Bay Lightning', 'Toronto Maple Leafs', 'Vancouver Canucks',
               'Vegas Golden Knights', 'Washington Capitals', 'Winnipeg Jets'],
              [76, 57, 107, 75, 111, 116, 68, 119, 81, 98, 74, 104, 122, 99, 113, 55, 
               97, 63, 84, 110, 73, 61, 103, 77, 60, 109, 110, 115, 92, 94, 100, 89],
              [228, 206, 253, 229, 291, 277, 213, 308, 258, 233, 227, 285, 
               337, 235, 305, 218, 262, 245, 229, 250, 224, 210, 269, 211, 
               213, 309, 285, 312, 246, 262, 270, 250],
              [266, 309, 218, 287, 206, 200, 289, 232, 297, 244, 310, 251, 
               242, 232, 249, 317, 250, 302, 231, 204, 264, 294, 222, 261, 
               284, 239, 228, 252, 231, 244, 242, 253]],
        index=["TeamName", "Points", "GF", "GA"]).T
    
    # All columns to convert to numeric
    cols = team_standings.columns.drop("TeamName")

    # Convert columns to numeric
    team_standings[cols] = team_standings[cols].apply(pd.to_numeric, errors="coerce")    
 
    # Add a rank variable
    team_standings["Rank"] = team_standings.Points.rank(method="first", ascending=False)
    
    # Add the top n players with their team's final position
    top_n_standings = top_n.merge(team_standings)
    
    return top_n_standings


if __name__ == "__main__":
    # Read the play by play, shift and player data
    pbp_df, shifts, players = read_data()
    
    # Remove JSON only shifts and shifts with duration 0 or smaller
    shifts = shifts.loc[shifts.Source.ne("JSON") & 
                        shifts.Duration.gt(0)].copy()
    
    # Compute the number of x played
    minutes_played = compute_minutes_played(shifts)

    # Get the top n players per team and the team's final standing
    top_n_standings = team_players(shifts, players, n=18)    

    # Save as csv
    top_n_standings.to_csv("../Data/top_n.csv", index=False)

    # Convert the play by play data from wide to long
    events_long = transform_wide_to_long(pbp_df)
    
    # Adjust the coordinates such that teams are always attacking in the same direction
    events_long = adjust_coordinates(events_long)

    # Compute the number of goals, assists and points
    goals, assists, points = compute_goals_and_assists_and_points(events_long)
    
    # Compute Corsi and Fenwick
    corsi, fenwick = compute_corsi_and_fenwick(events_long)
    
    # Compute the number of giveaways
    giveaways = compute_giveaways(events_long)
    
    # Compute the number of takeaways
    takeaways = compute_takeaways(events_long)
    
    # Compute faceoff win percentage
    faceoff_win_pct = compute_faceoff_win_percentage(events_long)

    # Compute the number of hits
    hits = compute_hits(events_long)
    
    # Compute the number of penalties and penalty minutes
    penalties, pim = compute_penalties_and_pim(events_long)
    
    # Compute zone starting percentage
    zone_starts = compute_zone_starts(pbp_df)
    
    # Compute the number of figths
    fighting = compute_fights(events_long)
    
    # Compute average coordinates for different event groups
    avg_coords = compute_median_coordinates(events_long)
    
    # Compute the number of shots by type
    shot_types = compute_shot_types(events_long)
    
    # Compute shooting percenages
    shot_pct = compute_shot_pct(events_long)
    
    # Compute the number of penalties by group
    penalty_types = compute_penalty_types(events_long)
    
    # Compute the number of blocked shots
    blocked_shots = compute_blocked_shots(events_long)
    
    # Compute +/-
    plus_minus = compute_plus_minus(pbp_df)
    
    # Remove duplicated shifts
    game_shifts_dict = remove_duplicated_shifts(shifts)
    
    # Remove penalties that does not affect manpower
    penalty_goals = remove_penalties_that_does_not_make_shorthanded(pbp_df)

    # Adjust the start/end time for stacked penalties
    all_penalties = adjust_time_for_stacked_penalties(penalty_goals, pbp_df)

    # Compute the manpower per team during all points in the game
    manpower = compute_manpower(all_penalties, pbp_df)
    
    # Compute the TOI during various manpower situations for all players
    toi_player = compute_toi_player(manpower, game_shifts_dict, players)
    
    # Compute the TOI during various manpower situations for all team
    toi_team = compute_toi_team(manpower)
    
    # Combine player and team TOI
    toi = toi_player.merge(toi_team, on=["GameId", "Team"],
                           suffixes=("_player", "_team"),
                           how="inner")
    
    # Compute goals created
    goals_created = compute_goals_created(events_long, players)
    
    # Compute PP TOI % for the entire season
    PP_TOI_pct = toi.groupby("PlayerId").agg(
        {"PP TOI_player": sum, "PP TOI_team": sum}).apply(
        lambda x: x[0] / x[1], axis=1).reset_index().rename(columns={0: "PP%"})
    
    # Compute PP TOI % for the entire season
    SH_TOI_pct = toi.groupby("PlayerId").agg(
        {"SH TOI_player": sum, "SH TOI_team": sum}).apply(
        lambda x: x[0] / x[1], axis=1).reset_index().rename(columns={0: "SH%"})
    
    # Compute EV TOI % for the entire season
    EV_TOI_pct = toi.groupby("PlayerId").agg(
        {"EV TOI_player": sum, "EV TOI_team": sum}).apply(
        lambda x: x[0] / x[1], axis=1).reset_index().rename(columns={0: "EV%"})
            
    # Compute the % of TOI a player participated in during 5 on 5 
    five_on_five_TOI_pct = toi.groupby("PlayerId").agg(
        {"5v5_player": sum, "5v5_team": sum}).apply(
        lambda x: x[0] / x[1], axis=1).reset_index().rename(columns={0: "5v5%"})
    
    # Compute the % of TOI a player participated in during overtime
    OT_TOI_pct = toi.groupby("PlayerId").agg(
        {"OT TOI_player": sum, "OT TOI_team": sum}).apply(
        lambda x: x[0] / x[1] if x[1] > 0 else 0, axis=1).reset_index().rename(columns={0: "OT%"})
           
    # Compute total TOI for each player
    toi_season = toi.groupby("PlayerId")[["EV TOI_player", "PP TOI_player", 
                                          "SH TOI_player", "OT TOI_player"]].sum(
                                              ).sum(axis=1).reset_index().rename(
                                              columns={0: "TOI"})
                        
    # Compute TOI during 5v5 
    five_on_five_TOI = toi.groupby("PlayerId")["5v5_player"].sum().reset_index().rename(
        columns={"5v5_player": "5v5 TOI"})
            
    # Combine all statistics into one
    data = points.merge(corsi, how="outer").merge(fenwick, how="outer").merge(
        giveaways, how="outer").merge(takeaways, how="outer").merge(
        faceoff_win_pct[["PlayerId", "FaceoffsWon", "FaceoffWonPct"]], how="outer").merge(
        hits[["PlayerId", "Hits", "HitNet"]], how="outer").merge(
        penalties[["PlayerId", "Penalties", "PenaltyNet"]], how="outer").merge(
        pim, how="outer").merge(
        zone_starts[["PlayerId", "OZS%", "DZS%"]], how="outer").merge(fighting, how="outer").merge(
        avg_coords.drop(["XmedianFaceoff", "YmedianFaceoff",
                         "XmedianOther", "YmedianOther"], axis=1), how="outer").merge(
        shot_types, how="outer").merge(blocked_shots, how="outer").merge(
        blocked_shots, how="outer").merge(plus_minus, how="outer").merge(
        PP_TOI_pct, how="outer").merge(SH_TOI_pct, how="outer").merge(
        EV_TOI_pct, how="outer").merge(OT_TOI_pct, how="outer").merge(
        five_on_five_TOI_pct, how="outer").merge(goals_created, how="outer").merge(
        toi_season, how="outer").merge(five_on_five_TOI, how="outer").merge(
        players[["PlayerId", "height", "weight"]], how="left").merge(
        penalty_types, how="outer").merge(shot_pct, how="outer")     
                                                 
    # Add filter for minimum minutes played
    sufficient_playtime = minutes_played.loc[minutes_played["Minutes"].ge(200), "PlayerId"]

    # Keep only players with minimum 200 minutes played
    data = data.loc[data.PlayerId.isin(sufficient_playtime)]
         
    # Remove goalkeepers
    data = data.loc[~data.PlayerId.isin(players.loc[players.Position.eq("G"), "PlayerId"])]
           
    # Read/compute possession data
    possession = compute_possession_metrics(pbp_df, players)
    
    # Combine data and possession data
    final_data = data.merge(possession.rename(columns={"CFOn%": "CF%", "CF%": "CF%Rel",
                                                       "FFOn%": "FF%", "FF%": "FF%Rel",
                                                       "xGOn%": "xG%", "xG%": "xG%Rel"})[
        ["PlayerId", "CorsiOn", "CF%", "CF%Rel",
         "FenwickOn", "FF%", "FF%Rel", "xGOn", "xG%", "xG%Rel"]])
    
    # Read xg data
    xg = pd.read_csv("../Data/xg.csv").drop(["fullName"], axis=1)

    # Add xG data
    final_data = final_data.merge(xg, how="left", on="PlayerId",
                                  suffixes=("", "_xg")).fillna(0)
    
    # Save as csv
    final_data.set_index("PlayerId").to_csv("../Data/PCA.csv", index=True)
        
    
