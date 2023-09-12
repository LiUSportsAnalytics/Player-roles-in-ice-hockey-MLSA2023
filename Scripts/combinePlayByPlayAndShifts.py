#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from itertools import groupby, chain
from typing import List, Dict, Tuple
from pandera.typing import DataFrame


def get_rle(col: str) -> List[int]:
    """
    Create a run-length encoding for a given column.
    Parameters
    ----------
    col : str
        The column to generate a run-length encoding for.
    Returns
    -------
    rle : list
        A list of repeated items (rle) for each value.
    """
    # Create run-length encoding of col
    rle = [sum(1 for i in g) for k, g in groupby(col)]
    
    # Convert to a list of repeated items to match length of df
    rle = list(chain(*[i * [i] for i in rle]))
    
    return rle


def remove_duplicated_shifts(game_shifts_df: DataFrame) -> Dict[int, DataFrame]:
    """
    Remove or flag shifts that are/could be duplicates.

    Parameters
    ----------
    game_shifts_df : DataFrame
        Data frame containing all shifts from a set of game(s).

    Returns
    -------
    game_shifts_dict : dict
        Dictionary containing the game id as a key and all game shifts as the 
        values, where the duplicates have been removed or flagged as possible 
        duplicates.

    """
    # Dictionary to store game shifts
    game_shifts_dict = {}
    
    for gameId in game_shifts_df["GameId"].unique():
        # Find all shifts from the game
        all_game_shifts_bool = game_shifts_df["GameId"].isin([gameId])
        
        # Get all shifts from the game 
        all_game_shifts = game_shifts_df.loc[all_game_shifts_bool].copy()

        # Create a new column to flag for potential to flag for potential extra players
        all_game_shifts["PotentialExtraPlayer"] = False

        # See if the team has cases where an extra player is tagged
        for team, team_shifts in all_game_shifts.groupby("TeamName"):
            # The shift started at the start/end of the game
            start_of_game = team_shifts["ShiftStart"] == 0
            end_of_game = team_shifts["ShiftEnd"].isin([1200, 2400, 3600])
            
            # The start time of the shift is not present among team end times
            extra_start_player = ~team_shifts["ShiftStart"].isin(team_shifts["ShiftEnd"])

            # The end time of the shift is not present among team start times
            extra_end_player = ~team_shifts["ShiftEnd"].isin(team_shifts["ShiftStart"])
            
            # The shift occurred in regulation
            regulation = team_shifts["PeriodNumber"] < 4

            # Players who played an "extra" shift which they did not actually play
            extra = team_shifts.loc[((extra_start_player & ~start_of_game) | 
                                     (extra_end_player   & ~end_of_game)# | (little_rest)
                                     ) & regulation]
            
            if len(extra) > 0:
                # Specify a potential extra shift
                all_game_shifts.loc[extra.index, "PotentialExtraPlayer"] = True

        # Remove shifts of duration 0 or smaller
        all_game_shifts.loc[all_game_shifts["Duration"].ge(0)]        

        # Duplicated end time
        dupe_end = all_game_shifts.loc[all_game_shifts.duplicated(["PlayerId", "ShiftEnd"], keep=False)].copy()
        
        # Sort by duration, keep only the shortest
        dupe_end.sort_values(["PlayerId", "ShiftEnd", "Duration"], inplace=True)

        # Remove the longest duplicate per group
        drop_end_times = dupe_end.duplicated(["PlayerId", "ShiftEnd"], keep="first")
        
        # Drop the duplicated rows from the data
        all_game_shifts = all_game_shifts.drop(drop_end_times[drop_end_times].index)
        
        # Duplicated start time
        dupe_start = all_game_shifts.loc[all_game_shifts.duplicated(["PlayerId", "ShiftStart"], keep=False)].copy()
        
        # Sort by duration, keep only the shortest
        dupe_start.sort_values(["PlayerId", "ShiftStart", "Duration"], inplace=True)

        # Remove the longest duplicate per group
        drop_start_times = dupe_start.duplicated(["PlayerId", "ShiftStart"], keep="first")
        
        # Drop the duplicated rows from the data
        all_game_shifts = all_game_shifts.drop(drop_start_times[drop_start_times].index)

        # Store game shifts in dictionary
        game_shifts_dict[gameId] = all_game_shifts

    return game_shifts_dict


def remove_penalties_that_does_not_make_shorthanded(pbp_data: DataFrame) -> DataFrame:
    """
    Adjust for, and remove, all penalties that are coincidental and/or do not 
    make a team go shorthanded. E.g., fighting, coincidental minors during 4v4 etc.

    Parameters
    ----------
    pbp_data : DataFrame
        Play by play data from a given season.

    Returns
    -------
    penalty_goals : DataFrame
        A data frame with all penalties and (powerplay) goals.

    """
    # Select all goal and penalty events
    penalty_goals = pbp_data.loc[pbp_data.EventType.isin(["PENALTY", "GOAL"]) &
                                 pbp_data.PeriodNumber.le(4) & 
                                 (pbp_data.PenaltyMinutes.gt(0) | pbp_data.Manpower.eq("PP")) &
                                 pbp_data.PenaltyType.ne("Penalty Shot"),
                                 ["GameId", "AwayTeamName", "HomeTeamName", "EventType", 
                                  "Team", "Manpower", 
                                  "Player1", "PenaltyType", "PenaltyMinutes", 
                                  "TotalElapsedTime"]]
    
    # Keep a copy of penalty minutes
    penalty_goals["UnadjustedPenaltyMinutes"] = penalty_goals["PenaltyMinutes"]
    
    # Drop misconducts penalties
    penalty_goals = penalty_goals.loc[~penalty_goals.PenaltyType.isin(["Misconduct", 
                                                                       "Game Misconduct"])]
    
    # Remove other 0/10 minutes penalties
    penalty_goals = penalty_goals.loc[~penalty_goals.PenaltyMinutes.eq(0) &
                                      ~(penalty_goals.PenaltyMinutes.eq(10) &
                                        penalty_goals.PenaltyType.ne("Match penalty"))].copy()
    
    # Sort values
    penalty_goals.sort_values(["GameId", "TotalElapsedTime", "PenaltyMinutes", "PenaltyType"], inplace=True)
    
    # Find two players getting the same major penalty at the same time (e.g. Fighting)
    double_major_penalty = (penalty_goals.shift(-1).GameId.eq(penalty_goals.shift(0).GameId) & 
                            penalty_goals.shift(-1).Team.ne(penalty_goals.shift(0).Team) & 
                            penalty_goals.shift(-1).PenaltyMinutes.eq(penalty_goals.shift(0).PenaltyMinutes) &
                            penalty_goals.shift(-1).TotalElapsedTime.eq(penalty_goals.shift(0).TotalElapsedTime) & 
                            penalty_goals.shift(-1).PenaltyMinutes.eq(5) &
                            penalty_goals.shift(-1).PenaltyType.eq("Fighting")) # This might not always work?
    
    # Get both penalties
    double_majors = penalty_goals.loc[double_major_penalty | double_major_penalty.shift(1)]
    
    # Remove both penalties as they don't affect the team
    penalty_goals.drop(double_majors.index, inplace=True)
    
    # Sort values for determining RLE
    penalty_goals.sort_values(["GameId", "TotalElapsedTime", "Team", "PenaltyMinutes"], inplace=True)
    
    # Find all penalties
    penalty_cond = penalty_goals.EventType.eq("PENALTY")
    
    # Compute the RLE for time and duration for penalties
    penalty_goals.loc[penalty_cond, "TimeRLE"] = get_rle(penalty_goals.loc[penalty_cond].TotalElapsedTime)
    penalty_goals.loc[penalty_cond, "DurationRLE"] = get_rle(penalty_goals.loc[penalty_cond].PenaltyMinutes)

    # Find coincidental penalties
    coincidental = penalty_goals.loc[
        ((penalty_goals.TimeRLE.gt(2) & penalty_goals.TotalElapsedTime.lt(3600)) |
         (penalty_goals.TimeRLE.ge(2) & penalty_goals.TotalElapsedTime.ge(3600))) &
        penalty_goals.DurationRLE.gt(1)].copy()
    
    # Update and compute the RLE for total elapsed time and team
    coincidental["TimeRLE"] = get_rle(coincidental.TotalElapsedTime)
    coincidental["TeamRLE"] = get_rle(coincidental.Team)

    # Drop the combinations that has a major or higher attached to it
    # coincidental = coincidental.loc[~coincidental.TimeRLE.le(2)]

    # Sort values
    coincidental.sort_values(["GameId", "TotalElapsedTime", "TimeRLE", "TeamRLE"], inplace=True)

    # The indices to drop after the loop
    index_to_drop = pd.Int64Index([])
    for idx, group in coincidental.groupby(["GameId", "TotalElapsedTime", "TimeRLE"]): 
        # The lowest RLE among the two teams
        min_team_rle = int(group.TeamRLE.min())

        # The highest RLE among the two teams
        max_team_rle = int(group.TeamRLE.max())
        
        # If the max team RLE and time RLE are the same (they shouldnt be)
        if max_team_rle == int(group.TimeRLE.max()):
            max_team_rle -= 1
        
        # If the RLE's are the same for both teams (2+2=4) remove both
        if max_team_rle == min_team_rle:
            n_to_drop = (max_team_rle + min_team_rle)
        # Otherwise keep only the final one
        else:
            n_to_drop = (max_team_rle + min_team_rle) - 1 
        
        # Find the indices of the rows to drop
        group_index_to_drop = group.head(n_to_drop).index
        
        # Add indices to the list to be dropped
        index_to_drop = index_to_drop.union(group_index_to_drop)
                
    # Drop all unwanted indices
    penalty_goals.drop(index_to_drop, inplace=True)
    
    # One penalty is given two simultaneous penalties
    double_penalty_player = (penalty_goals.shift(-1).Player1.eq(penalty_goals.shift(0).Player1) &
                             penalty_goals.shift(-1).TotalElapsedTime.eq(penalty_goals.shift(0).TotalElapsedTime) & 
                             penalty_goals.shift(-1).PenaltyMinutes.eq(penalty_goals.shift(0).PenaltyMinutes) &
                             penalty_goals.shift(-1).PenaltyMinutes.eq(2))

    # Instead of having two separate penalties, combine them together
    penalty_goals.loc[double_penalty_player, "PenaltyMinutes"] = 4

    # Remove the other double penalty
    penalty_goals = penalty_goals.loc[~double_penalty_player.shift(1, fill_value=False)]

    # Set 5 penalty minutes for match penalties
    penalty_goals.loc[penalty_goals.PenaltyMinutes.gt(5), "PenaltyMinutes"] = 5

    # Compute the theoretical end of all penalties
    penalty_goals["PenaltyEndTime"] = penalty_goals["TotalElapsedTime"] + 60 * penalty_goals["PenaltyMinutes"]
    
    # Games that end up in overtime
    overtime_games = pbp_data.loc[pbp_data.PeriodNumber.ge(4), "GameId"].unique()

    # Penalties do not carry over to overtime if there is none
    penalty_goals.loc[penalty_goals.PenaltyEndTime.gt(3600) &
                  penalty_goals.TotalElapsedTime.le(3600) &
                  ~penalty_goals.GameId.isin(overtime_games), "PenaltyEndTime"] = 3600
    
    # Sort values
    penalty_goals.sort_values(["GameId", "TotalElapsedTime", "PenaltyMinutes", "PenaltyType"], inplace=True)

    # If two (or more) penalties are called at the same time    
    penalty_goals["SimultaneousPenalty"] = (
        (penalty_goals.GameId.shift(-1).eq(penalty_goals.GameId.shift(0)) &
         penalty_goals.Team.shift(-1).ne(penalty_goals.Team.shift(0)) & 
         penalty_goals.TotalElapsedTime.shift(-1).eq(penalty_goals.TotalElapsedTime.shift(0)) &
         penalty_goals.PenaltyMinutes.shift(-1).eq(penalty_goals.PenaltyMinutes.shift(0)) &
         penalty_goals.PenaltyMinutes.shift(-1).gt(2)
         ) |
        (penalty_goals.GameId.shift(1).eq(penalty_goals.GameId.shift(0)) &
         penalty_goals.Team.shift(1).ne(penalty_goals.Team.shift(0)) & 
         penalty_goals.TotalElapsedTime.shift(1).eq(penalty_goals.TotalElapsedTime.shift(0))  &
         penalty_goals.PenaltyMinutes.shift(1).eq(penalty_goals.PenaltyMinutes.shift(0)) &
         penalty_goals.PenaltyMinutes.shift(1).gt(2)
         )
        )
    
    # Find simultaneous fighting penalties that remain
    sim_fight = penalty_goals.loc[penalty_goals.PenaltyType.eq("Fighting") &
                                  penalty_goals.SimultaneousPenalty]
    
    # Remove simultaneous fighting penalties that remain
    penalty_goals.drop(sim_fight.index, inplace=True)
    
    # 2 + 2 and a double minor are cancelled out
    coincident_double = penalty_goals.loc[penalty_goals.SimultaneousPenalty & 
                                          penalty_goals.PenaltyMinutes.eq(4)]
    
    # Remove coincidental double minors
    penalty_goals.drop(coincident_double.index, inplace=True)
    
    # Penalties within the final 5 minutes of regulation or overtime
    final_five = penalty_goals.loc[penalty_goals.TotalElapsedTime.ge(3300) &
                                   penalty_goals.EventType.eq("PENALTY")]
   
    # Find the pairs of penalties that occur in the final 5 minutes and are uneven in severity
    uneven_penalties_final_five = (
        (final_five.GameId.shift(-1).eq(final_five.GameId.shift(0)) &
         final_five.Team.shift(-1).ne(final_five.Team.shift(0)) & 
         final_five.TotalElapsedTime.shift(-1).eq(final_five.TotalElapsedTime.shift(0)) &
         final_five.PenaltyMinutes.shift(-1).ne(final_five.PenaltyMinutes.shift(0))
         ) |
        (final_five.GameId.shift(1).eq(final_five.GameId.shift(0)) &
         final_five.Team.shift(1).ne(final_five.Team.shift(0)) & 
         final_five.TotalElapsedTime.shift(1).eq(final_five.TotalElapsedTime.shift(0))  &
         final_five.PenaltyMinutes.shift(1).ne(final_five.PenaltyMinutes.shift(0))
         )
        )
    
    # Find the penalties that occur otherwise (time-wise) and has uneven coincidental penalties
    uneven_penalties_other = (
        (penalty_goals.GameId.shift(-1).eq(penalty_goals.GameId.shift(0)) &
         penalty_goals.Team.shift(-1).ne(penalty_goals.Team.shift(0)) & 
         penalty_goals.TotalElapsedTime.lt(3300) &
         penalty_goals.TotalElapsedTime.shift(-1).eq(penalty_goals.TotalElapsedTime.shift(0)) &
         penalty_goals.PenaltyMinutes.shift(-1).ne(penalty_goals.PenaltyMinutes.shift(0)) &
         penalty_goals.PenaltyMinutes.shift(-1).le(4) & 
         penalty_goals.PenaltyMinutes.le(4)
         ) |
        (penalty_goals.GameId.shift(1).eq(penalty_goals.GameId.shift(0)) &
         penalty_goals.Team.shift(1).ne(penalty_goals.Team.shift(0)) & 
         penalty_goals.TotalElapsedTime.lt(3300) &
         penalty_goals.TotalElapsedTime.shift(1).eq(penalty_goals.TotalElapsedTime.shift(0))  &
         penalty_goals.PenaltyMinutes.shift(1).ne(penalty_goals.PenaltyMinutes.shift(0)) &
         penalty_goals.PenaltyMinutes.shift(1).le(4) & 
         penalty_goals.PenaltyMinutes.le(4)
         )
        )

    # Keep only the rows that are True
    uneven_penalties_other_true = uneven_penalties_other[uneven_penalties_other]
    uneven_penalties_final_five_true = uneven_penalties_final_five[uneven_penalties_final_five]

    # Combine the indices
    uneven_index = uneven_penalties_final_five_true.index.union(uneven_penalties_other_true.index)

    # Get all uneven pairs in the final 5 minutes
    uneven_penalties = penalty_goals.loc[uneven_index].copy()
    
    for idx, penalty_pairs in uneven_penalties.groupby(["GameId", "TotalElapsedTime"]):
        # Get the shortest duration among the two penalties
        lowest_penalty = penalty_pairs.UnadjustedPenaltyMinutes.min()
        
        # Get the longest duration among the two penalties
        highest_penalty = penalty_pairs.UnadjustedPenaltyMinutes.max()
        
        # If the penalties has been adjusted and there is actually no double minor/major
        if lowest_penalty == highest_penalty:
            continue
        
        # Compute the difference in duration between the penalties
        penalty_difference = highest_penalty - lowest_penalty
        
        # Get the index of the penalty with the shortest duration
        penalty_with_lowest = penalty_pairs.iloc[[np.argmin(penalty_pairs.UnadjustedPenaltyMinutes)]].index
       
        # Get the index of the penalty with the longest duration
        penalty_with_highest = penalty_pairs.iloc[[np.argmax(penalty_pairs.UnadjustedPenaltyMinutes)]].index
        
        # For the longest penalty: update the penalty minutes and penalty end
        penalty_goals.loc[penalty_with_highest, "PenaltyMinutes"] = penalty_difference
        penalty_goals.loc[penalty_with_highest, "PenaltyEndTime"] = idx[1] + 60 * penalty_difference
        
        # For the shortest penalty: remove
        penalty_goals.drop(penalty_with_lowest, inplace=True)
        
    # Add column to indicate if penalty was ended by a goal
    penalty_goals["EndedByGoal"] = 0
    
    # Loop over all goals
    for goal in penalty_goals.loc[penalty_goals.EventType.eq("GOAL")].itertuples():
        # Id of the game
        game_id = goal.GameId
        
        # The team that scored the goal
        scoring_team = goal.Team
        
        # The time the goal was scored
        scoring_time = goal.TotalElapsedTime
        
        # If the home team was the scoring tema
        home_team_scored = goal.HomeTeamName == scoring_team
        
        # The condceding team
        conceding_team = goal.AwayTeamName if home_team_scored else goal.HomeTeamName
        
        # All penalties in the game
        penalties = (penalty_goals.GameId.eq(game_id) & 
                     penalty_goals.Team.eq(conceding_team) &
                     penalty_goals.EventType.eq("PENALTY") &
                     ~penalty_goals.SimultaneousPenalty)
        
        # If the goal was scored within the start and end of a powerplay
        within_time = (penalty_goals.GameId.eq(game_id) & 
                       penalty_goals.PenaltyEndTime.gt(scoring_time) & 
                       penalty_goals.TotalElapsedTime.le(scoring_time)) 
        
        try:
            # First penalty
            first_penalty_index = penalty_goals.loc[penalties & within_time].index[0]
            p = penalty_goals.loc[first_penalty_index, "PenaltyMinutes"]
            
            # If there is a double minor and a goal is scored
            if scoring_time + 120 > penalty_goals.loc[first_penalty_index, "PenaltyEndTime"]:    
                scored_during_the_first_penalty_during_double_minor = False
            else:
                scored_during_the_first_penalty_during_double_minor = True
                
            if p == 2 or (p == 4 and not scored_during_the_first_penalty_during_double_minor):
                # Remove the penalty
                penalty_goals.loc[first_penalty_index, "PenaltyEndTime"] = scoring_time
                
                # Add to indicate it was ended by a goal
                penalty_goals.loc[first_penalty_index, "EndedByGoal"] = 1
                
            elif p == 4 and scored_during_the_first_penalty_during_double_minor:
                # Remove the first minor penalty only
                penalty_goals.loc[first_penalty_index, "PenaltyEndTime"] = scoring_time + 120
                penalty_goals.loc[first_penalty_index, "PenaltyMinutes"] = 2
                
                # Add to indicate it was ended by a goal
                penalty_goals.loc[first_penalty_index, "EndedByGoal"] += 1
                
            elif p in [3, 5]:
                # Add to indicate it was a goal was scored
                penalty_goals.loc[first_penalty_index, "EndedByGoal"] += 1
                # Major penalties are not interruped
                continue
            
        except IndexError:
            # In case a goal is tagged as being Powerplay, but there is no penalty to kill
            continue
    
    # Create column to indicate if a penalty occurred while another penalty was already present
    penalty_goals["PenaltyDuringPenalty"] = False
    
    # Find penalties occuring during another penalty
    for penalty in penalty_goals.loc[penalty_goals.EventType.eq("PENALTY")].itertuples():
        # Id of the game
        game_id = penalty.GameId
        
        # The team that scored the goal
        scoring_team = penalty.Team
        
        # The time the penalty occurred
        penalty_tme = penalty.TotalElapsedTime
        
        # All penalties in the game
        penalties = (penalty_goals.GameId.eq(game_id) & 
                     penalty_goals.EventType.eq("PENALTY"))
        
        # If the penalty occurred scored within the start and end of a powerplay
        within_time = (penalty_goals.GameId.eq(game_id) & 
                       penalty_goals.PenaltyEndTime.ge(penalty_tme) & 
                       penalty_goals.TotalElapsedTime.lt(penalty_tme)) 
        
        if len(penalty_goals.loc[penalties & within_time]) > 0:
            penalty_goals.loc[penalty.Index, "PenaltyDuringPenalty"] = True
            
    # Coincidental penalties during overtime
    coincident_penalty_during_overtime = penalty_goals.loc[penalty_goals.SimultaneousPenalty &
                                                           penalty_goals.TotalElapsedTime.gt(3600)]
    # Remove the coincidental penalties that occur during overtime
    penalty_goals.drop(coincident_penalty_during_overtime.index, inplace=True)

    # Coincidental penalties during another penalty
    coincident_penalty_during_penalty = penalty_goals.loc[penalty_goals.SimultaneousPenalty &
                                                          penalty_goals.PenaltyDuringPenalty]
    
    # Remove the coincidental penalties that occur during another penalty
    penalty_goals.drop(coincident_penalty_during_penalty.index, inplace=True)
        
    return penalty_goals 


def adjust_time_for_stacked_penalties(penalty_goals: DataFrame, 
                                      pbp_data: DataFrame) -> DataFrame:
    """
    Adjust the start and end time of penalties that are stacked and do not begin
    counting down until a previous penalty has expired.

    Parameters
    ----------
    penalty_goals : DataFrame
        A data frame with all penalties and (powerplay) goals.
    pbp_data : DataFrame
        Play by play data from a given season.

    Returns
    -------
    penalties : DataFrame
        A data frame with all penalties, alongside their start, end and duration.

    """
    # Get all penalties
    penalties = penalty_goals.loc[penalty_goals.EventType.eq("PENALTY")].copy()

    # Initialize NaN values for number of players
    penalties["AwayPlayers"] = np.nan
    penalties["HomePlayers"] = np.nan
    
    # Placeholder values
    current_game_id = 0
    penalty_end_time = 0
    
    for penalty in penalties.itertuples():
        # If the game id has changed or the penalty has no other penalty running simultaneously
        if penalty.GameId != current_game_id or penalty.TotalElapsedTime > penalty_end_time:
            # Default values for how many players are present
            starting_home_players = 6
            starting_away_players = 6
    
        # Get all penalties that are currently running
        current_penalties = penalties.loc[penalties.GameId.eq(penalty.GameId) &
                                          penalties.PenaltyEndTime.ge(penalty.TotalElapsedTime) &
                                          penalties.TotalElapsedTime.le(penalty.TotalElapsedTime)]
    
        # Find all penalties currently running for each team
        current_home_penalties = current_penalties.loc[current_penalties.Team.eq(penalty.HomeTeamName)]
        current_away_penalties = current_penalties.loc[current_penalties.Team.eq(penalty.AwayTeamName)]
        
        # Compute how many players are (probably) on the ice for each team
        starting_home_players = 6 - len(current_home_penalties)
        starting_away_players = 6 - len(current_away_penalties)

        # Save the number of home and away players (presumably) on the ice
        penalties.loc[penalty.Index, "HomePlayers"] = starting_home_players
        penalties.loc[penalty.Index, "AwayPlayers"] = starting_away_players
        
        # Save the values for the next iteration
        current_game_id = penalty.GameId
        penalty_end_time = penalty.PenaltyEndTime

    # Sort values as the original order
    penalties.sort_values(["GameId", "TotalElapsedTime", "Team", "PenaltyMinutes"],
                          ascending=[True, True, True, False], inplace=True)
    
    # Penalties becoming stacked at the same time
    stacked_penalties = penalties.loc[penalties.HomePlayers.lt(4) | penalties.AwayPlayers.lt(4)].copy()
    
    # Reset the index to so it becomes a column
    stacked_penalties.reset_index(inplace=True)
    
    # Compute the number of penalties per team at the same time
    group_size = stacked_penalties.reset_index().groupby(["GameId", "TotalElapsedTime", "Team"],
                                                         as_index=False).size()
    
    # Add group size to the original data frame
    stacked_penalties = stacked_penalties.merge(group_size)
    
    # Reset to the old index
    stacked_penalties.set_index("index", inplace=True)
    
    # Get the first two observations for each group (if applicable)
    first_two_idx = stacked_penalties.reset_index().groupby(["GameId", "TotalElapsedTime", "Team"],
                                                            as_index=False)["index"].head(2)
    # Determine 3+ penalties running at the same time
    stacked_penalties["ThreePlus"] = True
    
    # Determine if there were less than 3 penalties simultaneously    
    stacked_penalties.loc[first_two_idx , "ThreePlus"] = False
    
    for stack in stacked_penalties.itertuples():
        # Decide which penalties should be affected
        if stack.size <= 2 or (stack.size > 2 and stack.ThreePlus):
            # The id of the game
            game_id = stack.GameId
            
            # The time the penalty occurred
            penalty_time = stack.TotalElapsedTime
            
            # The team who received the penalty
            penalty_team = stack.Team
            
            # The duration of the penalty
            pen_min = stack.PenaltyMinutes
            
            # Penalties running at the same time
            simultaneous_penalty = penalties.loc[penalties.GameId.eq(game_id) &
                                                 penalties.TotalElapsedTime.le(penalty_time) &
                                                 penalties.PenaltyEndTime.ge(penalty_time) &
                                                 penalties.Team.eq(penalty_team)]    
            
            # The end time of the first penalty
            first_penalty_end = simultaneous_penalty.PenaltyEndTime.min()
            
            # Update the start time of the stacked penalty to be that of the first penalty expriring
            penalties.loc[stack.Index, "TotalElapsedTime"] = first_penalty_end
            
            # Update the end time of the stacked penalty to be that of the first penalty expriring + duration
            penalties.loc[stack.Index, "PenaltyEndTime"] = first_penalty_end + 60 * pen_min

        # In case there are three penalties running simultaneously
        if stack.HomePlayers < 4:
            # Three penalties at the same time for the home team
            penalties.loc[stack.Index, "HomePlayers"] = 4
            
        if stack.AwayPlayers < 4:
            # Three penalties at the same time for the away team
            penalties.loc[stack.Index, "AwayPlayers"] = 4
   
    # Find the game ids that go into overtime
    overtime_games = pbp_data.loc[pbp_data.PeriodNumber.ge(4), "GameId"].unique()

    # Penalties do not carry over to overtime if there is none
    penalties.loc[penalties.PenaltyEndTime.gt(3600) &
                  penalties.TotalElapsedTime.le(3600) &
                  ~penalties.GameId.isin(overtime_games), "PenaltyEndTime"] = 3600
    
    # Penalties stop when the game is over
    penalties.loc[penalties.PenaltyEndTime.gt(3900) &
                  penalties.TotalElapsedTime.le(3900), "PenaltyEndTime"] = 3900
    
    return penalties


def compute_manpower(penalties: DataFrame, 
                     pbp_data: DataFrame) -> DataFrame:
    """
    Compute the manpower, i.e., how many players were on the ice during specific
    time-points in a game, for all games.

    Parameters
    ----------
    penalties : DataFrame
        A data frame with all penalties, alongside their start, end and duration.
    pbp_data : DataFrame
        Play by play data from a given season.

    Returns
    -------
    game_manpower : DataFrame
        A data frame with information of the different manpower situations that
        occurred during the game, alongside their duration..

    """
    # Store all games in this dictionary
    game_manpower_dict ={}
    
    # Loop over all games and their penalties
    for game_id, game_penalties in tqdm(penalties.groupby("GameId"),
                                        total=len(penalties.groupby("GameId"))):
        # Sort the penalties in the game by when they start and end
        game_penalties.sort_values(["TotalElapsedTime", "PenaltyEndTime"], inplace=True)
        
        # Get all the starting and end times of penalties
        penalty_start_end_times = np.ravel((np.ravel(game_penalties.TotalElapsedTime),
                                            np.ravel(game_penalties.PenaltyEndTime)))
        
        # Keep only the unique start and end times
        unique_times = np.unique(penalty_start_end_times)
        
        # Create a data frame containing the first row and to be appended to
        game_df = pd.DataFrame([[0, unique_times[0], 6, 6, 0, 0]], 
                               columns=["Start", "End", "NrHome", "NrAway", 
                                        "PenaltyMinutes", "EndedByGoal"])
        
        # Loop over all unique times
        for time in np.sort(unique_times):
            # Find penalties that have shared time
            penalty = game_penalties.loc[game_penalties.TotalElapsedTime.ge(time) |
                                         game_penalties.PenaltyEndTime.gt(time)].copy()
            
            # If the shared time is below the one investigated, set to nan
            penalty.loc[penalty.TotalElapsedTime.le(time), "TotalElapsedTime"] = np.nan
            penalty.loc[penalty.PenaltyEndTime.le(time), "PenaltyEndTime"] = np.nan
            
            with warnings.catch_warnings():
                # Ignore NaN axis warning
                warnings.filterwarnings(action='ignore', message='All-NaN axis encountered')
                
                # Get the minimum 
                first_penalty_end = np.nanmin([penalty.TotalElapsedTime.min(),
                                               penalty.PenaltyEndTime.min()])
            
            # Compute the penalties that can run simultaneously as the end time
            penalty = game_penalties.loc[game_penalties.PenaltyEndTime.ge(first_penalty_end) &
                                         game_penalties.TotalElapsedTime.lt(first_penalty_end)]
            
            # Compute the number of total penalties
            nr_penalties = len(penalty)
            
            # Compute number of penalties for the home team
            nr_home_penalties = len(penalty.loc[penalty.Team.eq(penalty.HomeTeamName)])
            
            # Compute the number of away penalties
            nr_away_penalties = nr_penalties - nr_home_penalties
            
            # Compute the number of players on the ice for each team
            if time > 3600:
                nr_home_players = 4 + nr_away_penalties
                nr_away_players = 4 + nr_home_penalties
            else:    
                nr_home_players = 6 - nr_home_penalties
                nr_away_players = 6 - nr_away_penalties
            
            # Get the penalty duration
            penalty_minutes = penalty.UnadjustedPenaltyMinutes.iloc[0] if len(penalty) > 0 else 0
            
            # If the penalty had a goal scored during it
            ended_by_goal = penalty.EndedByGoal.iloc[0] if len(penalty) > 0 else 0
            
            # Append to the data frame
            game_df.loc[len(game_df)] = [time, first_penalty_end, nr_home_players, nr_away_players, 
                                         penalty_minutes, ended_by_goal]
      
        # Get the final time point in the game during overtime (or NaN)
        final_time_point_overtime = pbp_data.loc[
            pbp_data.GameId.eq(game_id) &
            pbp_data.PeriodNumber.eq(4)].TotalElapsedTime.max()
                
        # See if there was an overtime or not
        no_overtime = np.isnan(final_time_point_overtime)

        # If there are no penalties
        no_penalties = (game_df.NrHome.eq(game_df.NrAway) &
                        game_df.NrHome.eq(6))

        if no_overtime:
            # Specify the end time of penalties that is the final one (regulation)
            game_df.loc[game_df.Start.le(3600) & 
                        game_df.End.isna(), "End"] = 3600
        else:
            # No penalties leading into overtime
            no_penalties_to_overtime = no_penalties & game_df.End.gt(3600)
            
            # Condition for speciying end time when there are no penalties carrying over
            end_time_in_regulation_no_penalties = (game_df.Start.le(3600) & 
             (no_penalties_to_overtime | game_df.End.isna())
             )
            
            # If there are penalties carrying over to overtime
            end_time_in_regulation_penalties = (game_df.Start.le(3600) & 
             ((~no_penalties & game_df.End.gt(3600)) | game_df.End.isna())
             )
            
            # If the end time should be specified for regulation
            game_df.loc[end_time_in_regulation_no_penalties, "End"] = 3600
            
            # Penalties that do not carry over to overtime
            penalties_not_carrying_over = game_df.loc[end_time_in_regulation_no_penalties]
            
            # Penalties that do carry over to overtime
            penalties_carrying_over = game_df.loc[end_time_in_regulation_penalties]
            
            if len(penalties_not_carrying_over) != 0:
                # If there are any penalties in overtime 
                penalties_in_overtime = game_df.loc[game_df.Start.ge(3600) &
                                                    game_df.NrHome.ne(game_df.NrAway)]
                if len(penalties_in_overtime) > 0:
                    end_first_ot_situation = penalties_in_overtime.Start.min()
                    game_df.loc[len(game_df)] = [3600, end_first_ot_situation, 4, 4, 0, 0]
                else:    
                    game_df.loc[len(game_df)] = [3600, final_time_point_overtime, 4, 4, 0, 0]
                    
            elif len(penalties_carrying_over) != 0:
                # The index of the row to be changed
                penalty_index = penalties_carrying_over.index[0]
                
                # Specify end time of regulation
                game_df.loc[penalty_index, "End"] = 3600 
                
                # Compute number of players in overtime
                home_players = (4 + (6 - game_df.loc[penalty_index, "NrAway"]))
                away_players = (4 + (6 - game_df.loc[penalty_index, "NrHome"]))
                
                # End point of the manpower situation
                end_point = min(final_time_point_overtime, penalties_carrying_over.End.values[0])
                
                # Specify the manpower situation in overtime
                manpower_penalty_overtime = pd.DataFrame([[3600, end_point,
                                                          home_players, away_players, 0, 0]],
                                                         columns=game_df.columns)
                
                # Add new row with the manpower situation to be added
                game_df = pd.concat([game_df[:penalty_index+1], 
                                     manpower_penalty_overtime,
                                     game_df[penalty_index+1:]
                                     ]).reset_index(drop=True)
             
        # Penalties in overtime expire when the game ends
        game_df.loc[game_df.Start.le(3900) &
                    game_df.End.gt(3900), "End"] = final_time_point_overtime
            
        # Specify the end time of penalties that is the final one (overtime)
        game_df.loc[game_df.Start.gt(3600) & 
                    game_df.End.isna(), "End"] = final_time_point_overtime
        
        # Save in the dictionary
        game_manpower_dict[game_id] = game_df
            
    # Find the game ids that do not contain a penalty for either team
    games_without_penalties = ~pbp_data.GameId.drop_duplicates().isin(penalties.GameId)
    
    # Keep only the game ids that have no penalty
    games_without_penalties = games_without_penalties[games_without_penalties]
    
    # Extract the game ids themselves
    game_ids_without_penalties = pbp_data.loc[games_without_penalties.index, "GameId"]
    
    # Initialize a data frame for the game without penalties
    no_penalty_df = pd.DataFrame([[0, 3600, 6, 6, 0, 0]], 
                                 columns=["Start", "End", "NrHome", "NrAway", 
                                          "PenaltyMinutes", "EndedByGoal"])
    
    # Loop over all games without a penalty
    for game_id in game_ids_without_penalties:
        # Add the placeholder data frame
        game_manpower_dict[game_id] = no_penalty_df

    # Combine all games and their manpower into one data frame
    game_manpower = pd.concat(game_manpower_dict).reset_index(level=0).rename(
        columns={"level_0": "GameId"}).reset_index(drop=True)
    
    # Add information regarding home and away team
    game_manpower = game_manpower.merge(pbp_data[["GameId", "HomeTeamName", "AwayTeamName"]].drop_duplicates(), 
                how="left") 
    
    # Determine the powerplay team
    game_manpower["PowerplayTeam"] = game_manpower.apply(lambda x: x.HomeTeamName if x.NrHome > x.NrAway else x.AwayTeamName, axis=1)
    
    # Determine the shorthanded team
    game_manpower["ShorthandedTeam"] = game_manpower.apply(lambda x: x.HomeTeamName if x.NrHome < x.NrAway else x.AwayTeamName, axis=1)
    
    # Compute the duration for each situation
    game_manpower["Duration"] = game_manpower["End"] - game_manpower["Start"]
    
    # If the manpower was equal, no team played in powerplay or shorthanded
    game_manpower.loc[game_manpower.NrHome == game_manpower.NrAway, "PowerplayTeam"] = np.nan
    game_manpower.loc[game_manpower.NrHome == game_manpower.NrAway, "ShorthandedTeam"] = np.nan
    
    # Remove durations of length 0
    game_manpower = game_manpower.loc[game_manpower.Duration.ne(0)].copy()
    
    # Sort values to proper order
    game_manpower.sort_values(["GameId", "Start"], inplace=True)
    
    # Add a bool to signify if the time was in overtime
    game_manpower["Overtime"] = False
    game_manpower.loc[game_manpower.Start.ge(3600), "Overtime"] = True
    
    # Indicate the manpower situation
    game_manpower["Manpower"] = np.select(
        [(game_manpower.NrHome.eq(6) & game_manpower.NrAway.eq(6)),
         (game_manpower.NrHome.eq(5) & game_manpower.NrAway.eq(5)),
         (game_manpower.NrHome.eq(4) & game_manpower.NrAway.eq(4)),
         (game_manpower.NrHome.eq(6) & game_manpower.NrAway.eq(5)) |
         (game_manpower.NrHome.eq(5) & game_manpower.NrAway.eq(6)),
         (game_manpower.NrHome.eq(6) & game_manpower.NrAway.eq(4)) |
         (game_manpower.NrHome.eq(4) & game_manpower.NrAway.eq(6)),
         (game_manpower.NrHome.eq(5) & game_manpower.NrAway.eq(4)) |
         (game_manpower.NrHome.eq(4) & game_manpower.NrAway.eq(5))
         ],
        ["5v5", "4v4", "3v3", "5v4", "5v3", "4v3"]
        )
    
    # Remove rows that have an end time prior to the start (impossible)
    game_manpower = game_manpower.loc[game_manpower.End.ge(game_manpower.Start)]
    
    # Adjust game misconducts to have a 5 minute penalty duration
    game_manpower.loc[game_manpower.PenaltyMinutes.gt(5), "PenaltyMinutes"] = 5
    
    return game_manpower

    
def combine_manpower_and_pbp(pbp_data: DataFrame, game_manpower: DataFrame) -> Tuple[DataFrame,
                                                                                     DataFrame,
                                                                                     DataFrame]:
    """
    Combine manpower information with pbp.

    Parameters
    ----------
    pbp_data : DataFrame
        Play-by-play data from a given season.
    game_manpower : DataFrame
        Manpower information from each game.

    Returns
    -------
    game_manpower : DataFrame
        Game manpower information.
    pbp_long : DataFrame
        All events with manpower information.
    non_ev_pbp : DataFrame
        All events in powerplay or while shorthanded.

    """
    
    # Get starting time and manpower from previous row
    game_manpower["PreviousStart"] = game_manpower.groupby("GameId")["Start"].shift(1, fill_value=0)
    game_manpower["PreviousManpower"] = pd.concat([game_manpower.GameId, game_manpower[["NrHome", "NrAway"]].min(axis=1)],
                                                  axis=1).groupby("GameId")[0].shift(1, fill_value=6)
    
    # Add an id for the manpower situation
    game_manpower["ManpowerId"] = range(1, len(game_manpower)+1)
    
    # Combine all events with the manpower information
    pbp_long = pbp_data.merge(game_manpower.rename(columns={"Manpower": "EventManpower",
                                                       "Overtime": "OvertimeManpower"}), how="left", 
                         on=["GameId", "AwayTeamName", "HomeTeamName"])
    
    # Goals scored in regulation
    regulation_pbp = (pbp_long.PeriodNumber.le(3) &
                      (pbp_long.EventType.eq("GOAL") &
                       pbp_long.TotalElapsedTime.between(pbp_long.Start+1,
                                                         pbp_long.End)) | 
                      (pbp_long.EventType.ne("GOAL") &
                       pbp_long.TotalElapsedTime.between(pbp_long.Start,
                                                         pbp_long.End-1)))
    
    # Goals scored in overtime
    overtime_pbp = (pbp_long.PeriodNumber.eq(4) &
                    pbp_long.TotalElapsedTime.between(pbp_long.Start,
                                                      pbp_long.End))
    
    # Keep only events that fall between the correct manpower time-windows
    pbp_long = pbp_long.loc[(regulation_pbp | overtime_pbp)]
    
    # Keep only the goals
    goals_pbp = pbp_long.loc[pbp_long.EventType.eq("GOAL")].copy()
    
    # Add goals scored information back to manpower
    game_manpower = game_manpower.merge(goals_pbp.groupby(["ManpowerId"], as_index=False).size(),
                                        on="ManpowerId", how="left").rename(
                                            columns={"size": "GoalsScored"}).fillna({"GoalsScored": 0})
    
    # Convert column to integer
    game_manpower["GoalsScored"] = game_manpower["GoalsScored"].astype(int)
             
    # Events in special teams                               
    non_ev_pbp = pbp_long.loc[pbp_long.PowerplayTeam.notna()].copy()

    # Compute the time from start of penalty until goal
    non_ev_pbp["TimeUntilGoal"] = non_ev_pbp.TotalElapsedTime - non_ev_pbp.Start
    
    # Adjust the time for 5v3 that become 5v4
    non_ev_pbp.loc[non_ev_pbp.PreviousManpower.eq(4) & non_ev_pbp.PeriodNumber.le(3), 
                   "TimeUntilGoal"] = non_ev_pbp.TotalElapsedTime - non_ev_pbp.PreviousStart
        
    return game_manpower, pbp_long, non_ev_pbp
    

def compute_toi_player(game_manpower: DataFrame, 
                       game_shifts_dict: Dict,
                       players: DataFrame) -> DataFrame:
    """
    Compute the TOI for all players during different manpower situations

    Parameters
    ----------
    game_manpower : DataFrame
        A data frame with information of the different manpower situations that
        occurred during the game, alongside their duration.
    game_shifts_dict : Dict
        A dictionary containing all shifts from all games.

    Returns
    -------
    player_toi : DataFrame
        A data frame with total time spent in different manpower situations as 
        well as PP/SH/OT/EV.

    """
    # Dictionary to store the results per game
    shifts_manpower = {}
    
    # Loop over all games
    for game_id in tqdm(game_shifts_dict.keys()):
        # Get the shifts from the game
        shifts = game_shifts_dict[game_id].copy()
    
        # Remove shifts from outside the first four periods
        shifts = shifts.loc[shifts.PeriodNumber.le(4)]
        
        # Get the different manpower situations in the game
        manpower = game_manpower.loc[game_manpower.GameId.eq(game_id)]
    
        # All different manpower situations
        shifts[["5v5", "5v4", "4v5", "5v3", "3v5", "4v3", "3v4", "4v4", "3v3"]] = 0
        
        for manpower_situation in manpower.itertuples():
            # Start and end of the manpower situation
            start = manpower_situation.Start
            end = manpower_situation.End
            
            # The current manpower situation, e.g., 4v4, 5v4 etc
            pp_situation = manpower_situation.Manpower
            
            # The team currently having a powerplay
            pp_team = manpower_situation.PowerplayTeam
            
            # The opposite coin of the manpower situation
            sh_situation = pp_situation[::-1] 
            
            # If the team who is playing powerplay is the current team with shifts
            same_team = shifts.TeamName.eq(pp_team)
                    
            # Fully contained within a manpower situation
            fully_contained = (shifts.ShiftStart.ge(start) & shifts.ShiftEnd.ge(start) &
                               shifts.ShiftStart.le(end) & shifts.ShiftEnd.le(end))
                    
            # Shifts that start prior to and end after the penalty
            start_contained = (shifts.ShiftStart.ge(start) & shifts.ShiftStart.le(end) &
                               shifts.ShiftEnd.ge(end))
            
            # Shifts that started after the penalty began and ended after it expired
            end_contained = (shifts.ShiftStart.le(start) & shifts.ShiftEnd.le(end) &
                             shifts.ShiftEnd.ge(start))
            
            # Shifts spanning entire manpowers, e.g. goalkeepers
            entire_manpower = (shifts.ShiftStart.lt(start) & shifts.ShiftEnd.gt(end))
            
            # =================================================================== #
            #                        FULLY CONTAINED                              #
            # When a shift is fully contained it is equal to the duration of the  #
            # shift since it does not contain other manpower situations.          #
            # =================================================================== #
            # Powerplay team
            shifts.loc[fully_contained & same_team, pp_situation] = (
                shifts.loc[fully_contained & same_team, "Duration"])
            
            # Shorthanded team
            shifts.loc[fully_contained & ~same_team, sh_situation] = (
                shifts.loc[fully_contained & ~same_team, "Duration"])
                
            # =================================================================== #
            #                        START CONTAINED                              #
            # When a shift is start contained the duration is the difference      #
            # between its start time and the end time of penalty                  #
            # =================================================================== #
            # Powerplay team
            shifts.loc[start_contained & ~fully_contained & ~end_contained & same_team, pp_situation] = (
                shifts.loc[start_contained & ~fully_contained & ~end_contained & same_team, pp_situation] +
                shifts.loc[start_contained & ~fully_contained & ~end_contained & same_team].apply(
                    lambda x: (end - x.ShiftStart), axis=1)
                )
            
            # Shorthanded team
            shifts.loc[start_contained & ~fully_contained & ~end_contained & ~same_team, sh_situation] = (
                shifts.loc[start_contained & ~fully_contained & ~end_contained & ~same_team, sh_situation] +
                shifts.loc[start_contained & ~fully_contained & ~end_contained & ~same_team].apply(
                    lambda x: (end - x.ShiftStart), axis=1)
                )
            
           
            # =================================================================== #
            #                          END CONTAINED                              #
            # When a shift is end contained the duration is the difference        #
            # between its end time and the start time of penalty                  #
            # =================================================================== #        
            # Powerplay team
            shifts.loc[end_contained & ~fully_contained & ~start_contained & same_team, pp_situation] = (
                shifts.loc[end_contained & ~fully_contained & ~start_contained & same_team, pp_situation] +
                shifts.loc[end_contained & ~fully_contained & ~start_contained & same_team].apply(
                    lambda x: (x.ShiftEnd - start), axis=1)
                )
            
            # Shorthanded team
            shifts.loc[end_contained & ~fully_contained & ~start_contained & ~same_team, sh_situation] = (
                shifts.loc[end_contained & ~fully_contained & ~start_contained & ~same_team, sh_situation] +
                shifts.loc[end_contained & ~fully_contained & ~start_contained & ~same_team].apply(
                    lambda x: (x.ShiftEnd - start), axis=1)
                )
    
            
            # =================================================================== #
            #                         ENTIRE MANPOWER                             #
            # When a shift spans through entire different manpower situations,    #
            # e.g., goaltenders.
            # =================================================================== #
            # Powerplay team
            shifts.loc[entire_manpower & same_team, pp_situation] = shifts.loc[entire_manpower & same_team, pp_situation].apply(
                lambda x: x + end - start if pd.notna(x) else end - start)
            
            # Shorthanded team
            shifts.loc[entire_manpower & ~same_team, sh_situation] = shifts.loc[entire_manpower & ~same_team, sh_situation].apply(
                lambda x: x + end - start if pd.notna(x) else end - start)
            

        # Assert the duration is the same as the various manpower situations
        assert all(shifts["Duration"] == shifts[["5v5", "5v4", "4v5", "5v3", "3v5", 
                                                 "4v3", "3v4", "4v4", "3v3"]].sum(axis=1))
        
        # Save the game shifts
        shifts_manpower[game_id] = shifts
        
    # Concatenate all game shifts to one data frame
    all_shifts_manpower = pd.concat(shifts_manpower).reset_index(drop=True)

    # Add indicator for overtime shifts
    all_shifts_manpower["Overtime"] = False
    all_shifts_manpower.loc[all_shifts_manpower.ShiftStart.ge(3600), "Overtime"] = True

    # Compute TOI for all different manpower situations
    player_toi = all_shifts_manpower.drop(["ShiftStart", "PeriodNumber", "ShiftEnd",
                                           "Duration", "PotentialExtraPlayer"], 
                                          axis=1).groupby(["GameId", "PlayerId", "TeamName",
                                                           "Overtime"], as_index=False).sum()
                                                           
    # Compute TOI in overtime
    player_toi["OT TOI"] = player_toi.loc[player_toi.Overtime].iloc[:, 4:].sum(axis=1)

    # Remove overtime column and sum overall
    player_toi = player_toi.drop("Overtime", axis=1).groupby(["GameId", "PlayerId", "TeamName"], 
                                                             as_index=False).sum()
                                                           
    # Compute TOI in even strength situations
    player_toi["EV TOI"] = player_toi[["5v5", "4v4", "3v3"]].sum(axis=1)
    
    # Compute TOI in powerplay situations
    player_toi["PP TOI"] = player_toi[["5v4", "5v3", "4v3"]].sum(axis=1)
    
    # Compute TOI in shorthanded situations
    player_toi["SH TOI"] = player_toi[["4v5", "3v5", "3v5"]].sum(axis=1)
 
    # Rename columns
    player_toi.rename(columns={"TeamName": "Team"}, inplace=True)
 
    # Reorder columns
    player_toi = player_toi[["GameId", "PlayerId", "Team", "5v5", "4v4", "3v3", 
                             "5v4", "5v3", "4v3", "4v5", "3v5", "3v4",
                             "EV TOI", "PP TOI", "SH TOI", "OT TOI"]]
    # Find all goalkeepers
    goalkeepers = players.loc[players.Position.eq("G")]

    # Remove goalkeepers
    player_toi_no_goalie = player_toi.loc[~player_toi.PlayerId.isin(goalkeepers.PlayerId)]
     
    return player_toi_no_goalie


def compute_toi_team(game_manpower: DataFrame) -> DataFrame:
    """
    Compute the TOI in different manpower situations for each team and game.

    Parameters
    ----------
    game_manpower : DataFrame
        A data frame with information of the different manpower situations that
        occurred during the game, alongside their duration.

    Returns
    -------
    team_toi : DataFrame
        A data frame with summarized time spent in different manpower situations
        per game.

    """
    # Copy to avoid changing in-place
    game_manpower_no_na = game_manpower.copy()
    
    # Add even strength information to cover both teams
    game_manpower_no_na.loc[game_manpower_no_na.PowerplayTeam.isna(), 
                            ["PowerplayTeam", "ShorthandedTeam"]] = "Even"
    
    # Compute the time spent in each manpower and PP/SH scenario
    game_manpower_team_time = game_manpower_no_na.groupby(["GameId", "PowerplayTeam", "Manpower",
                                                           "ShorthandedTeam", "Overtime"], 
                                                          as_index=False)["Duration"].sum()

    # Get all powerplay team perspectives
    powerplay = game_manpower_team_time.pivot(index=["GameId", "PowerplayTeam", "Overtime"], 
                                              columns="Manpower",
                                              values="Duration").reset_index().rename(
                                                  columns={"PowerplayTeam": "Team"})
    # Get all shorthanded team perspectives
    shorthanded = game_manpower_team_time.pivot(index=["GameId", "ShorthandedTeam", "Overtime"], 
                                                columns="Manpower", 
                                                values="Duration").reset_index().rename(
                                                    columns={"ShorthandedTeam": "Team"})
    
    # Create new columns with the shorthanded perspective
    shorthanded["4v5"] = shorthanded["5v4"]
    shorthanded["3v5"] = shorthanded["5v3"]
    shorthanded["3v4"] = shorthanded["4v3"]
    
    # Nullify the wrong perspective
    shorthanded["5v4"] = shorthanded["5v3"] = shorthanded["4v3"] = np.nan

    # Compute TOI in overtime
    powerplay["OT TOI"] = powerplay.loc[powerplay.Overtime].iloc[:, 3:].sum(axis=1)
    shorthanded["OT TOI"] = shorthanded.loc[shorthanded.Overtime].iloc[:, 3:].sum(axis=1)

    # Remove overtime column and sum overall per team
    powerplay = powerplay.drop("Overtime", axis=1).groupby(["GameId", "Team"], as_index=False).sum()
    shorthanded = shorthanded.drop("Overtime", axis=1).groupby(["GameId", "Team"], as_index=False).sum()

    # Combine powerplay and shorthanded prespective into one dataframe
    pp_and_sh = pd.merge(shorthanded.drop(["5v4", "5v3", "4v3"], axis=1),
                         powerplay.drop(["5v5", "4v4", "3v3"], axis=1),
                         on=["GameId", "Team"], how="outer").fillna(0)

    # When one of the OT TOI columns is zero but the other is not
    x_is_zero = pp_and_sh["OT TOI_x"].eq(0) & pp_and_sh["OT TOI_y"].ne(0)

    # Update one of the OT TOI columns to contain all info
    pp_and_sh.loc[x_is_zero, "OT TOI_x"] = pp_and_sh.loc[x_is_zero, "OT TOI_y"] 

    # Remove the unneded column
    pp_and_sh.drop(["OT TOI_y"], axis=1, inplace=True)
    
    # Remove the unneded column
    pp_and_sh.rename(columns={"OT TOI_x": "OT TOI"}, inplace=True)

    # Compute the team's PP/SH OT TOI (i.e. not even strength)
    team_OT = pp_and_sh.loc[pp_and_sh.Team.ne("Even")].drop_duplicates(["GameId", "OT TOI"]).groupby(
    "GameId", as_index=False)["OT TOI"].sum()

    # Compute the team's even strength OT TOI )
    even_OT = pp_and_sh.loc[pp_and_sh.Team.eq("Even")].groupby(
    "GameId", as_index=False)["OT TOI"].sum()

    # Combine team and even strength OT TOI
    all_OT = pd.merge(team_OT, even_OT, how="outer").groupby("GameId", as_index=False)["OT TOI"].sum()

    # Compute the total TOI in overtime
    pp_and_sh_OT = pp_and_sh.drop("OT TOI", axis=1).merge(
        all_OT)
    
    # Remove the team "Even" and add time in even strength to both teams
    team_toi = pp_and_sh_OT.loc[pp_and_sh_OT.Team.eq("Even")].drop(["Team", "4v5", "3v5", "3v4",
                                                                    "4v3", "5v3", "5v4"], axis=1).merge(
               pp_and_sh_OT.loc[pp_and_sh_OT.Team.ne("Even")].drop(["3v3", "4v4", "5v5", "OT TOI"], axis=1),
               on=["GameId"], how="outer")
  
    # Where there is no team name given, i.e. no powerplay/shorthanded
    missing_team = team_toi.Team.isna()
    
    # Loop over all games with missing teams
    for team in team_toi.loc[missing_team].itertuples():
        # Id of the row
        idx = team.Index
        
        # Id of the game where teams are missing
        game_id = team.GameId
        
        # Get the homeand away team from the game
        home_team = game_manpower_no_na.loc[game_manpower_no_na.GameId.eq(game_id), "HomeTeamName"].unique()[0]
        away_team = game_manpower_no_na.loc[game_manpower_no_na.GameId.eq(game_id), "AwayTeamName"].unique()[0]
        
        # Create a tuple for the new rows to be added
        new_row_home_team = team[1:6] + (home_team, ) + team[7:]
        new_row_away_team = team[1:6] + (away_team, ) + team[7:]
        
        # Add the new rows to the data frame
        team_toi = pd.concat([team_toi.loc[:idx-1],
                              pd.DataFrame([new_row_home_team], columns=team_toi.columns),
                              pd.DataFrame([new_row_away_team], columns=team_toi.columns),
                              team_toi.loc[idx+1:]])
                                            
    # Replace NA with 0 for the newly added values
    team_toi.fillna(0, inplace=True)
    
    # Reset the index for the newly added values
    team_toi.reset_index(drop=True, inplace=True)
                      
    # Compute TOI in even strength situations
    team_toi["EV TOI"] = team_toi[["5v5", "4v4", "3v3"]].sum(axis=1)
    
    # Compute TOI in powerplay situations
    team_toi["PP TOI"] = team_toi[["5v4", "5v3", "4v3"]].sum(axis=1)
  
    # Compute TOI in shorthanded situations
    team_toi["SH TOI"] = team_toi[["4v5", "3v5", "3v4"]].sum(axis=1)

    # Reorder columns                                                               
    team_toi = team_toi[["GameId", "Team", "5v5", "4v4", "3v3", 
                         "5v4", "5v3", "4v3", "4v5", "3v5", "3v4",
                         "EV TOI", "PP TOI", "SH TOI", "OT TOI"]]

    return team_toi
    
    
def add_players_on_ice(game_shifts_df: DataFrame, pbp_data: DataFrame) -> DataFrame:
    """
    Add players on the ice to play by play data.

    Parameters
    ----------
    game_shifts_df : DataFrame
        All shifts from a season stored in a data frame.
    pbp_data : DataFrame
        Play by play data from a given season.

    Returns
    -------
    pbp_data_on_ice : DataFrame
        Play by play data with players on ice during all events.

    """
    # Initialize empty nested dictionary
    pbp_dict = {idx: {} for idx in pbp_data["GameId"].unique()}
    
    # Remove duplicated shifts
    game_shifts_dict = remove_duplicated_shifts(game_shifts_df)

    # Loop over all game ids and their events
    for game_idx, game_events in tqdm(pbp_data.groupby(["GameId", "TotalElapsedTime"]),
                                      total=len(pbp_data.groupby(["GameId", "TotalElapsedTime"]))):

        # Get the home team name
        home_team = game_events.iloc[0]["HomeTeamName"]
        
        # Get all game shifts from the game
        all_game_shifts = game_shifts_dict[game_idx[0]]

        # Go over all events in the play by play data
        for event, event_idx in zip(game_events.itertuples(), game_events.index):
            # Initialize empty list
            pbp_dict[game_idx[0]][event_idx] = []

            # Default value for events that cause stoppage in play     
            next_event_breaks = False

            # All events after the current event in order but at the same time
            next_events_at_same_time = game_events.loc[game_events.EventNumber > event.EventNumber,
                                                       "EventType"]
            
            # If the next event is a sequence breaking event
            break_event = next_events_at_same_time.isin(["STOPPAGE", "FACEOFF", "PERIOD END", "GAME END"])
            
            # If there are no other events other than the given ones
            only_goal_penalty_events = game_events.EventType.isin(["GOAL", "SHOT", "PENALTY"])
            
            # If the next event breaks the action sequence
            if ((any(break_event) or all(only_goal_penalty_events)) and 
                event.EventType in ["GOAL", "SHOT", "PENALTY"]):
                next_event_breaks = True
            
            # There was no goal/stoppage in regulation
            regulation_no_goal_bool = ((event.PeriodNumber <= 3) & (not next_event_breaks)) 
            
            # There was a goal/stoppage in regulation
            regulation_goal_bool = ((event.PeriodNumber <= 3) & (next_event_breaks))
            
            # There was no goal/stoppage in overtime
            overtime_no_goal_bool = ((event.PeriodNumber > 3) &
                                     (not next_event_breaks)
                                     )
            
            # There was a goal/stoppage in overtime
            overtime_goal_bool = ((event.PeriodNumber > 3) & 
                                  (next_event_breaks)
                                  )
            
            # Regulation time - events that are not goals
            regulation_no_goal = ((event.TotalElapsedTime >= all_game_shifts["ShiftStart"]) &
                                  (event.TotalElapsedTime < all_game_shifts["ShiftEnd"]) & 
                                  (regulation_no_goal_bool)
                                 )
            
            # Goals in regulation time
            regulation_goal = ((event.TotalElapsedTime > all_game_shifts["ShiftStart"]) &
                               (event.TotalElapsedTime <= all_game_shifts["ShiftEnd"]) & 
                               (regulation_goal_bool)
                              )
            
            # Overtime - events that are not goals
            overtime_no_goal = ((event.TotalElapsedTime >= all_game_shifts["ShiftStart"]) &
                                (event.TotalElapsedTime < all_game_shifts["ShiftEnd"]) & 
                                (overtime_no_goal_bool) 
                                )
            
            # Goals in overtime
            overtime_goal = ((event.TotalElapsedTime > all_game_shifts["ShiftStart"]) &
                             (event.TotalElapsedTime < all_game_shifts["ShiftEnd"] + 1) & 
                             (overtime_goal_bool) 
                            )
            
            # Find which players were on the ice during the given time frame
            on_ice = np.where(regulation_no_goal | regulation_goal | overtime_no_goal | overtime_goal)
            
            # Get the player id and team names for all players on the ice during the vent
            all_players_on_ice = all_game_shifts.iloc[on_ice][["PlayerId", "TeamName", "potentialExtraPlayer"]]

            # Remove duplicate entries for players
            all_players_on_ice = all_players_on_ice.drop_duplicates(["PlayerId"])

            # Loop over both team and all players
            for team, team_players in all_players_on_ice.groupby("TeamName"):
                # Determine which team the player is from
                side = "Home" if team == home_team else "Away"
                
                # Sort based on player id
                team_players.sort_values("PlayerId", inplace=True)
                
                # If there are additional players on the team
                if len(team_players) > 6:
                    team_players = team_players.loc[~team_players["potentialExtraPlayer"]].copy()
                
                # An integer for each player
                team_players["id"] = range(1, len(team_players)+1)
                
                # Create a string of the form "HomePlayer1"
                team_players["id"] = team_players["id"].apply(lambda x: f"{side}Player{x}")
                
                # Remove column for checking extra players
                team_players.drop(["potentialExtraPlayer"], axis=1, inplace=True)
                
                # Get the player id of all players on the ice for one side
                team_players_on_ice = team_players.set_index("id").T.drop("TeamName").reset_index(drop=True)
                
                # Remove index name
                team_players_on_ice.rename_axis(None, axis=1, inplace=True)
                
                # Add team players on ice to the dictionary
                pbp_dict[game_idx[0]][event_idx].append(team_players_on_ice.T)
                
            # If there are players on the ice
            if len(pbp_dict[game_idx[0]][event_idx]) > 0:
                pbp_dict[game_idx[0]][event_idx] = pd.concat(pbp_dict[game_idx[0]][event_idx]).T
            # No players on the ice
            else:
                pbp_dict[game_idx[0]][event_idx] = pd.DataFrame(columns=["HomePlayer1", "HomePlayer2", "HomePlayer3",
                                                                         "HomePlayer4", "HomePlayer5", "HomePlayer6",
                                                                         "HomePlayer7", "HomePlayer8", "HomePlayer9",
                                                                         "AwayPlayer1", "AwayPlayer2", "AwayPlayer3",
                                                                         "AwayPlayer4", "AwayPlayer5", "AwayPlayer6",
                                                                         "AwayPlayer7", "AwayPlayer8", "AwayPlayer9"]
                                                                )
                    
    # Loop over all game ids
    for gameId in pbp_data["GameId"].unique():
        # Remove duplicate columns if there are any
        non_duplicated = {k: v.loc[:,~v.columns.duplicated()] for k, v in pbp_dict[gameId].items()}
        
        # Save results
        pbp_dict[gameId] = pd.concat(non_duplicated).reset_index(level=1, drop=True)
    
    # See if this works
    pbp_on_ice_df = pd.concat(pbp_dict).reset_index().rename(columns={
        "level_0": "GameId", "level_1": "EventId"})
    
    # Combine players on ice with play by play data
    pbp_data_on_ice = pbp_data.reset_index().rename(columns={"index": "EventId"}).merge(
        pbp_on_ice_df, on=["GameId", "EventId"], how="left")
    
    return pbp_data_on_ice


def add_players_on_ice_merge(game_shifts_df: DataFrame, pbp_data: DataFrame) -> DataFrame:
    """
    Add players on the ice to play by play data by merging pbp and shift data.

    Parameters
    ----------
    game_shifts_df : DataFrame
        All shifts from a season stored in a data frame.
    pbp_data : DataFrame
        Play by play data from a given season.

    Returns
    -------
    pbp_with_shifts : DataFrame
        Play by play data with players on ice during all events.

    """
    
    # Get all events from each game
    pbp_dict = {game_id: pbp_data.loc[pbp_data.GameId.eq(game_id)].copy() for 
                game_id in pbp_data["GameId"].unique()}
    
    # Remove duplicate game shifts
    game_shifts_dict = remove_duplicated_shifts(game_shifts_df)

    # Store all games with shifts in a dictionary
    pbp_with_shifts_dict = {}
    
    for shift_tuple, events in tqdm(zip(game_shifts_dict.items(), pbp_dict.values()),
                                    total=len(game_shifts_dict)):
        # Copy to avoid changing in-place    
        events = events.copy()
        
        # Get the game id
        game_id = shift_tuple[0]
        
        # Get the game shifts
        shifts = shift_tuple[1]
        
        # Determine which team played at home
        home_team = events.HomeTeamName.unique()[0]
        
        # Combine event data with shifts to get all available information
        event_shifts = events.merge(shifts, left_on="GameId", right_on="gameId", how="outer")

        # Keep only players who were (most likely) on the ice during the event
        on_ice = event_shifts.loc[(event_shifts.TotalElapsedTime.between(event_shifts.ShiftStart, 
                                                                         event_shifts.ShiftEnd-1) &
                                   event_shifts.EventType.isin(["PERIOD START", "FACEOFF"]) &
                                   event_shifts.PeriodNumber.le(3)) |
                                  (event_shifts.TotalElapsedTime.between(event_shifts.ShiftStart+1,
                                                                         event_shifts.ShiftEnd) &
                                   ~event_shifts.EventType.isin(["PERIOD START", "FACEOFF"])) |
                                  (event_shifts.TotalElapsedTime.between(event_shifts.ShiftStart, 
                                                                         event_shifts.ShiftEnd-1) &
                                   event_shifts.EventType.isin(["PERIOD START", "FACEOFF"]) &
                                   event_shifts.PeriodNumber.gt(3)) |
                                  (event_shifts.TotalElapsedTime.between(event_shifts.ShiftStart, 
                                                                         event_shifts.ShiftEnd) &
                                   event_shifts.EventType.isin(["GOAL"]) &
                                   event_shifts.PeriodNumber.gt(3)) 
                                  ].copy()
    
        # Combine all player ids for each event number into a string
        player_ids_on_ice = on_ice[["TeamName", "EventNumber", "PlayerId"]].astype(str).groupby(
            ["TeamName", "EventNumber"])["PlayerId"].agg(','.join).unstack().T
    
        # Determine if the first team is the home team
        first_is_home = player_ids_on_ice.iloc[:, 0].name == home_team
    
        # Players on the ice for first and second team, respectively
        f1 = player_ids_on_ice.iloc[:, 0].str.split(pat=",", expand=True)
        f2 = player_ids_on_ice.iloc[:, 1].str.split(pat=",", expand=True)

        # Rename columns
        if first_is_home:
            f1.columns = [f"HomePlayer{i}" for i in range(1, len(f1.columns)+1)]    
            f2.columns = [f"AwayPlayer{i}" for i in range(1, len(f2.columns)+1)]    
        else:
            f1.columns = [f"AwayPlayer{i}" for i in range(1, len(f1.columns)+1)]    
            f2.columns = [f"HomePlayer{i}" for i in range(1, len(f2.columns)+1)]    
            
        # Combine home and away players into one data frame
        home_away_players = pd.merge(f1.reset_index(), f2.reset_index())
        
        # Convert all columns to float
        home_away_players = home_away_players.astype(float)
        
        # Convert eventnumber to integer
        home_away_players["EventNumber"] = home_away_players["EventNumber"].astype(int)
        
        # Sort by event number
        home_away_players.sort_values("EventNumber", inplace=True)
        
        # Combine event data with home and away players in the wide format
        events_with_players = events.merge(home_away_players, on="EventNumber")
        
        # Find players that may be duplicates
        possible_duplicates = on_ice.loc[on_ice.potentialExtraPlayer, 
                                         ["TeamName", "EventNumber", "PlayerId",
                                          "potentialExtraPlayer"]]
        
        # Loop over all shifts that maybe be erroneous
        for possible_duplicate in possible_duplicates.itertuples():
            
            # If it was the home team with the duplicate
            is_home = possible_duplicate.TeamName == home_team
            
            # See if there have been tagged with more than 6 players on the ice simultaneously
            cond = (events_with_players.EventNumber.eq(possible_duplicate.EventNumber) &
                    (is_home and "HomePlayer7" in events_with_players.columns) |
                    (~is_home and "AwayPlayer7" in events_with_players.columns))
            
            # Where the player is on the ice
            player = np.where(events_with_players.iloc[:, 30:].values == 
                              possible_duplicate.PlayerId)
            
            # In case there is no actual duplicate
            if len(player[0]) == 0:
                continue
            
            # Find the common index
            duplicate_idx = np.intersect1d(events_with_players.loc[cond].index, player[0])
            
            # Find the column where the index are duplicated
            duplicate_col = player[1][np.in1d(player[0], duplicate_idx)]
            
            for dupe_idx, dupe_col in zip(duplicate_idx, duplicate_col):
                # Find the column name of the duplicate
                col_name = events_with_players.columns[30+dupe_col]
                
                # Replace the duplicate player with nan
                events_with_players.loc[dupe_idx, col_name] = np.nan
        
        # Shift the home player columns to the left to fill NA holes        
        home = events_with_players.loc[:, events_with_players.columns.str.startswith("HomePlayer")].apply(
            lambda x: pd.Series(x.dropna().to_numpy()), axis=1)

        # Save the shifted player values
        events_with_players.loc[:, [f"HomePlayer{i}" for i in range(1, len(home.columns)+1)]] = home.values

        # Shift the away player columns to the left to fill NA holes
        away = events_with_players.loc[:, events_with_players.columns.str.startswith("AwayPlayer")].apply(
            lambda x: pd.Series(x.dropna().to_numpy()), axis=1)
        
        # Save the shifted player values
        events_with_players.loc[:, [f"AwayPlayer{i}" for i in range(1, len(away.columns)+1)]] = away.values
        
        for team in ["Home", "Away"]:
            # Find if there are any columns that are extra, i.e., PlayerX, where X = 7/8/9
            player_cond = (events_with_players.columns.str.startswith(f"{team}Player") &
                                events_with_players.columns.str.contains("7|8|9|10|11|12"))
            # If there are no extra columns
            if all(~player_cond):
                continue
            
            # Get the data corresponding to extra columns
            player_extra_columns = events_with_players.loc[:, player_cond]
            
            # Find rows that might have duplicated player ids
            player_duplicate = np.where(player_extra_columns.notna())
            
            # For the players who are duplicated, set the PlayerX to nan
            events_with_players.loc[player_duplicate[0], 
                                    player_extra_columns.columns[player_duplicate[1]]] = np.nan
            
            # Remove duplicated columns
            events_with_players = events_with_players.loc[:, ~events_with_players.columns.duplicated()].copy()
            
        # Find columns with only NA values
        only_na_columns = events_with_players.notna().sum(axis=0)
        
        # Remove columns with only NA
        events_with_players = events_with_players.loc[:, only_na_columns.ne(0)].copy()
        
        # Add penalty shootout events too not drop any events
        events_with_players = pd.concat([
            events_with_players, 
            events.loc[events.TotalElapsedTime.eq(4800)]]).reset_index(drop=True)
        
        # Save in dictionary
        pbp_with_shifts_dict[game_id] = events_with_players
        
    # Combine all games into one data frame
    pbp_with_shifts = pd.concat(pbp_with_shifts_dict).reset_index(drop=True)

    return pbp_with_shifts
        

if __name__ == "__main__":

    # =========================================================================
    # Computing time on ice statistics.
    # =========================================================================
    for season in range(2010, 2022):
        
        print(f"Processing season {season}")
        # Read the data for the given season
        #game_shifts = pd.read_csv(f"../Data/GameShifts/game_shifts_{season}.csv")
        pbp = pd.read_csv(f"C:/Users/Rasmus/Downloads/PlayerRolesData/PBP/pbp_{season}.csv")
        #players = pd.read_csv("../Data/players.csv", low_memory=False)
        
        # Add events that signify that a goalie has been pulled
        #pbp.loc[~pbp.HomeGoalieOnIce.fillna(False)] # Home goalie pulled
        #pbp.loc[~pbp.AwayGoalieOnIce.fillna(False)] # Away goalie pulled
        
        # Remove duplicated shifts
        #game_shifts_dict = remove_duplicated_shifts(game_shifts)
        
        # Remove penalties that does not affect manpower
        penalty_goals = remove_penalties_that_does_not_make_shorthanded(pbp)

        # Adjust the start/end time for stacked penalties
        penalties = adjust_time_for_stacked_penalties(penalty_goals, pbp)

        # Compute the manpower per team during all points in the game
        manpower = compute_manpower(penalties, pbp)
        
        # Combine manpower and pbp
        manpower, pbp_manpower, non_ev_pbp = combine_manpower_and_pbp(pbp, manpower)
        
        # Save as csv
        manpower.to_csv(f"C:/Users/Rasmus/Downloads/PlayerRolesData/PowerPlay/manpower_{season}.csv", index=False)
        pbp_manpower.to_csv(f"C:/Users/Rasmus/Downloads/PlayerRolesData/PowerPlay/pbp_manpower_{season}.csv", index=False)
        non_ev_pbp.to_csv(f"C:/Users/Rasmus/Downloads/PlayerRolesData/PowerPlay/non_ev_pbp_{season}.csv", index=False)
        
    #     # Compute the TOI during various manpower situations for all players
    #     toi_player = compute_toi_player(manpower, game_shifts_dict, players)
        
    #     # Compute the TOI during various manpower situations for all team
    #     toi_team = compute_toi_team(manpower)
        
    #     # Combine player and team TOI
    #     toi = toi_player.merge(toi_team, on=["GameId", "Team"],
    #                            suffixes=("_player", "_team"),
    #                            how="inner")
        
    #     # Compute PP TOI % for the entire season
    #     PP_TOI_pct = toi.groupby("PlayerId").agg(
    #         {"PP TOI_player": sum, "PP TOI_team": sum}).apply(
    #         lambda x: x[0] / x[1], axis=1).reset_index().rename(columns={0: "PP%"})
        
    #     # Compute PP TOI % for the entire season
    #     SH_TOI_pct = toi.groupby("PlayerId").agg(
    #         {"SH TOI_player": sum, "SH TOI_team": sum}).apply(
    #         lambda x: x[0] / x[1], axis=1).reset_index().rename(columns={0: "SH%"})
                
     
    # # =========================================================================
    # # Combining play by play with skaters on the ice
    # # =========================================================================
    
    # for season in range(2010, 2022):
    #     print(f"Processing season {season}")
    #     # Read the data for the given season
    #     game_shifts = pd.read_csv(f"../Data/game_shifts_{season}.csv")
    #     pbp = pd.read_csv(f"../Data/pbp_{season}.csv")
        
    #     # Combine play by play data with players on ice
    #     pbp_with_players = add_players_on_ice_merge(game_shifts, pbp)
        
    #     # Save as csv file
    #     pbp_with_players.to_csv(f"../Data/pbp_{season}_with_shifts.csv", index=False)
