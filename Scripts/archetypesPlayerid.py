#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
from difflib import get_close_matches
import numpy as np
import sys

if __name__ == "__main__":
    # Read data
    archetypes = pd.read_excel("../Data/playerArchetypes.xlsx")
    players = pd.read_csv("../Data/players.csv", sep=",", low_memory=False)
    
    if "fullName" in archetypes.columns:
        sys.exit()
    
    # Find all players in the desired season
    players = players.loc[players.groupsEarnedThruSeason.ge(20212022), ["id", "fullName"]]
    
    # Convert all caps name to proper formatting
    archetypes["Name"] = archetypes.NAME.str.title()
    
    # Combine archetype with player ids
    player_archetypes = archetypes.merge(players, left_on=["Name"], right_on=["fullName"], how="outer")
    
    # Players who did not get a name in the merge
    missing_name = player_archetypes.loc[player_archetypes.fullName.isna()].copy()
    
    # Players who did not get an archetype in the merge
    missing_archetype = player_archetypes.loc[player_archetypes.Name.isna()].copy()
    
    # Find the most similar matching name among the non-matched ones
    missing_name["fullName"] = missing_name.Name.apply(lambda x: get_close_matches(x, missing_archetype.fullName,
                                                                                   n=1, cutoff=0.7))
    
    # Convert the list to a string
    player_archetypes.loc[missing_name.index, "fullName"] = missing_name.fullName.apply(lambda x: x[0] if len(x) > 0 else np.nan)
    
    # Remove players without an archetype 
    player_archetypes = player_archetypes.dropna(axis=0, subset=["TEAM", "fullName"])
    
    # Players who need to have a player id added
    missing_player_id = player_archetypes.loc[player_archetypes.id.isna(), ["fullName"]]
    
    # Get the player id for the missing players
    player_id_for_missing_players = players.loc[players.fullName.isin(missing_player_id.fullName)]
    
    # Combine player id and full name into the data
    player_archetypes.loc[missing_player_id.index, "id"] = missing_player_id.reset_index().merge(
        player_id_for_missing_players).set_index("index")
    
    # Create a new column for names
    player_archetypes["Type"] = player_archetypes.TYPE.map({
        "TWF": "Two-Way Forward", 
        "TWD": "Two-Way Defenseman", 
        "DFD": "Defensive Defenseman", 
        "SNP": "Sniper", 
        "PLY": "Playmaker", 
        "PWF": "Power Forward", 
        "GRN": "Grinder", 
        "OFD": "Offensive Defenseman", 
        "ENF": "Enforcer"})
    
    # Rename columns
    player_archetypes.rename(columns={"id": "PlayerId"}, inplace=True)
    
    # Save to csv
    player_archetypes.to_csv("../Data/playerArchetypes.csv", index=False)