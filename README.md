# Identifying Player Roles in Ice Hockey

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![R](https://img.shields.io/badge/r-%23276DC3.svg?style=for-the-badge&logo=r&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-005C84?style=for-the-badge&logo=mysql&logoColor=white)

## Description

In ice hockey, each team consists of a set of players. Each player has their own unique playing style but is it possible to identify which role they most likely have within the team? That is the question we seek to answer, and the method of choice is fuzzy c-means clustering on principal components obtained from the original variables. 

## Instruction

### Data

The data used in this repository originate from NHL's API and contains all play-by-play events from the 2021-2022 NHL season. Information regarding salary and cap hits are from capfriendly.com.

### Variables

From the data a set of 40+ candidate variables are computed from the play-by-play data. 

### Repository structure

This project contains the most up-to-date version of the code. The structure is the following:
- **Scripts**: contains the Python scripts to derive the variables and R scripts to perform the analysis.
- **Data**: contains additional data-files that might be needed. 

## Usage

1. Store the required data in the corresponding folders (not yet fully described in the event of future changes) in the **Data/** directory.
2. Run the ```playerRoles.py``` script to store the results in a temporary file for ease of transferability in step 3.
3. To obtain the results from modeling, run the ```playerRolesFuzzyClustering.R``` script. 

## Links
- [Sports Analytics @ LiU](https://www.ida.liu.se/research/sportsanalytics/)
