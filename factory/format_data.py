"""
Formatting the play-by-play data into the format into tabular lineup data.
"""

from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from multiprocessing import Pool
import timeit
import os

def format_data():

    # make sure folder exists to save 
    if not os.path.exists('data/lineup_scores/'):
        os.makedirs('data/lineup_scores/')
        
    # play-by-play files saved by season
    # go through season files and get game lineups
    pbp_files = sorted(glob('data/raw_pbp/*_pbp.csv'))
        
    for pbp_file in pbp_files:

        print(f'Processing season : {pbp_file}')

        start_time = timeit.default_timer()

        season = pbp_file.split('/')[-1].split('_pbp')[0]
        save_file = f'data/lineup_scores/{season}.pkl'

        if not os.path.exists(save_file):
            data = _process_season(pbp_file)
            data['season'] = season
            data.to_pickle(save_file)

        end_time = timeit.default_timer()
        time = (end_time - start_time) / 60

        print(f'Finished {pbp_file} in {time}')

        
def _process_season(pbp_file, processors=None):
    
    # open season play by plays
    season_pbp = pd.read_csv(pbp_file)
    
    ########################
    # Format all season data
    #
    
    # fix bad column name
    season_pbp.rename(columns={'Unnamed: 0' : 'GAME_IDX'}, inplace=True)
    # make sure games sorted properly
    season_pbp.sort_values(['GAME_ID', 'GAME_IDX'], inplace=True)
    
    # calculate elapsed time since start of the game, slow-ish
    # start with converting timestamp to seconds
    seconds_temp = season_pbp.PCTIMESTRING.str.split(':', 
                                                     expand=True).astype(np.int64).apply(lambda x: 
                                                                                         x[0]*60 + x[1], 
                                                                                         axis=1)
    # depending on period, calculate elapsed time from start of game
    # if regulation: (720 - seconds_temp) + (720 * (season_pbp.PERIOD - 1))
    # if extra: 2880 + (300 - seconds_temp) + (300 * (season_pbp.PERIOD - 5))) 
    season_pbp['SECONDS_ELAPSED'] = np.where(season_pbp.PERIOD < 5, 
                                             720 * season_pbp.PERIOD - seconds_temp,  
                                             300 * season_pbp.PERIOD - seconds_temp + 1680)
    
    ####################################
    # Get lineups from play-by-play data
    #
    
    # create groupby to track progress
    game_groups = season_pbp.groupby(['GAME_ID', 'PERIOD'])
    
    # create a progress bar
    pbar = tqdm(total=len(game_groups))
    # function to update progress bar and results
    season_lineups = []
    def update(a):
        season_lineups.append(a)
        pbar.update()
        
    # create pool to parallelize
    pool = Pool(processors)
    
    # extract lineups from play-by-play data
    for _, game_pbp in game_groups:
        p = pool.apply_async(_process_pbp, 
                             args=(game_pbp, ), 
                             callback=update)
    
    # wait for parallel processes to finish
    pool.close()
    pool.join()
    # close when all processes finished
    pbar.close()
    
    # format lineup data
    season_lineups = pd.concat(season_lineups, ignore_index=True)
    
    season_lineups.sort_values(['GAME_ID', 'GAME_IDX'], inplace=True)
    season_lineups.reset_index(drop=True, inplace=True)
    
    ############################
    # Add scores to game lineups 
    #
    
    # add score at start of game to forward fill
    season_lineups.loc[season_lineups.groupby('GAME_ID').GAME_IDX.idxmin(), 'SCORE'] = '0 - 0'
    
    # split score string, visitor score listed first
    season_lineups['HOMESCORE'] = season_lineups.SCORE.str.split('-').str.get(1).str.strip()
    season_lineups['VISITSCORE'] = season_lineups.SCORE.str.split('-').str.get(0).str.strip()
    
    season_lineups['HOMESCORE'] = season_lineups.groupby('GAME_ID').HOMESCORE.ffill().astype(int)
    season_lineups['VISITSCORE'] = season_lineups.groupby('GAME_ID').VISITSCORE.ffill().astype(int)
    
    game_scores = season_lineups.groupby(['GAME_ID', 'PERIOD', 'GROUPINGS'])
    
    # create a progress bar
    pbar = tqdm(total=len(game_scores))
    # function to update progress bar and results
    lineup_scores = []
    def update(a):
        lineup_scores.append(a)
        pbar.update()

    # create pool to parallelize
    pool = Pool(processors)

    # extract each pbp file
    for (game_id, period_id, group_id), g in game_scores:
        p = pool.apply_async(_fill_scores, 
                             args=(g, game_id, period_id, group_id), 
                             callback=update)
        
    # wait for parallel processes to finish
    pool.close()
    pool.join()
    # close when all processes finished
    pbar.close() 
    
    # formatting all lineup scores
    lineup_scores = pd.DataFrame(lineup_scores, columns=['game_id', 'period_id', 'group_id', 'seconds', 'home_lineup', 
                                                         'visit_lineup', 'home_points', 'visit_points'])
    lineup_scores = lineup_scores[lineup_scores.seconds > 0].copy()
    
    # expand lineups to columns
    data = []

    for _, row in lineup_scores.iterrows():
        sec = row['seconds']
        hl = row['home_lineup']
        vl = row['visit_lineup']
        hp = row['home_points']
        vp = row['visit_points']

        data.append([sec, hl, vl, hp])
        data.append([sec, vl, hl, vp])

    data = pd.DataFrame(data, columns=['seconds', 'offense_lineup', 'defense_lineup', 'points'])
    
    data[['off1', 'off2', 'off3', 'off4', 'off5']] = pd.DataFrame(data.offense_lineup.to_list())
    data[['def1', 'def2', 'def3', 'def4', 'def5']] = pd.DataFrame(data.defense_lineup.to_list())
    
    data.drop(columns=['offense_lineup', 'defense_lineup'], inplace=True)
    
    return data

def _process_pbp(game_pbp):
    
    # get home team id and visitor team id
    teams, counts = np.unique(game_pbp.loc[~game_pbp.HOMEDESCRIPTION.isna(), 'PLAYER1_TEAM_ID'], 
                              return_counts=True)
    home_id = int(teams[counts.argmax()])
    teams, counts = np.unique(game_pbp.loc[game_pbp.HOMEDESCRIPTION.isna(), 'PLAYER1_TEAM_ID'],
                              return_counts=True)
    visit_id = int(teams[counts.argmax()])

    lineups = _get_lineups(game_pbp, home_id, visit_id)

    game_pbp = game_pbp.merge(lineups, on='GAME_IDX')    

    game_pbp.HOME_LINEUP.bfill(inplace=True)
    game_pbp.VISIT_LINEUP.bfill(inplace=True)
    game_pbp.HOME_LINEUP.ffill(inplace=True)
    game_pbp.VISIT_LINEUP.ffill(inplace=True)
    game_pbp.dropna(subset=['HOME_LINEUP', 'VISIT_LINEUP'], inplace=True)

    game_pbp['HOME_LINEUP'] = game_pbp.HOME_LINEUP.apply(lambda x: tuple(sorted(x)))
    game_pbp['VISIT_LINEUP'] = game_pbp.VISIT_LINEUP.apply(lambda x: tuple(sorted(x)))

    game_pbp.sort_values('GAME_IDX', inplace=True)

    lineup_groups = []
    for t, (k, g) in enumerate(itertools.groupby(game_pbp.HOME_LINEUP + game_pbp.VISIT_LINEUP)):
        lineup_groups.extend([t]*len(list(g)))
    game_pbp['GROUPINGS'] = lineup_groups    

    return game_pbp


def _fill_scores(g, game_id, period_id, group_id):
    
    # Fills in score over elapsed time
    home_points = g.HOMESCORE.max() - g.HOMESCORE.min()
    visit_points = g.VISITSCORE.max() - g.VISITSCORE.min()
    seconds = g.SECONDS_ELAPSED.max() - g.SECONDS_ELAPSED.min()
    home_lineup = g.HOME_LINEUP.iloc[0]
    visit_lineup = g.VISIT_LINEUP.iloc[0]
    return [game_id, period_id, group_id, seconds, home_lineup, 
            visit_lineup, home_points, visit_points]

def _get_lineups(game_pbp, home_id, visit_id):
    
    # count players forward in game
    lineups = _create_lineup(game_pbp, home_id, visit_id)
    # find where missing
    missing = lineups.loc[lineups.isna().any(1), 'GAME_IDX'].values
    # count players backwards in game to fill starting lineups
    lineups_back = _create_lineup(game_pbp.iloc[::-1], home_id, visit_id, reverse=True)
    # combine forwards and backwards
    lineups_back.index = lineups_back.index + lineups.index.max() + 1
    lineups = pd.concat([lineups, lineups_back])
    
    # go through home lineups and take the non-NA lineup or the forward if both available
    home_lineups = []
    for k,g in  lineups.groupby('GAME_IDX').HOME_LINEUP:
        g.dropna(inplace=True)
        if len(g) > 0:
            home_lineups.append([k, tuple(sorted(g.loc[g.index.min()]))])
    home_lineups = pd.DataFrame(home_lineups, columns=['GAME_IDX', 'HOME_LINEUP'])
    # repeat with visitor lineups
    visit_lineups = []
    for k,g in  lineups.groupby('GAME_IDX').VISIT_LINEUP:
        g.dropna(inplace=True)
        if len(g) > 0:
            visit_lineups.append([k, tuple(sorted(g.loc[g.index.min()]))])
    visit_lineups = pd.DataFrame(visit_lineups, columns=['GAME_IDX', 'VISIT_LINEUP'])
    
    lineups = pd.merge(home_lineups, visit_lineups, on='GAME_IDX', how='outer')
    
    return lineups

def _create_lineup(game_pbp, home_id, visit_id, reverse=False):
    
    lineups = []
    homecourt = set()
    visitcourt = set()

    for _, row in game_pbp.iterrows():

        # skip action if performed by team
        if row.PLAYER1_ID in [home_id, visit_id]:
            continue

        # if action is a sub, switch players oncourt
        # if processed reverse, player switch is reversed
        if row.EVENTMSGTYPE == 8:

            # if processing forwards in time
            if reverse == False:
                # first player is on court, exiting
                # second player is entering
                # make sub for home team
                if row.PLAYER1_TEAM_ID == home_id:
                    # add existing list, in case somehow first action
                    lineups[-1][1].add(row.PLAYER1_ID)
                    # sub player out
                    homecourt.discard(row.PLAYER1_ID)
                    # sub new player in
                    homecourt.add(row.PLAYER2_ID)
                # make sub for the visitors    
                else:
                    lineups[-1][2].add(row.PLAYER1_ID)
                    visitcourt.discard(row.PLAYER1_ID)
                    visitcourt.add(row.PLAYER2_ID)

            # if processing backwards in time
            else:
                if row.PLAYER1_TEAM_ID == home_id:
                    lineups[-1][1].add(row.PLAYER2_ID)
                    homecourt.discard(row.PLAYER2_ID)
                    homecourt.add(row.PLAYER1_ID)  
                else:
                    lineups[-1][2].add(row.PLAYER2_ID)
                    visitcourt.discard(row.PLAYER2_ID)
                    visitcourt.add(row.PLAYER1_ID)

        # if the action is not a sub, add the players to the court sets
        else:
            
            if isinstance(row.PLAYER1_NAME, str):
                if row.PLAYER1_TEAM_ID == home_id:
                    homecourt.add(row.PLAYER1_ID)
                else:
                    visitcourt.add(row.PLAYER1_ID)

            if isinstance(row.PLAYER2_NAME, str):
                if row.PLAYER2_TEAM_ID == home_id:
                    homecourt.add(row.PLAYER2_ID)
                else:
                    visitcourt.add(row.PLAYER2_ID)

            if isinstance(row.PLAYER3_NAME, str):
                if row.PLAYER3_TEAM_ID == home_id:
                    homecourt.add(row.PLAYER3_ID)
                else:
                    visitcourt.add(row.PLAYER3_ID)

        # store game_idx lineups
        lineups.append([row.GAME_IDX, homecourt.copy(), visitcourt.copy()])

    # format all game_idx lineups
    lineups = pd.DataFrame(lineups, columns=['GAME_IDX', 'HOME_LINEUP', 'VISIT_LINEUP'])
    
    lineups['HOME_LINEUP'] = np.where(lineups.HOME_LINEUP.apply(len) == 5, lineups.HOME_LINEUP, np.nan)
    lineups['VISIT_LINEUP'] = np.where(lineups.VISIT_LINEUP.apply(len) == 5, lineups.VISIT_LINEUP, np.nan)
    
    lineups.reset_index(drop=True, inplace=True)

    return lineups