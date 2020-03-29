import os
import shutil
import requests
from tqdm.auto import tqdm
from multiprocessing import Pool
import pandas as pd
import numpy as np
from params import *

# notes on pbp files:
# action is based on EVENTMSGTYPE and PERSON1TYPE who
# is performing the action
# here are the EVENTMSGTYPE scrounged from the web
# 1 - Make 
# 2 - Miss 
# 3 - Free Throw 
# 4 - Rebound 
# 5 - out of bounds / Turnover / Steal 
# 6 - Personal Foul 
# 7 - Violation 
# 8 - Substitution 
# 9 - Timeout 
# 10 - Jumpball 
# 11 - Ejection
# 12 - Start Q1? 
# 13 - Start Q2?
# 18 - Unknown


def get_pbp_data():
    # Download play by play data from eightthirtyfour
    # Their extended datasets contain all players on the 
    # court at each play-by-play event. The dataset starts 
    # in the 2008-2009 season and ends the 2018-2019 season.
    url_base = 'https://eightthirtyfour.com/nba/pbp/'
    # list extended files
    csv_files = [f'events_{y-1}-{y}_pbp.csv' 
                 for y in pbp_seasons]
    local_files = []
    for csv_file in csv_files:

        local_file = f'{etf_base}/{csv_file}'
        local_files.append(local_file)

        if not os.path.exists(local_file):
            # downloaded if file doesn't exist locally
            if verbose:
                print(f'Downloading {csv_file}')

                url = url_base +  csv_file
                with requests.get(url, stream=True) as r:
                    with open(local_file, 'wb') as f:
                        shutil.copyfileobj(r.raw, f) 

    return local_files


def get_player_data():
    # get player id list, if not downloaded
    url = 'https://eightthirtyfour.com/nba/pbp/playerlist.csv'

    if not os.path.exists(player_file):

        if verbose:
            print('Saving player id list')
        
        with requests.get(url, stream=True) as r:
            with open(player_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)


def extract_possession(pbp_file):
    df = _read_and_clean(pbp_file)
    df_poss = _get_possessions(df, pbp_file)
    return df_poss


def multi_extract_possession(pbp_files):
    # Go through each pbp file and extract
    # possession and outcome. Parallelized

    # create a progress bar
    pbar = tqdm(total=len(pbp_files))
    # function to update progress bar
    def update(*a):
        pbar.update()
    # create pool to parallelize
    pool = Pool()
    # store results, turn to dataframe after
    df_poss = []
    # extract each pbp file
    for i in range(pbar.total):
        p = pool.apply_async(extract_possession, 
                             args=(pbp_files[i],), 
                             callback=update)
        # add pool results to list
        df_poss.append(p)
    # wait for parallel processes to finish
    pool.close()
    pool.join()
    # close when all processes finished
    pbar.close()
    # combine parallel job results into dataframe
    df_poss = pd.concat([p.get() for p in df_poss], 
                        ignore_index=True)
    # save to possession file
    df_poss.to_csv(possession_file)


def _runs_of_ones_array(bits):
    # fast python way to find runs, from: 
    # https://stackoverflow.com/a/1066838
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return list(zip(run_starts, run_ends))


def _read_and_clean(pbp_file):
    # read and prep pbp file
    
    # read pbp file, some col need to be treated 
    # as obj and need more memory
    df = pd.read_csv(pbp_file, 
                     low_memory=False)

    df = pd.read_csv(pbp_file, low_memory=False)
    # fix game index column name
    df.rename(columns={'Unnamed: 0' : 'GAME_IDX'}, inplace=True)
    # make sure games sorted properly
    df.sort_values(['GAME_ID', 'GAME_IDX'], inplace=True)
    # remove any games with irregular times
    g = df.groupby('GAME_ID').TIME.apply(
        lambda x: (x <= x.shift(-1).fillna(method='ffill')).all())
    df = df[~df.GAME_ID.isin(g[g==False].index)].copy()
    # remove any games with irregular scores
    g1 = df.groupby('GAME_ID').HOME_SCORE.apply(
        lambda x: (x.diff().fillna(0) >= 0).all())
    g2 = df.groupby('GAME_ID').AWAY_SCORE.apply(
        lambda x: (x.diff().fillna(0) >= 0).all())
    bad_games = np.concatenate([g1[g1==False].index, 
                                g2[g2==False].index])
    df = df[~df.GAME_ID.isin(bad_games)].copy() 
    # remove acts not necessary for determining possession
    df = df[~df.EVENTMSGTYPE.isin([6,7,8,9,10,11,18])].copy() 

    return df   


def _get_possessions(df, pbp_file):

    # determine the team (home/visitor) of make actor 
    # (PERSON1TYPE)
    df['MAINACTOR'] = np.where(df.PERSON1TYPE==0, 'OTHER', 
                               np.where(df.PERSON1TYPE.isin([2,4]), 
                                        'HOME', 'VISITOR'))
    # find possessions (runs of MAINACTOR)
    # store possession results, turn to dataframe after
    poss_res = []   
    # go through each game
    for gameid, g, in df.groupby('GAME_ID'):
        # find home team possessions
        # look for runs where MAINACTOR is home
        home_poss = _runs_of_ones_array(g.MAINACTOR=='HOME')
        # store home possession runs in dataframe
        home_poss = pd.DataFrame(home_poss, 
                                 columns=['start', 'end'])
        # add that possession team is home
        home_poss['POSS_TEAM'] = 'HOME'
        # repeat with visitor possessions
        visitor_poss = _runs_of_ones_array(g.MAINACTOR=='VISITOR')
        visitor_poss = pd.DataFrame(visitor_poss, 
                                    columns=['start', 'end'])
        visitor_poss['POSS_TEAM'] = 'VISITOR'
        # combine home and visitor possession runs
        poss_df = pd.concat([home_poss, visitor_poss], 
                            ignore_index=True)
        # sort game in possession order
        poss_df.sort_values('start', inplace=True)
        # number the possesssions
        poss_df['POSS_NUM'] = np.arange(len(poss_df))
        # find proper starting game index
        poss_df['GAME_IDX'] = g.iloc[poss_df.start].GAME_IDX.values
        # add game id to possessions
        poss_df['GAME_ID'] = gameid
        # remove the possession times, not needed
        poss_df.drop(columns=['start', 'end'], inplace=True)
        # add to running results
        poss_res.append(poss_df)
    # combine possessions into dataframe      
    poss_res = pd.concat(poss_res, ignore_index=True)

    # add new possession data to full dataframe
    # this will add possession numbers to corresponding 
    # pbp row. some rows will not get a match.
    # these NA rows are filled in below
    df = df.merge(poss_res, on=['GAME_ID', 'GAME_IDX'], 
                  how='outer')  
    df.reset_index(drop=True, inplace=True)
    # set missing teams to OTHER. these are the events with no
    # possession (like jump ball)
    df.loc[df.EVENTMSGTYPE.isin([12,13]), 
           'POSS_TEAM'] = 'OTHER'
    # set possession num to -1 for events with no possession
    df.loc[df.EVENTMSGTYPE.isin([12,13]), 
           'POSS_NUM'] = -1
    # fill in missing teams
    df['POSS_TEAM'] = df.groupby('GAME_ID').POSS_TEAM.apply(
        lambda x: x.fillna(method='ffill')) 
    # fill in missing possession numbers
    df['POSS_NUM'] = df.groupby('GAME_ID').POSS_NUM.apply(
        lambda x: x.fillna(method='ffill'))    
    # fill any missing poss number with -1 
    df.POSS_NUM.fillna(-1, inplace=True)
    # determine points scored per possession
    points = df.groupby('GAME_ID').apply(
        lambda x: np.where(x.POSS_TEAM=='HOME', 
                           x.HOME_SCORE.diff(), 
                           x.AWAY_SCORE.diff()))
    points = np.concatenate(points.values)
    points = np.nan_to_num(points, 0)
    df['POSS_POINTS'] = points
    # remove no possesssions
    df = df[df.POSS_NUM>=0].copy() 
    # group pbp events into possessions
    df_poss = df.groupby(['GAME_ID', 'POSS_NUM', 
        'POSS_TEAM']).agg({'POSS_POINTS':np.sum, 
        'PERIOD':'last', 'HOME_PLAYER_ID_1':'last',
        'HOME_PLAYER_ID_2':'last', 'HOME_PLAYER_ID_3':'last', 
        'HOME_PLAYER_ID_4':'last', 'HOME_PLAYER_ID_5':'last',
        'AWAY_PLAYER_ID_1':'last', 'AWAY_PLAYER_ID_2':'last',
        'AWAY_PLAYER_ID_3':'last', 'AWAY_PLAYER_ID_4':'last', 
        'AWAY_PLAYER_ID_5':'last'}).astype(np.int64)
    # add season from filename
    season = ''.join(i for i in pbp_file.split('/')[-1] 
                     if i.isdigit())
    df_poss['SEASON'] = season
    # make sure possession dataframe index sorted
    df_poss.reset_index(inplace=True)   
    # assign home/visitor as offense/defense
    # need to split df_poss into home and away
    # then rename columns to off and def
    df_home = df_poss[df_poss.POSS_TEAM=='HOME'].copy()
    df_visitor = df_poss[df_poss.POSS_TEAM=='VISITOR'].copy()
    df_home.rename(columns={
        'HOME_PLAYER_ID_1':'OFF1', 'HOME_PLAYER_ID_2':'OFF2', 
        'HOME_PLAYER_ID_3':'OFF3', 'HOME_PLAYER_ID_4':'OFF4',
        'HOME_PLAYER_ID_5':'OFF5', 'AWAY_PLAYER_ID_1':'DEF1',
        'AWAY_PLAYER_ID_2':'DEF2', 'AWAY_PLAYER_ID_3':'DEF3',
        'AWAY_PLAYER_ID_4':'DEF4', 'AWAY_PLAYER_ID_5':'DEF5'}, 
        inplace=True)
    df_visitor.rename(columns={
        'AWAY_PLAYER_ID_1':'OFF1', 'AWAY_PLAYER_ID_2':'OFF2', 
        'AWAY_PLAYER_ID_3':'OFF3', 'AWAY_PLAYER_ID_4':'OFF4',
        'AWAY_PLAYER_ID_5':'OFF5', 'HOME_PLAYER_ID_1':'DEF1',
        'HOME_PLAYER_ID_2':'DEF2', 'HOME_PLAYER_ID_3':'DEF3',
        'HOME_PLAYER_ID_4':'DEF4', 'HOME_PLAYER_ID_5':'DEF5'}, 
        inplace=True)
    # recombine home/away now as off/def in order
    df_poss = pd.concat([df_home, df_visitor], 
                        sort=True)[df_home.columns]

    return df_poss    


