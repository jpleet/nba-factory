import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from glob import glob
from tqdm import tqdm
import requests
import time
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score

def plot_top_success(read_directory='data/contribution_predictions/',
                     save_directory='data/plots/top_success/'):
    
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    filename = save_directory + f'top_r2.png'
    
    if os.path.exists(filename):
        return
    
    print('Getting plotting data')
    all_res = _get_data(read_directory)
    
    print('Plot cross-val test')
    _plot_cross_val_test(all_res, save_directory)
    
    print('Plot winning importance')
    _plot_winning_importance(all_res, save_directory)
    
    print('Plot championship importance')
    _plot_championship_importance(all_res, save_directory)


def _plot_championship_importance(all_res, save_directory, top = 6):
    
    save_file = save_directory + 'championship_importance.png'
    
    if os.path.exists(save_file):
        return
    
    xs = []
    ys = []
    teams = []

    for season in all_res:

        team_df = all_res[season][0]
        team_stats = all_res[season][1]
        champion = all_res[season][2]

        for team, g in team_df.groupby('TEAM'):
            x = g.nlargest(top, 'TIME')[['off_norm', 'def_norm']].unstack().values
            y = 1 if team in champion else 0

            xs.append(x)
            ys.append(y)
            teams.append(team + '_' + season)

    xs = np.vstack(xs)
    ys = np.array(ys)

    fts = []
    for ntree in tqdm([50, 75, 100, 125, 150, 175, 200]):

        for i in np.where(ys==1)[0]:

            xs_temp = xs[[x for x in range(len(xs)) if x != i]]
            ys_temp = ys[[y for y in range(len(xs)) if y != i]]

            rfr = BalancedRandomForestClassifier(n_estimators=ntree)
            rfr.fit(xs_temp, ys_temp)
            ft = rfr.feature_importances_
            fts.append(ft)
            
    fts = np.vstack(fts)
    
    feature_names = ['off' + str(i+1) for i in range(top)] + ['def' + str(i+1) for i in range(top)]
    
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(feature_names)):
        ax.boxplot(fts[:, i], positions=[i])
    ax.set_xticklabels(feature_names)
    ax.set_ylabel('Feature Importance', labelpad=10)
    ax.set_title('Championship Feature Importance')
    
    plt.savefig(save_file)
    plt.close()
    
    
def _plot_winning_importance(all_res, save_directory, top = 6):
    
    save_file = save_directory + 'winning_importance.png'
    
    if os.path.exists(save_file):
        return

    xs = []
    ys = []
    teams = []
    champion_marker = []

    for season in all_res:

        team_df = all_res[season][0]
        team_stats = all_res[season][1]
        champion = all_res[season][2]

        for team, g in team_df.groupby('TEAM'):
            x = g.nlargest(top, 'TIME')[['off_norm', 'def_norm']].unstack().values
            y = float(team_stats.loc[team_stats.Team.str.contains(team), 'W/L%'].values[0])

            xs.append(x)
            ys.append(y)
            teams.append(team + '_' + season)

            champion_marker.append('star' if team in champion else 'circle')

    xs = np.vstack(xs)
    ys = np.array(ys)

    fts = []
    for ntree in tqdm([50, 75, 100, 125, 150, 175, 200]):
        rfr = RandomForestRegressor(n_estimators=ntree)
        rfr.fit(xs, ys)
        ft = rfr.feature_importances_
        fts.append(ft)
    fts = np.vstack(fts)
    
    feature_names = ['off' + str(i+1) for i in range(top)] + ['def' + str(i+1) for i in range(top)]
    
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(feature_names)):
        ax.boxplot(fts[:, i], positions=[i])
    ax.set_xticklabels(feature_names)
    ax.set_ylabel('Feature Importance', labelpad=10)
    ax.set_title('Winning Feature Importance')
    
    plt.savefig(save_file)
    plt.close()
    
    
def _plot_cross_val_test(all_res, save_directory):
    
    save_file = save_directory + 'cross_val_test.png'
    
    if os.path.exists(save_file):
        return
    
    cv_res = []

    for top in tqdm([1,2,3,4,5,6,7,8,9]):

        xs_time = []
        xs_sort = []
        ys = []
        teams = []
        champion_marker = []

        for season in all_res:

            team_df = all_res[season][0]
            team_stats = all_res[season][1]
            champion = all_res[season][2]

            for team, g in team_df.groupby('TEAM'):
                x_time = g.nlargest(top, 'TIME')[['off_norm', 'def_norm']].unstack().values
                x_sort = np.concatenate([g.off_norm.nlargest(top).values,  
                                         g.def_norm.nsmallest(top).values])            

                y = float(team_stats.loc[team_stats.Team.str.contains(team), 'W/L%'].values[0])

                xs_time.append(x_time)
                xs_sort.append(x_sort)
                ys.append(y)
                teams.append(team + '_' + season)
                champion_marker.append('star' if team in champion else 'circle')

        xs_time = np.vstack(xs_time)
        xs_sort = np.vstack(xs_sort)
        ys = np.array(ys)

        all_times = []
        all_sorts = []
        for ntree in [50, 100, 150, 200]:

            rfr = RandomForestRegressor(n_estimators=ntree)
            cvs = cross_val_score(rfr, xs_time, ys, cv=5, scoring='r2')
            all_times.append(cvs)

            rfr = RandomForestRegressor(n_estimators=ntree)
            cvs = cross_val_score(rfr, xs_sort, ys, cv=5, scoring='r2')
            all_sorts.append(cvs)

        all_times = np.concatenate(all_times)
        all_sorts = np.concatenate(all_sorts)

        cv_res.append([top, all_times, all_sorts]) 
        
        
    fig, ax = plt.subplots(1,2, figsize=(16,6), sharex=True, sharey=True)
    for (top, times, sorts) in cv_res:
        ax[0].boxplot(sorts, positions=[top])
        ax[0].set_title('Top Sorted', fontsize=18)
        ax[1].boxplot(times, positions=[top])
        ax[1].set_title('Time Sorted', fontsize=18)

    ax[0].plot([t for t,_,_ in cv_res], [np.median(c) for _, _, c in cv_res])
    ax[1].plot([t for t,_,_ in cv_res], [np.median(c) for _, c, _ in cv_res])

    ax[0].set_ylabel('R-square', labelpad=10, fontsize=16)
    ax[0].set_xlabel('Top-N', labelpad=10, fontsize=16)
    ax[1].set_xlabel('Top-N', labelpad=10, fontsize=16)
    
    plt.savefig(save_file)
    plt.close()
    
    
def _get_data(read_directory):
    
    player_info = pd.read_csv('data/raw_pbp/playerlist.csv', index_col=[0])
    pred_files = sorted(glob(read_directory + '*.pkl'))

    all_res = {}
    
    for pred_file in tqdm(pred_files):

        df = pd.read_pickle(pred_file)    

        # aggregate contributions
        df['weight'] = 1 / df.cv_mse

        player_off = df.groupby('PERSON_ID').apply(lambda x: (x['weight'] * x['off_predict']).sum() / x['weight'].sum()).reset_index()
        player_off.rename(columns={0:'off'}, inplace=True)

        player_def = df.groupby('PERSON_ID').apply(lambda x: (x['weight'] * x['def_predict']).sum() / x['weight'].sum()).reset_index()
        player_def.rename(columns={0:'def'}, inplace=True)

        player_time = df.groupby('PERSON_ID').TIME.min().reset_index()
        player_time.rename(columns={0:'time'}, inplace=True)

        res = pd.merge(player_off, player_def, on='PERSON_ID').merge(player_time, on='PERSON_ID')
        res = res.merge(player_info[['PERSON_ID', 'DISPLAY_FIRST_LAST']], on='PERSON_ID')

        season = pred_file.split('/')[-1].split('.')[0]
        season_pbp = pd.read_csv(f'data/raw_pbp/{season}_pbp.csv')

        team_df = pd.DataFrame(np.vstack([season_pbp[['PLAYER1_ID', 'PLAYER1_TEAM_NICKNAME']].dropna().drop_duplicates().values,
                                          season_pbp[['PLAYER2_ID', 'PLAYER2_TEAM_NICKNAME']].dropna().drop_duplicates().values,
                                          season_pbp[['PLAYER3_ID', 'PLAYER3_TEAM_NICKNAME']].dropna().drop_duplicates().values]), 
                               columns=['PERSON_ID', 'TEAM'])
        team_df.drop_duplicates(inplace=True)
        team_df = team_df.merge(res, on='PERSON_ID', how='left').dropna()

        team_df['off_norm'] = (team_df['off'] - team_df['off'].mean()) / team_df['off'].std()
        team_df['def_norm'] = (team_df['def'] - team_df['def'].mean()) / team_df['def'].std()

        # download reg record
        end_year = season.split('-')[-1]
        east, west = pd.read_html(f'https://www.basketball-reference.com/leagues/NBA_20{end_year}.html')[0:2]
        team_stats = pd.concat([east.rename(columns={'Eastern Conference' : 'Team'}), 
                                west.rename(columns={'Western Conference' : 'Team'})])

        # get championship team
        with requests.get(f'https://www.basketball-reference.com/leagues/NBA_20{end_year}.html') as resp:
            text = resp.content
        soup = BeautifulSoup(text.decode('utf-8'), features='lxml')
        champion = soup.find("strong", text="League Champion").find_next_sibling().text


        all_res[season] = (team_df, team_stats, champion)

        # sleep to not hit br too fast
        time.sleep(1)
        
    return all_res