import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from glob import glob
from tqdm import tqdm
import requests
import time
from bs4 import BeautifulSoup

def plot_aggregate_success(func1, func2, 
                           read_directory='data/contribution_predictions/',
                           save_directory='data/plots/aggregate_success/'):
    
    print(f'Plotting {func1}-{func2}')
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    filename = save_directory + f'{func1}_{func2}.html'
    
    if os.path.exists(filename):
        return
    
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
        
    season_preds = []
    champions = []

    for season in all_res:

        team_preds = all_res[season][0].groupby('TEAM').agg({'off_norm' : func1, 'def_norm' : func2}).reset_index()

        team_stats = all_res[season][1]
        team_wls = []
        for team in team_preds.TEAM:
            wl = float(team_stats.loc[team_stats.Team.str.contains(team), 'W/L%'].values[0])
            team_wls.append(wl)
        team_preds['WL'] = team_wls
        team_preds['TEAM'] = team_preds.TEAM + '_' + season

        champions.append(all_res[season][2].split(' ')[-1] + '_' + season)

        season_preds.append(team_preds)

    season_preds = pd.concat(season_preds, ignore_index=True)
    season_preds['is_champion'] = season_preds.TEAM.isin(champions)
    
    fig = go.Figure(go.Scatter(x = season_preds['off_norm'], 
                               y = season_preds['def_norm'], 
                               mode = 'markers', 
                               marker = dict(color = season_preds['WL'], 
                                             size = season_preds['WL']*50,
                                             symbol = np.where(season_preds.is_champion==True, 'star', 'circle'),
                                             colorscale = 'Inferno',
                                             colorbar=dict(title="Win%"),
                                             showscale=True),
                               hovertemplate = ('<b>%{hovertext}</b><br>' + 
                                                'Off : %{x:.2f}<br>' + 
                                                'Def : %{y:.2f}<br>' +
                                                'Win% : %{marker.color}'
                                                '<extra></extra>'),
                               hovertext = season_preds['TEAM']))

    fig.update_layout(title={'text': 'Team Average Individual PPM with Season Win Percent',
                             'y':0.98, 'x':0.5, 
                             'xanchor': 'center', 'yanchor': 'top'},
                      autosize = True,
                      xaxis = dict(title = 'Offensive PPM (normalized)'),
                      yaxis = dict(title = 'Defensive PPM (normalized)'))

    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="red")

    pio.write_html(fig, file=filename, auto_open=False)
    
    return
