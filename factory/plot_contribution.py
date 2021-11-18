import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from glob import glob
from tqdm import tqdm

def plot_contribution(read_directory='data/contribution_predictions/',
                      save_directory='data/plots/contribution/'):
    
    print('Plotting')
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    player_info = pd.read_csv('data/raw_pbp/playerlist.csv', index_col=[0])
    pred_files = sorted(glob(read_directory + '*.pkl'))

    for pred_file in tqdm(pred_files):
        
        season = pred_file.split('/')[-1].split('.')[0]
        
        filename = save_directory + pred_file.split('/')[-1].replace('.pkl', '.html')

        if os.path.exists(filename):
            continue

        df = pd.read_pickle(pred_file)

        df['weight'] = 1 / df.cv_mse

        player_off = df.groupby('PERSON_ID').apply(lambda x: (x['weight'] * x['off_predict']).sum() / x['weight'].sum()).reset_index()
        player_off.rename(columns={0:'off'}, inplace=True)

        player_def = df.groupby('PERSON_ID').apply(lambda x: (x['weight'] * x['def_predict']).sum() / x['weight'].sum()).reset_index()
        player_def.rename(columns={0:'def'}, inplace=True)

        player_time = df.groupby('PERSON_ID').TIME.min().reset_index()
        player_time.rename(columns={0:'time'}, inplace=True)

        res = pd.merge(player_off, player_def, on='PERSON_ID').merge(player_time, on='PERSON_ID')
        res = res.merge(player_info[['PERSON_ID', 'DISPLAY_FIRST_LAST']], on='PERSON_ID')
                
        # add team name
        team_df = _get_team_df(season)
        res = team_df.merge(res, on='PERSON_ID', how='left').dropna()
        
        teams = sorted(res.TEAM.unique())

        fig = go.Figure(go.Scatter(x = res['off'], 
                                   y = res['def'], 
                                   mode = 'markers', 
                                   marker = {'size' : res['TIME']/5000},
                                   customdata = (res['TIME']/60).round(),
                                   hovertemplate = ('<b>%{hovertext}</b><br>' +
                                                    'Off : %{x:.2f}<br>' +
                                                    'Def : %{y:.2f}<br>' +
                                                    'Minutes : %{customdata}' + 
                                                    '<extra></extra>'),
                                   hovertext = res['DISPLAY_FIRST_LAST']))

        fig.update_layout(title={'text': season,
                                 'y':0.98, 'x':0.5, 
                                 'xanchor': 'center', 'yanchor': 'top'},
                          autosize = True,
                          xaxis = dict(title = 'Offensive PPM'), 
                          yaxis = dict(title = 'Defensive PPM'))

        fig.add_vline(x=res['off'].mean(), line_width=2, line_dash="dash", line_color="red")
        fig.add_hline(y=res['off'].mean(), line_width=2, line_dash="dash", line_color="red")
        
        buttons = [dict(method='update',
                        label='All',
                        args=[{'x': [res['off']],
                               'y': [res['def']],
                               'customdata' : [(res['TIME']/60).round()],
                               'hovertext' : [res['DISPLAY_FIRST_LAST']],
                               'marker' : [{'size' : res['TIME']/5000, 
                                            'text' : res['DISPLAY_FIRST_LAST']}]}])]

        for team in teams:

            buttons.append(dict(method='update',
                                label=team,
                                args=[{'x': [res.loc[res.TEAM==team, 'off']],
                                       'y': [res.loc[res.TEAM==team, 'def']],
                                       'customdata' : [(res.loc[res.TEAM==team, 'TIME']/60).round()],
                                       'hovertext' : [res.loc[res.TEAM==team, 'DISPLAY_FIRST_LAST']],
                                       'marker' : [{'size' : res.loc[res.TEAM==team, 'TIME']/5000, 
                                                    'text' : res.loc[res.TEAM==team, 'DISPLAY_FIRST_LAST']}]}]))


        fig.update_layout(updatemenus=[dict(buttons=buttons, direction='down', x=0.1, y=1.1, showactive=True)])

        pio.write_html(fig, file=filename, auto_open=False)
        
        
def _get_team_df(season):
    
    season_file = f'data/raw_pbp/{season}_pbp.csv'
    season_pbp = pd.read_csv(season_file)
    
    team_df = pd.DataFrame(np.vstack([season_pbp[['PLAYER1_ID', 'PLAYER1_TEAM_NICKNAME']].dropna().drop_duplicates().values,
                                      season_pbp[['PLAYER2_ID', 'PLAYER2_TEAM_NICKNAME']].dropna().drop_duplicates().values,
                                      season_pbp[['PLAYER3_ID', 'PLAYER3_TEAM_NICKNAME']].dropna().drop_duplicates().values]), 
                           columns=['PERSON_ID', 'TEAM'])
    team_df.drop_duplicates(inplace=True)
    
    return team_df