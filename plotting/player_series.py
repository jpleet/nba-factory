import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.append('../')
from params import *

def plot(df):

    if verbose:
        print('Creating player PPP season plots')

    # limit df to single seasons
    season_df = df[df.seasons!='all'].copy()
    # make sure seasons sorted
    season_df.sort_values('seasons', inplace=True)
    # store season means
    season_means = season_df.groupby('seasons')[['off_wmn', 
                                                 'def_wmn']].mean()
    season_means.columns = ['off_mean', 'def_mean']
    # go through players and save their series
    for name in individual_series:

        player_df = season_df[season_df.name==name]
        player_df = player_df.merge(season_means, 
                                     on='seasons', how='left')
        # get colors
        color_step = 255 / series_bars
        green_map = cm.get_cmap('Greens')
        red_map = cm.get_cmap('Reds')
        # create x values for plot
        x = np.arange(len(player_df))
        # create plot
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        
        # plot offense
        ax[0].plot(player_df.off_wmn, marker='o', 
                   label='offense', color='black', lw=3, 
                   markersize=8)
        ax[0].plot(player_df.off_mean, color='grey')
        # add good colors
        step = (1.75 - player_df.off_mean.min()) / series_bars
        for b in range(series_bars):
            y1 = player_df.off_mean + step * b
            y2 = (player_df.off_mean + step) + step * b
            ax[0].fill_between(x, y1, y2, 
                color=green_map(int(b*color_step)))
        # add bad colors
        step = (player_df.off_mean.max() - 0.5) / series_bars
        for b in range(series_bars):
            y1 = player_df.off_mean - step * b
            y2 = (player_df.off_mean - step) - step * b
            ax[0].fill_between(x, y1, y2, 
                color=red_map(int(b*color_step)))
        # add hatch to red (to help colorblind)
        ax[0].fill_between(x, player_df.off_mean, 
                           [0.5]*len(player_df), 
                           facecolor="none", hatch="x", 
                           edgecolor="grey")
        # adjust offensive ticks and labels
        ax[0].set_xticks(range(len(player_df)))
        ax[0].set_xticklabels(player_df.seasons, 
                              rotation=45)
        ax[0].set_ylim(0.5, 1.75)
        ax[0].set_ylabel('Offensive PPP', 
                         labelpad=10, fontsize=16)

        # plot defense
        ax[1].plot(player_df.def_wmn, marker='o', 
                   label='defense', color='black', lw=3, 
                   markersize=8)
        ax[1].plot(player_df.def_mean, color='grey')
        # add good colors
        step = (player_df.def_mean.max() - 0.5) / series_bars
        for b in range(series_bars):
            y1 = player_df.def_mean - step * b
            y2 = (player_df.def_mean - step) - step * b
            ax[1].fill_between(x, y1, y2, 
                color=green_map(int(b*color_step)))
        # add bad colors
        step = (1.75 - player_df.def_mean.min()) / series_bars
        for b in range(series_bars):
            y1 = player_df.def_mean + step * b
            y2 = (player_df.def_mean + step) + step * b
            ax[1].fill_between(x, y1, y2, 
                color=red_map(int(b*color_step)))   
        # add hatch to red (to help colorblind)     
        ax[1].fill_between(x, player_df.def_mean, 
                           [1.75]*len(player_df), 
                           facecolor="none", hatch="x", 
                           edgecolor="grey")
        # adjust defensive ticks and labels
        ax[1].set_xticks(range(len(player_df)))
        ax[1].set_xticklabels(player_df.seasons, 
                              rotation=45)
        ax[1].set_ylim(0.5, 1.75)
        ax[1].set_ylabel('Defensive PPP', 
                         labelpad=10, fontsize=16)
        # add title to plot
        plt.suptitle(name, fontsize=20)
        # make good spacing
        plt.subplots_adjust(wspace=0.3)
        # save
        sname = name.replace(' ', '')
        plt.savefig(f'{prediction_folder}/ppp_{sname}.png')
        # close plot
        plt.close()


