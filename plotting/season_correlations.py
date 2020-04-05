import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import linregress

import sys
sys.path.append('../')
from params import *

def plot(full_df):

    bref_year = correlation_season[4:]
    season_df = full_df[full_df.seasons == 
                        (correlation_season,)].copy()

    # rating comparison
    if verbose:
        print('Comparing rating metrics')

    _plot_rating_comparison(season_df, bref_year)
    _plot_winshares_comparison(season_df, bref_year)
    _plot_per_comparison(season_df, bref_year)


def _plot_rating_comparison(df, bref_year):
    # url with individual rating values
    rating_url = ('https://www.basketball-reference.com/' + 
                f'leagues/NBA_{bref_year}_per_poss.html')
    # download ratings
    bref_per_poss = pd.read_html(rating_url)[0]
    # format ratings
    bref_per_poss = bref_per_poss[bref_per_poss.Player != 
                                  'Player'].copy()
    bref_per_poss['ORtg'] = bref_per_poss.ORtg.astype(np.float32)
    bref_per_poss['DRtg'] = bref_per_poss.DRtg.astype(np.float32)
    # multiple rows exist if player traded in season
    # for those, only looking at Total rating
    bref_player_ratings = []
    for k, g in bref_per_poss.groupby(['Player', 'Rk']):
        if np.isin(g.Tm, 'TOT')[0]:
            row = g[g.Tm=='TOT']
        else:
            row = g
        bref_player_ratings.append(row)
    bref_player_ratings = pd.concat(bref_player_ratings)
    # merge with season ppp values
    pred_ratings = df.merge(bref_player_ratings, 
                            left_on='name', 
                            right_on='Player')
    pred_ratings.dropna(inplace=True, subset=['ORtg', 'DRtg'])
    # normalize offensive ratings and PPP
    bref_ortg_norm = ((pred_ratings.ORtg - 
                       pred_ratings.ORtg.mean()) 
                      / pred_ratings.ORtg.std())
    pred_off_wmn = ((pred_ratings.off_wmn - 
                     pred_ratings.off_wmn.mean()) 
                    / pred_ratings.off_wmn.std())
    # fit regression model to offensive data
    slope, intercept, r_value, off_p_value, std_err = linregress(
        pred_off_wmn, bref_ortg_norm)
    off_x = np.linspace(pred_off_wmn.min(), pred_off_wmn.max())
    off_y = off_x * slope + intercept
    # normalize defensive ratings and PPP
    bref_drtg_norm = ((pred_ratings.DRtg - 
                       pred_ratings.DRtg.mean()) 
                      / pred_ratings.DRtg.std())
    pred_def_wmn = ((pred_ratings.def_wmn - 
                     pred_ratings.def_wmn.mean()) 
                    / pred_ratings.def_wmn.std())
    # fit regression model to defensive data
    slope, intercept, r_value, def_p_value, std_err = linregress(
        pred_def_wmn, bref_drtg_norm)
    def_x = np.linspace(pred_def_wmn.min(), pred_def_wmn.max())
    def_y = def_x * slope + intercept
    # plot points and regression lines
    # plot offense
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].scatter(pred_off_wmn, bref_ortg_norm)
    ax[0].plot(off_x, off_y, color='black')
    ax[0].set_xlabel('Normalized Offensive PPP', 
                     labelpad=10, fontsize=14)
    ax[0].set_ylabel('Normalized Offensive Rating', 
                     labelpad=10, fontsize=14)
    at = AnchoredText('p-value = {:.2E}'.format(off_p_value), 
                      frameon=True, loc='upper right', 
                      prop=dict(bbox=dict(facecolor='white', 
                                          alpha=0.5)))
    ax[0].add_artist(at)
    # plot defense
    ax[1].scatter(pred_def_wmn, bref_drtg_norm)
    ax[1].plot(def_x, def_y, color='black')
    ax[1].set_xlabel('Normalized Defensive PPP', 
                     labelpad=10, fontsize=14)
    ax[1].set_ylabel('Normalized Defensive Rating', 
                     labelpad=10, fontsize=14)
    at = AnchoredText('p-value = {:.2E}'.format(def_p_value), 
                      frameon=True, loc='upper right', 
                      prop=dict(bbox=dict(facecolor='white', 
                                          alpha=0.5)))
    ax[1].add_artist(at)
    # add title to plot
    plt.suptitle(f'Comparing Predictions with Ratings ({bref_year})', 
                 fontsize=16)
    # better space two plots
    plt.subplots_adjust(wspace=0.5)
    # save results
    plt.savefig(f'{prediction_folder}/compare_ppp_rtg_{bref_year}.png')
    plt.close()


def _plot_winshares_comparison(df, bref_year):
    # url with individual rating values
    winshare_url = ('https://www.basketball-reference.com/' + 
                    f'leagues/NBA_{bref_year}_advanced.html')
    # download winshares
    bref_advanced = pd.read_html(winshare_url)[0]
    # format winshare data
    bref_advanced = bref_advanced[bref_advanced.Player != 
                                  'Player'].copy()
    bref_advanced['OWS'] = bref_advanced.OWS.astype(np.float32)
    bref_advanced['DWS'] = bref_advanced.DWS.astype(np.float32)
    # traded players have multiple rows, get only total for them
    bref_player_ws = []
    for k, g in bref_advanced.groupby(['Player', 'Rk']):
        if np.isin(g.Tm, 'TOT')[0]:
            row = g[g.Tm=='TOT']
        else:
            row = g
        bref_player_ws.append(row)
    bref_player_ws = pd.concat(bref_player_ws)
    # merge with season PPP data
    pred_advanced = df.merge(bref_player_ws, 
                             left_on='name', 
                             right_on='Player')#.dropna()
    pred_advanced.dropna(inplace=True, subset=['OWS', 'DWS'])
    # normalize values
    pred_advanced['off_wmn_norm'] = ((pred_advanced.off_wmn - 
                                      pred_advanced.off_wmn.mean()) 
                                     / pred_advanced.off_wmn.std())
    pred_advanced['def_wmn_norm'] = ((pred_advanced.def_wmn - 
                                      pred_advanced.def_wmn.mean()) 
                                     / pred_advanced.def_wmn.std())
    pred_advanced['OWS_norm'] = ((pred_advanced.OWS - 
                                  pred_advanced.OWS.mean()) 
                                 / pred_advanced.OWS.std())
    pred_advanced['DWS_norm'] = ((pred_advanced.DWS - 
                                  pred_advanced.DWS.mean()) 
                                 / pred_advanced.DWS.std())
    # calculate offensive linear regression
    ows_slope, ows_intercept, _, ows_p_value, _ = linregress(
        pred_advanced.off_wmn_norm, pred_advanced.OWS_norm)
    ows_x = np.linspace(pred_advanced.off_wmn_norm.min(), 
                        pred_advanced.off_wmn_norm.max())
    ows_y = ows_x * ows_slope + ows_intercept
    # calculate defensive linear regression
    dws_slope, dws_intercept, _, dws_p_value, _ = linregress(
        pred_advanced.def_wmn_norm, pred_advanced.DWS_norm)
    dws_x = np.linspace(pred_advanced.def_wmn_norm.min(), 
                        pred_advanced.def_wmn_norm.max())
    dws_y = dws_x * dws_slope + dws_intercept
    # plot points and regression lines
    # plot offense
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].scatter(pred_advanced.off_wmn_norm, pred_advanced.OWS_norm)
    ax[0].plot(ows_x, ows_y, color='black')
    ax[0].set_xlabel('Normalized Offensive PPP', 
                     labelpad=10, fontsize=14)
    ax[0].set_ylabel('Normalized OWS', 
                     labelpad=10, fontsize=14)
    at = AnchoredText('p-value = {:.2E}'.format(ows_p_value), 
                      frameon=True, loc='upper right', 
                      prop=dict(bbox=dict(facecolor='white', 
                                          alpha=0.5)))
    ax[0].add_artist(at)
    # plot defense
    ax[1].scatter(pred_advanced.def_wmn_norm, pred_advanced.DWS_norm)
    ax[1].plot(dws_x, dws_y, color='black')
    ax[1].set_xlabel('Normalized Defensive PPP', 
                     labelpad=10, fontsize=14)
    ax[1].set_ylabel('Normalized DWS', 
                     labelpad=10, fontsize=14)
    at = AnchoredText('p-value = {:.2E}'.format(dws_p_value), 
                      frameon=True, loc='upper right', 
                      prop=dict(bbox=dict(facecolor='white', 
                                          alpha=0.5)))
    ax[1].add_artist(at)
    # add title
    plt.suptitle(f'Comparing Predictions with Win Shares ({bref_year})', 
                 fontsize=16)
    # adjust plot width
    plt.subplots_adjust(wspace=0.5)
    # save
    plt.savefig(f'{prediction_folder}/compare_ppp_ws_{bref_year}.png')
    #close
    plt.close()


def _plot_per_comparison(df, bref_year):
    # url with indvidual per rating values
    per_url = ('https://www.basketball-reference.com/' + 
               f'leagues/NBA_{bref_year}_advanced.html')
    # download per  
    bref_advanced = pd.read_html(per_url)[0]
    # format per
    bref_advanced = bref_advanced[bref_advanced.Player != 
                                  'Player'].copy()
    bref_advanced['PER'] = bref_advanced.PER.astype(np.float32)
    # traded players have multiple rows, get only total for them
    bref_player_per = []
    for k, g in bref_advanced.groupby(['Player', 'Rk']):
        if np.isin(g.Tm, 'TOT')[0]:
            row = g[g.Tm=='TOT']
        else:
            row = g
        bref_player_per.append(row)
    bref_player_per = pd.concat(bref_player_per)
    pred_advanced = df.merge(bref_player_per, 
                             left_on='name', 
                             right_on='Player')#.dropna()
    pred_advanced.dropna(inplace=True, subset=['PER'])
    # normalize per
    pred_advanced['off_wmn_norm'] = ((pred_advanced.off_wmn - 
                                      pred_advanced.off_wmn.mean()) 
                                     / pred_advanced.off_wmn.std())
    pred_advanced['def_wmn_norm'] = ((pred_advanced.def_wmn - 
                                      pred_advanced.def_wmn.mean()) 
                                     / pred_advanced.def_wmn.std())
    pred_advanced['PER_norm'] = ((pred_advanced.PER - 
                                  pred_advanced.PER.mean()) 
                                 / pred_advanced.PER.std())
    # calulate per - PPP regression
    per_slope, per_intercept, _, per_p_value, _ = linregress(
        pred_advanced.off_wmn_norm, pred_advanced.PER_norm)
    per_x = np.linspace(pred_advanced.off_wmn_norm.min(), 
                        pred_advanced.off_wmn_norm.max())
    per_y = per_x * per_slope + per_intercept
    # plot per comparison
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(pred_advanced.off_wmn_norm, pred_advanced.PER_norm)
    ax.plot(per_x, per_y, color='black')
    ax.set_xlabel('Normalized Offensive PPP', 
                  labelpad=10, fontsize=14)
    ax.set_ylabel('Normalized PER', 
                  labelpad=10, fontsize=14)
    at = AnchoredText('p-value = {:.2E}'.format(per_p_value), 
                      frameon=True, loc='upper right', 
                      prop=dict(bbox=dict(facecolor='white', 
                                alpha=0.5)))
    ax.add_artist(at)
    plt.suptitle(f'Comparing Offensive PPP with PER ({bref_year})', 
                 fontsize=16)
    plt.savefig(f'{prediction_folder}/compare_ppp_per_{bref_year}.png')
    plt.close()

