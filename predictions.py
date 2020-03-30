import numpy as np
import pandas as pd
import scipy.sparse as sp
import xlearn
import os
from sklearn.datasets import dump_svmlight_file
from glob import glob
import joblib
from tqdm.auto import tqdm
from scipy.stats import linregress
import itertools
import io
from wurlitzer import pipes, STDOUT
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import cm
import mpld3
# hack to make mpld3 work 
# https://github.com/mpld3/mpld3/issues/434
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
from mpld3 import _display
_display.NumpyEncoder = NumpyEncoder

from params import *


def predict(experiment_id): 
    
    if individual_predictions:
        _individual_predictions(experiment_id)

    if tandem_predictions:
        _tandem_predictions(experiment_id)


def _individual_predictions(experiment_id):

    if verbose:
        print('Making individual experiments')
    
    # read in player info file
    player_df = pd.read_csv(player_file, index_col=[0])
    # clean up player info file
    player_df.drop_duplicates(subset=['PERSON_ID'], 
                              inplace=True)
    player_df.rename(columns={'DISPLAY_FIRST_LAST':'name'}, 
                     inplace=True)
    # load experiment information
    experiment_folder = f'{xlearn_folder}/{experiment_id}'
    experiment_file = f'{experiment_folder}/{experiment_id}.pkl'
    mids, mparams = joblib.load(experiment_file)

    # store experiment predictions
    all_preds = []
    # go through all models
    for i in tqdm(range(len(mids))):

        mid = mids[i]
        mparam = mparams[i]
        # load model info
        model_dict = joblib.load(f'{experiment_folder}/{mid}.pkl')
        # get length of predictor vector
        vec_len = model_dict['x_shape'][1]
        # set test and pred files
        test_file = '/tmp/test.txt'
        pred_file = '/tmp/pred.txt'
        # load base FM
        fm = xlearn.create_fm()

        # make offensive predictions
        # get offensive player info from model run
        off_df = model_dict['offense'].merge(player_df[['PERSON_ID', 
                                                        'name']], 
                                             left_on='player_id', 
                                             right_on='PERSON_ID', 
                                             how='left')    
        # create matrix with single offensive player per row
        off_rows = np.arange(len(off_df))
        off_cols = off_df.sparse_id
        off_data = np.repeat(1, len(off_rows))
        off_mat = sp.csr_matrix((off_data, (off_rows, off_cols)), 
                                 shape = (len(off_rows), vec_len))
        # save offensive test data to xlearn format
        dump_svmlight_file(off_mat, np.repeat(1, off_mat.shape[0]), 
                           test_file)
        # set FM to test file
        fm.setTest(test_file)
        # make predictions
        out = io.StringIO()
        with pipes(stdout=out, stderr=STDOUT):
            fm.predict(f'{experiment_folder}/{mid}_model.out', 
                       pred_file)
        # read in predictions
        with open(pred_file) as f:
            off_pred = f.read().split('\n')
        # set predictions to offensive dataframe
        off_df['off_ppp'] = [float(o) for o in off_pred if o]
        # clean up files
        [os.remove(f) for f in glob(test_file + '*')]
        [os.remove(f) for f in glob(pred_file + '*')]

        # make defensive predictions (same process as offense)
        def_df = model_dict['defense'].merge(player_df[['PERSON_ID', 
                                                        'name']], 
                                             left_on='player_id', 
                                             right_on='PERSON_ID', 
                                             how='left')    
        def_rows = np.arange(len(def_df))
        def_cols = def_df.sparse_id
        def_data = np.repeat(1, len(def_rows))
        def_mat = sp.csr_matrix((def_data, (def_rows, def_cols)), 
                                 shape = (len(def_rows), vec_len))
        dump_svmlight_file(def_mat, np.repeat(1, def_mat.shape[0]), 
                           test_file)
        fm.setTest(test_file)
        with pipes(stdout=out, stderr=STDOUT):
            fm.predict(f'{experiment_folder}/{mid}_model.out', 
                       pred_file)
        with open(pred_file) as f:
            def_pred = f.read().split('\n')
        def_df['def_ppp'] = [float(o) for o in def_pred if o]
        [os.remove(f) for f in glob(test_file + '*')]
        [os.remove(f) for f in glob(pred_file + '*')]       

        # merge predictions
        pred_df = pd.merge(off_df[['name', 'player_id', 
                                   'off_ppp', 'off_count']],
                           def_df[['name', 'player_id', 
                                   'def_ppp', 'def_count']],
                           on = ['name', 'player_id'])
        # add up all the possesion counts
        pred_df['total_count'] = (pred_df.off_count + 
                                  pred_df.def_count)
        # add extra model info to prediction dataframe
        pred_df['cv_loss'] = model_dict['cv_loss']
        pred_df['cv_weight'] = 1 / pred_df.cv_loss
        pred_df['factors'] = [mparam[1]] * len(pred_df)
        pred_df['seasons'] = [tuple(mparam[2])] * len(pred_df)
        pred_df['pminq'] = mparam[3]
        pred_df['gminq'] = mparam[4]
        pred_df['train_params'] = [mparam[5]] * len(pred_df)
        # add pred dataframe to all preds
        all_preds.append(pred_df)

    # combine all preds into one dataframe
    all_preds = pd.concat(all_preds, ignore_index=True)
    # clip outside predictions
    all_preds.loc[all_preds.off_ppp < 0, 'off_ppp'] = 0
    all_preds.loc[all_preds.def_ppp < 0 , 'def_ppp'] = 0
    # group by player and season
    # for each player take average of offensive and 
    # defensive PPP, weighted by the inverse of the 
    # prediction model's cross-validation loss.
    # so better models have more weight
    all_season_preds = []
    for k, g in all_preds.groupby(['name', 
                                   'player_id', 
                                   'seasons']):
        n, p, s = k
        off_wmn = np.average(g.off_ppp, 
                             weights=g.cv_weight)
        def_wmn = np.average(g.def_ppp, 
                             weights=g.cv_weight) 
        tot_poss = g.total_count.sum()
        all_season_preds.append([n, p, s, off_wmn, 
                                 def_wmn, tot_poss])
    all_season_preds = pd.DataFrame(all_season_preds, 
        columns=['name', 'player_id', 'seasons', 
                 'off_wmn', 'def_wmn', 'total_poss'])  

    # Make plots 
    if individual_plots:
        _make_individual_plots(all_season_preds)

    if individual_correlations:
        _make_individual_correlations(all_season_preds)

    if individual_series:
        _make_series(all_season_preds)


def _make_individual_plots(df):

    if verbose:
        print('Making individual plots')

    # go through each season parameter and make
    # scatter plots of individual player
    # offensive and defensive PPP
    for k, pred_df in df.groupby('seasons'):

        # format season parameter
        if len(k) == 0:
            s = min(pbp_seasons) - 1
            e = max(pbp_seasons)
            k = f'{s}-{e}'
        elif len(k) == 1:
            k = k[0][0:4] + '-' + k[0][4:]
        else:
            k = ', '.join(i[:4]+'-'+i[4:] for i in k)

        # scale marker sizes
        marker_size = np.exp((pred_df.total_poss - 
                              pred_df.total_poss.mean()) / 
                             pred_df.total_poss.std()) * 2
        # create plot
        #fig, ax = plt.subplots(figsize=(12,10))
        fig, ax = plt.subplots(figsize=(10,8))
        # add points to scatter plot
        sc = ax.scatter(pred_df.off_wmn, pred_df.def_wmn, 
                        s=marker_size)
        # add mean grid line
        ax.plot([pred_df.off_wmn.min(), 
                 pred_df.off_wmn.max()], 
                [pred_df.def_wmn.mean(), 
                 pred_df.def_wmn.mean()], 
                ls='--', color='grey')
        # add other mean grid line
        ax.plot([pred_df.off_wmn.mean(), 
                 pred_df.off_wmn.mean()], 
                [pred_df.def_wmn.min(), 
                 pred_df.def_wmn.max()], 
                ls='--', color='grey')
        # set title
        ax.set_title(k, fontsize=20)
        # add labels
        ax.set_ylabel('Defensive Points Per Possession', 
                      labelpad=20, fontsize=16)
        ax.set_xlabel('Offensive Points Per Possession', 
                       labelpad=20, fontsize=16)
        plt.tight_layout() # MAYBE REMOVE
        # create html labels for scatter plot
        base = ('https://ak-static.cms.nba.com/wp-content/' + 
                'uploads/headshots/nba/latest/260x190')
        labels = [(f'<img src="{base}/{pid}.png">' + 
                   f'<p>{nm}</p>') for _, (pid, nm) in 
                  pred_df[['player_id', 'name']].iterrows()]
        # create mpld3 tooltip
        tooltip = mpld3.plugins.PointHTMLTooltip(
            sc, labels=labels)
        # add tooltip to figure
        mpld3.plugins.connect(fig, tooltip)
        # save figure
        html = mpld3.fig_to_html(fig)
        sfile = f'{prediction_folder}/player_ppp_{k}.html'
        with open(sfile, 'w') as f:
            f.write(html)
        # close plot 
        plt.close()


def _make_individual_correlations(df):
    # look at one season and compare PPP values with 
    # existing basketball metrics downloaded
    # from basketball-refences
    bref_year = correlation_season[4:]

    season_preds = df[df.seasons == 
                      (correlation_season,)].copy()

    # rating comparison
    if verbose:
        print('Comparing rating metrics')

    _plot_rating_comparison(season_preds, bref_year)
    _plot_winshares_comparison(season_preds, bref_year)
    _plot_per_comparison(season_preds, bref_year)


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
                            right_on='Player')#.dropna(
                                #subset=['ORtg', 'DRtg'])
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
    ax[0].set_xlabel('Normalized Offensive Points Per Possession', 
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
    ax[1].set_xlabel('Normalized Defensive Points Per Possession', 
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
    ax[0].set_xlabel('Normalized Offensive Points Per Possession', 
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
    ax[1].set_xlabel('Normalized Defensive Points Per Possession', 
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
    ax.set_xlabel('Normalized Offensive Points Per Possession', 
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


def _make_series(df):

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
        ax[1].set_ylabel('Defensive Contribution', 
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


def _tandem_predictions(experiment_id):

    if verbose:
        print('Making tandem predictions')

    # read in player info file
    player_df = pd.read_csv(player_file, index_col=[0])
    # clean up player info file
    player_df.drop_duplicates(subset=['PERSON_ID'], 
                              inplace=True)
    player_df.rename(columns={'DISPLAY_FIRST_LAST':'name'}, 
                     inplace=True)
    # load experiment information
    experiment_folder = f'{xlearn_folder}/{experiment_id}'
    experiment_file = f'{experiment_folder}/{experiment_id}.pkl'
    mids, mparams = joblib.load(experiment_file)

    # variable to store all predictions
    all_preds = []
    # iterate over all models, make predictions
    # this is slow. eventually, pairs are limited
    # for plotting, but could limit here to speed
    # up the predictions.
    for i in tqdm(range(len(mids))):
        
        mid = mids[i]
        mparam = mparams[i]
        # load model info
        model_dict = joblib.load(f'{experiment_folder}/{mid}.pkl')
        # get length of predictor vector
        vec_len = model_dict['x_shape'][1]
        # set test and pred files
        test_file = '/tmp/test.txt'
        pred_file = '/tmp/pred.txt'
        # load base FM
        fm = xlearn.create_fm()

        # make offensive predictions
        # get offensive player info from model run
        off_df = model_dict['offense'].merge(player_df[['PERSON_ID', 
                                                        'name']], 
                                             left_on='player_id', 
                                             right_on='PERSON_ID', 
                                             how='left')
        # get all combinations of 2 offensive players
        off_duos = list(itertools.combinations(off_df.sparse_id, 
                                               2))
        # turn combinations into dataframe
        off_duos = pd.DataFrame(off_duos, columns=['sid1', 'sid2'])
        # unstack duos dataframe to easily make matrix
        off_unstack = off_duos.unstack().reset_index()
        off_unstack.columns = ['duo', 'row', 'col']
        off_mat = sp.csr_matrix(([1]*len(off_unstack), 
                                 (off_unstack.row.values, 
                                  off_unstack.col.values)), 
                                shape=(len(off_duos), vec_len))
        # save prediction data for xlearn
        dump_svmlight_file(off_mat, np.repeat(1, off_mat.shape[0]), 
                           test_file)
        # set predictions as test file for xlearn
        fm.setTest(test_file)
        # make predictions with model, saves to file
        out = io.StringIO()
        with pipes(stdout=out, stderr=STDOUT):
            fm.predict(f'{experiment_folder}/{mid}_model.out', 
                       pred_file)
        # read prediction
        with open(pred_file) as f:
            off_pred = f.read().split('\n')
        # add predictions to offensive dataframe
        off_duos['off_ppp'] = [float(o) for o in off_pred if o != '']    
        # add player id and name to duo pairs
        off_duos[['pid1', 'name1', 'oc1']] = off_duos.merge(
            off_df[['sparse_id', 'player_id', 'name', 'off_count']], 
            left_on='sid1', right_on='sparse_id', 
            how='left')[['player_id', 'name', 'off_count']]
        off_duos[['pid2', 'name2', 'oc2']] = off_duos.merge(
            off_df[['sparse_id', 'player_id', 'name', 'off_count']], 
            left_on='sid2', right_on='sparse_id', 
            how='left')[['player_id', 'name', 'off_count']]
        # clean up files
        [os.remove(f) for f in glob(test_file + '*')]
        [os.remove(f) for f in glob(pred_file + '*')] 

        # make defensive predictions
        # get defensive player info from model run    
        def_df = model_dict['defense'].merge(player_df[['PERSON_ID', 
                                                        'name']], 
                                             left_on='player_id', 
                                             right_on='PERSON_ID', 
                                             how='left')
        # get all combinations of 2 offensive players
        def_duos = list(itertools.combinations(def_df.sparse_id, 
                                               2))
        # turn combinations into dataframe
        def_duos = pd.DataFrame(def_duos, columns=['sid1', 'sid2'])
        # unstack duos dataframe to easily make matrix
        def_unstack = def_duos.unstack().reset_index()
        def_unstack.columns = ['duo', 'row', 'col']
        def_mat = sp.csr_matrix(([1]*len(def_unstack), 
                                 (def_unstack.row.values, 
                                  def_unstack.col.values)), 
                                shape=(len(def_duos), vec_len))
        # save prediction data for xlearn
        dump_svmlight_file(def_mat, np.repeat(1, def_mat.shape[0]), 
                           test_file)
        # set predictions as test file for xlearn
        fm.setTest(test_file)
        # make predictions with model, saves to file
        out = io.StringIO()
        with pipes(stdout=out, stderr=STDOUT):
            fm.predict(f'{experiment_folder}/{mid}_model.out', 
                       pred_file)
        # read prediction
        with open(pred_file) as f:
            def_pred = f.read().split('\n')
        # add predictions to offensive dataframe
        def_duos['def_ppp'] = [float(o) for o in def_pred if o != '']    
        # add player id and name to duo pairs
        def_duos[['pid1', 'name1', 'dc1']] = def_duos.merge(
            def_df[['sparse_id', 'player_id', 'name', 'def_count']], 
            left_on='sid1', right_on='sparse_id', 
            how='left')[['player_id', 'name', 'def_count']]
        def_duos[['pid2', 'name2', 'dc2']] = def_duos.merge(
            def_df[['sparse_id', 'player_id', 'name', 'def_count']], 
            left_on='sid2', right_on='sparse_id', 
            how='left')[['player_id', 'name', 'def_count']]
        # clean up files
        [os.remove(f) for f in glob(test_file + '*')]
        [os.remove(f) for f in glob(pred_file + '*')] 

        # merge predictions
        pred_duos = pd.merge(off_duos, def_duos, 
                            on = ['pid1', 'name1', 'pid2', 'name2'])
        # add extra info to prediction dataframe
        # TO DO: make this more dynamic
        pred_duos['factors'] = [tuple(mparam[1])] * len(pred_duos)
        pred_duos['seasons'] = [tuple(mparam[2])] * len(pred_duos)
        pred_duos['pminq'] = mparam[3]
        pred_duos['gminq'] = mparam[4]
        pred_duos['train_param'] = [mparam[5]] * len(pred_duos) 
        pred_duos['cv_loss'] = model_dict['cv_loss']
        pred_duos['cv_weight'] = 1 / model_dict['cv_loss']
        pred_duos['x_shape'] = [model_dict['x_shape']] * len(pred_duos)
        pred_duos['duo_min_poss'] = pred_duos[['oc1', 'oc2', 
                                               'dc1', 'dc2']].min(1) 
        # add pred to all preds
        all_preds.append(pred_duos)       

    # combine all preds into one dataframe
    all_preds = pd.concat(all_preds, ignore_index=True)
    # clip outside predictions
    all_preds.loc[all_preds.off_ppp < 0, 'off_ppp'] = 0
    all_preds.loc[all_preds.def_ppp < 0 , 'def_ppp'] = 0
    # group by duo and season
    if verbose:
        print('Combining duo predictions (slow)')
    # for each duo take average of offensive and 
    # defensive PPP, weighted by the inverse of the 
    # prediction model's cross-validation loss.
    # so better models have more weight
    all_season_preds = []
    for k, g in tqdm(all_preds.groupby(['pid1', 'name1', 
                                        'pid2', 'name2', 
                                        'seasons'])):
        pid1, nm1, pid2, nm2, seasons = k
        off_wmn = np.average(g.off_ppp, 
                             weights=g.cv_weight)
        def_wmn = np.average(g.def_ppp, 
                             weights=g.cv_weight) 
        min_poss = g.duo_min_poss.min()
        all_season_preds.append([pid1, nm1, pid2, nm2, seasons, 
                                 off_wmn, def_wmn, min_poss])
    all_season_preds = pd.DataFrame(all_season_preds, 
        columns=['pid1', 'name1', 'pid2', 'name2',
                 'seasons', 'off_wmn', 'def_wmn', 'min_duo_poss']) 

    # make interactive plot
    if verbose:
        print('Making interactive tandem plots')
    # need to limit the number of pairs, otherwise way too many
    for k, season_df in all_season_preds.groupby('seasons'):
        # limit df
        season_top = season_df[(
            season_df.min_duo_poss > 
            season_df.min_duo_poss.quantile(tandem_quantile))]
        # scale marker size
        marker_size = np.exp((season_top.min_duo_poss - 
                              season_top.min_duo_poss.mean()) 
                             / season_top.min_duo_poss.std()) * 2
        # format season parameter
        if len(k) == 0:
            s = min(pbp_seasons) - 1
            e = max(pbp_seasons) + 1
            k = f'{s}-{e}'
        elif len(k) == 1:
            k = k[0][0:4] + '-' + k[0][4:]
        else:
            k = ', '.join(i[:4]+'-'+i[4:] for i in k)
        # create figure
        #fig, ax = plt.subplots(figsize=(12,10))
        fig, ax = plt.subplots(figsize=(10,8))
        sc = ax.scatter(season_top.off_wmn, season_top.def_wmn, 
                        s=marker_size)
        ax.plot([season_df.off_wmn.min(), 
                 season_df.off_wmn.max()], 
                [season_df.def_wmn.mean(), 
                 season_df.def_wmn.mean()], 
                ls='--', color='grey')
        ax.plot([season_df.off_wmn.mean(), 
                 season_df.off_wmn.mean()], 
                [season_df.def_wmn.min(), 
                 season_df.def_wmn.max()], 
                ls='--', color='grey')  
        # set title
        ax.set_title(k, fontsize=20)
        # add labels
        ax.set_ylabel('Defensive Points Per Possession', 
                      labelpad=20, fontsize=16)
        ax.set_xlabel('Offensive Points Per Possession', 
                       labelpad=20, fontsize=16)
        plt.tight_layout()
        # create hover labels for scatter plot
        labels = [f'{n1} - {n2}' for i, (n1, n2) 
                  in season_top[['name1', 
                                 'name2']].iterrows()]
        # create mpld3 tooltip
        tooltip = mpld3.plugins.PointLabelTooltip(
            sc, labels=labels)
        # add tooltip to figure
        mpld3.plugins.connect(fig, tooltip)
        # save figure
        html = mpld3.fig_to_html(fig)
        sfile = f'{prediction_folder}/tandem_ppp_{k}.html'
        with open(sfile, 'w') as f:
            f.write(html)
        # close plot 
        plt.close()


