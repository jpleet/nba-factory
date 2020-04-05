import numpy as np
import pandas as pd
import scipy.sparse as sp
import xlearn
import os
from sklearn.datasets import dump_svmlight_file
from glob import glob
import joblib
from tqdm.auto import tqdm
import itertools
import io
from wurlitzer import pipes, STDOUT

from params import *


def predict(experiment_id): 
    
    outs = []

    if individual_predictions:
        out = _individual_predictions(experiment_id)
        outs.append(out)

    if tandem_predictions:
        out = _tandem_predictions(experiment_id)
        outs.append(out)

    return outs


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
                                                        'name',
                                                        'TEAM_NAME']], 
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
                                                        'name', 
                                                        'TEAM_NAME']], 
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
                                   'off_ppp', 'off_count', 
                                   'TEAM_NAME',]],
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
    num_seasons = all_preds.seasons.nunique()
    models_per_season = len(mids) // num_seasons
    all_season_preds = []
    for k, g in all_preds.groupby(['name', 
                                   'player_id', 
                                   'TEAM_NAME',
                                   'seasons']):
        if len(g) == models_per_season:
            # only plot players that have predictions
            # for all models. 
            n, p, t, s = k
            off_wmn = np.average(g.off_ppp.values, 
                                 weights=g.cv_weight.values)      
            def_wmn = np.average(g.def_ppp.values, 
                                 weights=g.cv_weight.values) 
            tot_poss = g.total_count.sum()
            all_season_preds.append([n, p, t, s, off_wmn, 
                                     def_wmn, tot_poss])
    all_season_preds = pd.DataFrame(all_season_preds, 
        columns=['name', 'player_id', 'team', 'seasons', 
                 'off_wmn', 'def_wmn', 'total_poss'])  

    # Make plots 
    if individual_plots:
        from plotting import player_season_scatter as pss_plot 
        pss_plot.plot(all_season_preds)

    if individual_correlations:
        from plotting import season_correlations as sc_plot
        sc_plot.plot(all_season_preds)

    if individual_series:
        from plotting import player_series as ser_plot
        ser_plot.plot(all_season_preds)

    return all_season_preds


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
    if tandem_plots:
        from plotting import tandem_season_scatter as tss_plot
        tss_plot.plot(all_season_preds)

    return all_season_preds



