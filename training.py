# script to train FMs from possession data

import numpy as np
import pandas as pd
import scipy.sparse as sp
import uuid
import xlearn
import joblib
import sys
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import io
from wurlitzer import pipes, STDOUT
from params import *

# harcode columns for easy usage
off_cols = ['OFF1', 'OFF2', 'OFF3', 'OFF4', 'OFF5']
def_cols = ['DEF1', 'DEF2', 'DEF3', 'DEF4', 'DEF5'] 


def train(experiment_id, factors, seasons, 
          player_min_poss_quant, group_min_poss_quant,
          train_param):
    
    model_id = uuid.uuid4()

    if verbose:
        print('Formatting possesion data')

    X_full, y_full, model_dict = _format_for_training(
        factors, seasons, player_min_poss_quant,
        group_min_poss_quant)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=valid_size)

    if verbose:
        print('Dumping data to xlearn format')

    train_file = f'/tmp/{model_id}_train.txt'
    valid_file = f'/tmp/{model_id}_valid.txt'
    full_train_file = f'/tmp/{model_id}_fulltrain.txt'
    dump_svmlight_file(X_train, y_train, train_file)
    dump_svmlight_file(X_val, y_val, valid_file)
    dump_svmlight_file(X_full, y_full, full_train_file)

    if verbose:
        print('Running cross validation')

    # setting up the FM
    fm = xlearn.create_fm()
    fm.setTrain(train_file)
    fm.setValidate(valid_file)
    
    # saving output from cross-validation
    out = io.StringIO()
    with pipes(stdout=out, stderr=STDOUT):
        fm.cv(train_param)
    out_val = out.getvalue()
    # get the cv loss
    cv_mse = float(out_val.split(
        'Average mse_loss:')[1].split('\n')[0].strip())

    if verbose:
        print('Running full training')

    # reset FM  and fully train
    fm = xlearn.create_fm()
    fm.setTrain(full_train_file)
    # saves xlearn console, though not stored
    out = io.StringIO()
    with pipes(stdout=out, stderr=STDOUT):
        fm.setTXTModel(
            f'{xlearn_folder}/{experiment_id}/{model_id}_model.txt')
        fm.fit(train_param, 
           f'{xlearn_folder}/{experiment_id}/{model_id}_model.out')
    full_out = out.getvalue()
    # store later, if ever needed

    if verbose:
        print('Cleaning and saving')

    model_dict['model_id'] = model_id
    model_dict['cv_loss'] = cv_mse
    model_dict['cv_val_size'] = valid_size
    model_dict['x_shape'] = X_full.shape

    joblib.dump(model_dict, 
                f'{xlearn_folder}/{experiment_id}/{model_id}.pkl')
    
    [os.remove(f) for f in glob(f'{train_file}*')]
    [os.remove(f) for f in glob(f'{valid_file}*')]
    [os.remove(f) for f in glob(f'{full_train_file}*')]

    if verbose:
        print(f'Finished {model_id}')

    return model_id


def _format_for_training(factors, seasons,
                         ply_min_poss_quant, 
                         grp_min_poss_quant):

    # read in extracted possesssions
    df_poss = pd.read_csv(possession_file)

    # limit to seasons:
    if len(seasons) > 0:
        df_poss = df_poss[df_poss.SEASON.isin(seasons)].copy()
        df_poss.reset_index(inplace=True, drop=True)

    # remove infrequent players
    # get all offensive player ids and counts
    off_pids, off_counts = np.unique(df_poss[off_cols].unstack(), 
                                     return_counts=True)
    # keep offensive ids above quantile threshold
    off_to_keep = off_pids[off_counts >= np.quantile(off_counts, 
                                                     ply_min_poss_quant)]
    # get all defensice player ids and counts
    def_pids, def_counts = np.unique(df_poss[def_cols].unstack(), 
                                     return_counts=True)
    # keep defensive ids above quantile threshold
    def_to_keep = def_pids[def_counts >= np.quantile(def_counts, 
                                                     ply_min_poss_quant)]
    # combine offensive and defensive ids to keep
    to_keep = np.concatenate([off_to_keep, def_to_keep])
    # limit possessions to only those with to_keep ids
    df_poss = df_poss[df_poss[off_cols + 
                              def_cols].isin(to_keep).all(1)].copy()
    df_poss.reset_index(drop=True, inplace=True)

    # go through factors and update sparse count and ids
    sparse_count = 0
    sparse_cols = []
    model_dict = {}

    for factor in factors:
        df_poss, sparse_count, sparse_cols, model_dict = _add_factor(
            factor, df_poss, sparse_count, sparse_cols, model_dict) 
    
    model_dict['factors'] = factors
    model_dict['seasons'] = seasons
    model_dict['player_min_poss_quant'] = ply_min_poss_quant

    # for each possession, group and sort the indices of the sparse
    # factors.
    df_poss['factor_idxs'] = list(map(tuple, 
                                      np.sort(df_poss[sparse_cols].to_numpy(), 
                                              axis=1)))

    # get average points per possession for each factor grouping
    df_factors = df_poss.groupby('factor_idxs').agg(
        {'POSS_POINTS':['sum', 'size', 'mean']}).reset_index()
    df_factors.columns = ['factor_idxs', 'points_sum', 'num_poss', 'ppp']
    # limit to factor groups with more possessions
    df_factors = df_factors[(df_factors.num_poss >= 
                             df_factors.num_poss.quantile(
                                grp_min_poss_quant))].copy()
    df_factors.reset_index(drop=True, inplace=True)
    model_dict['group_min_poss_quant'] = grp_min_poss_quant
    # expand factor idxs into own columns
    # get number of factors
    factor_lens = len(df_factors.factor_idxs.iloc[0])
    # create column names for sparse factors
    sp_cols = [f'spid_{c}' for c in range(factor_lens)]
    # add sparse columns to dataframe
    df_factors[sp_cols] = pd.DataFrame([list(d) for d in 
                                        df_factors.factor_idxs])
    # create matrix from factors
    df_mat = df_factors[sp_cols].unstack().reset_index()
    df_mat.columns = ['level', 'row', 'col']
    df_mat['data'] = 1
    mat = sp.csr_matrix((df_mat.data, (df_mat.row, df_mat.col)), 
                        shape=(len(df_factors), sparse_count))

    return mat, df_factors.ppp.values, model_dict


def _add_factor(factor, df_poss, sparse_count,
                sparse_cols, model_dict):

    # adds another factor to possessions

    if factor == 'offense':
        # create sparse ids for offensive players
        off_df = pd.DataFrame()
        pids, pid_counts = np.unique(df_poss[off_cols].unstack(),
                                     return_counts=True)
        off_df['player_id'] = pids
        off_df['off_count'] = pid_counts
        off_df['sparse_id'] = np.arange(len(off_df))
        for c in off_cols:
            df_poss[c +'_ID'] = pd.merge(df_poss[c], off_df, 
                                         left_on=c, right_on='player_id', 
                                         how='left')['sparse_id'].values
            sparse_cols.append(c + '_ID')
        sparse_count = off_df.sparse_id.max() + 1
        model_dict['offense'] = off_df
    
    if factor == 'defense':
        # create sparse ids for defensive players
        def_df = pd.DataFrame()
        pids, pid_counts = np.unique(df_poss[def_cols].unstack(),
                                     return_counts=True)
        def_df['player_id'] = pids
        def_df['def_count'] = pid_counts
        def_df['sparse_id'] = np.arange(len(def_df)) + sparse_count
        for c in def_cols:
            df_poss[c +'_ID'] = pd.merge(df_poss[c], def_df, 
                                         left_on=c, right_on='player_id', 
                                         how='left')['sparse_id'].values 
            sparse_cols.append(c + '_ID')
        sparse_count = def_df.sparse_id.max() + 1
        model_dict['defense'] = def_df
    
    if factor == 'team_possession':
        # create sparse ids for team possession
        team_poss_df = pd.DataFrame()
        team_poss_df['POSS_TEAM'] = df_poss.POSS_TEAM.unique()
        team_poss_df['sparse_id'] = np.arange(len(def_df)) + sparse_count
        df_poss['POSS_TEAM_ID'] = pd.merge(df_poss['POSS_TEAM'], team_poss_df, 
                                           on='POSS_TEAM', 
                                           how='left')['sparse_id'].values
        sparse_cols.append('POSS_TEAM_ID')
        sparse_count = team_poss_df.sparse_id.max() + 1
        model_dict['team_possession'] = team_poss_df
    
    if factor == 'period':
        # create sparse ids for game periods
        period_df = pd.DataFrame()
        period_df['PERIOD'] = np.sort(df_poss.PERIOD.unique())
        period_df['sparse_id'] = np.arange(len(period_df)) + sparse_count
        df_poss['PERIOD_ID'] = pd.merge(df_poss['PERIOD'], period_df, 
                                        on='PERIOD', 
                                        how='left')['sparse_id'].values
        sparse_cols.append('PERIOD_ID')
        sparse_count = period_df.sparse_id.max() + 1
        model_dict['period'] = period_df
    
    if factor == 'season':
        # create sparse ids for seasons 
        season_df = pd.DataFrame()
        season_df['SEASON'] = np.sort(df_poss.SEASON.unique())
        season_df['sparse_id'] = np.arange(len(season_df)) + sparse_count
        df_poss['SEASON_ID'] = pd.merge(df_poss['SEASON'], season_df, 
                                        on='SEASON', 
                                        how='left')['sparse_id'].values
        sparse_cols.append('SEASON_ID')
        sparse_count = season_df.sparse_id.max() + 1
        model_dict['season'] = season_df

    return df_poss, sparse_count, sparse_cols, model_dict