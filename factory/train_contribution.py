import os
import io
from glob import glob
from multiprocessing import Pool

import pandas as pd
import numpy as np
from tqdm import tqdm
from wurlitzer import pipes, STDOUT
import scipy.sparse as sp
from sklearn.preprocessing import power_transform
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import xlearn

def train_contribution(lineup_quantiles, season_quantiles, 
                       train_params, directory='data/contribution_predictions/'):
	
	if not os.path.exists(directory):
		os.makedirs(directory)
		
	season_files = sorted(glob('data/lineup_scores/*.pkl'))
	
	for season_file in season_files:

		filename = directory + season_file.split('/')[-1]

		if os.path.exists(filename):
			continue

		season_res = []

		for lineup_quantile in lineup_quantiles:

			for season_quantile in season_quantiles:

				print('Training', season_file, lineup_quantile, season_quantile)
				
				# single pool to prevent pandas memory leak
				with Pool(1) as pool:
					res = pool.starmap(_train_season, [(season_file, train_params,
														lineup_quantile, season_quantile)])[0]

				season_res.append(res)

		season_res = pd.concat(season_res, ignore_index=True)

		season_res.to_pickle(filename)
	
	
def _train_season(season_file, params, lineup_quantile, season_quantile, min_season_time=3600):
    
    res = []
    count = 0  
    
    df = pd.read_pickle(season_file)
    
    # group the same exact lineups, to get an overall season results
    df = df.groupby(['off1', 'off2', 'off3', 'off4', 'off5', 
                     'def1', 'def2', 'def3', 'def4', 'def5']).agg({'seconds' : 'sum', 
                                                                   'points' : 'sum', 
                                                                   'season' : 'first'}).reset_index()
    # limit to longer lineup times, reduces the number of 0 scores
    df = df[df.seconds > df.seconds.quantile(lineup_quantile)].copy()
    
    # calculate player season times
    player_time = pd.DataFrame()
    player_time['PERSON_ID'] = df[['off1', 'off2','off3', 'off4', 'off5']].stack().values
    player_time['TIME'] = df.seconds.repeat(5).values
    player_time = player_time.groupby('PERSON_ID').TIME.sum().reset_index()
    
    # need to limit season playing time to avoid bad data
    player_time = player_time[player_time.TIME > min_season_time].copy()
    # limit to season_quantile
    player_time = player_time[player_time.TIME > player_time.TIME.quantile(season_quantile)].copy()

    # only keep good lineups with players that meet time requirements
    df = df[df[['off1', 'off2', 'off3', 'off4', 'off5', 
                'def1', 'def2', 'def3', 'def4', 'def5']].isin(player_time.PERSON_ID.values).all(1)].copy()
    
    df.reset_index(inplace=True, drop=True)

    player_df = pd.DataFrame()
    player_df['PERSON_ID'] = df[['off1', 'off2', 'off3', 'off4', 'off5']].unstack().unique()
    player_df['OFF_ID'] = np.arange(len(player_df))
    player_df['DEF_ID'] = np.arange(len(player_df)) + len(player_df)

    df = df.merge(player_df.rename(columns={'OFF_ID' : 'off1_id', 'PERSON_ID' : 'off1'}).drop(columns='DEF_ID'), on='off1')
    df = df.merge(player_df.rename(columns={'OFF_ID' : 'off2_id', 'PERSON_ID' : 'off2'}).drop(columns='DEF_ID'), on='off2')
    df = df.merge(player_df.rename(columns={'OFF_ID' : 'off3_id', 'PERSON_ID' : 'off3'}).drop(columns='DEF_ID'), on='off3')
    df = df.merge(player_df.rename(columns={'OFF_ID' : 'off4_id', 'PERSON_ID' : 'off4'}).drop(columns='DEF_ID'), on='off4')
    df = df.merge(player_df.rename(columns={'OFF_ID' : 'off5_id', 'PERSON_ID' : 'off5'}).drop(columns='DEF_ID'), on='off5')

    df = df.merge(player_df.rename(columns={'DEF_ID' : 'def1_id', 'PERSON_ID' : 'def1'}).drop(columns='OFF_ID'), on='def1')
    df = df.merge(player_df.rename(columns={'DEF_ID' : 'def2_id', 'PERSON_ID' : 'def2'}).drop(columns='OFF_ID'), on='def2')
    df = df.merge(player_df.rename(columns={'DEF_ID' : 'def3_id', 'PERSON_ID' : 'def3'}).drop(columns='OFF_ID'), on='def3')
    df = df.merge(player_df.rename(columns={'DEF_ID' : 'def4_id', 'PERSON_ID' : 'def4'}).drop(columns='OFF_ID'), on='def4')
    df = df.merge(player_df.rename(columns={'DEF_ID' : 'def5_id', 'PERSON_ID' : 'def5'}).drop(columns='OFF_ID'), on='def5')
    
    # transform points per second into something more normal-like
    #target = power_transform((df.points / df.seconds).values.reshape(-1,1)).ravel()
    # simple points per minute
    target = df.points / df.seconds * 60
    
    cols = df[['off1_id', 'off2_id', 'off3_id', 'off4_id', 'off5_id', 
               'def1_id', 'def2_id', 'def3_id', 'def4_id', 'def5_id']].stack()
    rows = cols.index.get_level_values(0)
    z = [1] * len(rows)

    mat = sp.csr_matrix((z,(rows, cols)),shape=(len(df), len(player_df)*2))

    predict_file = f'/tmp/nba_predict.txt'
    dump_svmlight_file(sp.eye(len(player_df)*2), [1]*(len(player_df)*2), predict_file)

    X_train, X_val, y_train, y_val = train_test_split(mat, target, test_size=0.1)

    train_file = f'/tmp/nba_train.txt'
    valid_file = f'/tmp/nba_valid.txt'
    full_train_file = f'/tmp/nba_full_train.txt'
    dump_svmlight_file(X_train, y_train, train_file)
    dump_svmlight_file(X_val, y_val, valid_file)
    dump_svmlight_file(mat, target, full_train_file)

    for (lr, lmb, k) in tqdm(params, position=0, leave=True):

        train_param = {'task':'reg', 'init': 0.1, 'k':k, 'lr':lr, 'lambda':lmb}

        # setting up the FM
        fm = xlearn.create_fm()
        fm.setTrain(train_file)
        fm.setValidate(valid_file)

        out = io.StringIO()
        with pipes(stdout=out, stderr=STDOUT):
            fm.cv(train_param)
        out_val = out.getvalue()
        # get the cv loss
        cv_mse = float(out_val.split('Average mse_loss:')[1].split('\n')[0].strip())

        fm = xlearn.create_fm()
        fm.setTrain(full_train_file)
        fm.setTXTModel(f'/tmp/model.txt')

        out = io.StringIO()
        with pipes(stdout=out, stderr=STDOUT):
            fm.fit(train_param, f'/tmp/model.out')
        full_out_val = out.getvalue()

        # run prediction
        #fm = xlearn.create_fm()
        fm.setTest(predict_file)
        # make predictions
        pred_file = '/tmp/predict.txt'
        out = io.StringIO()
        with pipes(stdout=out, stderr=STDOUT):
            fm.predict(f'/tmp/model.out', pred_file)

        pred_df = pd.DataFrame()
        pred_values = []
        with open(pred_file) as f:
            for line in f:
                pred_values.append(float(line.replace('\n', '')))
        pred_df['predict'] = pred_values
        pred_df['LAT_ID'] = np.arange(len(pred_df))

        player_df_temp = player_df.copy()
        player_df_temp = player_df_temp.merge(pred_df.rename(columns={'predict' : 'off_predict', 'LAT_ID' : 'OFF_ID'}), on='OFF_ID', how='left')
        player_df_temp = player_df_temp.merge(pred_df.rename(columns={'predict' : 'def_predict', 'LAT_ID' : 'DEF_ID'}), on='DEF_ID', how='left')

        #player_df_temp['off_contribution_norm'] = ((player_df_temp.off_contribution - player_df_temp.off_contribution.mean()) / 
        #                                           (player_df_temp.off_contribution.std()))
        #player_df_temp['def_contribution_norm'] = ((player_df_temp.def_contribution - player_df_temp.def_contribution.mean()) / 
        #                                           (player_df_temp.def_contribution.std()))

        player_df_temp = player_df_temp.merge(player_time, on='PERSON_ID', how='left')

        player_df_temp['cv_mse'] = cv_mse
        player_df_temp['lineup_quantile'] = lineup_quantile
        player_df_temp['season_quantile'] = season_quantile
        player_df_temp['counter'] = count
        count += 1

        res.append(player_df_temp)
    
    res = pd.concat(res, ignore_index=True)
    res['season'] = season_file.split('/')[-1].split('.')[0]
    
    return res