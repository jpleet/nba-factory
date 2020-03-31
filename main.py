# Runs an experiment from parameters in params.py

import os
import joblib
from tqdm.auto import tqdm
from datetime import datetime
from params import *
from predictions import predict

# check if possession file exists.
if not os.path.exists(possession_file):
    # creating the possession data from play-by-play
    # only needs to be done once and saved for future
    # experiments, unless new play-by-plays added.
    import data_prep
    # check if pbp files downloaded
    pbp_files = data_prep.get_pbp_data()
    # check if player info file exists
    data_prep.get_player_data()
    # extract possession data from pbp files
    # mutliprocess to speed up a bit
    if verbose:
        print('Extracting possessions from pbp files')
    data_prep.multi_extract_possession(pbp_files)

# check to train or jump to experiment predictions
# determined if experiment_id given in params
if experiment_id is None:

    from training import train

    if verbose:
        print('Starting to train FMs')

    # create experiment ID
    experiment_id = ''.join([n for n in str(datetime.now())
                             if n.isnumeric()])
    # make folder to save experiment results
    os.makedirs(f'{xlearn_folder}/{experiment_id}')

    model_ids = []
    # TO DO: make more dynamic
    model_params = [(experiment_id, fac, seas, 
                     pmin, gmin, tp) 
                    for fac in factors 
                    for seas in seasons 
                    for pmin in player_min_poss_quantiles
                    for gmin in group_min_poss_quantiles
                    for tp in train_params]

    # train each parameter set
    for model_param in tqdm(model_params):
        mid = train(*model_param)
        model_ids.append(mid)

    # save experiment information
    joblib.dump([model_ids, model_params], 
                f'{xlearn_folder}/{experiment_id}/{experiment_id}.pkl')

# examine experiment results
if verbose:
    print('Making predictions')

predict(experiment_id)