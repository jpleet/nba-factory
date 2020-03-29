# Parameters for experiment

verbose = True

# change to skip training and go to predictions
experiment_id = '20200329155301778600' #None

# Training params
factors = [['offense', 'defense']]
seasons = [[], ['20182019'], 
           ['20172018'], ['20162017'], 
           ['20152016'], ['20142015'], 
           ['20132014'], ['20122013'], 
           ['20112012'], ['20102011'], 
           ['20092010'], ['20082009']]
#player_min_poss_quantiles = [0, 0.2, 0.3]
#group_min_poss_quantiles = [0, 0.2, 0.3]
player_min_poss_quantiles = [0.2, 0.3]
group_min_poss_quantiles = [0.2, 0.3]
#train_params = [{'task':'reg'}, 
#                {'task':'reg', 'k':5}, 
#                {'task':'reg', 'lr':0.2, 'lambda':0.002}, 
#                {'task':'reg', 'lr':0.2, 'lambda':0.002, 'k':5}]
train_params = [{'task':'reg'}]

# Prediction params
individual_predictions = False #True
individual_plots = False #True
individual_correlations = False #True
correlation_season = '' #'20182019'
"""
individual_series = ['Steve Nash', 'LeBron James', 
                     'Andrew Wiggins', 'Andrea Bargnani', 
                     'Kevin Durant', 'James Harden', 
                     'Giannis Antetokounmpo', 'Vince Carter']
"""
individual_series = None
series_bars = 100
tandem_predictions = True
tandem_quantile = 0.95

# Hard-coded parameters. don't need to change
possession_file = 'data/df_all_poss.csv' 
etf_base = 'data/eightthirtyfour'
player_file = f'{etf_base}/playerlist.csv'
pbp_seasons = range(2009, 2020)
xlearn_folder = 'data/xlearn'
valid_size = 0.2
prediction_folder = 'imgs'