# Parameters for experiment

# whether to print text while running
verbose = True

# change to skip training and go to predictions
experiment_id = None '20200330175259675368'

# Training params
# list of factors to consider in the FM
factors = [['offense', 'defense']]
# seasons of possessions to use when training
# an emtpy list means use all seasons
seasons = [[], ['20182019'], 
           ['20172018'], ['20162017'], 
           ['20152016'], ['20142015'], 
           ['20132014'], ['20122013'], 
           ['20112012'], ['20102011'], 
           ['20092010'], ['20082009']]
# remove players with number of possessions below
# threhsold determined by quantiles. the more data
# per player, the better the model -- at the expense
# of learning about less players 
player_min_poss_quantiles = [0.2, 0.3, 0.4]
# remove groups of factors with number of 
# possessions below threshold determined by 
# quantiles. again, controls the number of data
# points per player at the expense of learning
# about less players.
group_min_poss_quantiles = [0.2, 0.3, 0.4]
# parameters passed to train xlearn FMs
# all must be regressions. more exploration
# into parameters is needed.
train_params = [{'task':'reg'}]

# Prediction params
# whether to makes predictions of individual
# offensive and defensive PPPs
individual_predictions = True
# whether to plot individual PPPs
individual_plots = True
# whether to test individual PPPs against
# existing basketball metrics 
individual_correlations = True
# what year to test basketall metric
correlation_season = '20182019'
# creates plots of individaul PPPs over their 
# career. Add name to list. If None, skips.
individual_series = ['Steve Nash', 'LeBron James', 
                     'Andrew Wiggins', 
                     'Andrea Bargnani', 
                     'Kevin Durant', 'James Harden', 
                     'Giannis Antetokounmpo', 
                     'Vince Carter']
# variable to colour code career series plots
series_bars = 100
# whether to make predictions of offensive
# and defensive pairings
tandem_predictions = False#True
# what quantile of players to plot pairings
# predictions. Needs to be high, otherwise
# way too many points to plot
tandem_quantile = 0.95

# Hard-coded parameters. don't need to change
possession_file = 'data/df_all_poss.csv' 
etf_base = 'data/eightthirtyfour'
player_file = f'{etf_base}/playerlist.csv'
pbp_seasons = range(2009, 2020)
xlearn_folder = 'data/xlearn'
valid_size = 0.2
prediction_folder = 'imgs'