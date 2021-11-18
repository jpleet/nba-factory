from factory.download_data import download
from factory.format_data import format_data
from factory.train_contribution import train_contribution
from factory.plot_contribution import plot_contribution

from train_params import *

# try to download
download()
print()

# try to format
format_data()
print()

# train
train_contribution(lineup_quantiles, season_quantiles, params)
print()

# plot
plot_contribution()