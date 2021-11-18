lineup_quantiles = [0.2, 0.3, 0.4]
season_quantiles = [0.2, 0.3, 0.4]

lrs = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2]
lmbs = [0.00002, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
ks = [1, 2, 3]
params = [(lr, lmb, k) for lr in lrs for lmb in lmbs for k in ks]