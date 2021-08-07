# NBA FACTORIZATION MACHINES

## Introduction

Training factorization machines (FMs) to quantifying the offensive and defensive contribution of NBA players. From play-by-play data, lineups and points scored per second are extracted. An ensemble of regression FMs are trained on the lineup data to learn individual offensive and defensive latent variables. The model form looks like:

![](sparse_regression.png)

Predictions with the trained FMs are used to isolate each player's offensive and defensive contributions. Better players have higher offensive values (more points scored) and lower defensive values (less scored on). The notebooks contain more information, follow in order.

## Individual Contributions

Interactive hover scatter plots of individual contributions by season. Again, ideal players have high offensive values and low defensive values.

[2000-01](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2001-02](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2002-03](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2003-04](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2004-05](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2005-06](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2006-07](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2007-08](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2008-09](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2009-10](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2010-11](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2011-12](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2012-13](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2013-14](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2014-15](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2015-16](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2016-17](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2017-18](jpleet.github.io/nba-factory/plots/2000-01.html), 
[2018-19](jpleet.github.io/nba-factory/plots/2000-01.html) 

## Tandem Contributions

TO DO