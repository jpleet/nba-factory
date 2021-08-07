# NBA FACTORIZATION MACHINES

## Introduction

Training factorization machines (FMs) to quantifying the offensive and defensive contribution of NBA players. From play-by-play data, lineups and points scored per second are extracted. An ensemble of regression FMs are trained on the lineup data to learn individual offensive and defensive latent variables. The model form looks like:

![](sparse_regression.png)

Predictions with the trained FMs are used to isolate each player's offensive and defensive contributions. Better players have higher offensive values (more points scored) and lower defensive values (less scored on). The notebooks contain more information, follow in order.

## Individual Contributions

Interactive hover scatter plots of individual contributions by season. Again, ideal players have high offensive values and low defensive values.

[2000-01](https://jpleet.github.io/nba-factory/plots/2000-01.html), 
[2001-02](https://jpleet.github.io/nba-factory/plots/2001-02.html), 
[2002-03](https://jpleet.github.io/nba-factory/plots/2002-03.html), 
[2003-04](https://jpleet.github.io/nba-factory/plots/2003-04.html), 
[2004-05](https://jpleet.github.io/nba-factory/plots/2004-05.html), 
[2005-06](https://jpleet.github.io/nba-factory/plots/2005-06.html), 
[2006-07](https://jpleet.github.io/nba-factory/plots/2006-07.html), 
[2007-08](https://jpleet.github.io/nba-factory/plots/2007-08.html), 
[2008-09](https://jpleet.github.io/nba-factory/plots/2008-09.html), 
[2009-10](https://jpleet.github.io/nba-factory/plots/2009-10.html), 
[2010-11](https://jpleet.github.io/nba-factory/plots/2010-11.html), 
[2011-12](https://jpleet.github.io/nba-factory/plots/2011-12.html), 
[2012-13](https://jpleet.github.io/nba-factory/plots/2012-13.html), 
[2013-14](https://jpleet.github.io/nba-factory/plots/2013-14.html), 
[2014-15](https://jpleet.github.io/nba-factory/plots/2014-15.html), 
[2015-16](https://jpleet.github.io/nba-factory/plots/2015-16.html), 
[2016-17](https://jpleet.github.io/nba-factory/plots/2016-17.html), 
[2017-18](https://jpleet.github.io/nba-factory/plots/2017-18.html), 
[2018-19](https://jpleet.github.io/nba-factory/plots/2018-19.html)

## Tandem Contributions

TO DO: find the best teammates

## All Seasons Field-aware Factorization Machine

TO DO: are FFMs better