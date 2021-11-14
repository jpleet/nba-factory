# NBA FACTORIZATION MACHINES

Quantifying the offensive and defensive contribution of NBA players with factorization machines (FMs). From play-by-play data, lineups and points scored per second are extracted. An ensemble of regression FMs are trained on the lineup data to learn individual offensive and defensive latent variables. The model form looks like:

![](sparse_regression.png)

The scripts contain more information.

## Individual Contributions

Created from running `main_contribution.py`

Predictions from the FMs are used to isolate each player's offensive and defensive contributions, a measure of points per minute in a controlled game. Ideal players have higher offensive values (more points scored, x-axis) and lower defensive values (less scored on, y-axis). See these links for interactive hover plots. The dropdown menu can select individual teams.

[2000-01](https://jpleet.github.io/nba-factory/data/plots/2000-01.html), 
[2001-02](https://jpleet.github.io/nba-factory/data/plots/2001-02.html), 
[2002-03](https://jpleet.github.io/nba-factory/data/plots/2002-03.html), 
[2003-04](https://jpleet.github.io/nba-factory/data/plots/2003-04.html), 
[2004-05](https://jpleet.github.io/nba-factory/data/plots/2004-05.html), 
[2005-06](https://jpleet.github.io/nba-factory/data/plots/2005-06.html), 
[2006-07](https://jpleet.github.io/nba-factory/data/plots/2006-07.html), 
[2007-08](https://jpleet.github.io/nba-factory/data/plots/2007-08.html), 
[2008-09](https://jpleet.github.io/nba-factory/data/plots/2008-09.html), 
[2009-10](https://jpleet.github.io/nba-factory/data/plots/2009-10.html), 
[2010-11](https://jpleet.github.io/nba-factory/data/plots/2010-11.html), 
[2011-12](https://jpleet.github.io/nba-factory/data/plots/2011-12.html), 
[2012-13](https://jpleet.github.io/nba-factory/data/plots/2012-13.html), 
[2013-14](https://jpleet.github.io/nba-factory/data/plots/2013-14.html), 
[2014-15](https://jpleet.github.io/nba-factory/data/plots/2014-15.html), 
[2015-16](https://jpleet.github.io/nba-factory/data/plots/2015-16.html), 
[2016-17](https://jpleet.github.io/nba-factory/data/plots/2016-17.html), 
[2017-18](https://jpleet.github.io/nba-factory/data/plots/2017-18.html), 
[2018-19](https://jpleet.github.io/nba-factory/data/plots/2018-19.html)

Thoughout a season, there's more lineup pairings between teammates than opposing players, so the FMs likely better learn interactions between teammates, making these predicted contribution values more relative to team dynamics, which explains why some non-All-Stars stand out.

## Individual Contributions and Team Success

TO DO

Created from running `main_contribution_success.py`

Examining if individual contributions within a team correspond to team sucess.

## Teammates

TO DO

Who are good teammates? Prediction together in a control environment. Maybe create matrix plot of above/below individual max OR use both players individual and if above
