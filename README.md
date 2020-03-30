# The NBA Factory Machine

## Overview

Play-by-play data is first formatted into sparse possession results, that looks like this:

![](imgs/sparse_regression.png).

Each row is a grouping of offensive players, defensive players, and other possible factors along with the group's average points per possession (PPP). FMs are trained on this sparse possession data to learn how players contribute to offensive and defensive PPP. The trained FMs can then be used to predict PPP for different combinations of offensive and defensive players. 

### Case 1: Individual predictions

The simplest case is looking at predictions of single offensive and defensive players. If only one player is selected, the FMs predict how much that player's presence alone influences the average PPP. The ideal player would have the highest PPP on offense and the lowest PPP on defense (bottom right of these interactive plots).   

Invidual player contibutions in the 2018-2019 season:
[imgs/player_ppp_2018-2019.html](imgs/player_ppp_2018-2019.html) 

Individual player contributions from 2008 to 2019:
[img/player_ppp_2008-2019.html](imgs/player_ppp_2018-2019.html) 

The size of the points are relative to the player's number of possessions. The FMs can better learn about players with more possession data, so there's likely more confidence in larger points. 

These individual contribution predictions, to me, make sense and also comply with existing basketball metrics.
There are strong correlations between individual offensive and defensive PPP and individual [offensive](https://en.wikipedia.org/wiki/Offensive_rating) and [defensive](https://en.wikipedia.org/wiki/Defensive_rating) ratings, respectively.

![](imgs/compare_ppp_rtg_2019.png)

Individual offensive and defensive PPP strongly correlates with Offensive and Defensive [Win Shares](https://en.wikipedia.org/wiki/Win_Shares). The defensive metrics negatively correlate because larger Win Shares and lower defensive PPP are better.

![win shares comparison](imgs/compare_ppp_ws_2019.png)

And there is also a strong correlation between individual offensive PPP and [PER](https://en.wikipedia.org/wiki/Player_efficiency_rating).  

![](imgs/compare_ppp_per_2019.png)

So these indvidual PPP predictions capture the same information as other basketball metrics. But unlike these other metrics, this framework can be extended to measure the interactions between players.

### Case 2: Tandem predictions

The next simplest case is when only two offensive or two defensive players are selected for prediction. The results represent how the duo together influence the average PPP. Again, ideal pairings have higher predicted offensive PPP and lower predicted defensive PPP (bottom right of these interactive plots).

Tandom contribtions for the 2018-2019 season:
[imgs/tandem_ppp_2018-2019.html](imgs/tandem_ppp_2018-2019.html)

Only duos with the most possessions are shown because there are too many possible combinations to plot. 

There are endless possibilities to explore with the framework 


## Technical Details   

TO DO
blah. . .
