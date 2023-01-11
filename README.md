# NBA FACTORIZATION MACHINES

A framework to quantify NBA players using [factorization machines](https://ieeexplore.ieee.org/document/5694074) (FMs). An FM takes formatted play-by-play data and learns the interactions and contributions of NBA players to scoring and defending. 

Overview: Section 1 describes the data formatting. Section 2 isolates individual offensive and defensive contributions. Section 3 finds the best teammates. Section 4 discusses extensions and future analyses. The notebooks contain more information.

## 1. Data Preparation

### Download

See: `download_data.ipynb`

Play-by-play and rotation data are downloaded (really slowly) from the NBA-API.

### Format Data

See: `format_data.ipynb`
 
The rotations are converted to lineups and merged with the scores in the play-by-play data. For each lineup, the amount of home and visitor points scored over their playing time are tallied. The lineup data for a game looks like:

| Home Lineup      | Visitor Lineup    | Home Points | Visitor Points | Playing Time (s) | Quarter |
|------------------|-------------------|-------------|----------------|------------------|---------|
| (h1,h2,h3,h4,h5) | (v1,v2,v3,v4,v5)  | 10          | 6              | 240              | 1       |
| (h2,h3,h4,h5,h6) | (v1,v2,v3,v4,v5)  | 4           | 8              | 200              | 1       |
| (h2,h3,h4,h5,h6) | (v3,v4,v5,v6,v7)  | 12          | 14             | 280              | 1       |
| ...              | ...               | ...         | ...            | ...              | ...     |

This lineup data is then filtered, depending on the training requirements, and converted to a sparse format that looks like:

| off_1 | off_2 | ... | off_n | def_1 | def_2 | ... | def_n | extra |   | points per time  (normalized) |
|-------|-------|-----|-------|-------|-------|-----|-------|-------|---|-------------------------------|
| 0     | 0     | ... | 0     | 0     | 1     | ... | 0     | ...   | = | ppt1                          |
| 0     | 1     | ... | 0     | 0     | 0     | ... | 0     | ...   | = | ppt2                          |
| ...   | ...   | ... | ...   | ...   | ...   | ... | ...   | ...   | = | ...                           |

Each lineup has two inputs into this sparse format: 
- one row with the home team's offensive IDs (off_id) set, visitor team's defensive IDs set (def_id), and home points per time
- one row with the visitor team's offensive IDs (off_id) set, home team's defensive IDs set (def_id), and visitor points per time

There can be extra features, like is offense the home team or the quarter. For now there are no extra features, but something probably worth investigating later.

In the next sections, FMs learn how the presence, absence, and interaction of players affect points per time (or some normalized metric of contribution). 

## 2. Individual Contributions

See: `individual_contributions.ipynb`

For each season, several FMs are trained on the sparse lineup data and their predictions are combined to infer individual offensive and defensive contributions. 

A single FM will learn the latent factors of players from the training data and then be used to make several predictions:
1. with no players set to predict a baseline control value
2. with each individual offensive player set on its own to isolate offensive contributions
3. with each individual defensive player set on its own to isolate defensive contributions
Lastly, the predicted contributions are centered from the control.

Because there isn't an immense amount of data, FMs are both bootstrapped and ensembled. Bootstrapping involves repeatedly training on a subset of the data and then averaging the predicted contributions. Here, the left out data is used as validation and the validation accuracy is applied as a weight when averaging the predicted contributions. FMs also have several training parameters (like number of latent factors), but instead of trying to find optimal parameters, the bootstrap samples are trained on different parameters and all these FMs are ensembled to produce single individual offensive and defensive predictions.  

Links below are interactive season results. Better players have higher offensive and lower defensive contributions. Teams can be selected from the drop-down menu.

[1996-97](https://jpleet.github.io/nba-factory/results/individual/1996-97.html), 
[1997-98](https://jpleet.github.io/nba-factory/results/individual/1997-98.html), 
[1998-99](https://jpleet.github.io/nba-factory/results/individual/1998-99.html), 
[1999-00](https://jpleet.github.io/nba-factory/results/individual/1990-00.html), 
[2000-01](https://jpleet.github.io/nba-factory/results/individual/2000-01.html), 
[2001-02](https://jpleet.github.io/nba-factory/results/individual/2001-02.html), 
[2002-03](https://jpleet.github.io/nba-factory/results/individual/2002-03.html), 
[2003-04](https://jpleet.github.io/nba-factory/results/individual/2003-04.html), 
[2004-05](https://jpleet.github.io/nba-factory/results/individual/2004-05.html), 
[2005-06](https://jpleet.github.io/nba-factory/results/individual/2005-06.html), 
[2006-07](https://jpleet.github.io/nba-factory/results/individual/2006-07.html), 
[2007-08](https://jpleet.github.io/nba-factory/results/individual/2007-08.html), 
[2008-09](https://jpleet.github.io/nba-factory/results/individual/2008-09.html), 
[2009-10](https://jpleet.github.io/nba-factory/results/individual/2009-10.html), 
[2010-11](https://jpleet.github.io/nba-factory/results/individual/2010-11.html), 
[2011-12](https://jpleet.github.io/nba-factory/results/individual/2011-12.html), 
[2012-13](https://jpleet.github.io/nba-factory/results/individual/2012-13.html), 
[2013-14](https://jpleet.github.io/nba-factory/results/individual/2013-14.html), 
[2014-15](https://jpleet.github.io/nba-factory/results/individual/2014-15.html), 
[2015-16](https://jpleet.github.io/nba-factory/results/individual/2015-16.html), 
[2016-17](https://jpleet.github.io/nba-factory/results/individual/2016-17.html), 
[2017-18](https://jpleet.github.io/nba-factory/results/individual/2017-18.html), 
[2018-19](https://jpleet.github.io/nba-factory/results/individual/2018-19.html),
[2019-20](https://jpleet.github.io/nba-factory/results/individual/2019-20.html),
[2020-21](https://jpleet.github.io/nba-factory/results/individual/2020-21.html),
[2021-22](https://jpleet.github.io/nba-factory/results/individual/2021-22.html)

These individual contributions have strong correlations with other metrics of NBA players. 

-- ADD PLOTS

But unlike other metrics, this framework allows for interactions and can predict who works better together.

## 2. Teammate Dynamics

See: `teammate_dynamics.ipynb`

Over an NBA season, players have far more interactions with their teammates than opponents and FMs can probably better infer teammate dynamics.

The same bootstrap ensemble methods from above are repeated, except now the FMs predict on all combinations of teammates.

[1996-97](https://jpleet.github.io/nba-factory/results/teammates/1996-97.html), 
[1997-98](https://jpleet.github.io/nba-factory/results/teammates/1997-98.html), 
[1998-99](https://jpleet.github.io/nba-factory/results/teammates/1998-99.html), 
[1999-00](https://jpleet.github.io/nba-factory/results/teammates/1990-00.html), 
[2000-01](https://jpleet.github.io/nba-factory/results/teammates/2000-01.html), 
[2001-02](https://jpleet.github.io/nba-factory/results/teammates/2001-02.html), 
[2002-03](https://jpleet.github.io/nba-factory/results/teammates/2002-03.html), 
[2003-04](https://jpleet.github.io/nba-factory/results/teammates/2003-04.html), 
[2004-05](https://jpleet.github.io/nba-factory/results/teammates/2004-05.html), 
[2005-06](https://jpleet.github.io/nba-factory/results/teammates/2005-06.html), 
[2006-07](https://jpleet.github.io/nba-factory/results/teammates/2006-07.html), 
[2007-08](https://jpleet.github.io/nba-factory/results/teammates/2007-08.html), 
[2008-09](https://jpleet.github.io/nba-factory/results/teammates/2008-09.html), 
[2009-10](https://jpleet.github.io/nba-factory/results/teammates/2009-10.html), 
[2010-11](https://jpleet.github.io/nba-factory/results/teammates/2010-11.html), 
[2011-12](https://jpleet.github.io/nba-factory/results/teammates/2011-12.html), 
[2012-13](https://jpleet.github.io/nba-factory/results/teammates/2012-13.html), 
[2013-14](https://jpleet.github.io/nba-factory/results/teammates/2013-14.html), 
[2014-15](https://jpleet.github.io/nba-factory/results/teammates/2014-15.html), 
[2015-16](https://jpleet.github.io/nba-factory/results/teammates/2015-16.html), 
[2016-17](https://jpleet.github.io/nba-factory/results/teammates/2016-17.html), 
[2017-18](https://jpleet.github.io/nba-factory/results/teammates/2017-18.html), 
[2018-19](https://jpleet.github.io/nba-factory/results/teammates/2018-19.html),
[2019-20](https://jpleet.github.io/nba-factory/results/teammates/2019-20.html),
[2020-21](https://jpleet.github.io/nba-factory/results/teammates/2020-21.html),
[2021-22](https://jpleet.github.io/nba-factory/results/teammates/2021-22.html)

It's also fun to see who hypothetically would be the best teammates.

[2021-22](https://jpleet.github.io/nba-factory/results/teammates/2021-22_all.html)

