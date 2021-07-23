"""
Script to download data.
Data comes from: https://eightthirtyfour.com/data
The longest dataset of NBA play-by-plays I can find
Downloaded into data/raw_pbp
"""

import os
import requests
import shutil
from multiprocessing import Pool
from tqdm import tqdm

def download_season(season):
    """
    Download a single season file
    """
    
    season_name = f'{season}-{str(season+1)[2:]}'
    filename = f'data/raw_pbp/{season_name}_pbp.csv'
    if not os.path.exists(filename):
        
        url_base = 'https://eightthirtyfour.com/nba/pbp/'

        url = url_base +  season_name + '_pbp.csv'
        with requests.get(url, stream=True) as r:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f) 
    return True
                
def download_player_list():
    """
    Download player ID list
    """
    
    filename = 'data/raw_pbp/playerlist.csv'
    if not os.path.exists(filename):
        
        url = 'https://eightthirtyfour.com/nba/pbp/playerlist.csv'
        
        with request.get(url) as r:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    
if __name__ == "__main__":
    
    print('Downloading player list')
    download_player_list()
    
    print('Downloading seasons')
    
    # create list of seasons
    seasons = range(2000, 2019)
    
    # iterate over seasons and download in parallel
    
    # create a progress bar
    pbar = tqdm(total=len(seasons))
    # function to update progress bar and results
    def update(a):
        pbar.update()

    # create pool to parallelize
    pool = Pool()

    # download each season in parallel
    for season in seasons:
        p = pool.apply_async(download_season, 
                             args=(season, ), 
                             callback=update)

    # wait for parallel processes to finish
    pool.close()
    pool.join()
    # close when all processes finished
    pbar.close() 
    
    