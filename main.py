"""
Main.py
Author: SMM Kuilboer
Date: 20 08 2024
Description:

4 games, different size of players
growing number of samples
"""

# Importing libraries
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import os

from tqdm import tqdm

# Importing files
from mycode.utils.game import Games
from mycode.utils.error import Errors
from mycode.approx import Approx
from mycode.utils.logger import setup_logger

APPROX = Approx()
GAMES = Games()
    
def stack_variables(setup):

    stacked_variables = []
    for game in setup:
        for method in game['method'][::-1]:
            for n_players in game['players'][::-1]:
                for sz in game['samplesizes'][::-1]:
                    for _ in range(game['iteration']):
                        stacked_variables.append(
                            (method,
                            game['game'], 
                            n_players, 
                            sz)
                        )
    
    return stacked_variables

def process(variables):

    # get the path from the temp file
    path = open('temp_pathname.txt', 'r').read().strip()

    # unpack variables
    method, game, n, samplesize = variables

    # set up the logger based on the PID
    logger = setup_logger()
    logger.info(f"Starting process: {method}, {game}, {n}, {samplesize}")

    approxi = Approx()
    approx_method = getattr(approxi, method)
    game_method = getattr(GAMES, game)

    _ = time.time()
    exp = approx_method(n=n, V=game_method, m=samplesize, logging=True)
    output = exp.run()

    time_processed = time.time() - _

    return_dict = {
        "method": method,
        "game": exp.V.name,
        "samplesize": samplesize,
        "n": n,
        "time": time_processed,
    }

    # error_class = Errors()
    # for error_method in error_methods:
    #     return_dict[error_method] = compare(error_class, n, game, output, error_method)

    # Create the DataFrame
    df = pd.DataFrame(output.values(), index=output.keys(), columns=['Shapley Value'])
    
    # Define the file path
    random_timer = str(time.time() * 1e6).split('.')[0]
    file_path = f"{path}/data/{random_timer}.txt"
    
    # Write the information and DataFrame to the text file
    with open(file_path, 'w') as f:
        # Write the top 5 rows of information
        f.write(f"Method: {method}\n")
        f.write(f"Game: {exp.V.name}\n")
        f.write(f"Sample Size: {samplesize}\n")
        f.write(f"Number of Players: {n}\n")
        f.write(f"Processing Time: {time_processed}\n")
        f.write("\n")
        
        # Write the DataFrame in CSV format
        df.to_csv(f, mode='a')

    logger.info(f"Finished process with variables: {method}, {game}, {n}, {samplesize}")

    return return_dict

def start_multiprocess(load, path, processors=mp.cpu_count()-2):

    os.makedirs(f'{path}/data', exist_ok=True)

    with open(f'temp_pathname.txt', 'w') as f:
        f.write(f"{path}\n")
    
    # Check if the corresponding file exists if the samplesize is fixed
    if load[-1] == 'fixed':
        if not os.path.exists('temp_fixedm.txt'):
            raise FileNotFoundError("The file 'temp_fixedm.txt' does not exist but the samplesize is fixed.")
    
    num_processors = processors
    pool = mp.Pool(num_processors)

    print(f'Processes shared over {num_processors}/{mp.cpu_count()} processors.')
    trigger = True
    
    with open(f'{path}/overview.csv', 'w') as f:

        for result in tqdm(pool.imap_unordered(process, load), 
                        total=len(load), 
                        desc='Processing'):
            if trigger:
                string = ','.join([str(key) for key in result.keys()])
                f.write(f"{string}\n")
                trigger = False
            
            string = ','.join([str(value) for (key, value) in result.items() if key != 'output'])
            f.write(f"{string}\n")

        pool.close()
        pool.join()

    try:
        os.remove(f'temp_pathname.txt')
    except FileNotFoundError:
        pass
    
    print(f"Finished processing.\n Output is written to {path}.")   



if __name__ == "__main__":
    start_multiprocess() # Only works when a setup is given as 
                         # variable in the function call