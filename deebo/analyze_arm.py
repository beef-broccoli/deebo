# methods to analyze arm selection algorithms

import pandas as pd
import numpy as np
import statistics

# TODO:
# 1. plot the probability of choosing each individual arm as a function of specified horizon
# - only plot top n, in case of a lot of options
# - might be cases whre nothing is selected
# - how to manage situatin where multiple best arms are requested?

# 2. plot the average number of experiments reqruied to find the best arm(s),
# only considering the simulations where best arm is actually found


def average_n_experiments_exact_match(acquisition_log, best_arm_log, best_arm_indexes):
    """
    only considering the simulations where all best arms are correctly identified,
    this function calculates the average number of experiments needed.

    Parameters
    ----------
    acquisition_log: str
        file path for full acquisition log
    best_arm_log: str
        file path for identified best arm log for each simulation
    best_arm_indexes:

    Returns
    -------

    """
    best_arms = pd.read_csv(best_arm_log, index_col=0)
    acquisition_log = pd.read_csv(acquisition_log, index_col=0)

    # from best_arms log, find exact matches with supplied best_arms_indexes
    # 1. construct numpy array with the same shape as best_arm_log with SORTED best_arms_indexes, pad with -1
    # 2. subtract and sum each row. Exact matches will return a sum of 0
    # 3. get index by argwhere which row returned 0
    # 4. select rows with num_sims matching the indexes
    a = best_arms.to_numpy()
    a.sort(axis=1)
    b = np.array(best_arm_indexes)
    b.sort()
    b = np.pad(b, (a.shape[1]-b.shape[0], 0), 'constant', constant_values=(-1, 69))  # pad -1 in front
    match = np.zeros(a.shape)
    match[...] = b
    matched_indexes = np.argwhere(np.sum(a-b, axis=1) == 0)
    matched_indexes = list(matched_indexes.flatten())
    acquisition_log = acquisition_log.loc[acquisition_log['num_sims'].isin(matched_indexes)]

    # for acquisition log, iterate through all valid indexes, and find the biggest horizon
    horizons = [max(acquisition_log.loc[acquisition_log['num_sims'] == index, 'horizon']) for index in matched_indexes]
    average = statistics.mean(horizons)

    return average



if __name__ == '__main__':
    average_n_experiments_exact_match(
        './logs/tests/successive_elim.csv',
        './logs/tests/successive_elim_best.csv',
        [4,3],
    )
