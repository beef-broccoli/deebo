import numpy as np
import pandas as pd
from tqdm import tqdm
import utils


def test_algorithm(algo, arms, num_sims, horizon):

    cols = ['num_sims', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in tqdm(range(num_sims), leave=False):

        algo.reset(len(arms))
        cumulative_reward = 0

        for t in range(horizon):
            chosen_arm = algo.select_next_arm()  # algorithm selects an arm
            reward = arms[chosen_arm].draw()  # chosen arm returns reward
            cumulative_reward = cumulative_reward + reward  # calculate cumulative reward over time horizon
            algo.update(chosen_arm, reward)  # algorithm updates chosen arm with reward
            ar[sim*horizon+t, :] = [sim, t, chosen_arm, reward, cumulative_reward]  # logs info

    df = pd.DataFrame(ar, columns=cols)

    return df


def test_algo_arm(algo, arms, num_sims, max_horizon):
    """

    Parameters
    ----------
    algo: algos_arm algo
    arms
    num_sims
    horizon

    Returns
    -------

    """

    cols = ['num_sims', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.zeros((num_sims*max_horizon, len(cols)))
    best_arms = np.negative(np.ones((num_sims, len(arms))))  # -1 to distinguish empty ones; could initialize smaller with n_candidates

    for sim in tqdm(range(num_sims), leave=False):

        algo.reset(len(arms))
        cumulative_reward = 0

        for t in range(max_horizon):
            chosen_arm = algo.select_next_arm()  # algorithm selects an arm
            if chosen_arm is None:
                break
            reward = arms[chosen_arm].draw()  # chosen arm returns reward
            cumulative_reward = cumulative_reward + reward  # calculate cumulative reward over time horizon
            algo.update(chosen_arm, reward)  # algorithm updates chosen arm with reward
            ar[sim*max_horizon+t, :] = [sim, t, chosen_arm, reward, cumulative_reward]  # logs info

        best_arms[sim, :] = utils.fill_list(algo.best_arms, len(arms), -1)

    # remove all zero rows: time horizons that are not used because requested # of candidates are found
    ar = ar[~np.all(ar==0, axis=1)]

    return pd.DataFrame(ar, columns=cols), pd.DataFrame(best_arms)
