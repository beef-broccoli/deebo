import numpy as np
import pandas as pd
from tqdm import tqdm
import utils



def test_algorithm_regret(algo, arms, num_sims, horizon):

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


def test_algorithm_regret_with_best_arm(algo, arms, num_sims, horizon):

    cols = ['num_sims', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward', 'ranking']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in tqdm(range(num_sims), leave=False):

        algo.reset(len(arms))
        cumulative_reward = 0

        for t in range(horizon):
            chosen_arm = algo.select_next_arm()  # algorithm selects an arm
            reward = arms[chosen_arm].draw()  # chosen arm returns reward
            cumulative_reward = cumulative_reward + reward  # calculate cumulative reward over time horizon
            algo.update(chosen_arm, reward)  # algorithm updates chosen arm with reward
            ar[sim*horizon+t, :] = [sim, t, chosen_arm, reward, cumulative_reward, algo.ranking[-1]]  # logs info

    df = pd.DataFrame(ar, columns=cols)

    return df


# this approach doesn't have any real benefits,
def batched_test_algorithm(algos, arms, num_sims, horizon):
    n_rounds = horizon // len(algos)  # num of complete rounds
    residual = horizon % len(algos)  # residual experiments that need to be handled

    # for logging
    cols = ['num_sims', 'round', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward', 'by_algo']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in tqdm(range(num_sims), leave=False):
        for algo in algos:
            algo.reset(len(arms))
        cumulative_reward = 0
        t = 0

        for r in range(n_rounds):

            # each algorithm selects one option
            chosen_arms = list(map(lambda x: x.select_next_arm(), algos))
            rewards = list(map(lambda x: arms[x].draw(), chosen_arms))

            for ii in range(len(chosen_arms)):
                for algo in algos:
                    algo.update(chosen_arms[ii], rewards[ii])
                cumulative_reward = cumulative_reward + rewards[ii]
                ar[sim*horizon+t, :] = [sim, r, t, chosen_arms[ii], rewards[ii], cumulative_reward, ii]  # logs info
                t = t+1  # advance time point

        # TODO: handle residual experiments

    return pd.DataFrame(ar, columns=cols)


def test_algorithm_arm(algo, arms, num_sims, max_horizon):
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


