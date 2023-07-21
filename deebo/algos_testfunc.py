import numpy as np
import pandas as pd
from tqdm import tqdm
import utils


def test_algorithm_regret(algo, arms, num_sims, horizon):
    """
    testing function for regret-type bandit algorithms

    Parameters
    ----------
    algo: algos_regret.RegretAlgorithm
        a regret bandit algorithm
    arms: list of arms
        a list of arms from arms.py
    num_sims: int
        total number of simulations
    horizon: int
        maximum time horizon for each simulation

    Returns
    -------
    dataframe of testing result logs

    """

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


def test_algorithm_regret_multidraw(algo, arms, num_sims, horizon, n_exps=1):
    """
    For algorithms that use sampling (like Thompson sampling), initialize one algorithm and propose multiple experiments
    at each round to conduct batch experiments.

    Parameters
    ----------
    algo: RegretAlgorithm object
    arms: list of Arm object
    num_sims: int
        number of simulations
    horizon: int
        time horizon for each simulation
    n_exps: int
        number of experiments per batch

    Returns
    -------
    dataframe of testing result logs

    """
    n_rounds = horizon // int(n_exps)  # num of complete rounds
    n_residual = horizon % int(n_exps)  # residual experiments that need to be handled

    cols = ['num_sims', 'round', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in tqdm(range(num_sims), leave=False):

        algo.reset(len(arms))
        cumulative_reward = 0
        t = 0

        for r in range(n_rounds+1):
            chosen_arms = []
            if r == n_rounds:
                n = n_residual  # last round, using residual experiments
            else:
                n = n_exps  # first n_rounds rounds, each batch has n_exps exps
            for e in range(n):
                chosen_arms.append(algo.select_next_arm())  # for each round, select next arm n_exps times
            for chosen_arm in chosen_arms:
                reward = arms[chosen_arm].draw()
                algo.update(chosen_arm, reward)
                cumulative_reward = cumulative_reward + reward
                ar[sim*horizon+t, :] = [sim, r, t, chosen_arm, reward, cumulative_reward]  # logs info
                t = t + 1

    df = pd.DataFrame(ar, columns=cols)

    return df


def test_algorithm_regret_multialgos(algo_list, arms, num_sims, horizon):
    """
    if a batch of n experiments is desired, initialize n experiments, and at each round each algorithm proposes an
    experiment

    Parameters
    ----------
    algo_list: Collection
        list of algorithms
    arms: list of Arm objects
    num_sims: int
        number of simulations
    horizon: int
        time horizon for each simulation

    Returns
    -------
    dataframe of testing result logs

    """
    # designed for ucb-type algorithms, with external exploration round
    n_rounds = (horizon-len(arms)) // len(algo_list)  # num of complete rounds
    n_residual = (horizon-len(arms)) % len(algo_list)  # residual experiments that need to be handled
    if n_residual > 0:
        n_rounds = n_rounds + 1

    # for logging
    cols = ['num_sims', 'round', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward', 'by_algo']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in tqdm(range(num_sims), leave=False):
        algos = algo_list
        for algo in algos:
            algo.reset(len(arms))
        cumulative_reward = 0
        t = 0

        # initial exploration round
        exploration_reward = [arm.draw() for arm in arms]
        for ii in range(len(exploration_reward)):
            for algo in algos:
                algo.update(ii, exploration_reward[ii])
            cumulative_reward = cumulative_reward + exploration_reward[ii]
            ar[sim*horizon+t, :] = [sim, -1, t, ii, exploration_reward[ii], cumulative_reward, -1]
            t = t + 1

        # now all ucb algos updated with exploration result, ready for acquisition
        for r in range(n_rounds):
            if r == n_rounds-1 and n_residual > 0:  # last extra round, dealing with residual experiments
                maxes = [max(algo.emp_means) for algo in algos]  # the highest emp mean identified for any arm for each algo
                indexes = np.argsort(maxes)[-n_residual:]
                algos = [algos[i] for i in indexes]

            # each algorithm selects one option
            chosen_arms = list(map(lambda x: x.select_next_arm(), algos))
            rewards = list(map(lambda x: arms[x].draw(), chosen_arms))

            for ii in range(len(chosen_arms)):
                for algo in algos:
                    algo.update(chosen_arms[ii], rewards[ii])
                cumulative_reward = cumulative_reward + rewards[ii]
                if r == n_rounds-1 and n_residual > 0:
                    ar[sim*horizon+t, :] = [sim, r, t, chosen_arms[ii], rewards[ii], cumulative_reward, indexes[ii]]
                else:
                    ar[sim*horizon+t, :] = [sim, r, t, chosen_arms[ii], rewards[ii], cumulative_reward, ii]  # logs info
                t = t+1  # advance time point

    return pd.DataFrame(ar, columns=cols)


def test_algorithm_arm(algo, arms, num_sims, max_horizon, n_candidates=1):
    """

    Parameters
    ----------
    algo: algos_arm.EliminationAlgorithm()
    arms
    num_sims
    horizon

    Returns
    -------
    acquisition history (if sim is terminated by algo before specified time limit, all empty entries are removed)
    best arms identified by algo (# specified by user, padded with -1) (shape: n_sim, n_arms)
    rankings based on # of samples at the end of each sim (shape: n_sim, n_arms)
    the round at which simulation terminates (shape: n_sim, 1)


    """

    cols = ['num_sims', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.negative(np.ones((num_sims*max_horizon, len(cols))))
    best_arms = np.negative(np.ones((num_sims, len(arms))))  # -1 to distinguish empty ones; could initialize smaller with n_candidates
    rankings = np.negative(np.ones((num_sims, len(arms))))  # rankings at the end of each simulations
    termination_round = np.negative(np.ones((num_sims,)))  # the round where each simulation terminates

    for sim in tqdm(range(num_sims), leave=False):

        algo.reset()
        cumulative_reward = 0

        for t in range(max_horizon):
            chosen_arm = algo.select_next_arm()  # algorithm selects an arm
            if chosen_arm is None:  # no next experiment; optimal arm has been found
                termination_round[sim] = t-1
                break
            reward = arms[chosen_arm].draw()  # chosen arm returns reward
            cumulative_reward = cumulative_reward + reward  # calculate cumulative reward over time horizon
            algo.update(chosen_arm, reward)  # algorithm updates chosen arm with reward
            ar[sim*max_horizon+t, :] = [sim, t, chosen_arm, reward, cumulative_reward]  # logs info

        best_arms[sim, :] = utils.fill_list(algo.best_arms, len(arms), -1)
        rankings[sim, :] = algo.rankings

    # remove all zero rows: time horizons that are not used because requested # of candidates are found
    ar = ar[~np.all(ar==-1, axis=1)]

    return pd.DataFrame(ar, columns=cols), pd.DataFrame(best_arms), pd.DataFrame(rankings), pd.DataFrame(termination_round)


