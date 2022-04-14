import itertools
import pandas as pd
import numpy as np

from algos import Random, EpsilonGreedy, AnnealingEpsilonGreedy
from chem_arms import ChemArm


def chem_test_algorithm(algo, arms, num_sims, horizon):

    cols = ['num_sims', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in range(num_sims):

        for arm in arms:
            arm.reset()

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


def chem_test_1():

    # build chem arms
    dataset_url = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-conditions.csv'
    names = ('base_smiles', 'solvent_smiles')  # same names with column name in df
    base = ['O=C([O-])C.[K+]', 'O=C([O-])C(C)(C)C.[K+]']
    solvent = ['CC(N(C)C)=O', 'CCCC#N']
    vals = list(itertools.product(base, solvent))  # sequence has to match what's in "names"
    arms = list(map(lambda x: ChemArm(x, names, dataset_url), vals))

    # since we have data, can figure out which is best arm, and calculate average for all arms
    best_avg = 0
    best_arm = ''
    best_index = 0
    all_avg = {}
    for idx, arm in enumerate(arms):
        avg = np.average(arm.data)
        all_avg[idx] = round(avg, 2)
        if avg > best_avg:
            best_avg = avg
            best_arm = arm.val
            best_index = idx

    # parameters for testing
    algos = [Random([], []),
             EpsilonGreedy(0.1, [], []),
             EpsilonGreedy(0.5, [], []),
             AnnealingEpsilonGreedy([], [])]
    fp = './logs/chem_test_1/'
    exp_list = ['random',
                'eps_greedy_0.1',
                'eps_greedy_0.5',
                'annealing_eps_greedy']
    fn_list = [exp + '.csv' for exp in exp_list] # for saving results
    num_sims = 10
    time_horizon = 10

    # log testing params and other info
    log_fp = fp + 'log.txt'
    with open(log_fp, 'w+') as f:
        f.write('ARM INFO\n'
                f'dataset url: {dataset_url}\n'
                f'component names: {names}\n'
                f'component values: {vals}\n'
                f'average for all arms {all_avg}\n'
                f'best arm is arm {best_index} {best_arm} with average {round(best_avg, 2)}\n'
                f'\n'
                f'ALGORITHM EVALUATED:\n'
                f'{exp_list}\n'
                f'\n'
                f'EXPERIMENT PARAMETERS:\n'
                f'{num_sims} simulations\n'
                f'time horizon: {time_horizon}\n')

    # testing
    for i in range(len(algos)):
        algo = algos[i]
        algo.reset(len(arms))
        result = chem_test_algorithm(algo, arms, num_sims, time_horizon)
        result.to_csv(fp + fn_list[i])

    return


if __name__ == '__main__':
    chem_test_1()
