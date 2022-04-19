import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm

from algos import Random, EpsilonGreedy, AnnealingEpsilonGreedy, Boltzmann, AnnealingBoltzmann
from analyze import plot_average_reward, plot_cumulative_reward, plot_probs_choosing_best_arm, calculate_baseline
from chem_arms import ChemArm


def chem_test_algorithm(algo, arms, num_sims, horizon):

    cols = ['num_sims', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in tqdm(range(num_sims), leave=False):

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
    n_datapoints = {}
    for idx, arm in enumerate(arms):
        avg = np.average(arm.data)
        all_avg[idx] = round(avg, 3)
        n_datapoints[idx] = len(arm.data)
        if avg > best_avg:
            best_avg = avg
            best_arm = arm.val
            best_index = idx

    # calculate baseline
    baseline = calculate_baseline(arms)  # TODO: log baseline

    # parameters for testing
    algos = [Random([], []),
             EpsilonGreedy(0.1, [], []),
             EpsilonGreedy(0.25, [], []),
             EpsilonGreedy(0.5, [], []),
             EpsilonGreedy(0.75, [], []),
             AnnealingEpsilonGreedy([], []),
             Boltzmann(0.1, [], []),
             Boltzmann(0.5, [], []),
             Boltzmann(1, [], []),
             AnnealingBoltzmann([], []),
             ]
    fp = './logs/chem_test_1/'
    exp_list = ['random',
                'eps_greedy_0.1',
                'eps_greedy_0.25',
                'eps_greedy_0.5',
                'eps_greedy_0.75',
                'annealing_eps_greedy',
                'softmax_0.1',
                'softmax_0.5',
                'softmax_1',
                'annealing_softmax']
    assert len(algos) == len(exp_list), 'num of algos need to match num of exp names supplied'
    fn_list = [exp + '.csv' for exp in exp_list]  # for saving results
    num_sims = 1000
    time_horizon = 150

    # log testing params and other info
    log_fp = fp + 'log.txt'
    with open(log_fp, 'w+') as f:
        f.write('ARM INFO:\n'
                f'dataset url: {dataset_url}\n'
                f'component names: {names}\n'
                f'component values: {vals}\n'
                f'number of data points: {n_datapoints}\n'
                f'average for all arms {all_avg}\n'
                f'best arm is arm {best_index} {best_arm} with average {round(best_avg, 3)}\n'
                f'\n'
                f'ALGORITHM EVALUATED:\n'
                f'{exp_list}\n'
                f'\n'
                f'EXPERIMENT PARAMETERS:\n'
                f'{num_sims} simulations\n'
                f'time horizon: {time_horizon}\n')

    # testing
    for i in tqdm(range(len(algos))):
        algo = algos[i]
        algo.reset(len(arms))
        result = chem_test_algorithm(algo, arms, num_sims, time_horizon)
        result.to_csv(fp + fn_list[i])

    return


def chem_test_1_analyze():

    exp_list = ['random',
                'softmax_0.1',
                'annealing_eps_greedy']

    fn_list = [str(e) + '.csv' for e in exp_list]
    legend_list = ['random',
                   'softmax (0.1)',
                   'eps greedy (annealing)',
                   ]
    assert len(exp_list) == len(legend_list)
    plot_probs_choosing_best_arm(fn_list, legend_list, baseline=0, best_arm_index=2, fp='./logs/chem_test_1/', title='', legend_title='algos')
    #plot_average_reward(fn_list, legend_list, fp='./logs/chem_test_1/', title='', legend_title='algos')
    return


if __name__ == '__main__':
    #chem_test_1()
    chem_test_1_analyze()
