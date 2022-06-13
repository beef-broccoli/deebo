import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm

from algos_regret import *
from analyze import plot_average_reward, plot_cumulative_reward, plot_probs_choosing_best_arm, calculate_baseline
from arms_chem import ChemArmSim, ChemArmSimBinary


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


def _chem_test_1():

    # build chem arms
    dataset_url = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-conditions.csv'
    names = ('base_smiles', 'solvent_smiles')  # same names with column name in df
    base = ['O=C([O-])C.[K+]', 'O=C([O-])C(C)(C)C.[K+]']
    solvent = ['CC(N(C)C)=O', 'CCCC#N']
    vals = list(itertools.product(base, solvent))  # sequence has to match what's in "names"
    arms = list(map(lambda x: ChemArmSim(x, names, dataset_url), vals))

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
    algos = [ETC([], [], explore_limit=20),
             UCB1([], [], []),
             UCB1Tuned([], [], [], []),
             UCBV([], [], [], [], []),
             ThompsonSampling([], [], [], [])
             ]
    fp = './logs/chem_test_1/'
    exp_list = ['ETC', 'ucb1', 'ucb1tuned', 'ucbv', 'TS']
    assert len(algos) == len(exp_list), 'num of algos need to match num of exp names supplied'
    fn_list = [exp + '.csv' for exp in exp_list]  # for saving results
    num_sims = 1000
    time_horizon = 100

    # log testing params and other info
    # TODO: it's overwriting, instead of appending
    # log_fp = fp + 'log.txt'
    # with open(log_fp, 'w+') as f:
    #     f.write('ARM INFO:\n'
    #             f'dataset url: {dataset_url}\n'
    #             f'component names: {names}\n'
    #             f'component values: {vals}\n'
    #             f'number of data points: {n_datapoints}\n'
    #             f'average for all arms {all_avg}\n'
    #             f'best arm is arm {best_index} {best_arm} with average {round(best_avg, 3)}\n'
    #             f'\n'
    #             f'ALGORITHM EVALUATED:\n'
    #             f'{exp_list}\n'
    #             f'\n'
    #             f'EXPERIMENT PARAMETERS:\n'
    #             f'{num_sims} simulations\n'
    #             f'time horizon: {time_horizon}\n')

    # testing
    for i in tqdm(range(len(algos))):
        algo = algos[i]
        algo.reset(len(arms))
        result = chem_test_algorithm(algo, arms, num_sims, time_horizon)
        result.to_csv(fp + fn_list[i])

    return


def _chem_test_2():

    # build chem arms
    dataset_url = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv'
    names = ('ligand_name')  # same names with column name in df
    df = pd.read_csv(dataset_url)
    vals = ['A-caPhos', 'BrettPhos', 'Cy-BippyPhos', 'PCy3 HBF4', 'X-Phos']
    arms = list(map(lambda x: ChemArmSimBinary(x, names, dataset_url, cutoff=0.8), vals))

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

    # # calculate baseline
    # baseline = calculate_baseline(arms)  # TODO: log baseline

    # parameters for testing
    algos = [Random([], []),
            ETC([], [], explore_limit=20),
             AnnealingEpsilonGreedy([], []),
             Boltzmann(0.1, [], []),
             UCB1([], [], []),
             UCB1Tuned([], [], [], []),
             UCBV([], [], [], [], []),
             ThompsonSampling([], [], [], [])
             ]
    fp = './logs/chem_test_2/'
    exp_list = ['random', 'ETC', 'eps_greedy_annel', 'softmax_0.1','ucb1', 'ucb1tuned', 'ucbv', 'TS']
    assert len(algos) == len(exp_list), 'num of algos need to match num of exp names supplied'
    fn_list = [exp + '.csv' for exp in exp_list]  # for saving results
    num_sims = 1000
    time_horizon = 50

    # log testing params and other info
    # TODO: it's overwriting, instead of appending
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

def _chem_test_2_analyze():

    exp_list = ['random',
                'eps_greedy_annel',
                'softmax_0.1',
                'ucb1tuned',
                'TS']

    # exp_list = ['random',
    #             'ucb1',
    #             'ucb1tuned',
    #             'ucbv',
    #             'TS']

    fn_list = [str(e) + '.csv' for e in exp_list]
    legend_list = ['random',
                   'annealing ε greedy',
                   'softmax (0.1)',
                    'UCB1-Tuned',
                    'TS']
    #
    # legend_list = ['random',
    #             'UCB1',
    #             'UCB1-Tuned',
    #             'UCBV',
    #             'TS']

    assert len(exp_list) == len(legend_list)
    plot_probs_choosing_best_arm(fn_list, legend_list, baseline=0, best_arm_index=2, fp='./logs/chem_test_2/', title='', legend_title='algos')
    #plot_average_reward(fn_list, legend_list, fp='./logs/chem_test_1/', title='', legend_title='algos')
    return


def _test_binary():
    pass




if __name__ == '__main__':
    _chem_test_2_analyze()
