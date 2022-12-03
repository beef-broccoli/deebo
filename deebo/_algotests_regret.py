from arms import BernoulliArm
from algos_regret import *
from algos_arm import *
from algos_testfunc import *
from utils import means_from_scenario, make_dir

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import sys
from pathlib import Path


# test all algorithms with defined scenarios
def test_all(scenario=1, algos=None, n_sims=1000, n_horizon=250, folder_name=None):

    if not folder_name:
        sys.exit('Supply folder name for saving result')

    if not algos:
        # skipping reinforcement comparison, too difficult to tune
        algos = [epsilon_greedy,
                 softmax,
                 pursuit,
                 ucb1,
                 ucb1_tuned,
                 moss,
                 ts_beta,
                 ucbv,
                 ucb2,
                 exp3,
                 dmed,
                 ]

    for algo in tqdm(algos):
        algo(scenario=scenario,
             n_sims=n_sims,
             n_horizon=n_horizon,
             folder_name=folder_name)

    return None


def test_algo_for_all_scenarios(algo, scenarios, n_sims=1000, n_horizon=250, folder_name=None):

    for scenario in scenarios:
        algo(scenario=scenario,
             n_sims=n_sims,
             n_horizon=n_horizon,
             folder_name=folder_name)

    return None


def epsilon_greedy(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/eps_greedy'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    # test for epsilon greedy
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(n_arms, eps)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, n_sims, n_horizon)
        filename = 'epsilon_' + str(eps) + '.csv'
        results.to_csv(output_dir / filename)

    # test for epsilon greedy with annealing
    algo = AnnealingEpsilonGreedy(n_arms)
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    results.to_csv(output_dir / 'annealing.csv')

    return None


def softmax(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/softmax'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    # test for Boltzmann
    for tau in [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = Boltzmann(n_arms, tau)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, n_sims, n_horizon)
        filename = 'tau_' + str(tau) + '.csv'
        results.to_csv(output_dir / filename)

    # test for Boltzmann with annealing
    algo = AnnealingBoltzmann(n_arms)
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    results.to_csv(output_dir / 'annealing.csv')

    return None


def pursuit(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/pursuit'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    lrs = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

    for lr in lrs:
        algo = Pursuit(n_arms, lr)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, n_sims, n_horizon)
        filename = 'lr_' + str(lr) + '.csv'
        results.to_csv(output_dir / filename)

    return None


# mod needed for testing
def reinforcement_comparison(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/rc'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    alphas = [0.05, 0.1, 0.2]
    betas = [0.05, 0.1, 0.2]

    for a, b in tqdm(itertools.product(alphas, betas)):
            algo = ReinforcementComparison(n_arms, a, b)
            algo.reset(n_arms)
            results = test_algorithm(algo, arms, n_sims, n_horizon)
            filename = 'rc_alpha_' + str(a) + '_beta_' + str(b) + '.csv'
            results.to_csv(output_dir / filename)

    return None


def ucb1(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = UCB1(n_arms)
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    filename = 'ucb1.csv'
    results.to_csv(output_dir / filename)

    return None


def ucb1_tuned(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = UCB1Tuned(n_arms)
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    filename = 'ucb1_tuned.csv'
    results.to_csv(output_dir / filename)

    return None


def moss(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = MOSS(n_arms)
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    filename = f'moss.csv'
    results.to_csv(output_dir / filename)

    return None


def etc(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/etc'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    exploration_limit = n_horizon // n_arms
    exp_len = np.arange(exploration_limit)+1

    for e in exp_len:
        algo = ETC(n_arms, explore_limit=e)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, n_sims, n_horizon)
        filename = f'{e}_exp_per_arm.csv'
        results.to_csv(output_dir / filename)
    return None


def ts_beta(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = ThompsonSampling(n_arms)
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    filename = 'TS_beta.csv'
    results.to_csv(output_dir / filename)

    return None


def ucbv(scenario, n_sims, n_horizon, folder_name):
    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = UCBV(n_arms)
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    filename = 'ucbv.csv'
    results.to_csv(output_dir / filename)

    return None


def ucb2(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/ucb2'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        algo = UCB2(n_arms, alpha=a)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, n_sims, n_horizon)
        filename = f'ucb2_{a}.csv'
        results.to_csv(output_dir / filename)

    return None


def exp3(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/exp3'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    for g in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        algo = EXP3(n_arms, gamma=g)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, n_sims, n_horizon)
        filename = f'gamma_{g}.csv'
        results.to_csv(output_dir / filename)


def dmed(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = DMED(n_arms, modified=True)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    if algo.modified:
        filename = f'dmed_modified.csv'
    else:
        filename = f'dmed.csv'
    results.to_csv(output_dir / filename)


def _test_successive_elimination(scenario=1):  # TODO: migrate this

    if scenario == 1:
        means = [0.1, 0.2, 0.3, 0.4, 0.9]
    elif scenario == 2:
        means = [0.1, 0.1, 0.1, 0.1, 0.2]
    elif scenario == 3:
        means = [0.1, 0.25, 0.5, 0.75, 0.9]
    else:
        exit(1)

    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = SuccessiveElimination(n_arms, delta=0.1)
    results, bests = test_algo_arm(algo, arms, 5, 100)
    results_fn = f'successive_elim.csv'
    bests_fn = f'successive_elim_best.csv'
    fp = f'./logs/tests/'
    if input('save result (might overwrite)? : ') == 'y':
        results.to_csv(fp+results_fn)
        bests.to_csv(fp+bests_fn)


if __name__ == '__main__':
    #test_all(scenario=4, n_sims=1000, n_horizon=500, folder_name='./logs')
    test_algo_for_all_scenarios(etc, [4], folder_name='./baseline_logs')