from arms import BernoulliArm
from algos_regret import *
from algos_arm import *
from algos_testfunc import *

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import sys
from pathlib import Path


def _means_from_scenario(scenario=0):
    """

    Parameters
    ----------
    scenario: test scenarios with preset means

    Returns
    -------
    mean reward for each arm

    """
    if scenario == 1:
        means = [0.1, 0.2, 0.3, 0.4, 0.9]
    elif scenario == 2:
        means = [0.1, 0.1, 0.1, 0.1, 0.2]
    elif scenario == 3:
        means = [0.1, 0.25, 0.5, 0.75, 0.9]
    else:
        means = None
        sys.exit('invalid test scenario number')
    return means


def _make_dir(dir):
    p = Path(dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# test all algorithms with defined scenarios
def test_all(scenario=0, algos=None, n_sims=1000, n_horizon=250):

    if not algos:
        # ignorin
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
        algo(scenario=scenario, n_sims=n_sims, n_horizon=n_horizon)

    return


def epsilon_greedy(scenario, n_sims, n_horizon):

    fp_prefix = f'./tests/scenario{scenario}/eps_greedy'
    output_dir = _make_dir(fp_prefix)

    means = _means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    # test for epsilon greedy
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(n_arms, eps, [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, n_sims, n_horizon)
        filename = 'epsilon_' + str(eps) + '.csv'
        results.to_csv(output_dir / filename)

    # test for epsilon greedy with annealing
    algo = AnnealingEpsilonGreedy(n_arms, [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, n_sims, n_horizon)
    results.to_csv(output_dir / 'annealing.csv')

    return


def softmax(scenario, n_arms, arms):

    fp_prefix = f'./logs/scenario{scenario}/softmax/'

    # test for Boltzmann
    for tau in []:
        algo = Boltzmann(n_arms, tau, [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)

        filename = 'tau_' + str(tau) + '.csv'
        fp = fp_prefix + filename
        results.to_csv(fp)

    # test for Boltzmann with annealing
    algo = AnnealingBoltzmann(n_arms, [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)

    filename = 'annealing.csv'
    fp = './logs/scenario2/softmax/' + filename
    results.to_csv(fp)

    return


def pursuit():

    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    lrs = [0.025]

    for l in tqdm(lrs):
        algo = Pursuit(l, [], [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)
        filename = 'lr_' + str(l) + '.csv'
        fp = './logs/scenario2/pursuit/' + filename
        results.to_csv(fp)

    return


def reinforcement_comparison():

    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    alphas = [0.05, 0.1, 0.2]
    betas = [0.05, 0.1, 0.2]

    for a, b in tqdm(itertools.product(alphas, betas)):
            algo = ReinforcementComparison(a, b, [], [], [], [], [])
            algo.reset(n_arms)
            results = test_algorithm(algo, arms, 1000, 250)
            filename = 'rc_alpha_' + str(a) + '_beta_'+ str(b) + '.csv'
            fp = './logs/scenario2/rc/' + filename
            results.to_csv(fp)

    return


def ucb1():

    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = UCB1()
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = 'ucb1_test.csv'
    fp = './logs/scenario2/optim/' + filename
    if input('save result (might overwrite)? : ') == 'y':
        results.to_csv(fp)

    return


def ucb1_tuned():

    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = UCB1Tuned([], [], [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = 'ucb1_tuned.csv'
    fp = './logs/scenario2/optim/' + filename
    results.to_csv(fp)

    return


def moss(scenario=1):

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

    algo = MOSS()
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = f'moss.csv'
    fp = f'./logs/scenario{scenario}/optim/' + filename
    if input('save result (might overwrite)? : ') == 'y':
        results.to_csv(fp)


def etc(scenario=1):

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

    exp_len = np.arange(15)+1

    for e in exp_len:
        algo = ETC([], [], e)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)
        filename = 'etc_' + str(e) + '.csv'
        fp = f'./logs/scenario{scenario}/ETC/' + filename
        results.to_csv(fp)


def ts_beta():
    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = ThompsonSampling([], [], [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = 'TS.csv'
    fp = './logs/scenario2/optim/' + filename
    results.to_csv(fp)


def ucbv():
    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = UCBV([], [], [], [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = 'ucbv.csv'
    fp = './logs/scenario2/optim/' + filename
    results.to_csv(fp)


def ucb2(scenario=1):

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

    for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = UCB2([], [], [], [], alpha=a)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1, 250)
        filename = f'ucb2_{a}.csv'
        fp = f'./logs/scenario{scenario}/optim/' + filename
        #results.to_csv(fp)


def exp3(scenario=3):
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

    for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EXP3(gamma=g)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)
        filename = f'gamma_{g}.csv'
        fp = f'./logs/scenario{scenario}/exp3/' + filename
        results.to_csv(fp)


def dmed(scenario=1):

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

    algo = DMED(n_arms, modified=True)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = f'dmed_modified.csv'
    fp = f'./logs/scenario{scenario}/' + filename
    if input('save result (might overwrite)? : ') == 'y':
        results.to_csv(fp)


def _test_successive_elimination(scenario=1):

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
    results, bests = test_algo_arm(algo, arms, 1000, 250)
    results_fn = f'successive_elim.csv'
    bests_fn = f'successive_elim_best.csv'
    fp = f'./logs/tests/'
    if input('save result (might overwrite)? : ') == 'y':
        results.to_csv(fp+results_fn)
        bests.to_csv(fp+bests_fn)


if __name__ == '__main__':
    test_all(scenario=1, algos=[epsilon_greedy])