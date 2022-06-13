from arms import BernoulliArm
from algos_regret import *
from algos_arm import *
from algos_testfunc import test_algorithm

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools


def _test_epsilon_greedy():
    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    # test for epsilon greedy
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(eps, [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)
        filename = 'epsilon_' + str(eps) + '.csv'
        fp = './logs/scenario2/eps_greedy/' + filename
        results.to_csv(fp)

    # test for epsilon greedy with annealing
    algo = AnnealingEpsilonGreedy([], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)

    filename = 'annealing.csv'
    fp = './logs/scenario2/eps_greedy/' + filename
    results.to_csv(fp)

    return


def _test_softmax():

    means = [0.1, 0.1, 0.1, 0.1, 0.2]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    # test for Boltzmann
    for tau in tqdm([0.025, 0.01]):
        algo = Boltzmann(tau, [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)

        filename = 'tau_' + str(tau) + '.csv'
        fp = './logs/scenario2/softmax/' + filename
        results.to_csv(fp)

    # # test for Boltzmann with annealing
    # algo = AnnealingBoltzmann([], [])
    # algo.reset(n_arms)
    # results = test_algorithm(algo, arms, 1000, 250)
    #
    # filename = 'annealing.csv'
    # fp = './logs/scenario2/softmax/' + filename
    # results.to_csv(fp)

    return


def _test_pursuit():

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


def _test_reinforcement_comparison():

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


def _test_ucb1():

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


def _test_ucb1_tuned():

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


def _test_moss(scenario=1):

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


def _test_etc(scenario=1):

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


def _test_ts_beta():
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


def _test_ucbv():
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


def _test_ucb2(scenario=1):

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


def _test_exp3(scenario=3):
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


def _test_dmed(scenario=1):

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

    algo = SuccessiveElimination(n_arms, delta=0.2)
    results = test_algorithm(algo, arms, 1, 250)
    filename = f'successive_elim.csv'
    fp = f'./logs/tests/' + filename
    if input('save result (might overwrite)? : ') == 'y':
        results.to_csv(fp)


if __name__ == '__main__':
    _test_successive_elimination()