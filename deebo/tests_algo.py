from arms import BernoulliArm
from algos_stochastic import *

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

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


def _test_epsilon_greedy():
    means = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    # test for epsilon greedy
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(eps, [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)
        filename = 'epsilon_' + str(eps) + '.csv'
        fp = './logs/epsilon_greedy_test_small_diff/' + filename
        results.to_csv(fp)

    # test for epsilon greedy with annealing
    algo = AnnealingEpsilonGreedy([], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)

    filename = 'annealing_epsilon_greedy.csv'
    fp = './logs/epsilon_greedy_test_small_diff/' + filename
    results.to_csv(fp)

    return


def _test_softmax():

    means = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    # test for Boltzmann
    for tau in [0.05]:
        algo = Boltzmann(tau, [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)

        filename = 'tau_' + str(tau) + '.csv'
        fp = './logs/Boltzmann_test/' + filename
        results.to_csv(fp)

    # # test for Boltzmann with annealing
    # algo = AnnealingBoltzmann([], [])
    # algo.reset(n_arms)
    # results = test_algorithm(algo, arms, 1000, 250)
    #
    # filename = 'annealing_boltzmann_test.csv'
    # fp = './logs/Boltzmann_test/' + filename
    # results.to_csv(fp)

    return


def _test_pursuit():

    means = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    lrs = [0.05, 0.005]

    for l in lrs:
        algo = Pursuit(l, [], [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)
        filename = 'pursuit_lr_' + str(l) + '.csv'
        fp = './logs/Pursuit/' + filename
        results.to_csv(fp)

    return


def _test_reinforcement_comparison():

    means = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    alphas = [0.01, 0.025, 0.05]
    betas = [0.2, 0.3, 0.4]

    for a, b in tqdm(itertools.product(alphas, betas)):
            algo = ReinforcementComparison(a, b, [], [], [], [], [])
            algo.reset(n_arms)
            results = test_algorithm(algo, arms, 1000, 250)
            filename = 'rc_alpha_' + str(a) + '_beta_'+ str(b) + '.csv'
            fp = './logs/reinforcement_comparison/' + filename
            results.to_csv(fp)

    return


def _test_ucb1():

    means = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = UCB1([], [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = 'ucb1_test.csv'
    fp = './logs/ucb1/' + filename
    results.to_csv(fp)

    return


def _test_ucb1_tuned():

    means = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = UCB1Tuned([], [], [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = 'ucb1_tuned_test.csv'
    fp = './logs/ucb1/' + filename
    results.to_csv(fp)

    return


def _test_etc():

    means = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    exp_len = [1,5,10,25,40]

    for e in exp_len:
        algo = ETC([], [], e)
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)
        filename = 'etc_' + str(e) + '.csv'
        fp = './logs/ETC/' + filename
        results.to_csv(fp)


def _test_ts_beta():
    means = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))

    algo = ThompsonSampling([], [], [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)
    filename = 'TS_test.csv'
    fp = './logs/TS/' + filename
    results.to_csv(fp)


if __name__ == '__main__':

    _test_ucb1_tuned()