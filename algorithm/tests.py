from arms import BernoulliArm
from algos import *

import random
import numpy as np
import pandas as pd


def test_algorithm(algo, arms, num_sims, horizon):

    cols = ['num_sims', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in range(num_sims):

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
    for tau in [0.6, 0.7, 0.8, 0.9, 1.0]:
        algo = Boltzmann(tau, [], [])
        algo.reset(n_arms)
        results = test_algorithm(algo, arms, 1000, 250)

        filename = 'tau_' + str(tau) + '.csv'
        fp = './logs/Boltzmann_test/' + filename
        results.to_csv(fp)

    # test for Boltzmann with annealing
    algo = AnnealingBoltzmann([], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1000, 250)

    filename = 'annealing_boltzmann_test.csv'
    fp = './logs/Boltzmann_test/' + filename
    results.to_csv(fp)

    return


def _test_pursuit():

    # algo = Pursuit(0.05, [], [], [])
    # algo.reset(n_arms)
    # results = test_algorithm(algo, arms, 1, 500)

    return


def _test_reinforcement_comparison():

    # algo = ReinforcementComparison(0.5, 0.5, [], [], [], [], [])
    # algo.reset(n_arms)
    # results = test_algorithm(algo, arms, 1, 50)

    return


if __name__ == '__main__':

    pass