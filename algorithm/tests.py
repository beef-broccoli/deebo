from arms import BernoulliArm
from algos import EpsilonGreedy

import random
import numpy as np
import pandas as pd


def test_algorithm(algo, arms, num_sims, horizon):

    cols = ['num_sims', 'horizon', 'chosen_arm', 'exploit', 'reward', 'cumulative_reward']
    ar = np.zeros((num_sims*horizon, len(cols)))

    for sim in range(num_sims):

        algo.reset(len(arms))
        cumulative_reward = 0

        for t in range(horizon):
            chosen_arm, exploit = algo.select_next_arm()  # algorithm selects an arm
            reward = arms[chosen_arm].draw()  # chosen arm returns reward
            cumulative_reward = cumulative_reward + reward  # calculate cumulative reward over time horizon
            algo.update(chosen_arm, reward)  # algorithm updates chosen arm with reward
            ar[sim*horizon+t, :] = [sim, t, chosen_arm, exploit, reward, cumulative_reward]  # logs info

    df = pd.DataFrame(ar, columns=cols)
    df.to_csv('logs.csv')

    return df


if __name__ == '__main__':

    means = [0.1, 0.2, 0.3]
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    print("Best arm is " + str(np.argmax(means)))
    print(means)

    algo = EpsilonGreedy(0.2, [], [])
    algo.reset(n_arms)
    results = test_algorithm(algo, arms, 1, 1000)
    
    # show average reward for each arm
    for i in [0,1,2]:
        print('Arm {0}, average reward: {1}'.format(i, round(np.average(results.loc[results['chosen_arm'] == i]['reward'].to_numpy()), 3)))

    # check exploit percentage
    print('Average explore percentage: {0}'.format(round(1-np.average(results['exploit'].to_numpy()), 5)))