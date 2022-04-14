import itertools
import pandas as pd
import numpy as np

from algos import Random
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


def test_random():

    dataset_url = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-conditions.csv'
    names = ('base_smiles', 'solvent_smiles')  # same names with column name in df
    base = ['O=C([O-])C.[K+]', 'O=C([O-])C(C)(C)C.[K+]']
    solvent = ['CC(N(C)C)=O', 'CCCC#N']
    vals = list(itertools.product(base, solvent))  # sequence has to match what's in "names"
    arms = list(map(lambda x: ChemArm(x, names, dataset_url), vals))

    # since we have data, figure out which is best arm
    best_avg = 0
    best_arm = ''
    for arm in arms:
        avg = np.average(arm.data)
        if avg > best_avg:
            best_avg = avg
            best_arm = arm.val
    print('best arm is {0} with average {1}'.format(best_arm, best_avg))

    algo = Random([], [])
    algo.reset(len(arms))

    test_result = chem_test_algorithm(algo, arms, 100, 10)

    test_result.to_csv('./logs/tests/test.csv')

    return


if __name__ == '__main__':
    test_random()
