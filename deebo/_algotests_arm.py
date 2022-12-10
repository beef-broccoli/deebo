# test methods for arm identification algorithms
import numpy as np

from algos_testfunc import test_algorithm_arm
from algos_arm import SuccessiveElimination, LilUCB
from arms import BernoulliArm
from utils import make_dir, means_from_scenario


def lilucb(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = LilUCB(n_arms, parameter_set=2)
    algo.reset(n_arms)
    results, best_arms = test_algorithm_arm(algo, arms, n_sims, n_horizon)
    filename = 'test_lilucb.csv'
    results.to_csv(output_dir / filename)
    best_arm_fp = 'test_lilucb_bestarm.csv'
    best_arms.to_csv(output_dir / best_arm_fp)

    return None

if __name__ == '__main__':
    lilucb(1, 1000, 50, './logs/arms/')
