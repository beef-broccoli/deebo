# test methods for arm identification algorithms
import numpy as np

from algos_testfunc import test_algorithm_arm
from algos_arm import *
from arms import BernoulliArm
from utils import make_dir, means_from_scenario


def test_func(algo, scenario, n_sims, n_horizon, folder_name):
    """
    test function with specified algo and scenario, with other params

    Parameters
    ----------
    algo: deebo.algos_arm.EliminationAlgorithm or deebo.algos_arm.SequentialHalving
        algorithm to be tested
    scenario: int
        pre-specified scenario with different means for different
    n_sims: int
        number of simulations
    n_horizon: int
        number of maxmimum time horizons (algo might terminate before this time)
    folder_name: str
        name of the folder where results are saved into

    Returns
    -------
    None

    """

    fp = folder_name + f'/scenario{scenario}/{algo.__str__()}/'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo.reset()
    results, best_arms, rankings, termination_rounds = test_algorithm_arm(algo, arms, n_sims, n_horizon)

    # save results
    filename = f'{algo.__str__()}_log.csv'
    results.to_csv(output_dir / filename)
    best_arm_fp = f'{algo.__str__()}_best_arms.csv'
    best_arms.to_csv(output_dir / best_arm_fp)
    rankings_fp = f'{algo.__str__()}_rankings.csv'
    rankings.to_csv(output_dir / rankings_fp)
    termination_rounds_fp = f'{algo.__str__()}_termination_rounds.csv'
    termination_rounds.to_csv(output_dir / termination_rounds_fp)

    return None


if __name__ == '__main__':
    algo = SequentialHalving(n_arms=5, time_limit=50)
    #algo = LilUCB(n_arms=5)
    test_func(algo, 3, 500, 250, './logs/arms')
