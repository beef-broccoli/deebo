# test methods for arm identification algorithms
import numpy as np

from algos_testfunc import test_algorithm_arm
from algos_arm import SuccessiveElimination
from arms import BernoulliArm


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
    results, bests = test_algorithm_arm(algo, arms, 5, 100)
    results_fn = f'successive_elim.csv'
    bests_fn = f'successive_elim_best.csv'
    fp = f'./logs/tests/'
    if input('save result (might overwrite)? : ') == 'y':
        results.to_csv(fp+results_fn)
        bests.to_csv(fp+bests_fn)
