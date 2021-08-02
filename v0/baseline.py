# implement baseline for bandit optimization
# baseline 1: random sample all ligands, equal samples drawn for each one with total experiments limited
# baseline 2: model substrate(s), evaluate ligands with all other conditions set, mimic chemist workflow

import math
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_results_suzuki


def main():
    names, _, results = get_results_suzuki('ligand')
    names, samples_n, samples = baseline_random(names, results, 100)
    plt.boxplot(samples.T, vert=False, labels=names)
    plt.show()


def baseline_random(names, results, total_exps=100):  # n=total number of experiments
    sample_n = total_exps//len(names) + 1  # make sure every component is evaluated evenly
    # for every row of results np array, randomly pick n elements

    def f(a):
        index = np.random.choice(results.shape[1], sample_n, replace=False)
        return a[index]

    samples = np.apply_along_axis(f, 1, results)
    return names, sample_n, samples


if __name__ == '__main__':
    main()

