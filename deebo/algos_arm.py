"""
Implemented algorithms for best arm identification in multi-armed bandit problem

"""

import random
import math
import numpy as np


class SuccessiveElimination:  # successive elimination

    def __init__(self, n_arms, counts=None, emp_means=None, explore_limit=1):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.limit = explore_limit  # how many rounds per arm
        self.best_arm = -1
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.best_arm = -1
        return

    def select_next_arm(self):
        if sum(self.counts) == self.limit*len(self.counts):  # exploration just complete, pick the best arm
            self.best_arm = np.argmax(self.emp_means)

        if self.best_arm == -1:  # no best arm set, still in the exploration phase
            return np.argmin(self.counts)  # plays the arm with lowest count until exploration ends
        else:  # commit
            return self.best_arm

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return