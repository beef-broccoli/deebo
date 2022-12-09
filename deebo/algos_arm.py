"""
Implemented algorithms for best arm identification in multi-armed bandit problem

Successive Elimination
Exponential-gap elimination: requires repeated sampling of one arm, way too many experiments needed

"""

import random
import math
import numpy as np


class Elimination:

    def __init__(self):
        pass


# NOTE: with low t, the confidence interval might be too big and cannot distinguish between arms that are close
class SuccessiveElimination:  # successive elimination

    def __init__(self, n_arms, counts=None, emp_means=None, current_set=None, delta=0.1, n_candidates=1):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.current_set = current_set if current_set else list(np.arange(n_arms))  # set of arms to be played
        self.to_play = list(self.current_set)  # arms to be played. Used to keep track of what's left
        self.delta = delta  # (0,1), parameter for any time confidence interval
        self.n_candidates = n_candidates  # algo terminates when desired number of candidates is reached
        self.best_arms = None

        if n_candidates >= n_arms:
            raise ValueError('Requested # of candidates is bigger than # of possible arms')

        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.current_set = np.arange(n_arms)
        self.to_play = list(self.current_set)
        self.best_arms = None
        return

    def _confidence_interval(self, option=1):
        t = sum(self.counts)
        if option == 1:
            u = math.sqrt(
                math.log(4.0*pow(t,2)/self.delta)
                / (2*t)
            )
        elif option == 2:
            u = math.sqrt(
                (2*math.log(1/self.delta) + 6*math.log(math.log(1/self.delta)) + 3*math.log(math.log(math.e*t)))
                / t
            )
        else:
            u = -1
            exit('Invalid option for confidence interval')
        return u

    def select_next_arm(self):

        if self.best_arms is not None:  # best_arms has been previously identified
            return None  # do not return next arm
        else:
            if not self.to_play:  # to_play empty, played all arms in current set. Need to evaluate
                # eliminate bad arms
                u = self._confidence_interval()
                emp_means_current_set = np.array([self.emp_means[idx] for idx in self.current_set])
                lower_bound = np.max(emp_means_current_set) - u
                plus_interval = emp_means_current_set + u
                idxs = plus_interval > lower_bound  # boolean array; what to keep for current set of arms
                self.current_set = self.current_set[idxs]
                self.to_play = list(self.current_set)
                # required number of best arms found
                if len(self.to_play) == self.n_candidates:
                    self.best_arms = self.to_play.copy()
                    return None
            return self.to_play.pop()

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


# TODO:
class LilUCB:  # successive elimination

    def __init__(self, n_arms, counts=None, emp_means=None, current_set=None, delta=0.1, n_candidates=1):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.delta = delta  # (0,1), parameter for any time confidence interval
        self.n_candidates = n_candidates  # algo terminates when desired number of candidates is reached
        self.best_arms = None

        if n_candidates >= n_arms:
            raise ValueError('Requested # of candidates is bigger than # of possible arms')

        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.current_set = np.arange(n_arms)
        self.to_play = list(self.current_set)
        self.best_arms = None
        return

    def _confidence_interval(self, option=1):
        t = sum(self.counts)
        if option == 1:
            u = math.sqrt(
                math.log(4.0*pow(t,2)/self.delta)
                / (2*t)
            )
        elif option == 2:
            u = math.sqrt(
                (2*math.log(1/self.delta) + 6*math.log(math.log(1/self.delta)) + 3*math.log(math.log(math.e*t)))
                / t
            )
        else:
            u = -1
            exit('Invalid option for confidence interval')
        return u

    def select_next_arm(self):

        if self.best_arms is not None:  # best_arms has been previously identified
            return None  # do not return next arm
        else:
            if not self.to_play:  # to_play empty, played all arms in current set. Need to evaluate
                # eliminate bad arms
                u = self._confidence_interval()
                emp_means_current_set = np.array([self.emp_means[idx] for idx in self.current_set])
                lower_bound = np.max(emp_means_current_set) - u
                plus_interval = emp_means_current_set + u
                idxs = plus_interval > lower_bound  # boolean array; what to keep for current set of arms
                self.current_set = self.current_set[idxs]
                self.to_play = list(self.current_set)
                # required number of best arms found
                if len(self.to_play) == self.n_candidates:
                    self.best_arms = self.to_play.copy()
                    return None
            return self.to_play.pop()

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return