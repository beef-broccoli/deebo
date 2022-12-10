"""
Implemented algorithms for best arm identification in multi-armed bandit problem

Successive Elimination
Lil UCB

Did not implement:
Exponential-gap elimination: requires repeated sampling of one arm, way too many experiments needed

"""

import random
import math
import numpy as np


# parent class for elimination type algorithms
class EliminationAlgorithm:

    def __init__(self, n_arms, counts=None, emp_means=None, delta=0.1, n_candidates=1):
        if n_candidates >= n_arms:
            raise ValueError('Requested # of candidates is bigger than # of possible arms')
        if (delta >= 1) or (delta <= 0):
            raise ValueError('confidence interval (delta) from 0 to 1')

        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.delta = delta  # delta in (0,1), confidence interval
        self.n_candidates = n_candidates  # algo terminates when desired number of candidates is reached
        self.best_arms = None
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.best_arms = None
        return

    def select_next_arm(self):
        pass

    def update(self, chosen_arm, reward):
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        # update empirical means
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


# NOTE: with low t, the confidence interval might be too big and cannot distinguish between arms that are close
class SuccessiveElimination(EliminationAlgorithm):  # successive elimination

    def __init__(self, n_arms, counts=None, emp_means=None, current_set=None, delta=0.1, n_candidates=1):
        EliminationAlgorithm.__init__(self, n_arms, counts, emp_means, delta, n_candidates)
        self.current_set = current_set if current_set else list(np.arange(n_arms))  # set of arms to be played
        self.to_play = list(self.current_set)  # arms to be played. Used to keep track of what's left
        return

    def reset(self, n_arms):
        EliminationAlgorithm.reset(self, n_arms)
        self.current_set = np.arange(n_arms)
        self.to_play = list(self.current_set)
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


class LilUCB(EliminationAlgorithm):

    def __init__(self, n_arms, counts=None, emp_means=None, delta=0.1, n_candidates=1, variance=0.25,
                 ucbs=None, parameter_set=2, initial_exploration_round=1):
        EliminationAlgorithm.__init__(self, n_arms, counts, emp_means, delta, n_candidates)
        self.variance = variance  # variance for sub-gaussian arm
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)]  # ucb values calculated with means and counts
        self.ranking = []
        # parameter set according to the paper
        if parameter_set == 1:  # theoretical guarantee
            self.eps = 0.01
            self.beta = 1
            self.lamb = 9
        elif parameter_set == 2:  # better performance in practice
            self.eps = 0
            self.beta = 0.5
            self.lamb = 10/n_arms + 1
            self.delta = self.delta/5  # scale input delta
        else:
            raise ValueError('Parameter set invalid')
        self.initial_exploration_round = initial_exploration_round

    def reset(self, n_arms):
        EliminationAlgorithm.reset(self, n_arms)
        self.ucbs = [0.0 for col in range(n_arms)]
        self.ranking = []

    def _confidence_bound(self, count):
        # expression inside square root
        s = (2 * self.variance * (1+self.eps) * math.log(math.log((1+self.eps)*count)/self.delta+1e-7)) / count
        if s < 0:  # for all arms, count needs to be at least 2 for the square root to work. This works around count=1
            s = 1  # slightly bigger than count=2 case
        u = (1+self.beta) * (1+math.sqrt(self.eps)) * math.sqrt(s)
        return u

    def select_next_arm(self):
        if self.best_arms is not None:  # best_arms has been previously identified
            return None  # stop; do not return next arm
        elif sum(self.counts) < self.initial_exploration_round*len(self.counts):  # run a first pass through all arms
            return np.argmin(self.counts)
        else:
            s = sum(self.counts)
            for count in self.counts:
                if count >= 1 + self.lamb*(s-count):
                    self.best_arms = self.ranking[-self.n_candidates:]  # output last {n_candidates} from ranking
                    return None  # stop; do not return next arm
            # calculate ucb here only limits math error for log. All arms have at least count=1 at this point
            self.ucbs = [e + self._confidence_bound(count) for e, count in zip(self.emp_means, self.counts)]
            return np.argmax(self.ucbs)

    def update(self, chosen_arm, reward):
        EliminationAlgorithm.update(self, chosen_arm, reward)
        self.ranking = list(np.arange(len(self.counts))[np.argsort(self.counts)])
        return
