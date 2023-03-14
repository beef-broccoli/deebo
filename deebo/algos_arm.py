"""
Implemented algorithms for best arm identification in multi-armed bandit problem

- Successive Elimination
- lil UCB
- LUCB++
    I don't quite understand how this method can work in a fixed reward interval setting.
    Basically the emp. average of (k)th and the (k+1)th best arms have to be two confidence intervals apart
    But a lot of cases the differences in true average are too small to make this work.
    Maybe i didn't choose the right confidence interval?


Did not implement:
Exponential-gap elimination: requires repeated sampling of one arm, way too many experiments needed

Most of these algorithms actively eliminate arms, so algorithms can be stopped when n arms are left (user specified n)
lil ucb's stop condition is different, it only stops for the best arm.

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

        self.n_arms = n_arms
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.delta = delta  # delta in (0,1), confidence interval
        self.n_candidates = n_candidates  # algo terminates when desired number of candidates is reached
        self.best_arms = None
        self.rankings = []
        return

    def __str__(self):
        return None

    def reset(self):  # no longer resetting n_arms or n_candidates
        self.counts = [0 for col in range(self.n_arms)]
        self.emp_means = [0.0 for col in range(self.n_arms)]
        self.best_arms = None
        self.rankings = []
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
        self.rankings = list(np.arange(len(self.counts))[np.argsort(self.counts)])  # worst -> best, based on counts
        return

    def _confidence_interval(self, t, delta, option=1):
        if option == 1:  # standard for successive elimination
            u = math.sqrt(
                math.log(4.0*pow(t,2)/delta)
                / (2*t)
            )
        elif option == 2:  # state of art for successive elimination according to lecture notes, proposed by kauffman 2016
            u = math.sqrt(
                (2*math.log(1/delta) + 6*math.log(math.log(1/delta)) + 3*math.log(math.log(math.e*t)))
                / t
            )
        elif option == 3:  # same kauffman 2016 paper (theorem 8), used in LUCB++ paper
            # beta = math.log(1/delta) + 3*math.log(math.log(1/delta)) + 1.5*math.log(math.log(math.e*t/2))
            pass
        else:
            exit('Invalid option for confidence interval')
        return u


# NOTE: with low t, the confidence interval might be too big and cannot distinguish between arms that are close
class SuccessiveElimination(EliminationAlgorithm):  # successive elimination

    def __init__(self, n_arms, counts=None, emp_means=None, current_set=None, delta=0.1, n_candidates=1):
        EliminationAlgorithm.__init__(self, n_arms, counts, emp_means, delta, n_candidates)
        self.current_set = current_set if current_set else list(np.arange(n_arms))  # set of arms to be played
        self.to_play = list(self.current_set)  # arms to be played. Used to keep track of what's left
        return

    def __str__(self):
        return f'successive_elimination_choose{self.n_candidates}'

    def reset(self):
        EliminationAlgorithm.reset(self)
        self.current_set = np.arange(self.n_arms)
        self.to_play = list(self.current_set)
        return

    def select_next_arm(self):

        if self.best_arms is not None:  # best_arms has been previously identified
            return None  # do not return next arm
        else:
            if not self.to_play:  # to_play empty, played all arms in current set. Need to evaluate
                # eliminate bad arms
                u = self._confidence_interval(sum(self.counts), self.delta/self.n_arms, option=1)  # U(t, δ/n), tested
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
    """
    lil ucb is not eliminating any arms per se.
    The more times an arm is sampled, the better it is. Same logic as regret algo rankings
    This implements a stop condition for the best arm only, but can still output the n best arms.

    """

    def __init__(self, n_arms, counts=None, emp_means=None, delta=0.1, n_candidates=1, variance=0.25,
                 ucbs=None, parameter_set=2, initial_exploration_round=1):
        EliminationAlgorithm.__init__(self, n_arms, counts, emp_means, delta, n_candidates)
        self.variance = variance  # variance for sub-gaussian arm
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)]  # ucb values calculated with means and counts
        self.parameter_set = parameter_set
        # parameter set according to the paper
        if self.parameter_set == 1:  # theoretical guarantee
            self.eps = 0.01
            self.beta = 1
            self.lamb = 9
        elif self.parameter_set == 2:  # better performance in practice
            self.eps = 0
            self.beta = 0.5
            self.lamb = 10/n_arms + 1
            self.delta = self.delta/5  # scale input delta
        else:
            raise ValueError('Parameter set invalid')
        self.initial_exploration_round = initial_exploration_round

    def __str__(self):
        return f'lilucb_{self.n_candidates}_choose{self.n_candidates}_param{self.parameter_set}'

    def reset(self):
        EliminationAlgorithm.reset(self)
        self.ucbs = [0.0 for col in range(self.n_arms)]

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
                    self.best_arms = self.rankings[-self.n_candidates:]  # output last {n_candidates} from rankings
                    return None  # stop; do not return next arm
            # calculate ucb here only limits math error for log. All arms have at least count=1 at this point
            self.ucbs = [e + self._confidence_bound(count) for e, count in zip(self.emp_means, self.counts)]
            return np.argmax(self.ucbs)


class LUCBPlusPlus(EliminationAlgorithm):
    def __init__(self, n_arms, counts=None, emp_means=None, delta=0.1, n_candidates=1):
        EliminationAlgorithm.__init__(self, n_arms, counts, emp_means, delta, n_candidates)
        self.to_play = []

    def __str__(self):
        return f'lucb++_choose{self.n_candidates}'

    def select_next_arm(self):
        if self.best_arms is not None:  # best_arms has been previously identified
            return None  # stop; do not return next arm
        elif sum(self.counts) < len(self.counts):  # run a first pass through all arms
            return np.argmin(self.counts)
        elif len(self.to_play) > 0:
            # every comparison between two sets yield two arms to sample, one gets played immediately
            # the other is saved in to_play()
            return self.to_play.pop()
        else:
            # rank arms by current empirical average
            emp_average_rankings = list(np.arange(len(self.counts))[np.argsort(self.emp_means)])

            # find top and bottom k arms
            top = emp_average_rankings[-self.n_candidates:]
            bottom = emp_average_rankings[:-self.n_candidates]
            # add confidence interval for all of them
            top_mean_with_conf = [self.emp_means[int(t)]-self._confidence_interval(self.counts[int(t)], self.delta/(2*(len(self.counts)-self.n_candidates)), option=2) for t in top]
            bottom_mean_with_conf = [self.emp_means[int(t)]+self._confidence_interval(self.counts[int(t)], self.delta/(2*self.n_candidates), option=2) for t in bottom]
            # confidence interval values look too big here. Impossible to converge

            if min(top_mean_with_conf) > max(bottom_mean_with_conf):  # (k)th element is confidently bigger than (k+1)th element
                self.best_arms = top  # all top k arms are found, terminate
                return None
            else:
                self.to_play.append(bottom[np.argmax(bottom_mean_with_conf)])  # save the (k+1)th arm
                return top[np.argmin(top_mean_with_conf)]  # return the (k)th arm


class SequentialHalving:
    def __init__(self, n_arms, time_limit, counts=None, emp_means=None):
        """

        Parameters
        ----------
        n_arms: int
            number of arms
        time_limit: int
            time horizon limit
        counts: list of int
            sampling counts for all arms
        emp_means: list of int
            empirical means for all arms
        """

        self.n_arms = n_arms
        self.time_limit = time_limit
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        # self.n_candidates = n_candidates  # no n_candidates here because of halving
        self.best_arms = None
        self.rankings = []
        self.current_set = list(np.arange(n_arms))

        # set initial to_play
        n_sample = math.floor(
            self.time_limit / (len(self.current_set) * math.ceil(math.log(self.n_arms, 2)))
        )
        # populate to_play list
        self.to_play = [val for val in self.current_set for _ in range(n_sample)]
        return

    def __str__(self):
        return f'sequential_halving'

    def reset(self):
        self.counts = [0 for col in range(self.n_arms)]
        self.emp_means = [0.0 for col in range(self.n_arms)]
        self.best_arms = None
        self.rankings = []
        self.current_set = list(np.arange(self.n_arms))
        n_sample = math.floor(self.time_limit / (len(self.current_set) * math.ceil(math.log(self.n_arms, 2))))
        self.to_play = [val for val in self.current_set for _ in range(n_sample)]

    def select_next_arm(self):
        if self.best_arms is not None:  # best_arms has been previously identified
            return None  # do not return next arm
        else:
            if not self.to_play:  # to_play empty, played all arms in current set. Need to evaluate
                # evalutate empirical mean for all arms remained in current_set
                current_set_emp_means = [self.emp_means[c] for c in self.current_set]
                top_half_arg = np.argsort(current_set_emp_means)[-math.ceil(len(current_set_emp_means)/2):]
                self.current_set = [self.current_set[t] for t in top_half_arg]

                if len(self.current_set) == 1:  # only the best arm is returned because of the halving
                    self.best_arms = self.current_set
                    return None

                # sample the new current set
                n_sample = math.floor(
                    self.time_limit / (len(self.current_set) * math.ceil(math.log(self.n_arms, 2)))
                )

                # populate to_play list
                self.to_play = [val for val in self.current_set for _ in range(n_sample)]

            return self.to_play.pop()

    def update(self, chosen_arm, reward):
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        # update empirical means
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        self.rankings = list(np.arange(len(self.counts))[np.argsort(self.counts)])  # worst -> best, based on counts
        return

