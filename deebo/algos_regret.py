#!/usr/bin/env python

"""
implemented algorithms for regret minimization in a multi armed bandit problem

- ETC: explore-then-commit
- Random: random selection of arms
- EpsilonGreedy: epsilon greedy algorithm
- AnnealingEpsilonGreedy: epsilon greedy with annealing (decaying epsilon)
- Boltzmann: softmax algorithm
- AnnealingBoltzmann: softmax with annealing (decaying tau)
- Pursuit
- ReinforcementComparison
- UCB1, UCB1-Tuned, MOSS, UCB-V, UCB2, DMED
- Thompson Sampling
- EXP3

dummy best arm: most played arm for each algorihtm

"""


import random
import math
import numpy as np
from scipy.stats import beta, norm
from utils import zero_nor_one


# parent class
class RegretAlgorithm:

    def __init__(self, n_arms, counts=None, emp_means=None):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.ranking = []  # ranks from worst to best
        return

    def __str__(self):
        return None

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.ranking = []
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
        # update ranking
        self.ranking = list(np.arange(len(self.counts))[np.argsort(self.counts)])
        return


class ETC(RegretAlgorithm):  # explore then commit

    def __init__(self, n_arms, counts=None, emp_means=None, explore_limit=1):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.limit = explore_limit  # how many rounds per arm
        self.best_arm = -1
        return

    def __str__(self):
        return 'etc'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.best_arm = -1
        return

    def select_next_arm(self):
        if sum(self.counts) == self.limit*len(self.counts):  # exploration just complete, pick the best arm
            self.best_arm = np.argmax(self.emp_means)

        if self.best_arm == -1:  # no best arm set, still in the exploration phase
            return np.argmin(self.counts)  # plays the arm with lowest count until exploration ends
        else:  # commit
            return self.best_arm


class Random(RegretAlgorithm):
    # random selection of arms
    # or pure exploration

    def __str__(self):
        return 'random'

    def select_next_arm(self):
        return random.randrange(len(self.emp_means))


class Exploit(RegretAlgorithm):

    # exploit algorithms, always choose the highest
    # or pure greedy

    def __init__(self, n_arms, counts=None, emp_means=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        # set all initial emp_means to 2.0, so all arms are at least select once
        self.emp_means = emp_means if emp_means else [2.0 for col in range(n_arms)]
        return

    def __str__(self):
        return 'exploit'

    def select_next_arm(self):
        return np.random.choice(np.flatnonzero(np.array(self.emp_means) == max(self.emp_means)))


class EpsilonGreedy(RegretAlgorithm):

    def __init__(self, n_arms, epsilon, counts=None, emp_means=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.epsilon = epsilon
        return

    def __str__(self):
        return f'eps_greedy_{self.epsilon}'

    def select_next_arm(self):
        if random.random() > self.epsilon:
            return np.random.choice(np.flatnonzero(np.array(self.emp_means) == max(self.emp_means)))
            # return np.argmax(self.emp_means)  # argmax cannot break ties, bad for initial rounds
        else:
            return random.randrange(len(self.emp_means))


class AnnealingEpsilonGreedy(RegretAlgorithm):

    def __str__(self):
        return 'eps_greedy_annealing'

    def select_next_arm(self):
        t = np.sum(self.counts) + 1
        epsilon = 1/math.log(t + 1e-7)

        if random.random() > epsilon:
            return np.random.choice(np.flatnonzero(np.array(self.emp_means) == max(self.emp_means)))
        else:
            return random.randrange(len(self.emp_means))


class Boltzmann(RegretAlgorithm):  # aka softmax

    def __init__(self, n_arms, tau, counts=None, emp_means=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.tau = tau
        return

    def __str__(self):
        return f'softmax_{self.tau}'

    def select_next_arm(self):
        z = sum([math.exp(v / self.tau) for v in self.emp_means])
        probs = [math.exp(v / self.tau) / z for v in self.emp_means]
        return random.choices(np.arange(len(self.emp_means)), weights=probs, k=1)[0]


class AnnealingBoltzmann(RegretAlgorithm):

    def __str__(self):
        return 'softmax_annealing'

    def select_next_arm(self):
        t = np.sum(self.counts) + 1
        tau = 1/math.log(t + 1e-7)  # TODO: better annealing function

        z = sum([math.exp(v / tau) for v in self.emp_means])
        probs = [math.exp(v / tau) / z for v in self.emp_means]
        return random.choices(np.arange(len(self.emp_means)), weights=probs, k=1)[0]


class Pursuit(RegretAlgorithm):

    def __init__(self, n_arms, lr, counts=None, emp_means=None, probs=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.lr = lr  # learning rate
        self.probs = probs if probs else [float(1/n_arms) for col in range(n_arms)]
        return

    def __str__(self):
        return f'pursuit_{self.lr}'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.probs = [float(1/n_arms) for col in range(n_arms)]
        return

    def select_next_arm(self):
        return random.choices(np.arange(len(self.emp_means)), weights=self.probs, k=1)[0]

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)

        # update probs
        if np.sum(self.emp_means) == 0:  # np.argmax returns the first arm when all reward emp_means are 0, so make sure we don't update probs in that case
            pass
        else:
            for ii in range(len(self.counts)):
                current_prob = self.probs[ii]
                if ii == np.argmax(self.emp_means):
                    self.probs[ii] = current_prob + self.lr*(1-current_prob)
                else:
                    self.probs[ii] = current_prob + self.lr*(0-current_prob)

        return


class ReinforcementComparison(RegretAlgorithm):  # hard to tune with two parameters
    
    def __init__(self, n_arms, alpha, beta, counts=None, emp_means=None, preferences=None, exp_rewards=None, probs=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.alpha = alpha  # learning rate for expected reward
        self.beta = beta  # learning rate for preference
        self.preferences = preferences if preferences else [0.0 for col in range(n_arms)]
        self.exp_rewards = exp_rewards if exp_rewards else [0.0 for col in range(n_arms)]
        self.probs = probs if probs else [float(1/n_arms) for col in range(n_arms)]
        return

    def __str__(self):
        return f'rc_alpha_{self.alpha}_beta_{self.beta}'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.preferences = [0.0 for col in range(n_arms)]  # how to initialize?
        self.exp_rewards = [0.0 for col in range(n_arms)]  # how to initialize?
        self.probs = [float(1/n_arms) for col in range(n_arms)]
        return

    def select_next_arm(self):
        return random.choices(np.arange(len(self.emp_means)), weights=self.probs, k=1)[0]

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)

        # update preference
        self.preferences[chosen_arm] = self.preferences[chosen_arm] + self.beta * (reward - self.exp_rewards[chosen_arm])

        # update expected reward
        self.exp_rewards[chosen_arm] = (1-self.alpha) * self.exp_rewards[chosen_arm] + self.alpha * reward
        #print(self.exp_rewards)

        # update probs
        exp_preference = [math.exp(p) for p in self.preferences]
        s = np.sum(exp_preference)
        self.probs = [e / s for e in exp_preference]

        return


class UCB1(RegretAlgorithm):

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None, batch=False):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)]  # ucb values calculated with means and counts
        self.batch = batch
        return

    def __str__(self):
        return 'ucb1'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.ucbs = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        if not self.batch:
            if 0 in self.counts:  # run a first pass through all arms
                for arm in range(len(self.counts)):
                    if self.counts[arm] == 0:
                        return arm
            else:  # now select arm based on ucb value
                return np.argmax(self.ucbs)
        else:
            return np.argmax(self.ucbs)

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        # update ucb values
        bonuses = [math.sqrt((2 * math.log(sum(self.counts) + 1)) / float(self.counts[arm] + 1e-7)) for arm in range(len(self.counts))]
        self.ucbs = [e + b for e, b in zip(self.emp_means, bonuses)]
        return


class UCB1Tuned(RegretAlgorithm):  # seems like V value are a lot bigger than 1/4, but should be normal behavior with small t

    def __init__(self, n_arms, batch=False, counts=None, emp_means=None, m2=None, ucbs=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.m2 = m2 if m2 else [0.0 for col in range(n_arms)]  # M2(n) = var(n) * n, used to update variance (a more stable Welford's algo)
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)]  # ucb values calculated with means and counts
        self.batch = batch  # use this in batch mode or not
        # batch mode changes select_next_arm() behavior.
        # The first exploration round is done externally, and is skipped in batch mode to so not all algos are exploring at the same time
        return

    def __str__(self):
        return 'ucb1tuned'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.m2 = [0.0 for col in range(n_arms)]
        self.ucbs = [0.0 for col in range(n_arms)]
        return

    def __update_ucbs(self):
        Vs = [self.m2[arm] / (self.counts[arm]+1e-7) + math.sqrt(2 * math.log(sum(self.counts)+1) / float(self.counts[arm] + 1e-7)) for arm in range(len(self.counts))]
        mins = [min(1/4, v) for v in Vs]
        bonuses = [math.sqrt((math.log(sum(self.counts)+1)) / float(self.counts[arm] + 1e-7) * mins[arm]) for arm in range(len(self.counts))]
        self.ucbs = [e + b for e, b in zip(self.emp_means, bonuses)]
        return

    def select_next_arm(self):
        if not self.batch:  # not batch mode, with exploration round
            if 0 in self.counts:  # run a first pass through all arms
                for arm in range(len(self.counts)):
                    if self.counts[arm] == 0:
                        return arm
            else:  # now select arm based on ucb value
                return np.argmax(self.ucbs)
        else:  # batch mode, no exploration
            return np.argmax(self.ucbs)

    def update(self, chosen_arm, reward):
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        # update emp. means
        old_mean = self.emp_means[chosen_arm]
        new_mean = ((n - 1) / float(n)) * old_mean + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_mean
        # update m2 values (n*variance)
        self.m2[chosen_arm] = self.m2[chosen_arm] + (reward - old_mean) * (reward - new_mean)
        # update UCB value
        self.__update_ucbs()
        self.ranking = list(np.arange(len(self.counts))[np.argsort(self.counts)])
        return


class MOSS(UCB1):

    def __str__(self):
        return 'moss'

    # override
    def update(self, chosen_arm, reward):
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        # update emp means
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        # update ucb values
        bonuses = [math.sqrt(
            max(0.0, math.log((sum(self.counts)+1) / (len(self.counts)*(self.counts[arm]+1e-7))))
            /float(self.counts[arm]+1e-7)
        ) for arm in range(len(self.counts))]
        self.ucbs = [e + b for e, b in zip(self.emp_means, bonuses)]
        self.ranking = list(np.arange(len(self.counts))[np.argsort(self.counts)])
        return


class BayesUCBBeta(UCB1):
    # Bayes UCB algorithm with beta prior
    # Implementation 1: simply use standard deviations (with parameter c) as confidence bound

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None, alphas=None, betas=None, c=2, batch=False):
        UCB1.__init__(self, n_arms, counts, emp_means, ucbs, batch)
        self.alphas = alphas if alphas else [1.0 for col in range(n_arms)]
        self.betas = betas if betas else [1.0 for col in range(n_arms)]
        self.c = c  # num of std's to consider as confidence bound
        # c=1 is better for scenario 2, all others use c=2
        return

    def __str__(self):
        return f'bayes_ucb_beta_c={self.c}'

    def reset(self, n_arms):
        UCB1.reset(self, n_arms)
        self.alphas = [1.0 for col in range(n_arms)]
        self.betas = [1.0 for col in range(n_arms)]
        return

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)

        # update α and β
        self.alphas[chosen_arm] = self.alphas[chosen_arm] + reward
        self.betas[chosen_arm] = self.betas[chosen_arm] + (1-reward)

        # update UCB values
        means = [a/(a+b) for a, b in zip(self.alphas, self.betas)]
        stds = [self.c * beta.std(a, b) for a, b in zip(self.alphas, self.betas)]
        self.ucbs = [m + s for m, s in zip(means, stds)]


class BayesUCBBetaPPF(UCB1):
    # used to be callaed NewBayesUCBBeta
    # Bayes UCB algorithm with beta prior
    # Implementation 2: use a percent point function to compare posteriors for different arms. From original paper
    # scipy.Beta.ppf(1-1/t, alpha, beta)
    # https://github.com/Ralami1859/Stochastic-Multi-Armed-Bandit/blob/master/Modules/BayesUCB_RecommendArm.m

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None, alphas=None, betas=None, batch=False):
        UCB1.__init__(self, n_arms, counts, emp_means, ucbs, batch)
        self.alphas = alphas if alphas else [1.0 for col in range(n_arms)]
        self.betas = betas if betas else [1.0 for col in range(n_arms)]
        return

    def __str__(self):
        return 'bayes_ucb_beta_ppf'

    def reset(self, n_arms):
        UCB1.reset(self, n_arms)
        self.alphas = [1.0 for col in range(n_arms)]
        self.betas = [1.0 for col in range(n_arms)]
        return

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)

        # update α and β
        self.alphas[chosen_arm] = self.alphas[chosen_arm] + reward
        self.betas[chosen_arm] = self.betas[chosen_arm] + (1-reward)

        # update UCB values
        self.ucbs = [beta.ppf((1-1/sum(self.counts)), a, b) for a, b in zip(self.alphas, self.betas)]


class BayesUCBGaussianSquared(UCB1):
    # Bayes UCB algorithm with a gaussian prior, similar to ThompsonSamplingGaussianFixedVarSquared
    # the posterior update is missing the square root, but this is also effective
    # see testing for more details

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None, c=2, batch=False):
        UCB1.__init__(self, n_arms, counts, emp_means, ucbs, batch)
        self.c = c  # num of std's to consider as confidence bound
        # c=1 is better for scenario 2, all others use c=2
        return

    def __str__(self):
        return f'bayes_ucb_gaussian_squared_c={self.c}'

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        stds = [self.c * 1/(c+1) for c in self.counts]
        self.ucbs = [m + s for m, s in zip(self.emp_means, stds)]


class BayesUCBGaussianPPF(UCB1):
    # Used to be called NewBayesUCBGaussian
    # same as BayesUCBBetaPPF, but uses a gaussian prior with fixed variance

    def __str__(self):
        return f'bayes_ucb_gaussian_ppf'

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        stds = [1 / math.sqrt(c + 1) for c in self.counts]
        self.ucbs = [norm.ppf((1-1/sum(self.counts)), m, s) for m, s in zip(self.emp_means, stds)]


class BayesUCBGaussian(UCB1):
    # for ucb, use mean+c*posterior std/sqrt(N)
    # https://www.davidsilver.uk/wp-content/uploads/2020/03/XX.pdf

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None, c=2, assumed_sd=0.25, batch=False):
        UCB1.__init__(self, n_arms, counts, emp_means, ucbs, batch)
        self.c = c  # num of std's to consider as confidence bound
        self.assumed_sd = assumed_sd
        # c=1 is better for scenario 2, all others use c=2
        return

    def __str__(self):
        return f'bayes_ucb_gaussian_c={self.c}_assumed_sd={self.assumed_sd}'

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        stds = [self.c * self.assumed_sd/math.sqrt(cc+1e-7) for cc in self.counts]
        self.ucbs = [m + s for m, s in zip(self.emp_means, stds)]


    # class KLUCB(UCB1):

#
#     # override
#     def update(self, chosen_arm, reward):
#
#         # update counts
#         self.counts[chosen_arm] = self.counts[chosen_arm] + 1
#
#         # update emp means
#         n = self.counts[chosen_arm]
#         value = self.emp_means[chosen_arm]
#         new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
#         self.emp_means[chosen_arm] = new_value
#
#         # update UCB values
#
#         return


# batch mode?
class UCBV(RegretAlgorithm):

    def __init__(self, n_arms, counts=None, emp_means=None, sum_reward_squared=None, ucbs=None, vars=None, amplitude=1.0):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.sum_reward_squared = sum_reward_squared if sum_reward_squared else [0.0 for col in range(n_arms)]  # sum of reward^2, used to calculate variance
        self.vars = vars if vars else [0.0 for col in range(n_arms)]
        self.ucbs = ucbs if ucbs else [-1.0 for col in range(n_arms)]
        self.amplitude = amplitude
        return

    def __str__(self):
        return 'ucbv'

    def reset(self, n_arms, amplitude=1.0):
        RegretAlgorithm.reset(self, n_arms)
        self.sum_reward_squared = [0.0 for col in range(n_arms)]
        self.vars = [0.0 for col in range(n_arms)]
        self.ucbs = [-1.0 for col in range(n_arms)]
        self.amplitude = amplitude
        return

    def __update_ucbs(self):
        def exploration(t):
            return 1.2*math.log(t)  # exploration function proposed in the paper
        t = sum(self.counts) + 1
        b1s = [math.sqrt(2 * exploration(t) * self.vars[arm] / float(self.counts[arm] + 1e-7)) for arm in range(len(self.counts))]
        b2s = [3 * self.amplitude * exploration(t) / float(self.counts[arm] + 1e-7) for arm in range(len(self.counts))]
        self.ucbs = [e + b1 + b2 for e, b1, b2 in zip(self.emp_means, b1s, b2s)]
        return

    def select_next_arm(self):
        if 0 in self.counts: # run a first pass through all arms
            for arm in range(len(self.counts)):
                if self.counts[arm] == 0:
                    return arm
        else:  # now select arm based on ucb value
            return np.argmax(self.ucbs)

    def update(self, chosen_arm, reward):
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        # update empirical means
        old_mean = self.emp_means[chosen_arm]
        new_mean = ((n - 1) / float(n)) * old_mean + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_mean
        # update sum of reward^2
        self.sum_reward_squared[chosen_arm] += reward * reward
        # update vars
        self.vars = [self.sum_reward_squared[arm] / float(self.counts[arm] + 1e-7) - pow(self.emp_means[arm], 2) for arm in range(len(self.counts))]
        self.vars = [0 if v < 0 else v for v in self.vars]
        # update ucbs
        self.__update_ucbs()
        self.ranking = list(np.arange(len(self.counts))[np.argsort(self.counts)])
        return


# batch mode?
class UCB2(RegretAlgorithm):

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None, rs=None, alpha=0.5, current_arm=-1, play_time=0):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)]  # ucb values calculated with means and counts
        self.rs = rs if rs else [0.0 for col in range(n_arms)]  # r values as proposed in paper
        self.alpha = alpha  # parameter alpha as proposed in paper
        self.current_arm = current_arm  # current arm that needs to be played
        self.play_time = play_time  # from algo: need to play best arm tau(r+1)-tau(r) times
        return

    def __str__(self):
        return f'ucb2_{self.alpha}'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.ucbs = [0.0 for col in range(n_arms)]
        self.rs = [0.0 for col in range(n_arms)]
        self.current_arm = -1
        self.play_time = 0
        return

    def __tau(self, r):
        return math.ceil((1+self.alpha)**r)

    def __bonus(self, r):
        tau = self.__tau(r)
        n = sum(self.counts)  # total number of plays
        bonus = math.sqrt((1 + self.alpha) * math.log(math.e * n / tau) / (2 * tau))
        return bonus

    def __update_ucbs(self):
        bonuses = [self.__bonus(r) for r in self.rs]
        self.ucbs = [e + b for e, b in zip(self.emp_means, bonuses)]
        return

    def select_next_arm(self):
        if 0 in self.counts:  # run a first pass through all arms
            for arm in range(len(self.counts)):
                if self.counts[arm] == 0:
                    return arm
        elif self.play_time > 0:  # still playing the best arm determined
            self.play_time -= 1
            return self.current_arm
        else:  # need to select a new arm
            self.rs[self.current_arm] += 1  # finished playing best arm, increment r
            self.current_arm = np.argmax(self.ucbs)  # set a new best arm
            self.play_time = self.__tau(self.rs[self.current_arm]+1) - self.__tau(self.rs[self.current_arm]) - 1
            return self.current_arm

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        # update UCB value (actually not necessary at every t)
        self.__update_ucbs()
        return


class ThompsonSamplingBeta(RegretAlgorithm):
    # TS for bernoulli arms, beta distribution as conjugate priors

    def __init__(self, n_arms, counts=None, emp_means=None, alphas=None, betas=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.alphas = alphas if alphas else [1.0 for col in range(n_arms)]
        self.betas = betas if betas else [1.0 for col in range(n_arms)]
        return

    def __str__(self):
        return 'ts_beta'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.alphas = [1.0 for col in range(n_arms)]
        self.betas = [1.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        rng = np.random.default_rng()
        probs = rng.beta(self.alphas, self.betas)
        return np.argmax(probs)

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        # update for beta distribution
        self.alphas[chosen_arm] = self.alphas[chosen_arm] + reward
        self.betas[chosen_arm] = self.betas[chosen_arm] + (1-reward)
        return


class ThompsonSamplingGaussianFixedVar(RegretAlgorithm):
    # TS for gaussian arms with gaussian prior, assume unknown mean but known variance
    # can also be used non-parametric stochastic MAB with log regret
    # assume_sd (int, float): assumed standard deviation for the gaussian prior

    def __init__(self, n_arms, counts=None, emp_means=None, assumed_sd=0.25):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.assumed_sd = assumed_sd
        return

    def __str__(self):
        return f'ts_gaussian_assumed_sd_{self.assumed_sd}'

    def select_next_arm(self):
        stds = [self.assumed_sd/math.sqrt(c+1) for c in self.counts]
        rng = np.random.default_rng()
        probs = rng.normal(self.emp_means, stds)
        return np.argmax(probs)


class ThompsonSamplingGaussianFixedVarSquared(RegretAlgorithm):
    # TS for gaussian arms with gaussian prior, assume unknown mean but known variance
    # Assume a fixed variance of 1, but the variance is squared
    # ***this was a mistake when implemented, but actually works well
    # this can be used for Bernoulli bandits; see testing results for details

    def __init__(self, n_arms, counts=None, emp_means=None, assumed_sd=1):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.assumed_sd = assumed_sd
        return

    def __str__(self):
        return f'ts_gaussian_squared'

    def select_next_arm(self):
        stds = [self.assumed_sd/(c+1) for c in self.counts]
        rng = np.random.default_rng()
        probs = rng.normal(self.emp_means, stds)
        return np.argmax(probs)


class ThompsonSamplingGaussian(RegretAlgorithm):
    # TS for gaussian arms, assume unknown mean and unknown variance
    # gaussian-gamma prior

    def __init__(self, n_arms, counts=None, emp_means=None, alphas=None, betas=None):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.alphas = alphas if alphas else [1.0 for col in range(n_arms)]
        self.betas = betas if betas else [0.1 for col in range(n_arms)]
        return

    def __str__(self):
        return 'ts_gaussian_novar'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.alphas = [1.0 for col in range(n_arms)]
        self.betas = [0.1 for col in range(n_arms)]
        return

    def select_next_arm(self):
        rng = np.random.default_rng()
        precisions = rng.gamma(self.alphas, [1/b for b in self.betas])  # rng.gamma() uses θ (θ=1/β)
        variances = [1/(p+1e-7) for p in precisions]
        probs = rng.normal(self.emp_means, np.sqrt(variances))
        return np.argmax(probs)

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        # update for beta distribution
        n = 1
        nu = self.counts[chosen_arm]
        self.alphas[chosen_arm] = self.alphas[chosen_arm] + 0.5
        self.betas[chosen_arm] = self.betas[chosen_arm] + ((n * nu / (nu + n)) * (((reward - self.emp_means[chosen_arm])**2)/2))
        # print(self.emp_means)
        # print([math.sqrt(b/(a+1)) for a, b in zip(self.alphas, self.betas)])  # estimated SD
        # print('')
        return


class DMED(RegretAlgorithm):

    def __init__(self, n_arms, counts=None, emp_means=None, action_list=None, modified=False):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.action_list = action_list if action_list else []
        self.modified = modified  # if true, generate new list with less aggressive pruning. else follow original paper
        return

    def __str__(self):
        if self.modified:
            return 'dmed_modified'
        else:
            return 'dmed'

    def __kl(self, ps, qs):
        ps = [p+1e-7 if p == 0.0 else p for p in ps]
        ps = [p-1e-7 if p == 1.0 else p for p in ps]
        qs = [q+1e-7 if q == 0.0 else q for q in qs]
        qs = [q-1e-7 if q == 1.0 else q for q in qs]
        ys = [p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q)) for p, q in zip(ps, qs)]
        return ys

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.action_list = []
        return

    def select_next_arm(self):
        if 0 in self.counts:  # run a first pass through all arms
            for arm in range(len(self.counts)):
                if self.counts[arm] == 0:
                    return arm
        else:
            if not self.action_list:  # action list empty. Current loop ended. Construct new action list
                current_best = np.argmax(self.emp_means)
                ys = np.array(self.__kl(self.emp_means, [self.emp_means[current_best]]*len(self.emp_means)))
                # print(f'new list KL calc {ys}')
                # ass = sum(self.counts)/np.array(self.counts)
                # print(f'compare {ass}')
                if self.modified:
                    args = np.array(self.counts) * ys < math.log(sum(self.counts))
                else:
                    args = np.array(self.counts) * ys < np.log(sum(self.counts)/np.array(self.counts))
                self.action_list = list(np.arange(len(self.emp_means))[args])
            # print(self.action_list)
            return self.action_list.pop()


class EXP3(RegretAlgorithm):

    def __init__(self, n_arms, counts=None, emp_means=None, weights=None, probs=None, gamma=0.5):
        RegretAlgorithm.__init__(self, n_arms, counts, emp_means)
        self.weights = weights if weights else [1.0] * int(n_arms)
        self.probs = probs if probs else [1.0/int(n_arms)] * int(n_arms)
        self.gamma = gamma
        return

    def __str__(self):
        return 'exp3'

    def reset(self, n_arms):
        RegretAlgorithm.reset(self, n_arms)
        self.weights = [1.0] * int(n_arms)
        self.probs = [1.0/int(n_arms)] * int(n_arms)
        return

    def select_next_arm(self):  # self.probs updated here
        sum_weight = sum(self.weights)
        self.probs = [(1-self.gamma)*weight/sum_weight + self.gamma/len(self.counts) for weight in self.weights]
        return random.choices(np.arange(len(self.counts)), weights=self.probs, k=1)[0]

    def update(self, chosen_arm, reward):
        RegretAlgorithm.update(self, chosen_arm, reward)
        # update weights
        xs = [0.0 for n in range(len(self.counts))]
        xs[chosen_arm] = reward/self.probs[chosen_arm]
        self.weights = [weight * math.exp(self.gamma * x / len(self.counts)) for weight, x in zip(self.weights, xs)]
        return


if __name__ == '__main__':
    pass
