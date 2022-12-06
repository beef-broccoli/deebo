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

"""

import random
import math
import numpy as np
from utils import zero_nor_one


class ETC:  # explore then commit

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


class Random:  # random selection of arms

    def __init__(self, n_arms, counts=None, emp_means=None):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        return random.randrange(len(self.emp_means))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


class EpsilonGreedy:

    def __init__(self, n_arms, epsilon, counts=None, emp_means=None):
        self.epsilon = epsilon
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        if random.random() > self.epsilon:
            return np.random.choice(np.flatnonzero(np.array(self.emp_means) == max(self.emp_means)))
            # return np.argmax(self.emp_means)  # argmax cannot break ties, bad for initial rounds
        else:
            return random.randrange(len(self.emp_means))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


class AnnealingEpsilonGreedy:

    def __init__(self, n_arms, counts=None, emp_means=None):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        t = np.sum(self.counts) + 1
        epsilon = 1/math.log(t + 1e-7)

        if random.random() > epsilon:
            return np.argmax(self.emp_means)
        else:
            return random.randrange(len(self.emp_means))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


class Boltzmann:  # aka softmax

    def __init__(self, n_arms, tau, counts=None, emp_means=None):
        self.tau = tau
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        z = sum([math.exp(v / self.tau) for v in self.emp_means])
        probs = [math.exp(v / self.tau) / z for v in self.emp_means]
        return random.choices(np.arange(len(self.emp_means)), weights=probs, k=1)[0]

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


# TODO: better annealing function
class AnnealingBoltzmann:
    def __init__(self, n_arms, counts=None, emp_means=None):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        t = np.sum(self.counts) + 1
        tau = 1/math.log(t + 1e-7)

        z = sum([math.exp(v / tau) for v in self.emp_means])
        probs = [math.exp(v / tau) / z for v in self.emp_means]
        return random.choices(np.arange(len(self.emp_means)), weights=probs, k=1)[0]

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


#TODO: annealing learning rate
class Pursuit:

    def __init__(self, n_arms, lr, counts=None, emp_means=None, probs=None):
        self.lr = lr  # learning rate
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.probs = probs if probs else [float(1/n_arms) for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.probs = [float(1/n_arms) for col in range(n_arms)]
        return

    def select_next_arm(self):
        return random.choices(np.arange(len(self.emp_means)), weights=self.probs, k=1)[0]

    def update(self, chosen_arm, reward):

        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        # update reward
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value

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


class ReinforcementComparison:  # need more test, doesn't seem to work
    
    def __init__(self, n_arms, alpha, beta, counts=None, emp_means=None, preferences=None, exp_rewards=None, probs=None):
        self.alpha = alpha  # learning rate for expected reward
        self.beta = beta  # learning rate for preference 
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.preferences = preferences if preferences else [0.0 for col in range(n_arms)]
        self.exp_rewards = exp_rewards if exp_rewards else [0.0 for col in range(n_arms)]
        self.probs = probs if probs else [float(1/n_arms) for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.preferences = [0.0 for col in range(n_arms)]  # how to initialize?
        self.exp_rewards = [0.0 for col in range(n_arms)]  # how to initialize?
        self.probs = [float(1/n_arms) for col in range(n_arms)]
        return

    def select_next_arm(self):
        return random.choices(np.arange(len(self.emp_means)), weights=self.probs, k=1)[0]

    def update(self, chosen_arm, reward):
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        # update empirical means
        value = self.emp_means[chosen_arm]
        new_mean = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_mean

        # update preference
        self.preferences[chosen_arm] = self.preferences[chosen_arm] + self.beta * (reward - self.exp_rewards[chosen_arm])

        # update expected reward
        self.exp_rewards[chosen_arm] = (1-self.alpha) * self.exp_rewards[chosen_arm] + self.alpha * reward
        #print(self.exp_rewards)

        # update probs
        exp_preference = [math.exp(p) for p in self.preferences]
        s = np.sum(exp_preference)
        self.probs = [e / s for e in exp_preference]
        #print(self.probs)

        return


class UCB1:

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)] # ucb values calculated with means and counts
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.ucbs = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        if sum(self.counts) < len(self.counts):  # run a first pass through all arms
            for arm in range(len(self.counts)):
                if self.counts[arm] == 0:
                    return arm
        else:  # now select arm based on ucb value
            return np.argmax(self.ucbs)

    def update(self, chosen_arm, reward):
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        # update emp means
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        # update ucb values
        bonuses = [math.sqrt((2 * math.log(sum(self.counts) + 1)) / float(self.counts[arm] + 1e-7)) for arm in range(len(self.counts))]
        self.ucbs = [e + b for e, b in zip(self.emp_means, bonuses)]
        return


class UCB1Tuned:  # seems like V value are a lot bigger than 1/4, but should be normal behavior with small t

    def __init__(self, n_arms, counts=None, emp_means=None, m2=None, ucbs=None):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.m2 = m2 if m2 else [0.0 for col in range(n_arms)]  # M2(n) = var(n) * n, used to update variance (a more stable Welford's algo)
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)]  # ucb values calculated with means and counts
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
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
        if sum(self.counts) < len(self.counts):  # run a first pass through all arms
            for arm in range(len(self.counts)):
                if self.counts[arm] == 0:
                    return arm
        else:  # now select arm based on ucb value
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
        return


class MOSS(UCB1):

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
        return


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


class UCBV:

    def __init__(self, n_arms, counts=None, emp_means=None, sum_reward_squared=None, ucbs=None, vars=None, amplitude=1.0):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.sum_reward_squared = sum_reward_squared if sum_reward_squared else [0.0 for col in range(n_arms)]  # sum of reward^2, used to calculate variance
        self.vars = vars if vars else [0.0 for col in range(n_arms)]
        self.ucbs = ucbs if ucbs else [-1.0 for col in range(n_arms)]
        self.amplitude = amplitude
        return

    def reset(self, n_arms, amplitude=1.0):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
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
        if sum(self.counts) < len(self.counts):  # run a first pass through all arms
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
        return


class UCB2:

    def __init__(self, n_arms, counts=None, emp_means=None, ucbs=None, rs=None, alpha=0.5, current_arm=-1, play_time=0):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.ucbs = ucbs if ucbs else [0.0 for col in range(n_arms)]  # ucb values calculated with means and counts
        self.rs = rs if rs else [0.0 for col in range(n_arms)]  # r values as proposed in paper
        self.alpha = alpha  # parameter alpha as proposed in paper
        self.current_arm = current_arm  # current arm that needs to be played
        self.play_time = play_time  # from algo: need to play best arm tau(r+1)-tau(r) times
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
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
        if sum(self.counts) < len(self.counts):  # run a first pass through all arms
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
        # update counts
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        # update emp. means
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        # update UCB value (actually not necessary at every t)
        self.__update_ucbs()
        return


class ThompsonSampling:  # TS for bernoulli arms, beta distribution as conjugate priors

    def __init__(self, n_arms, counts=None, emp_means=None, alphas=None, betas=None):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.alphas = alphas if alphas else [1.0 for col in range(n_arms)]
        self.betas = betas if betas else [1.0 for col in range(n_arms)]
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.alphas = [1.0 for col in range(n_arms)]
        self.betas = [1.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        rng = np.random.default_rng()
        probs = rng.beta(self.alphas, self.betas)
        return np.argmax(probs)

    def update(self, chosen_arm, reward):
        # update emp means
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        # update for beta distribution
        self.alphas[chosen_arm] = self.alphas[chosen_arm] + reward
        self.betas[chosen_arm] = self.betas[chosen_arm] + (1-reward)
        return


class DMED:

    def __init__(self, n_arms, counts=None, emp_means=None, action_list=None, modified=False):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.action_list = action_list if action_list else []
        self.modified = modified  # if true, generate new list with less aggressive pruning. else follow original paper
        return

    def __kl(self, ps, qs):
        ps = [p+1e-7 if p == 0.0 else p for p in ps]
        ps = [p-1e-7 if p == 1.0 else p for p in ps]
        qs = [q+1e-7 if q == 0.0 else q for q in qs]
        qs = [q-1e-7 if q == 1.0 else q for q in qs]
        ys = [p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q)) for p, q in zip(ps, qs)]
        return ys

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.action_list = []
        return

    def select_next_arm(self):
        if sum(self.counts) < len(self.counts):  # run a first pass through all arms
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

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        return


class EXP3:

    def __init__(self, n_arms, counts=None, emp_means=None, weights=None, probs=None, gamma=0.5):
        self.counts = counts if counts else [0 for col in range(n_arms)]
        self.emp_means = emp_means if emp_means else [0.0 for col in range(n_arms)]
        self.weights = weights if weights else [1.0] * int(n_arms)
        self.probs = probs if probs else [1.0/int(n_arms)] * int(n_arms)
        self.gamma = gamma
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.weights = [1.0] * int(n_arms)
        self.probs = [1.0/int(n_arms)] * int(n_arms)
        return

    def select_next_arm(self):  # self.probs updated here
        sum_weight = sum(self.weights)
        self.probs = [(1-self.gamma)*weight/sum_weight + self.gamma/len(self.counts) for weight in self.weights]
        return random.choices(np.arange(len(self.counts)), weights=self.probs, k=1)[0]

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1  # update counts
        # update empirical means
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        # update weights
        xs = [0.0 for n in range(len(self.counts))]
        xs[chosen_arm] = reward/self.probs[chosen_arm]
        self.weights = [weight * math.exp(self.gamma * x / len(self.counts)) for weight, x in zip(self.weights, xs)]
        return


if __name__ == '__main__':
    pass