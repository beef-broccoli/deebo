#!/usr/bin/env python

"""
implemented algorithms for bandit optimization

- ETC: explore-then-commit
- Random: random selection of arms
- EpsilonGreedy: epsilon greedy algorithm
- AnnealingEpsilonGreedy: epsilon greedy with annealing (decaying epsilon)
- Boltzmann: softmax algorithm
- AnnealingBoltzmann: softmax with annealing (decaying tau)
- Pursuit
- ReinforcementComparison

Could write a parent class to incorporate common methods, but this flatter version is probably easier to understand
"""

import random
import math
import numpy as np


class ETC:  # explore then commit

    def __init__(self, counts, emp_means, explore_limit=1):
        self.counts = counts
        self.emp_means = emp_means
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

    def __init__(self, counts, emp_means):
        self.counts = counts
        self.emp_means = emp_means
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

    def __init__(self, epsilon, counts, emp_means):
        self.epsilon = epsilon
        self.counts = counts
        self.emp_means = emp_means  # empirical means of rewards
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        if random.random() > self.epsilon:
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


class AnnealingEpsilonGreedy:

    def __init__(self, counts, emp_means):
        self.counts = counts
        self.emp_means = emp_means  # reward value (as average)
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

    def __init__(self, tau, counts, emp_means):
        self.tau = tau
        self.counts = counts
        self.emp_means = emp_means  # reward value (average)
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


class AnnealingBoltzmann:
    def __init__(self, counts, emp_means):
        self.counts = counts
        self.emp_means = emp_means  # reward value (average)
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


class Pursuit:

    def __init__(self, lr, counts, emp_means, probs):
        self.lr = lr  # learning rate
        self.counts = counts
        self.emp_means = emp_means  # reward value (average)
        self.probs = probs
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
    
    def __init__(self, alpha, beta, counts, emp_means, preferences, exp_rewards, probs):
        self.alpha = alpha  # learning rate for expected reward
        self.beta = beta  # learning rate for preference 
        self.counts = counts  # num data points for each arm
        self.emp_means = emp_means  # empirical means of rewards for each arm
        self.preferences = preferences
        self.exp_rewards = exp_rewards
        self.probs = probs
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

    def __init__(self, counts, emp_means, ucbs):
        self.counts = counts
        self.emp_means = emp_means
        self.ucbs = ucbs  # ucb values calculated with means and counts
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.ucbs = [0.0 for col in range(n_arms)]
        return

    def __update_ucbs(self):
        bonuses = [math.sqrt((2 * math.log(sum(self.counts) + 1)) / float(self.counts[arm] + 1e-7)) for arm in range(len(self.counts))]
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
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.emp_means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.emp_means[chosen_arm] = new_value
        self.__update_ucbs()
        return


class UCB1Tuned:  # seems like V value are a lot bigger than 1/4, but should be normal behavior with small t

    def __init__(self, counts, emp_means, M2, ucbs):
        self.counts = counts
        self.emp_means = emp_means
        self.M2 = M2  # M2(n) = var(n) * n, used to update variance (a more stable Welford's algo)
        self.ucbs = ucbs  # ucb values calculated with means and counts
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.emp_means = [0.0 for col in range(n_arms)]
        self.M2 = [0.0 for col in range(n_arms)]
        self.ucbs = [0.0 for col in range(n_arms)]
        return

    def __update_ucbs(self):
        Vs = [self.M2[arm] / (self.counts[arm]+1e-7) + math.sqrt(2 * math.log(sum(self.counts)+1) / float(self.counts[arm] + 1e-7)) for arm in range(len(self.counts))]
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
        # update M2 values (n*variance)
        self.M2[chosen_arm] = self.M2[chosen_arm] + (reward - old_mean) * (reward - new_mean)
        # update UCB value
        self.__update_ucbs()
        return


class UCBV:

    def __init__(self, counts, emp_means, sum_reward_squared, ucbs, vars, amplitude=1.0):
        self.counts = counts
        self.emp_means = emp_means
        self.sum_reward_squared = sum_reward_squared  # sum of reward^2, used to calculate variance
        self.vars = vars
        self.ucbs = ucbs
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

    def __init__(self, counts, emp_means, ucbs, rs, alpha=0.5, current_arm=-1, play_time=0):
        self.counts = counts
        self.emp_means = emp_means
        self.ucbs = ucbs  # ucb values calculated with means and counts
        self.rs = rs  # r values as proposed in paper
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

    def __init__(self, counts, emp_means, alphas, betas):
        self.counts = counts
        self.emp_means = emp_means
        self.alphas = alphas
        self.betas = betas
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


if __name__ == '__main__':
    pass
