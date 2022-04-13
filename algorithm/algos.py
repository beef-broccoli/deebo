import random
import math
import numpy as np


class EpsilonGreedy:  # could implement decay epsilon

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


class ReinforcementComparison:  # TODO: need more test, doesn't seem to work
    
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
        print(self.preferences)

        # update expected reward
        self.exp_rewards[chosen_arm] = (1-self.alpha) * self.exp_rewards[chosen_arm] + self.alpha * reward
        #print(self.exp_rewards)

        # update probs
        exp_preference = [math.exp(p) for p in self.preferences]
        s = np.sum(exp_preference)
        self.probs = [e / s for e in exp_preference]
        #print(self.probs)

        return
    

if __name__ == '__main__':
    a = EpsilonGreedy(epsilon=0, counts=[], emp_means=[])
    a.reset(5)
