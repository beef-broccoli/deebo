import random
import math
import numpy as np


class EpsilonGreedy:  # could implement decay epsilon

    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values  # reward value (as average)
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values), 1  # exploit == True
        else:
            return random.randrange(len(self.values)), 0  # exploit == False aka explore

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


class AnnealingEpsilonGreedy:

    def __init__(self, counts, values):
        self.counts = counts
        self.values = values  # reward value (as average)
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        t = np.sum(self.counts) + 1
        epsilon = 1/math.log(t + 1e-7)

        if random.random() > epsilon:
            return np.argmax(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


class Boltzmann:  # aka softmax

    def __init__(self, tau, counts, values):
        self.tau = tau
        self.counts = counts
        self.values = values  # reward value (average)
        return

    def reset(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_next_arm(self):
        z = sum([math.exp(v / self.tau) for v in self.values])
        probs = [math.exp(v / self.tau) / z for v in self.values]
        return random.choices(np.arange(len(self.values)), weights=probs, k=1)[0]

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


class AnnealingBoltzmann:
    pass



if __name__ == '__main__':
    a = EpsilonGreedy(epsilon=0, counts=[], values=[])
    a.reset(5)
