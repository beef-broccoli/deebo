import random
import numpy as np


class EpsilonGreedy():

    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values  # reward value (average)
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


if __name__ == '__main__':
    a = EpsilonGreedy(epsilon=0, counts=[], values=[])
    a.reset(5)
