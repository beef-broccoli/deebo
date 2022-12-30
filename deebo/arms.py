import random
from numpy.random import default_rng


class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def __str__(self):
        return f'Arm with Bernoulli distribution (p={self.p})'

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


class NormalArm:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def __str__(self):
        return f'Arm with normal distribution (mean={self.mean}, sd={self.sd})'

    def draw(self):
        rng = default_rng()
        return rng.normal(self.mean, self.sd)


class NormalArmZeroToOne:
    # if sample drawn from normal distribution is negative, return zero
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def __str__(self):
        return f'[0,1] arm with normal distribution (mean={self.mean}, sd={self.sd})'

    def draw(self):
        rng = default_rng()
        sample = rng.normal(self.mean, self.sd)
        if sample < 0:
            return 0
        elif sample > 1:
            return 1
        else:
            return sample


if __name__ == '__main__':
    pass