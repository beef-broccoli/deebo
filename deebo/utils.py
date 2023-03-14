import sys
from pathlib import Path


# check that number is neither 0 nor 1
def zero_nor_one(num):
    if num>1 or num<0:
        return num
    else:
        return min(max(float(num), 1e-7), 1-1e-7)


# fill a list to a desired length with supplied element
def fill_list(li, desired_len, to_fill):
    if li is None:
        return [to_fill]*desired_len
    else:
        return li + [to_fill]*(desired_len-len(li))


def scaler(x):
    # x on a scale of 0-100
    x = x/100
    if x > 1:
        return 1.0
    else:
        return x


def cutoffer(x, cutoff):
    if x>cutoff:
        return 1
    else:
        return 0


def means_from_scenario(scenario=0):
    """

    Parameters
    ----------
    scenario: test scenarios with preset means

    Returns
    -------
    mean reward for each stochastic arm

    """
    if scenario == 1:
        means = [0.1, 0.2, 0.3, 0.4, 0.9]
    elif scenario == 2:
        means = [0.1, 0.1, 0.1, 0.1, 0.2]
    elif scenario == 3:
        means = [0.1, 0.25, 0.5, 0.75, 0.9]
    elif scenario == 4:
        means = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif scenario == 5:
        means = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    else:
        means = None
        sys.exit('invalid test scenario number')
    return means


def plot_info_file_path_match(fp):

    if any(s in fp for s in ['scenario1', 'scenario2', 'scenario3']):
        best_arm_index = 4
    elif 'scenario4' in fp:
        best_arm_index = 8
    elif 'scenario5' in fp:
        best_arm_index = 18
    else:
        best_arm_index = -1

    if 'eps_greedy' in fp:
        return 'epsilon greedy algorithm', 'epsilon', best_arm_index
    elif 'softmax' in fp:
        return 'softmax algorithm', 'tau', best_arm_index
    elif 'optim' in fp:
        return 'optimism algorithm', 'algorithms', best_arm_index
    elif 'pursuit' in fp:
        return 'pursuit algorithm', 'learning rate', best_arm_index
    elif 'ucb2' in fp:
        return 'ucb2 algorithm', 'alpha', best_arm_index
    elif 'exp3' in fp:
        return 'exp3 algorithm', 'gamma', best_arm_index
    else:
        return 'title', 'parameter', best_arm_index


def make_dir(dir):
    p = Path(dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == '__main__':
    print(fill_list(None, 6, 9))