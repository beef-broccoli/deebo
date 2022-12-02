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
    else:
        means = None
        sys.exit('invalid test scenario number')
    return means


def make_dir(dir):
    p = Path(dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == '__main__':
    print(fill_list(None, 6, 9))