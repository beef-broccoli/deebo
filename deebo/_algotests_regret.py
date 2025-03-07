from arms import *
from algos_regret import *
from algos_arm import *
from algos_testfunc import *
from utils import means_from_scenario, make_dir

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import sys
from pathlib import Path


# test all algorithms with defined scenarios
def test_all(scenario=1, algos=None, n_sims=1000, n_horizon=250, folder_name=None):

    if not folder_name:
        sys.exit('Supply folder name for saving result')

    if not algos:
        # skipping reinforcement comparison, too difficult to tune
        algos = [epsilon_greedy,
                 softmax,
                 pursuit,
                 ucb1,
                 ucb1_tuned,
                 moss,
                 ts_beta,
                 ucbv,
                 ucb2,
                 exp3,
                 dmed,
                 ]

    for algo in tqdm(algos):
        algo(scenario=scenario,
             n_sims=n_sims,
             n_horizon=n_horizon,
             folder_name=folder_name)

    return None


def test_algo_for_all_scenarios(algo, scenarios, n_sims=1000, n_horizon=250, folder_name=None):

    for scenario in scenarios:
        algo(scenario=scenario,
             n_sims=n_sims,
             n_horizon=n_horizon,
             folder_name=folder_name)

    return None


def epsilon_greedy(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/eps_greedy'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    # test for epsilon greedy
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(n_arms, eps)
        algo.reset(n_arms)
        results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
        filename = 'epsilon_' + str(eps) + '.csv'
        results.to_csv(output_dir / filename)

    # test for epsilon greedy with annealing
    algo = AnnealingEpsilonGreedy(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    results.to_csv(output_dir / 'annealing.csv')

    return None


def softmax(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/softmax'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    # test for Boltzmann
    for tau in [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = Boltzmann(n_arms, tau)
        algo.reset(n_arms)
        results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
        filename = 'tau_' + str(tau) + '.csv'
        results.to_csv(output_dir / filename)

    # test for Boltzmann with annealing
    algo = AnnealingBoltzmann(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    results.to_csv(output_dir / 'annealing.csv')

    return None


def pursuit(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/pursuit'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    lrs = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

    # pursuit with set learning rates
    for lr in lrs:
        algo = Pursuit(n_arms, lr)
        algo.reset(n_arms)
        results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
        filename = 'lr_' + str(lr) + '.csv'
        results.to_csv(output_dir / filename)

    return None


# mod needed for testing
def reinforcement_comparison(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/rc'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    alphas = [0.05, 0.1, 0.2]
    betas = [0.05, 0.1, 0.2]

    for a, b in tqdm(itertools.product(alphas, betas)):
            algo = ReinforcementComparison(n_arms, a, b)
            algo.reset(n_arms)
            results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
            filename = 'rc_alpha_' + str(a) + '_beta_' + str(b) + '.csv'
            results.to_csv(output_dir / filename)

    return None


def ucb1(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = UCB1(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = 'ucb1.csv'
    results.to_csv(output_dir / filename)

    return None


def ucb1_tuned(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = UCB1Tuned(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = f'ucb1_tuned-{n_sims}s-{n_horizon}r.csv'
    results.to_csv(output_dir / filename)

    return None


def moss(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = MOSS(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = f'moss.csv'
    results.to_csv(output_dir / filename)

    return None


def bayes_ucb_beta(scenario, n_sims, n_horizon, folder_name, c=2):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = BayesUCBBeta(n_arms, c=c)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = 'bayes_ucb_beta.csv'
    results.to_csv(output_dir / filename)

    return None


def ucb1tunednormal(scenario, n_sims, n_horizon, folder_name, c=2, assume_sd=0.5):

    fp = folder_name + f'/scenario{scenario}/TS/TS_squared/'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    sds = [0.25 for i in range(n_arms)]
    def build(x, y):
        return NormalArmZeroToOne(x, y)
    arms = list(map(build, means, sds))
    # arms = list(map(lambda x: BernoulliArm(x), means))

    algo = ThompsonSamplingGaussianFixedVarSquared(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = 'realsd_0.25.csv'
    results.to_csv(output_dir / filename)

    return None


def old_bayes_ucb_beta(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = BayesUCBBeta(n_arms, c=2)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = f'bayes_ucb_beta_c=2.csv'
    results.to_csv(output_dir / filename)

    return None


def etc(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/etc'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    exploration_limit = n_horizon // n_arms
    exp_len = np.arange(exploration_limit)+1

    for e in exp_len:
        algo = ETC(n_arms, explore_limit=e)
        algo.reset(n_arms)
        results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
        filename = f'{e}_exp_per_arm.csv'
        results.to_csv(output_dir / filename)
    return None


def ts_beta(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = ThompsonSamplingBeta(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = f'TS-{n_sims}s-{n_horizon}r.csv'
    results.to_csv(output_dir / filename)

    return None


def ucbv(scenario, n_sims, n_horizon, folder_name):
    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = UCBV(n_arms)
    algo.reset(n_arms)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    filename = 'ucbv.csv'
    results.to_csv(output_dir / filename)

    return None


def ucb2(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/ucb2'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        algo = UCB2(n_arms, alpha=a)
        algo.reset(n_arms)
        results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
        filename = f'ucb2_{a}.csv'
        results.to_csv(output_dir / filename)

    return None


def exp3(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/exp3'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    for g in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        algo = EXP3(n_arms, gamma=g)
        algo.reset(n_arms)
        results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
        filename = f'gamma_{g}.csv'
        results.to_csv(output_dir / filename)


def dmed(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/optim'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    algo = DMED(n_arms, modified=True)
    results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
    if algo.modified:
        filename = f'dmed_modified.csv'
    else:
        filename = f'dmed.csv'
    results.to_csv(output_dir / filename)


def _test_batched_multidraw(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/batch/multidraw/'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    # test for epsilon greedy
    for n_exps in [1,2,3,4,5]:
        algo = ThompsonSamplingBeta(n_arms)
        result = test_algorithm_regret_multidraw(algo, arms, n_sims, n_horizon, n_exps=n_exps)
        result.to_csv(output_dir / f'TS_{n_exps}.csv')

    return None


def _test_batched_multialgo(scenario, n_sims, n_horizon, folder_name):

    fp = folder_name + f'/scenario{scenario}/batch/multialgo/'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    # test for epsilon greedy
    n_exps = 4
    algos = [UCB1Tuned(n_arms, batch=True),
             BayesUCBBeta(n_arms, batch=True),
             BayesUCBGaussian(n_arms, batch=True, c=1),
             ThompsonSamplingGaussianFixedVar(n_arms)]
    result = test_algorithm_regret_multialgos(algos, arms, n_sims, n_horizon)
    result.to_csv(output_dir / f'UCB1Tuned-BayesUCBBeta-BayesUCBGaussian(c=1)-TSGaussianFixedVar.csv')

    return None


def test_TS_gaussian(scenario, n_sims, n_horizon):

    fp = f'./logs/scenario{scenario}/TS'
    output_dir = make_dir(fp)

    means = means_from_scenario(scenario)
    n_arms = len(means)
    arms = list(map(lambda x: BernoulliArm(x), means))

    sds = [0.1, 0.25, 0.5, 0.75, 1]
    for sd in sds:
        algo = ThompsonSamplingGaussianFixedVar(n_arms, assumed_sd=sd)
        results = test_algorithm_regret(algo, arms, n_sims, n_horizon)
        results.to_csv(output_dir/f'TS_gaussian_assumed_sd_{str(sd)}.csv')

    return None


def test_n_arms(folder_name, n_sims=500, n_horizon=15000):
    for s in [15]:
        ts_beta(scenario=s,
                n_sims=n_sims,
                n_horizon=n_horizon,
                folder_name=folder_name)


if __name__ == '__main__':
    #test_TS_gaussian(0, 1000, 250)

    #test_all(scenario=1, n_sims=50, n_horizon=100, folder_name='./logs/tests')
    #_test_batched_multialgo(2, 1000, 250, './logs/')
    #_test_batched(1, 1000, 250, 'logs/')
    #etc(scenario=5, n_sims=1000, n_horizon=500, folder_name='./baseline_logs/')

    #test_eps_greedy(1, 3, 200, './test/')

    test_n_arms('logs/scalability')

