import itertools
import pickle
import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import algos_regret

# 1. propose multiple experiments for different arms; this needs to wait for batched algorithms
# 2. how to integrate stop conditions, and use arm selection algorithm
# 3. propose experiment in a non-random way


class Scope:

    def __init__(self):
        self.data_dic = None  # store a copy of the dictionary that's used to build scope
        self.data = None  # dataframe that holds all experiment and result
        self.predictions = None  # predictions from regression model
        self.pre_accuray = None  # prediction accuracy
        self.arms = None  # a list of arms, e.g., [('a2', 'c1'), ('a2', 'c3'), ('a3', 'c1'), ('a3', 'c3')]
        self.arm_labels = None  # label names. e.g., ['component_a', 'component_b']
        self.current_experiment_index = None  # df index of the experiments currently running
        self.current_arms = None  # current arms being evaluated; corresponding to self.current_eperiment_index
        return

    def __str__(self):
        return str(self.data_dic)

    def build_scope(self, d):
        """
        build a dataframe with reaction components populated, except yield

        Parameters
        ----------
        d: dict
            dictionary of reaction components
            e.g., d = {
                'component_a': ['a1', 'a2', 'a3'],
                'component_b': ['b1', 'b2'],
                'component_c': ['c1', 'c2', 'c3', 'c4']
            }

        Returns
        -------
        None

        """
        if self.data:
            exit('scope already exists, cannot build')
        self.data_dic = d

        component_names = sorted(d)
        combinations = itertools.product(*(d[c] for c in component_names))
        self.data = pd.DataFrame(combinations, columns=component_names)
        self.data['yield'] = np.nan
        return

    def query(self, d):
        """
        Queries the reaction scope for yield with a dictionary

        Parameters
        ----------
        d: dict
            dictionary of reaction components and values to be queried for yield
            e.g., d = {
                'component_a': 'a2',
                'component_b': 'b1',
                'component_c': 'c3',
            }

        Returns
        -------
        {int, float, np.nan}
            yield from query, or np.nan if no result is found
        """

        assert self.data.shape[1] - 1 == len(d.items()), 'Missing reaction components when querying data'
        names = set(list(self.data.columns))
        names.remove('yield')  # remove yield from column labels, then match
        assert names == set(d.keys()), 'labels not fully matched, cannot update'

        component_names = sorted(d)  # sort query dictionary by component name
        scope = self.data.sort_index(axis=1)  # sort scope data by component name (column)
        values = [d[c] for c in component_names]  # get values for each component

        # can directly match with a list of value, since both query dict and dataframe are sorted
        y = scope[np.equal.outer(scope.to_numpy(copy=False), values).any(axis=1).all(axis=1)]['yield']
        if y.empty:
            print(f'Query {d} does not exist in this scope')
            return np.nan  # no such result exists in the scope
        elif np.isnan(list(y)[0]):
            print(f'No result yet for {d}')
            return np.nan
        else:
            assert len(list(y)) == 1, f'multiple result exist for query {d}'
            return list(y)[0]

    def update_with_dict(self, d):
        """
        Update scope with reaction yield

        Parameters
        ----------
        d: dict
            dictionary of reaction component (similar to d in query(), with yield).
            e.g.,d = {
                'component_a': 'a2',
                'component_b': 'b1',
                'component_c': 'c3',
                'yield': 69,
            }

        Returns
        -------
        None

        """
        assert self.data.shape[1] == len(d.items()), 'missing reaction components or yield'
        assert 'yield' in list(d.keys()), 'missing yield, cannot update'
        assert type(d['yield']) in [int, float], 'yield value is not numerical'
        assert set(list(self.data.columns)) == set(d.keys()), 'labels not fully matched, cannot update'

        y = d.pop('yield')
        component_names = sorted(d)  # sort query dictionary by component name
        self.data = self.data.sort_index(axis=1)  # sort scope data by component name (column)
        values = [d[c] for c in component_names]  # get values for each component
        # match and update yield
        boo = np.equal.outer(self.data.to_numpy(copy=False), values).any(axis=1).all(axis=1)
        if not any(boo):
            print(f'No update; requested components do not exist for {d}')
        else:
            self.data.loc[boo, 'yield'] = y
        return

    def update_with_index(self, index, y):
        self.data.loc[index, 'yield'] = y
        return

    def predict(self):
        """
        Build regression model to predict for all experiments in scope

        Returns
        -------

        """
        ys = self.data['yield']
        train_mask = ys.notna()  # training data mask: index of experiments that have been run
        ys_train = ys[train_mask].to_numpy()

        Xs = self.data.drop(['yield'], axis=1)
        Xs = OHE().fit_transform(Xs).toarray()
        Xs_train = Xs[train_mask, :]
        Xs_test = Xs[~train_mask, :]

        model = RFR()
        model.fit(Xs_train, ys_train)
        self.pre_accuray = model.score(Xs_train, ys_train)
        self.predictions = model.predict(Xs)

        return

    def build_arms(self, d):
        """
        Function to build arms with a dictionary
            e.g., d = {'component_c': ['c1', 'c3'], 'component_a': ['a2', 'a3']}
            arms =[('a2', 'c1'), ('a2', 'c3'), ('a3', 'c1'), ('a3', 'c3')]

        Parameters
        ----------
        d: dict
            dictionary with component labels as keys, and components as values

        Returns
        -------
        None
        """
        if self.arms:
            exit('Arms already exist, call clear_arms() first before building')
        assert set(d.keys()).issubset(set(list(self.data.columns)) ), 'some labels do not exist in this scope'
        for component in d.keys():
            if not set(d[component]).issubset(set(self.data_dic[component])):
                exit('attempting to build arms with components not present in current scope.')

        self.arm_labels = list(sorted(d.keys()))
        self.arms = list(itertools.product(*(d[c] for c in self.arm_labels)))
        return

    def build_arm_dict(self, arm_index):
        arm = list(self.arms[int(arm_index)])
        return dict(zip(self.arm_labels, arm))

    def clear_arms(self):
        self.arms = None

    def propose_experiment(self, arm_index, mode='random', num_exp=1):
        """
        Propose an experiment for a specified arm.
        Will return None when all experiments for a given arm are sampled

        Parameters
        ----------
        arm_index: int
            index of the selected arm
        mode: {'random'}
            sampling method to select one experiment

        Returns
        -------

        """
        candidates = self.data.loc[self.data['yield'].isnull()]  # experiments without a yield
        for ii in range(len(self.arm_labels)):  # find available experiments for this arm
            candidates = candidates.loc[candidates[self.arm_labels[ii]] == self.arms[arm_index][ii]]

        if mode == 'random':
            try:
                sample = candidates.sample(num_exp)
            except ValueError:  # not enough available reactions for this arm to be sampled
                sample = None
                n = num_exp
                while n > 1:  # keep reducing the number of sample until its sample-able
                    n = n-1
                    try:
                        sample = candidates.sample(n)
                        break
                    except ValueError:
                        continue
            if sample is not None:
                self.current_experiment_index = sample.index
                self.current_arms = [arm_index]*len(self.current_experiment_index)
            else:
                self.current_experiment_index = None
                self.current_arms = None
            return sample
        else:  # other sampling modes
            pass


def propose_initial_experiments(scope_dict, arms_dict, algo, dir='./test/', num_exp=1):
    """
    Build an initial scope, propose initial experiments and save required objects to be loaded later

    Parameters
    ----------
    scope_dict: dict

    arms_dict: dict

    algo: any of the algo object implemented

    dir: str
        directory for all the files to be saved into

    Returns
    -------
    None

    """

    scope = Scope()
    scope.build_scope(scope_dict)
    scope.build_arms(arms_dict)

    chosen_arm = algo.select_next_arm()
    proposed_experiments = scope.propose_experiment(chosen_arm, num_exp=num_exp)

    if os.path.exists(f'{dir}history.csv'):
        os.remove(f'{dir}history.csv')
    if os.path.exists(f'{dir}log.csv'):
        os.remove(f'{dir}log.csv')
    proposed_experiments.to_csv(f'{dir}proposed_experiments.csv')  # save proposed experiments
    with open(f'{dir}algo.pkl', 'wb') as f:
        pickle.dump(algo, f)  # save algo object
    with open(f'{dir}scope.pkl', 'wb') as f:
        pickle.dump(scope, f)  # save scope object

    return


def update_and_propose(dir='./test/', num_exp=1):

    """
    After user filled out experimental result, load the result and update scope and algoritm, propose next experiments

    Parameters
    ----------
    dir: str
        directory where previous log files and results are stored

    Returns
    -------
    None

    """

    # load all files
    with open(f'{dir}algo.pkl', 'rb') as f:
        algo = pickle.load(f)  # load algo object
    with open(f'{dir}scope.pkl', 'rb') as f:
        scope = pickle.load(f)  # load scope object
    exps = pd.read_csv(f'{dir}proposed_experiments.csv', index_col=0)  # proposed experiments with results input from user
    try:
        log = pd.read_csv(f'{dir}log.csv')  # acquisition log for algorithm
    except FileNotFoundError:
        log = None
    try:
        history = pd.read_csv(f'{dir}history.csv', index_col=0)  # experiment history
    except FileNotFoundError:
        history = None

    # get results for proposed experiments
    rewards = np.array(list(exps['yield']))
    if np.isnan(rewards).any():
        exit('need to fill in yield')
    if ((rewards > 1).any()) or ((rewards < 0).any()):
        exit('adjust yield to be between 0 and 1')

    if log is not None:
        horizon = log['horizon'].iloc[-1] + 1
        cumulative_reward = log['cumulative_reward'].iloc[-1]
    else:  # first time logging
        horizon = 0
        cumulative_reward = 0.0

    cumulative_rewards = []
    current = cumulative_reward
    for ii in range(len(rewards)):
        current = current + rewards[ii]
        cumulative_rewards.append(current)
    horizons = list(np.arange(horizon, horizon+len(rewards)))
    chosen_arms = scope.current_arms
    new_log = pd.DataFrame(list(zip(horizons, chosen_arms, rewards, cumulative_rewards)),
                           columns=['horizon', 'chosen_arm', 'reward', 'cumulative_reward'])
    log = pd.concat([log, new_log])

    # update scope, algo, history
    for ii in range(len(rewards)):
        scope.update_with_index(scope.current_experiment_index[ii], rewards[ii])
        algo.update(scope.current_arms[ii], rewards[ii])
    scope.predict()
    new_history = pd.concat([history, exps])

    # propose new experiments
    chosen_arm = algo.select_next_arm()
    proposed_experiments = scope.propose_experiment(chosen_arm, num_exp=num_exp)

    # save files and objects again
    new_history.to_csv(f'{dir}history.csv')
    proposed_experiments.to_csv(f'{dir}proposed_experiments.csv')  # save proposed experiments
    log.to_csv(f'{dir}log.csv', index=False)  # save acquisition log
    with open(f'{dir}algo.pkl', 'wb') as f:
        pickle.dump(algo, f)  # save algo object
    with open(f'{dir}scope.pkl', 'wb') as f:
        pickle.dump(scope, f)  # save scope object

    return None


def simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo):

    """
    Method for simulation; skipping the saving and loading by user, query dataset with full results instead

    Returns
    -------

    """

    scope = Scope()
    scope.build_scope(scope_dict)
    scope.build_arms(arms_dict)

    # propose initial experiments
    log = pd.DataFrame(columns=['horizon', 'chosen_arm', 'reward', 'cumulative_reward'])
    chosen_arm = algo.select_next_arm()
    proposed_experiments = scope.propose_experiment(chosen_arm)
    history = pd.DataFrame(columns=proposed_experiments.columns)

    log.loc[len(log.index)] = [0, chosen_arm, np.nan, 0]

    return


def _test_human_in_the_loop():

    # # build scope
    # x = {'component_b': ['b1', 'b2'],
    #     'component_a': ['a1', 'a2', 'a3'],
    #      'component_c': ['c1', 'c2', 'c3', 'c4']
    # }
    # y = {'component_b': ['b1', 'b2'],
    #      'component_a': ['a1', 'a3']}
    #
    # algo = algos_regret.EpsilonGreedy(4, 0.5)
    # propose_initial_experiments(x, y, algo, num_exp=2)

    update_and_propose(num_exp=2)


def _test_simulate():

    # fetch ground truth data
    ground_truth = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
    def scaler(x):
        # x on a scale of 0-100
        x = x/100
        if x>1:
            return 1.0
        else:
            return x
    ground_truth['yield'] = ground_truth['yield'].apply(scaler)
    ground_truth = ground_truth[['ligand_name',
                                 'electrophile_id',
                                 'nucleophile_id',
                                 'yield']]
    ligands = ground_truth['ligand_name'].unique()
    elecs = ground_truth['electrophile_id'].unique()
    nucs = ground_truth['nucleophile_id'].unique()

    # build dictionary for acquisition
    scope_dict = {'ligand': ligands,
                  'elec': elecs,
                  'nuc': nucs}
    arms_dict = {'ligand': ligands}
    algo = algos_regret.AnnealingEpsilonGreedy


if __name__ == '__main__':
    dir = 'test/'
    with open(f'{dir}scope.pkl', 'rb') as f:
        scope = pickle.load(f)  # load algo object

    print(
        scope.data
    )
