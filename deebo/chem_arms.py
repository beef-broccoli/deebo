import itertools
import pathlib
import pickle
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


import algos_regret
import utils

# dev notes
# - propose multiple experiments for different arms; WAIT FOR BATCHED ALGORITHM
# - add round number; WAIT FOR BATCHED ALGORITHM
# - how to integrate stop conditions, and use arm selection algorithm
# - propose experiment in a non-random way
# - (maybe) prediction models with features
# - (maybe) better handle situation when no experiments are available. Set algo count very high to eliminate uncertainty?


class Scope:

    def __init__(self):
        self.data_dic = None  # store a copy of the dictionary that's used to build scope
        self.data = None  # dataframe that holds all experiment and result
        self.pre_accuracy = None  # prediction accuracy
        self.arms = None  # a list of arms, e.g., [('a2', 'c1'), ('a2', 'c3'), ('a3', 'c1'), ('a3', 'c3')]
        self.arm_labels = None  # arm label names. e.g., ['component_a', 'component_b']
        self.current_experiment_index = None  # df index of the experiments currently running
        self.current_arms = None  # current arms being evaluated; corresponding to self.current_eperiment_index
        return

    def __str__(self):
        return str(self.data_dic)

    def reset(self):
        """
        For simulations, reset the scope object. Keep all arm parameters, but remove all data
        Returns
        -------

        """
        d = self.data_dic
        self.data = None
        self.build_scope(d)
        self.pre_accuracy = None
        self.current_experiment_index = None
        self.current_arms = None
        return

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
        if self.data is not None:
            exit('scope already exists, cannot build')

        component_names = sorted(d)
        combinations = itertools.product(*(d[c] for c in component_names))
        self.data = pd.DataFrame(combinations, columns=component_names)
        # check column dtypes for int or float. These cause issues for value matching when querying/updating
        # add column name as prefix to force string
        for col in self.data.columns:
            if self.data[col].dtype == 'int64' or self.data[col].dtype == 'float64':
                print(f'Warning: attempting to pass int/float as values for {col}.\n'
                      f'Renaming values with prefix, e.g., {self.data[col][0]} -> {col} {self.data[col][0]}\n'
                      f'Modifications to values needed when building arms')
                new_val = [col+' '+str(v) for v in list(self.data[col])]
                self.data[col] = new_val
                d[col] = list(set(new_val))
        self.data_dic = d
        self.data['yield'] = np.nan
        self.data['prediction'] = np.nan

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

        assert self.data.shape[1] - 1 == len(d.items()), f'Missing reaction components when querying {d}'
        names = set(list(self.data.columns))
        names.remove('yield')  # remove yield from column labels, then match
        assert names == set(d.keys()), f'labels not fully matched for {d}, cannot query'

        component_names = sorted(d)  # sort query dictionary by component name
        scope = self.data.sort_index(axis=1)  # sort scope data by component name (column)
        values = [d[c] for c in component_names]  # get values for each component

        # can directly match with a list of value, since both query dict and dataframe are sorted
        y = scope[np.equal.outer(scope.to_numpy(copy=False), values).any(axis=1).all(axis=1)]['yield']
        # WARNING: np.equal seems to cause very weird inequality problem if any dataframe element is int
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
        if Xs_train.shape[0] > 2:  # to suppress sklearn warning when calculating R2 score with <2 samples
            self.pre_accuracy = model.score(Xs_train, ys_train)
        self.data['prediction'] = model.predict(Xs)

        return

    def recommend(self):

        df = self.data.copy()
        # supplement the experiments without yield with predicted yield
        yields = np.array(df['yield'])
        pres = np.array(df['prediction'])
        mask = np.isnan(yields)
        yields[mask] = 0
        pres[~mask] = 0
        df['yield'] = yields + pres

        df['arm'] = df[self.arm_labels].apply(tuple, axis=1)
        df = df.drop(self.arm_labels, axis=1)
        columns_to_groupby = [c for c in df.columns if c not in ['yield', 'arm']]
        recommendations = df[df.groupby(columns_to_groupby)['yield'].transform(max) == df['yield']]

        return recommendations

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
        assert set(d.keys()).issubset(set(list(self.data.columns))), 'some labels do not exist in this scope'
        for component in d.keys():
            if not set(d[component]).issubset(set(self.data_dic[component])):
                exit('attempting to build arms with components not present in current scope.')

        self.arm_labels = list(sorted(d.keys()))
        self.arms = list(itertools.product(*(d[c] for c in self.arm_labels)))
        return

    def build_arm_dict(self, arm_index):
        """
        Build a dictionary {(component_a, component_b): (a1, b2)}. For query purposes
        Parameters
        ----------
        arm_index: int
            arm index

        Returns
        -------
        dict

        """
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
        num_exp: int
            the number of experiments to be proposed for this arm

        Returns
        -------

        """
        candidates = self.data.loc[self.data['yield'].isnull()]  # experiments without a yield
        for ii in range(len(self.arm_labels)):  # find available experiments for this arm
            candidates = candidates.loc[candidates[self.arm_labels[ii]] == self.arms[arm_index][ii]]

        if mode == 'random':  # random sampling
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
        elif mode == 'highest':  # choose the n highest predicted yield
            sample = candidates.nlargest(num_exp, 'prediction')
            if len(sample.index) != 0:
                self.current_experiment_index = sample.index
                self.current_arms = [arm_index] * len(self.current_experiment_index)
            else:
                sample = None  # set empty dataframe to None; important for downstream function
                self.current_experiment_index = None
                self.current_arms = None
            return sample
        elif mode == 'random_highest':  # randomly choose one from n highest yields; n default to 5
            default_n_highest = 5  # get n experiments with highest predicted yield
            if num_exp >= default_n_highest:
                exit(f'requested # of experiments are too many for random highest mode. Default n_highest set to {default_n_highest}.')
            nlargest = candidates.nlargest(default_n_highest, 'prediction')
            try:
                sample = nlargest.sample(num_exp)
            except ValueError:
                sample = None
                n = num_exp
                while n > 1:  # keep reducing the number of sample until its sample-able
                    n = n - 1
                    try:
                        sample = nlargest.sample(n)
                        break
                    except ValueError:
                        continue
            if sample is not None:
                self.current_experiment_index = sample.index
                self.current_arms = [arm_index] * len(self.current_experiment_index)
            else:
                self.current_experiment_index = None
                self.current_arms = None
            return sample
        else:  # other sampling modes
            pass


def propose_initial_experiments(scope_dict, arms_dict, algo, dir='./test/', num_exp=1, propose_mode='random'):
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

    d = dict(zip(np.arange(len(scope.arms)), scope.arms))
    with open(f'{dir}arms.pkl', 'wb') as f:
        pickle.dump(d, f)  # save arms dictionary {arm index: arm names}

    chosen_arm = algo.select_next_arm()
    proposed_experiments = scope.propose_experiment(chosen_arm, num_exp=num_exp, mode=propose_mode)

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


def update_and_propose(dir='./test/', num_exp=1, propose_mode='random'):

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

    THRESHOLD = 100  # threshold for how many experiments to wait for algo to output a different arm to evaluate

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

    # get results from user for proposed experiments
    rewards = np.array(list(exps['yield']))
    if np.isnan(rewards).any():
        exit('need to fill in yield')
    if ((rewards > 1).any()) or ((rewards < 0).any()):
        exit('adjust yield to be between 0 and 1')

    # get some info from logs
    if log is not None:
        horizon = log['horizon'].iloc[-1] + 1
        cumulative_reward = log['cumulative_reward'].iloc[-1]
    else:  # first time logging
        horizon = 0
        cumulative_reward = 0.0

    # set up horizons, chosen_arms, reward and cumulative reward and update log
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
    new_history = pd.concat([history, exps]).drop(columns=['prediction'])

    # propose new experiments
    chosen_arm = algo.select_next_arm()
    proposed_experiments = scope.propose_experiment(chosen_arm, num_exp=num_exp, mode=propose_mode)
    if proposed_experiments is None:  # no experiments available for this arm
        threshold = 0
        print(f'No experiments available for arm {chosen_arm}: {scope.arms[chosen_arm]}. Trying to find new experiments')
        while proposed_experiments is None:
            if threshold > THRESHOLD:
                print(f'No experiments available for arm {chosen_arm}: {scope.arms[chosen_arm]} after '
                      f'{THRESHOLD} attempts; it might be the best arm')
                break
            algo.update(chosen_arm, algo.emp_means[chosen_arm])
            new_chosen_arm = algo.select_next_arm()
            if new_chosen_arm == chosen_arm:
                threshold = threshold + 1
                continue
            else:
                proposed_experiments = scope.propose_experiment(new_chosen_arm, num_exp=num_exp, mode=propose_mode)

    # save files and objects again
    new_history.to_csv(f'{dir}history.csv')
    if proposed_experiments is not None:
        proposed_experiments.to_csv(f'{dir}proposed_experiments.csv')  # save proposed experiments
    log.to_csv(f'{dir}log.csv', index=False)  # save acquisition log
    with open(f'{dir}algo.pkl', 'wb') as f:
        pickle.dump(algo, f)  # save algo object
    with open(f'{dir}scope.pkl', 'wb') as f:
        pickle.dump(scope, f)  # save scope object

    return None


def simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo, dir='./test/', num_sims=1000, num_exp=1, num_round=100, propose_mode='random'):
    """
    Method for simulation; skipping the saving and loading by user, query dataset with full results instead

    Parameters
    ----------
    scope_dict: dict
    arms_dict
    ground_truth
    algo
    dir: str
        directory to save files into
    num_sims: int
    num_exp: int
        number of experiments for the scope object
    num_round

    Returns
    -------

    """
    THRESHOLD = 100

    ground_truth_cols = set(ground_truth.columns)
    ground_truth_cols.remove('yield')
    assert set(scope_dict.keys()) == ground_truth_cols, \
        'ground truth and scope do not have the same fields'

    # numpy array for acquisition log
    log_cols = ['num_sims', 'round', 'experiment', 'horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    log_arr = np.zeros((num_sims * num_round * num_exp, len(log_cols)))

    # numpy array for experimental history. This is just the prefix
    history_prefix_cols = ['num_sims', 'round', 'experiment', 'horizon']
    history_prefix_arr = np.zeros((num_sims * num_round * num_exp, len(history_prefix_cols)))
    history = None  # actual experimental history

    def ground_truth_query(ground_truth, to_query):
        """
        helper function to query ground truth dataset for yield result

        Parameters
        ----------
        ground_truth: dataframe with all components + yield
        to_query: list of dictionaries of component name/values to be queried

        Returns
        -------
        a list of yields after query

        """

        # adapted from Scope.query()
        names = set(list(ground_truth.columns))
        names.remove('yield')  # remove yield from column labels, then match
        component_names = sorted(to_query[0])  # only need one list of component names from dict; they are all the same

        ground_truth = ground_truth.sort_index(axis=1)  # important to sort here with column name also
        ground_truth_np = ground_truth.to_numpy(copy=False)
        ys = []
        for d in to_query:
            assert ground_truth.shape[1] - 1 == len(d.items()), f'Missing reaction components when querying {d}'
            assert names == set(d.keys()), f'labels not fully matched for {d}, cannot query'
            values = [d[c] for c in component_names]
            y = ground_truth[np.equal.outer(ground_truth_np, values).any(axis=1).all(axis=1)]['yield']
            if y.empty or np.isnan(list(y)[0]):
                print(f'something went wrong for {d}, no results found')
            else:
                assert len(list(y)) == 1, f'multiple result exist for query {d}'
                ys.append(list(y)[0])
        return ys

    # initialize scope object
    scope = Scope()
    scope.build_scope(scope_dict)
    scope.build_arms(arms_dict)

    # write arm dictionary into file
    # example: {<arm_index1>: <arm_name1>, <arm_index2>: <arm_name2>}
    d = dict(zip(np.arange(len(scope.arms)), scope.arms))
    with open(f'{dir}arms.pkl', 'wb') as f:
        pickle.dump(d, f)

    # simulation starts
    for sim in tqdm(range(num_sims), desc='simulations'):

        # reset scope and algo; arm settings are kept
        scope.reset()
        algo.reset(len(scope.arms))
        cumulative_reward = 0

        for r in tqdm(range(num_round), leave=False, desc='rounds'):
            chosen_arm = algo.select_next_arm()  # choose an arm
            proposed_experiments = scope.propose_experiment(chosen_arm, num_exp=num_exp, mode=propose_mode)  # scope proposes experiment
            if proposed_experiments is None:
                threshold = 0
                print(
                    f'[simulation {sim}, round {r}]: no experiments available for arm {chosen_arm}: {scope.arms[chosen_arm]}. Trying to find new experiments')
                while proposed_experiments is None:
                    if threshold > THRESHOLD:
                        print(f'[simulation {sim}, round {r}]: no experiments available for arm {chosen_arm}: {scope.arms[chosen_arm]} after '
                              f'{THRESHOLD} attempts; it might be the best arm')
                        break
                    algo.update(chosen_arm, algo.emp_means[chosen_arm])
                    new_chosen_arm = algo.select_next_arm()
                    if new_chosen_arm == chosen_arm:
                        threshold = threshold + 1
                        continue
                    else:
                        proposed_experiments = scope.propose_experiment(new_chosen_arm, num_exp=num_exp, mode=propose_mode)
                        # TODO: what if new_chosen_arm also does not have experiments available
            if proposed_experiments is not None:
                to_query = proposed_experiments[scope_dict.keys()].to_dict('records')  # generate a list of dicts to query
                rewards = ground_truth_query(ground_truth, to_query)  # ground truth returns all yields
                proposed_experiments['yield'] = rewards  # mimic user behavior and fill proposed experiments with yield
                history = pd.concat([history, proposed_experiments], ignore_index=True)
            else:
                pass  # TODO: how to deal with logging and history when no exp is available

            for ii in range(len(rewards)):  # update cumulative_reward, scope, algo and log results
                cumulative_reward = cumulative_reward + rewards[ii]
                scope.update_with_index(scope.current_experiment_index[ii], rewards[ii])
                algo.update(scope.current_arms[ii], rewards[ii])
                log_arr[sim * num_round * num_exp + r*num_exp+ii, :] = [sim, r, ii, r*num_exp+ii, chosen_arm, rewards[ii], cumulative_reward]
                history_prefix_arr[sim * num_round * num_exp + r*num_exp+ii, :] = [sim, r, ii, r*num_exp+ii]
            scope.predict()  # update prediction models

    log_df = pd.DataFrame(log_arr, columns=log_cols)
    history_df = pd.concat([pd.DataFrame(history_prefix_arr, columns=history_prefix_cols), history], axis=1).drop(columns=['prediction'])

    history_df.to_csv(f'{dir}history.csv')
    log_df.to_csv(f'{dir}log.csv')

    return


if __name__ == '__main__':

    # # scope
    # x = {'component_a': ['a1', 'a2', 'a3'],
    #      'component_b': ['b1', 'b2'],
    #      'component_c': ['c1', 'c2', 'c3', 'c4']}
    #
    # y = {'component_a': ['a1', 'a3'],
    #      'component_b': ['b1', 'b2']}
    #
    # algo = algos_regret.EpsilonGreedy(4, 0.5)
    # #propose_initial_experiments(x, y, algo, num_exp=1, propose_mode='random_highest')
    # update_and_propose(num_exp=1, propose_mode='random_highest')
    # with open(f'./test/scope.pkl', 'rb') as f:
    #     scope = pickle.load(f)  # load scope object
    # print(scope.data)

    # fetch ground truth data
    ground_truth = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand-1.csv')

    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    ground_truth = ground_truth[['ligand_name',
                                 'electrophile_id',
                                 'nucleophile_id',
                                 'yield']]
    ligands = ground_truth['ligand_name'].unique()
    elecs = ground_truth['electrophile_id'].unique()
    nucs = ground_truth['nucleophile_id'].unique()

    # build dictionary for acquisition
    scope_dict = {'ligand_name': ligands,
                  'electrophile_id': elecs,
                  'nucleophile_id': nucs}
    arms_dict = {'ligand_name': ligands}

    algo = algos_regret.ThompsonSamplingGaussianFixedVar(len(ligands))
    num_sims = 400
    num_round = 100
    num_exp = 1
    dir_name = f'./dataset_logs/aryl-scope-ligand-1/TSGaussian-{num_sims}s-{num_round}r-{num_exp}e/'

    p = pathlib.Path(dir_name)
    p.mkdir(parents=True)

    simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo,
                                dir=dir_name, num_sims=num_sims,
                                num_round=num_round, num_exp=num_exp, propose_mode='random')

