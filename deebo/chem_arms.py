import itertools
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from algos_regret import EpsilonGreedy


class Scope:

    def __init__(self):
        self.data_dic = None  # store a copy of the dictionary that's used to build scope
        self.data = None  # dataframe that holds all experiment and result
        self.predictions = None  # predictions from regression model
        self.pre_accuray = None  # prediction accuracy
        self.arms = None  # a list of arms, e.g., [('a2', 'c1'), ('a2', 'c3'), ('a3', 'c1'), ('a3', 'c3')]
        self.arm_labels = None  # label names. e.g., ['component_a', 'component_b']
        return

    def __str__(self):
        return self.data_dic

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

    def update(self, d):
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

    def clear_arms(self):
        self.arms = None

    def propose_experiment(self, arm_index, mode='random'):
        """
        Propose an experiment for a specified arm

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
            return candidates.sample(1)
        else:
            return


def propose_initial_experiments(scope_dict, arms_dict, algo):

    exps = Scope()
    exps.build_scope(scope_dict)
    exps.build_arms(arms_dict)

    log = pd.DataFrame(columns=['horizon', 'chosen_arm', 'reward', 'cumulative_reward'])
    chosen_arm = algo.select_next_arm()
    experiments = exps.propose_experiment(chosen_arm)
    log.loc[len(log.index)] = [0, chosen_arm, np.nan, 0]

    folder = './test/'
    experiments.to_csv(f'{folder}proposed_expriments.csv')
    log.to_csv(f'{folder}log.csv')
    with open(f'{folder}algo.pkl', 'wb') as f:
        pickle.dump(algo, f)
    with open(f'{folder}scope.pkl', 'wb') as f:
        pickle.dump(exps, f)

    return


# TODO
def update_and_propose(algo, arms, horizon):
    cols = ['horizon', 'chosen_arm', 'reward', 'cumulative_reward']
    ar = np.zeros((horizon, len(cols)))

    algo.reset(len(arms))
    cumulative_reward = 0

    for t in range(horizon):
        chosen_arm = algo.select_next_arm()  # algorithm selects an arm
        reward = arms[chosen_arm].draw()  # chosen arm returns reward
        cumulative_reward = cumulative_reward + reward  # calculate cumulative reward over time horizon
        algo.update(chosen_arm, reward)  # algorithm updates chosen arm with reward
        ar[sim*horizon+t, :] = [sim, t, chosen_arm, reward, cumulative_reward]  # logs info

    df = pd.DataFrame(ar, columns=cols)
    return None


if __name__ == '__main__':

    # build scope
    x = {'component_b': ['b1', 'b2'],
        'component_a': ['a1', 'a2', 'a3'],
         'component_c': ['c1', 'c2', 'c3', 'c4']
    }
    y = {'component_b': ['b1', 'b2'],
         'component_a': ['a1', 'a3']}

    algo = EpsilonGreedy(4, 0.1)
    propose_initial_experiments(x, y, algo)




