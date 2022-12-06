import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Scope:

    def __init__(self):
        self.data = None  # dataframe that holds all
        self.predictions = None  # predictions from regression model
        self.pre_accuray = None  # prediction accuracy
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
        if self.data:
            exit('scope already exists, cannot build')

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


if __name__ == '__main__':

    x = {'component_b': ['b1', 'b2'],
        'component_a': ['a1', 'a2', 'a3'],
         'component_c': ['c1', 'c2', 'c3', 'c4']
    }
    s = Scope()
    s.build_scope(x)
    xx = {
        'component_b': 'b1',
        'component_a': 'a3',
        'yield': 11,
        'component_c': 'c1',
    }
    s.update(xx)

    xx = {
        'component_b': 'b2',
        'component_a': 'a2',
        'yield': 22,
        'component_c': 'c1',
    }
    s.update(xx)

    s.predict()
    print(s.pre_accuray)





