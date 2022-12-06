import itertools
import pandas as pd
import numpy as np


class Scope:

    def __init__(self):
        df = None
        return

    def scope_builder(self, d):
        """
        build a dataframe with reaction components populated, except yield

        Parameters
        ----------
        d: dictionary of reaction components
        e.g., d = {
            'component_a': ['a1', 'a2', 'a3'],
            'component_b': ['b1', 'b2'],
            'component_c': ['c1', 'c2', 'c3', 'c4']
        }

        Returns
        -------
        pd.DataFrame

        """
        component_names = sorted(d)
        combinations = itertools.product(*(d[c] for c in component_names))
        df = pd.DataFrame(combinations, columns=component_names)
        df['yield'] = np.nan

        return df

def scope_builder(d):
    """
    build a dataframe with reaction components populated, except yield

    Parameters
    ----------
    d: dictionary of reaction components
    e.g., d = {
        'component_a': ['a1', 'a2', 'a3'],
        'component_b': ['b1', 'b2'],
        'component_c': ['c1', 'c2', 'c3', 'c4']
    }

    Returns
    -------
    pd.DataFrame

    """
    component_names = sorted(d)
    combinations = itertools.product(*(d[c] for c in component_names))
    df = pd.DataFrame(combinations, columns=component_names)
    df['yield'] = np.nan

    return df


def scope_query(d, scope):
    """
    Queries the reaction scope for yield with a dictionary

    Parameters
    ----------
    d: dictionary of reaction components and values to be queried for yield
    e.g., d = {
        'component_a': 'a2',
        'component_b': 'b1',
        'component_c': 'c3',
    }

    scope: dataframe of entire reaction scope built by scope_builder()

    Returns
    -------

    """

    assert scope.shape[1] - 1 == len(d.items()), 'Missing reaction components when querying data'

    component_names = sorted(d)  # sort query dictionary by component name
    scope = scope.sort_index(axis=1)  # sort scope data by component name (column)
    values = [d[c] for c in component_names]  # get values for each component

    # can directly match with a list of value, since both query dict and dataframe are sorted
    y = scope[np.equal.outer(scope.to_numpy(copy=False), values).any(axis=1).all(axis=1)]['yield']
    if y.empty:
        print(f'No result from querying {d}')
        return np.nan  # no such result exists in the scope
    else:
        assert len(list(y)) == 1, f'multiple result exist for query {d}'
        return list(y)[0]


def scope_update(d, y, scope):

    return None



if __name__ == '__main__':

    x = {'component_b': ['b1', 'b2'],
        'component_a': ['a1', 'a2', 'a3'],
         'component_c': ['c1', 'c2', 'c3', 'c4']
    }
    df = scope_builder(x)
    fake_ys = np.arange(df.shape[0])
    df['yield'] = fake_ys

    print(df)

    d = {
        'component_b': 'b1',
        'component_a': 'a3',
    }

    print(
        scope_query(d, df)
    )


