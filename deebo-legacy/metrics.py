# calculate the ground truth for a dataset

import pandas as pd
import numpy as np


# check dataframe, ensure there are only components and one column is numerical yield
# maybe not, pandas defaults all types to object
def _check_dataframe(df):
    assert 'yield' in df.columns, 'dataframe does not have "yield" column'
    assert 'index' not in df.columns, 'dataframe has "index" column'
    # y = df.pop('yield').to_list()
    # assert all(isinstance(x, float) for x in y), 'yield has non-numerical values'
    return None


def evaluate(ground_truth_data, query_conditions, metric='75%'):
    """
    find ground truth data with query conditions and metrics

    :param ground_truth_data:
             condition1 condition2 ... conditionN yield
        1     xx         xx       ...   xx        xx
        2     xx         xx       ...   xx        xx
    :param query_conditions:
            condition1 condition2
        1      xx          xx
        2      xx          xx
    :param metric: name of the metrics for evaluation
    :return: target label and metric, best condition combinations according to metric, all conditions and their performance
    :rtype: tuple, tuple, pandas.Series
    """

    df1 = ground_truth_data
    df2 = query_conditions
    query_rows = df2.index
    query_columns = df2.columns

    _check_dataframe(df1)

    df1 = df1.merge(df2)
    gb = df1.groupby(by=list(query_columns.values))
    stats = gb.describe()
    #stats = ['mean', 'median', 'max', 'size', 'sem', 'std', 'quantile', ]

    in_describe = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    other_metrics = {''}  #TODO: other metrics (SEM, counts,

    #print(stats.loc[:, ('yield', metric)].name)
    if metric in in_describe:
        return stats.loc[:, ('yield', metric)].name, stats.loc[:, ('yield', metric)].idxmax(), stats.loc[:, ('yield', metric)]
    else:
        pass


def _basic_test():
    data = pd.DataFrame(np.array([['A1', 'B1', 'C1'],
                                  ['A2', 'B1', 'C1'],
                                  ['A1', 'B2', 'C1'],
                                  ['A2', 'B2', 'C1'],
                                  ['A1', 'B1', 'C2'],
                                  ['A2', 'B1', 'C2'],
                                  ['A1', 'B2', 'C2'],
                                  ['A2', 'B2', 'C2']
                                  ]), columns=['As', 'Bs', 'Cs'])
    data['yield'] = np.linspace(0, 100, 8)
    query_conditions = pd.DataFrame(np.array([['A1', 'B2'], ['A2', 'B2']]), columns=['As', 'Bs'])
    a, b = evaluate(data, query_conditions)
    assert a == ('A2', 'B2'), 'did not pass basic test'
    query_conditions = pd.DataFrame(np.array([['A1', 'B2', 'C1'], ['A2', 'B2', 'C2']]), columns=['As', 'Bs', 'Cs'])
    a, b = evaluate(data, query_conditions)
    assert a == ('A2', 'B2', 'C2'), 'did not pass basic test'
    return None

if __name__ == '__main__':
    _basic_test()