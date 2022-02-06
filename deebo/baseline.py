# establish some baselines of optimization

import pandas as pd

from metrics import evaluate


def analyze_dataset_structure(df):
    """

    :param df: dataframe to be analyzed
    :type df: pandas.DataFrame
    :return: dictionary of reaction components, dictionary of counts of reaction components
    :rtype: dict, dict
    """

    cols = list(df.columns)
    cols.remove('yield')

    # find components that have a 'names' field first
    has_name = []
    for c in cols:
        l = c.split('_')
        if l[-1] == 'name':
            has_name.append(c)

    # analyze reaction components in the dataset, and how many options for each component
    # name is preferred over smiles
    components = {}
    counts = {}
    for h in has_name:
        l = h.split('_')
        u = list(pd.unique(df[h]))
        components[l[0]] = u
        counts[l[0]] = len(u)

    for c in cols:
        l = c.split('_')
        if l[0] not in components:
            u = list(pd.unique(df[c]))
            components[l[0]] = u
            counts[l[0]] = len(u)

    return components, counts


def transform(df, name_prefered=True):
    pass


def single_component_baseline(df):
    pass


if __name__ == '__main__':

    prefix = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/'

    # datasets = [
    #     'aryl-scope-ligand.csv',
    #     'aryl-conditions.csv',
    #     'cn.csv',
    #     'deoxyf.csv',
    #     'suzuki.csv',
    #     'vbur.csv',
    #     ]

    # available data here
    # aryl-conditions.csv
    # aryl-scope-ligand.csv
    # cn.csv
    # deoxyf.csv
    # suzuki.csv
    # vbur.csv

    df = pd.read_csv(prefix + 'aryl-scope-ligand.csv')

    components, counts = analyze_dataset_structure(df)
    df = df[['ligand_name', 'electrophile_name', 'nucleophile_name', 'yield']]
    print(df)

