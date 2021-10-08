# logistic regression with binary data (partitioned at 20%)
# use LASSO as potential dimensionality reduction

# partition data (direct arylation scope set)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt


def partition(df, cutoff=20):

    df['success'] = df['yield'].apply(lambda x: x >= cutoff)
    df.drop(['yield'], axis=1, inplace=True)

    return df


# sanitize features
# input is feature set only (e.g. n ligands, x features), no objective
# scaler: standard or minmax or robust
def sanitize(df, var_threshold=1, corr_threshold=0.95, scaler='standard'):

    # drop columns with nan
    df = df.dropna(axis=1)

    # drop columns with non-numerical values
    df = df.select_dtypes(include=np.number)

    # drop columns with constant value (column variance=0)
    df = df.loc[:, ~np.isclose(0, df.var())]

    # drop columns with near-constant value (low variance)
    df = df.loc[:, df.var() > var_threshold]

    # drop columns that have high correlation >0.95
    # MAYBE: keep a list of correlations for interpretability
    del_cols = set()  # Set of all the names of deleted columns
    corr = df.corr().abs()
    for i in range(len(corr.columns)):
        for j in range(i):
            if (corr.iloc[i, j] >= corr_threshold) and (corr.columns[j] not in del_cols):
                colname = corr.columns[i]
                del_cols.add(colname)
                if colname in df.columns:
                    del df[colname]

    # scaler: standard or minmax or robust
    features = list(df.columns)
    if scaler == 'standard':
        sclr = StandardScaler()
    elif scaler == 'minmax':
        sclr = MinMaxScaler()
    elif scaler == 'robust':
        sclr = RobustScaler()
    else:
        print('\n requested scaler does not exist, using standard scaler. \n')
        sclr = StandardScaler()

    X = sclr.fit_transform(df.to_numpy())

    return X, features


# featurize ligand with kraken, one hot encode substrates
def featurize_kraken(df):
    
    lookup = pd.read_csv('../data/arylation/krakenID_lookup.csv')
    kraken = pd.read_csv('https://raw.githubusercontent.com/doyle-lab-ucla/ochem-data/main/kraken/kraken.csv')

    # preprocess kraken features with ligands first
    ligands = list(pd.unique(df['ligand_smiles']))
    IDs = lookup.loc[lookup['ligand_smiles'].isin(ligands)]
    IDs = list(IDs['kraken_id'])
    kraken = kraken.loc[kraken['ID'].isin(IDs)]  # selected ligands from kraken library

    ll = ['ID', 'ligand', 'can_smiles', 'smiles', 'smiles(dft_sheet)']
    identifiers = kraken[ll].copy()
    kraken.drop(ll, axis=1, inplace=True)

    kraken, features = sanitize(kraken, var_threshold=1, corr_threshold=0.9, scaler='standard')

    print(kraken.shape)
    
    return kraken  # change


df = pd.read_csv('../data/arylation/scope_ligand.csv')
df = df[['yield', 'ligand_smiles', 'electrophile_smiles', 'nucleophile_smiles']]  # dropped product_smiles

featurize_kraken(partition(df))