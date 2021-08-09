# cluster substrates
# testing clustering for substrates
# for testing only: deoxy, arylation, suzuki

# Note 08/08/21: clustering doesn't really work, maybe better for simple visualization and outlier detection


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from label import label_single_component, data_loader


# limited number of possible conditions, number of clusters can match number of conditions
def cluster_small_n(labeled_substrates, encoding='m2v', alg='kmeans'):

    # get a list of substrates, compute number of unique conditions
    substrates = labeled_substrates.pop('substrate_SMILES')
    combos = labeled_substrates[labeled_substrates.columns].astype(str).sum(axis=1)
    n_clusters = combos.nunique()

    # load encodings of smiles string
    if encoding == 'm2v':
        m2v = Path.cwd().parent / 'data/deoxy/substrate_mol2vec.csv'
        X = data_loader(m2v)
        X = X.drop(['names', 'SMILES'], axis=1)
    else:
        X = []

    # scale X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # initialize clustering
    if alg == 'kmeans':
        cluster_alg = KMeans(n_clusters, random_state=69).fit(X)
    else:
        cluster_alg = KMeans(n_clusters).fit(X)

    labels = cluster_alg.labels_

    # re-combine to make a dataframe with labels and cluster groups

    labeled_substrates['substrate_SMILES'] = list(substrates)
    labeled_substrates['cluster_labels'] = list(labels)

    print(labels)
    print(labeled_substrates[labeled_substrates['cluster_labels'] == 2])

    return None


# TODO: test
def plot():

    fp = Path.cwd().parent / 'data/deoxy/experiment_index.csv'
    sub = Path.cwd().parent / 'data/deoxy/substrate.csv'
    m2v = Path.cwd().parent / 'data/deoxy/substrate_mol2vec.csv'

    labeled = label_single_component(data_loader(fp), data_loader(sub), 'fluoride', mode='avg')
    y = labeled['best_fluoride_SMILES']
    X = data_loader(m2v)
    X = X.drop(['names', 'SMILES'], axis=1)

    pca = PCA(n_components=5)
    X_r = pca.fit_transform(X)

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'lightcoral', 'forestgreen']
    lw = 2

    target_values = ['O=S(C1=CC=C(C(F)(F)F)C=C1)(F)=O',
                     'none',
                     'FC(C(F)(S(=O)(F)=O)F)(F)C(F)(F)C(F)(F)F',
                     'O=S(C1=CC=C([N+]([O-])=O)C=C1)(F)=O',
                     'ClC1=CC=C(S(=O)(F)=O)C=C1',
                     'O=S(C1=CC=CC=N1)(F)=O'
                     ]
    target_names = ['3-CF3',
                    'none',
                    'PBSF',
                    '3-NO2',
                    '3-Cl',
                    'PyFluor']

    for color, i, target_name in zip(colors, target_values, target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')
    plt.show()


if __name__ == '__main__':
    fp = Path.cwd().parent / 'data/deoxy/experiment_index.csv'
    sub = Path.cwd().parent / 'data/deoxy/substrate.csv'
    cluster_small_n(
        label_single_component(data_loader(fp), data_loader(sub), 'fluoride', mode='avg'),
        alg='kmeans'
    )



