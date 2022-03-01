# arylation scope dataset, chemist baseline
#
# From 64 combinations, randomly pick 1 and find the best ligand out of 24
# Then apply the same ligand to all 64 combinations, calculate percentage of cases where this ligand is the best ligand
# Calculate the average percentage for all 64 combinations
#
# Notes:
# - 4E gives no yield for all ligands, so I'm only simulating through 63 combinations

import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from collections import Counter
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, DataStructs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import interpolate


# df = pd.read_csv('../data/arylation/scope_ligand.csv')
# df = df[['ligand', 'electrophile_pci_name', 'nucleophile_pci_name', 'yield']]
# df['combinations'] = df['electrophile_pci_name'].astype(str) + df['nucleophile_pci_name'].astype(str)
# df = df[['ligand', 'combinations', 'yield']]
#
# idx = df.groupby(['combinations'])['yield'].transform(max) == df['yield']
# df = df[idx]
# df = df.loc[df['combinations'] != '4E']
#
# counter = Counter(list(df['ligand']))
# df['percentage_all_combinations'] = df['ligand'].apply(lambda x: counter[x] / 64)
#
# print(
#     df.to_string()
#       )

df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
df = df[['ligand_name', 'electrophile_id', 'nucleophile_id', 'yield']]
ligands = list(df['ligand_name'].unique())

# plot all results. 4x6 for ligands, and each ligand is represented by a 8x8 block, overall 32x48
l = []
df = df.sort_values(by=['ligand_name', 'electrophile_id', 'nucleophile_id'])
ligand_names = list(df['ligand_name'].unique())
nuc_names = list(df['nucleophile_id'].unique())
elec_names = list(df['electrophile_id'].unique())


def plot_all_results():  # heatmap for all results, grouped by ligand

    for ligand in ligands:
        tempdf = df.loc[df['ligand_name'] == ligand]
        tempdf = tempdf.drop(['ligand_name'], axis=1)
        a = np.array(tempdf.groupby(['electrophile_id'], sort=True)['yield'].apply(list).to_list())
        # each row is a electrophile, each column is a nucleophile
        l.append(a)

    a1 = np.hstack(l[0:6])
    a2 = np.hstack(l[6:12])
    a3 = np.hstack(l[12:18])
    a4 = np.hstack(l[18:24])
    a = np.vstack([a1, a2, a3, a4])

    fig, ax = plt.subplots()
    im = ax.imshow(a, cmap='inferno')
    text_kwargs = dict(ha='center', va='center', fontsize=12, color='white')
    ii = 0
    for i in range(4):
        for j in range(6):
            ax.add_patch(Rectangle((8*j-0.5, 8*i-0.5), 8, 8, fill=False, edgecolor='white', lw=2))
            plt.text(8*j+3.5, 8*i+3.5, ligand_names[ii], **text_kwargs)
            ii = ii+1
    plt.axis('off')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('yield (%)', rotation=270)
    plt.show()
    return None


def plot_one_ligand_result():  # heatmap for one ligand, with numerical yield

    for ligand in ligands:
        tempdf = df.loc[df['ligand_name'] == ligand]
        tempdf = tempdf.drop(['ligand_name'], axis=1)
        a = np.array(tempdf.groupby(['electrophile_id'], sort=True)['yield'].apply(list).to_list())
        # each row is a electrophile, each column is a nucleophile
        l.append(a)

    a1 = np.hstack(l[0:6])
    a2 = np.hstack(l[6:12])
    a3 = np.hstack(l[12:18])
    a4 = np.hstack(l[18:24])
    a = np.vstack([a1, a2, a3, a4])

    p = a[8:16, 0:8]
    fig, ax = plt.subplots()
    im = ax.imshow(p, cmap='inferno', vmin=min(df['yield']), vmax=max(df['yield']))
    text_kwargs = dict(ha='center', va='center',fontsize=16, color='white')
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, p[i, j], ha="center", va="center", color="w")
    ax.set_xticks(np.arange(len(nuc_names)))
    ax.set_xticklabels(nuc_names)
    ax.set_xlabel('nucleophile')
    ax.set_ylabel('electrophile')
    ax.set_yticks(np.arange(len(elec_names)))
    ax.set_yticklabels(elec_names)
    plt.show()
    return None


def plot_best_ligand_with_diff_metric():  # 6 bar plots, each with top 5 ligands, and their performance wrt metric

    stats = df.groupby(by=['ligand_name']).describe()
    twentyfive = stats.loc[:, ('yield', '25%')].nlargest(5)  # 1st quantile top 5
    median = stats.loc[:, ('yield', '50%')].nlargest(5)  # 2nd quantile
    seventyfive = stats.loc[:, ('yield', '75%')].nlargest(5)  # 3rd quantile
    mean = stats.loc[:, ('yield', 'mean')].nlargest(5)  # average

    overtwenty = df.loc[df['yield'] > 20].groupby(by='ligand_name').size().nlargest(5)  # top 5, over 20%, count
    overeighty = df.loc[df['yield'] > 80].groupby(by='ligand_name').size().nlargest(5)  # top 5, over 80%, count

    # make color dictionary, one color for one ligand
    all_top_ligands = []
    for li in [twentyfive, median, seventyfive, mean, overtwenty, overeighty]:
        all_top_ligands = all_top_ligands + list(li.index)
    all_top_ligands = list(set(all_top_ligands))
    colors = {}
    colormap = plt.cm.tab10.colors
    for i in range(len(all_top_ligands)):
        colors[all_top_ligands[i]] = colormap[i]

    def get_colors(ll):
        out = []
        for l in ll:
            out.append(colors[l])
        return out

    def trim(ll):  # trim the long ligand names
        return [s[:10] for s in ll]

    figsize = (10,5)
    kwargs = {'aa': True, 'width': 0.5}
    plt.rcParams['savefig.dpi'] = 300
    figs, axs = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    axs[0, 0].bar(trim(list(twentyfive.index)), list(twentyfive.values), color=get_colors(list(twentyfive.index)), **kwargs)
    axs[0, 0].set_title('1st quantile (Q1)')
    axs[0, 0].set_ylabel('yield (%)')

    axs[0, 1].bar(trim(list(median.index)), list(median.values), color=get_colors(list(median.index)), **kwargs)
    axs[0, 1].set_title('median')

    axs[1, 0].bar(trim(list(seventyfive.index)), list(seventyfive.values), color=get_colors(list(seventyfive.index)), **kwargs)
    axs[1, 0].set_title('3rd quantile (Q3)')
    axs[1, 0].set_ylabel('yield (%)')

    axs[1, 1].bar(trim(list(mean.index)), list(mean.values), color=get_colors(list(mean.index)), **kwargs)
    axs[1, 1].set_title('average')

    axs[2, 0].bar(trim(list(overtwenty.index)), list(overtwenty.values), color=get_colors(list(overtwenty.index)), **kwargs)
    axs[2, 0].set_title('yield >20%')
    axs[2, 0].set_ylabel('count')

    axs[2, 1].bar(trim(list(overeighty.index)), list(overeighty.values), color=get_colors(list(overeighty.index)), **kwargs)
    axs[2, 1].set_title('yield >80%')

    plt.show()


def plot_results_with_model_substrates():  # a heatmap, with each substrate pair as model system, best ligand is identified
    fd = df.copy()
    fd['combo'] = fd['electrophile_id'].astype(str) + fd['nucleophile_id'].astype(str)
    #fd = fd.sort_values(by=['combo', 'ligand_name'])
    max = fd.loc[fd.groupby(by=['combo'])['yield'].idxmax()]
    #print(max.loc[max['plot']!=0]['ligand_name'].value_counts())

    def f(x):  # to assign colors
        if x == 'CgMe-PPh':
            return 1
        elif x == 'tBPh-CPhos':
            return 2
        elif x == 'Cy-BippyPhos':
            return 3
        elif x == 'Et-PhenCar-Phos':
            return 4
        elif x == 'PPh3':
            return 5
        else:
            return 6

    max['valid'] = df['yield'].apply(lambda x: 0 if x<75 else 1)  # 0 for plotting, if highest yield < 75%
    max['plot'] = df['ligand_name'].apply(f)
    max['plot'] = max['plot']*max['valid']
    max = max.pivot(index='nucleophile_id', columns='electrophile_id', values='plot')

    fig, ax = plt.subplots()
    im = ax.imshow(max, cmap='turbo')

    # grid line
    for i in range(8):
        for j in range(8):
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='white', lw=1))

    ax.set_xticks(np.arange(8), labels=list(max.columns))
    ax.set_yticks(np.arange(8), labels=list(max.index))
    ax.set_xlabel('aryl bromide')
    ax.set_ylabel('imidazole')
    values = list(np.arange(7))
    colors = [im.cmap(im.norm(value)) for value in values]
    ligand_color = ['Not optimized (<75%)', 'CgMe-PPh', 'tBPh-CPhos', 'Cy-BippyPhos', 'Et-PhenCar-Phos', 'PPh3', 'other ligands']
    patches = [mpatches.Patch(color=colors[i], label=ligand_color[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.spines['top'].set_visible(False)  # remove boundaries
    ax.spines['right'].set_visible(False)

    plt.rcParams['savefig.dpi'] = 300
    plt.show()

# random sampling baseline, for each ligand, sample n experiments, plot the top 5 ligands as bar plots
# parameters: identify the ligand of interest, and how many experiments per ligand
def plot_simulations_random_sampling(ligand='Cy-BippyPhos', n_exp_per_ligand=1):
    n_simulations = 10000
    best = []
    gb = df.groupby(by=['ligand_name'])
    for i in tqdm(range(n_simulations)):
        sample = gb.sample(n_exp_per_ligand).groupby('ligand_name').mean()
        best.append(sample['yield'].idxmax())

    c = Counter(best).most_common(6)  # outputs a list of tuples
    labels, values = zip(*c)
    percentage = np.array(values)/n_simulations*100
    percentage = [str(round(p, 1)) + '%' for p in percentage]

    fig, ax = plt.subplots()

    index = labels.index(ligand)  # find Cy-BippyPhos and assign a diff color
    colors = ['#1f77b4']*len(labels)
    colors[index] = '#ff7f0e'

    b = ax.bar(np.arange(len(labels)), values, color=colors, width=0.5)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel('ligand')
    ax.set_ylabel('N times identified as optimal')
    ax.tick_params(axis='x', labelrotation=25)
    ax.bar_label(b, percentage, padding=3)

    ax.spines['top'].set_visible(False)  # remove boundaries
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 300
    plt.show()


def plot_simulations_random_sampling_all():
    # data obtained from running simulations on colab (plot.ipynb)
    # same random sampling, except sample sizes vary from 1 to 64
    # count the number of times Cy-BippyPhos is the best ligand (out of 10000) simulations
    count = [944, 1201, 1388, 1551, 1703, 1760, 1881, 2013, 2113, 2271, 2393, 2369,
             2557, 2523, 2689, 2872, 2887, 2972, 3094, 3111, 3211, 3218, 3241, 3385,
             3480, 3537, 3597, 3707, 3853, 3829, 3968, 3995, 4103, 4235, 4318, 4320,
             4498, 4627, 4696, 4820, 4788, 4934, 5143, 5243, 5368, 5458, 5704, 5890,
             6009, 6197, 6412, 6542, 6754, 7014, 7391, 7632, 7968, 8357, 8664, 9062,
             9495, 9872, 10000, 10000]
    probs = [c/10000*100 for c in count]
    n_exps = [(i+1)*24 for i in np.arange(64)]

    fig, ax = plt.subplots()

    ax.plot([n_exps[0], n_exps[-1]], [probs[0], probs[-1]], c='lightgray', ls='--', zorder=1)  # straight line to demonstrate non-linear
    ax.scatter(n_exps[-1], probs[-1], c='#ff7f0e', s=100, zorder=5)  # full HTE
    ax.scatter(0, 1/24*100, c='#2ca02c', s=100, zorder=2)  # random guess with zero experiments
    ax.scatter(n_exps[:-1], probs[:-1], s=25, zorder=3)  # all random sampling points

    # interpolate with spline
    sample_position = [0, 10, 20, 30, 40, 50, 55, 58, 60, 63]
    spline_x = [n_exps[i] for i in sample_position]
    spline_y = [probs[i] for i in sample_position]
    cs = interpolate.CubicSpline(spline_x, spline_y)
    ax.plot(n_exps, cs(n_exps), c='k', zorder=4)

    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('number of experiments (24 x sample_size)')
    ax.set_ylabel('probability of finding Cy-BippyPhos (%)')

    plt.rcParams['savefig.dpi'] = 300
    plt.show()
    return None


def _calculate_random_sampling_deprecated():

    # bad implementation, too complicated

    roi = 2  # row number for the row of interest

    a = [2,2,6]
    b = [1,3,5]
    c = [3,4,5]
    d = [2,3,7]
    X = np.vstack([a,b,c])

    (n_rows, n_cols) = X.shape
    indexes = np.repeat(np.arange(n_cols), n_rows)
    X = X.flatten()
    args = np.argsort(X)
    X = X[args]
    indexes = indexes[args]

    rows_to_search = np.delete(np.arange(n_rows), roi)  # exclude the row of interest

    indexes_list = list(indexes)
    firsts = [indexes_list.index(r) for r in rows_to_search]
    max = np.amax(firsts)
    roi_index = [index+max for index, element in enumerate(indexes_list[max:]) if element == roi]  # roi index that need to be examined

    def count_and_multiply(r):
        ctr = Counter(indexes_list[:r])
        ctr.pop(roi, 'None')
        combo = np.prod(list(ctr.values()))
        return combo

    combos = [count_and_multiply(r) for r in roi_index]
    combos_all_possible = np.sum(combos)

    print(combos_all_possible/pow(n_cols, n_rows))


def calculate_random_sampling():

    # random sampling 1
    #
    # Goal: pick a reaction componenet with specific value, and calculate the probability of this value being the best
    # in a random sampling situation (only sample 1)
    #
    # General idea: p = (n_situations_desired) / (n_overall_combinations)
    #
    # Algorithm: start with array with shape (n_components_to_evaluate, n_data_for_each_component)
    # for example: (n_ligands, n_experiemtns_for_each_ligand)
    # - Identify the row (reaction component with a specific value)
    # - Loop through each value of the row, and make any other values bigger than this value in array 0
    # - Count non-zero values for each row
    # - Multiply all values (minus the row we are considering) to get the number of situations where roi value is highest

    # # test array
    # X = np.array(np.linspace(0,23,24))
    # np.random.shuffle(X)
    # X = X.reshape(4,6)
    # print(X)

    for ligand in ligands:
        tempdf = df.loc[df['ligand_name'] == ligand]
        tempdf = tempdf.drop(['ligand_name'], axis=1)
        a = np.array(tempdf.groupby(['electrophile_id'], sort=True)['yield'].apply(list).to_list())
        # each row is a electrophile, each column is a nucleophile
        l.append(a.flatten())

    X = np.vstack(l)

    def count_and_multiply(n):
        a = X.copy()
        a[a > n] = -1  # This line will cause overcount in cases where there are duplicate values (rare for actual reaction yield)
        smaller = np.sum(a != -1, 1)
        smaller = np.delete(smaller, roi)  # need to delete the row of interest
        assert smaller.shape[0] == a.shape[0]-1
        smaller = np.divide(smaller, X.shape[1])  # overflow here if np.prod() or math.prod() directly! need to divide and make smaller
        prod = math.prod(smaller)
        return prod/X.shape[1]  # one extra division for the roi

    roi = ligands.index('Cy-BippyPhos')  # row number for the row of interest
    ll = X[roi, :]
    combos = [count_and_multiply(n) for n in ll]
    probability = np.sum(combos)  # probability = n of (roi value bigger than all other rows)/ total number of combinations
    print(probability)

    # # test on all ligands, should sum to 1
    # probs = []
    # for i in range(X.shape[0]):
    #     # roi = ligands.index(ligand)
    #     roi = i
    #     ll = X[roi, :]
    #     combos = [count_and_multiply(n) for n in ll]
    #     probability = np.sum(combos)  # probability = n of (roi value bigger than all other rows)/ total number of combinations
    #     probs.append(probability)
    # print(probs)
    # print(np.sum(probs))


def cluster_substrates(draw_product=1):
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
    df = df[['electrophile_id', 'electrophile_smiles', 'nucleophile_id', 'nucleophile_smiles', 'product_smiles']].drop_duplicates(ignore_index=True)
    df['product_id'] = df['electrophile_id'].astype(str) + df['nucleophile_id'].astype(str)

    def smiles_to_ecfp(x):
        mol = Chem.MolFromSmiles(x)
        if mol is not None:
            fp_obj = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True)
            fp = np.zeros((0,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp_obj, fp)
            return fp
        else:  # some smiles invalid for rdkit
            return None

    for c in ['nucleophile', 'electrophile', 'product']:
        cname = c+'_ecfp'
        sname = c+'_smiles'
        df[cname] = df[sname].apply(smiles_to_ecfp)
        assert df[cname].isnull().sum() == 0, '{} smiles did not convert to fingerprint successfully'.format(c)

    if draw_product:
        p = df[['product_ecfp', 'product_id']].drop_duplicates(subset=['product_id'])
        Xs = np.stack(p['product_ecfp'].values)
        pro_names = list(p['product_id'])
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(Xs)
        pc1 = reduced[:, 0]
        pc2 = reduced[:, 1]
        pc3 = reduced[:, 2]

        kmeans = KMeans(n_clusters=3).fit(Xs)
        labels = kmeans.labels_

        # # 3d
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(pc1, pc2, pc3)
        # plt.show()

        # 2d
        fig, ax = plt.subplots()
        ax.scatter(pc1, pc2, c=labels, edgecolors='k')
        plt.show()
    else:
        n = df[['nucleophile_ecfp', 'nucleophile_id']].drop_duplicates(subset=['nucleophile_id'])
        e = df[['electrophile_ecfp', 'electrophile_id']].drop_duplicates(subset=['electrophile_id'])
        Xn = np.stack(n['nucleophile_ecfp'].values)
        nuc_names = list(n['nucleophile_id'])
        Xe = np.stack(e['electrophile_ecfp'].values)
        elec_names = list(e['electrophile_id'])

        # kmeans = KMeans(n_clusters=1).fit(X)
        # labels = kmeans.labels_

        pca = PCA(n_components=3)
        Xn_reduced = pca.fit_transform(Xn)
        Xe_reduced = pca.fit_transform(Xe)

        es = Xe_reduced[:, 0]
        ns = Xn_reduced[:, 0]

        fig, ax = plt.subplots()

        for e in es:
            ax.scatter(ns, [e]*len(ns), edgecolors='k')

        ax.set_xticks(ns)
        ax.set_xticklabels(nuc_names)
        ax.set_xlabel('nucleophile')

        ax.set_yticks(es)
        ax.set_yticklabels(elec_names)
        ax.set_ylabel('electrophile')

        plt.show()


    exit()
    for i, txt in enumerate(nuc_names):
        plt.annotate(txt, (Xn_reduced[:, 0][i], 1.1))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    exit()
    x_min, x_max = X_reduced[:, 0].min() - 0.5, X_reduced[:, 0].max() + 0.5
    y_min, y_max = X_reduced[:, 1].min() - 0.5, X_reduced[:, 1].max() + 0.5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()



if __name__ == '__main__':
    calculate_random_sampling()