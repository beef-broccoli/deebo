# arylation scope dataset, chemist baseline
#
# From 64 combinations, randomly pick 1 and find the best ligand out of 24
# Then apply the same ligand to all 64 combinations, calculate percentage of cases where this ligand is the best ligand
# Calculate the average percentage for all 64 combinations
#
# Notes:
# - 4E gives no yield for all ligands, so I'm only simulating through 63 combinations


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from collections import Counter
from tqdm import tqdm

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

def plot_all_results():  # heatmap for all results, grouped by ligand

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


def plot_results_with_model_substrates():
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

    plt.rcParams['savefig.dpi'] = 300
    plt.show()


def plot_simulations_need_to_rename():
    n_simulations = 10000
    best = []
    gb = df.groupby(by=['ligand_name'])
    for i in tqdm(range(n_simulations)):
        sample = gb.apply(lambda x: x.sample(1))
        best.append(sample['yield'].idxmax()[0])

    c = Counter(best).most_common(6)  # outputs a list of tuples
    labels, values = zip(*c)
    percentage = np.array(values)/n_simulations*100
    percentage = [str(round(p, 1)) + '%' for p in percentage]

    fig, ax = plt.subplots()

    index = labels.index('Cy-BippyPhos')  # find Cy-BippyPhos and assign a diff color
    colors = ['#1f77b4']*len(labels)
    colors[index] = '#ff7f0e'

    b = ax.bar(np.arange(len(labels)), values, color=colors, width=0.5)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel('ligand')
    ax.set_ylabel('N times identified as optimal')
    ax.tick_params(axis='x', labelrotation=25)
    ax.bar_label(b, percentage, padding=3)

    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 300
    plt.show()


def model_substrate_baseline():
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
    df = df[['ligand_name', 'electrophile_id', 'nucleophile_id', 'yield']]
    df['combinations'] = df['electrophile_id'].astype(str) + df['nucleophile_id'].astype(str)
    df = df[['ligand_name', 'combinations', 'yield']]
    idx = df.groupby(['combinations'])['yield'].transform(max) == df['yield']
    df = df[idx]

    counter = Counter(list(df['ligand_name']))
    df['percentage_all_combinations'] = df['ligand_name'].apply(lambda x: counter[x] / 64)
    df = df.loc[df['yield'] > 80]
    print(df)

    return None


#plot_all_results()
plot_simulations_need_to_rename()


