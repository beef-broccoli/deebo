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
import itertools
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 600
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib as mpl
from collections import Counter
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, DataStructs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import interpolate
import yaml
import random

classic_blue_hex = '#0f4c81'
coral_essence_hex = '#f26b5b'
lime_punch_hex = '#c0d725'
pink_tint_hex = '#dbcbbc'
pirate_black_hex = '#373838'
monument_hex = '#84898c'
jasmine_green_hex = '#7EC845'
cornhusk_hex = '#F3D5AD'
peach_quartz_hex = '#f5b895'
stucco_hex = "#A58D7F"
baby_blue_hex = "#B5C7D3"
provence_hex = '#658DC6'

la_gold = '#FDB927'
la_purple = '#552583'


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
df['electrophile_id'] = df['electrophile_id'].apply(lambda x: x.lstrip('e')).astype('int')  # for sorting purposes, so 10 is not immediately after 1
df['nucleophile_id'] = df['nucleophile_id'].apply(lambda x: x.lstrip('n'))
ligands = list(df['ligand_name'].unique())

# plot all results. 4x6 for ligands, and each ligand is represented by a 8x8 block, overall 32x48
df = df.sort_values(by=['ligand_name', 'electrophile_id', 'nucleophile_id'])
ligand_names = list(df['ligand_name'].unique())
nuc_names = list(df['nucleophile_id'].unique())
elec_names = list(df['electrophile_id'].unique())


def plot_all_results(binary=0, cutoff=80):  # heatmap for all results, grouped by ligand
    l = []
    averages = []

    for ligand in ligand_names:
        tempdf = df.loc[df['ligand_name'] == ligand]
        tempdf = tempdf.drop(['ligand_name'], axis=1)
        a = np.array(tempdf.groupby(['electrophile_id'], sort=True)['yield'].apply(list).to_list())
        averages.append(np.average(a))
        # each row is a electrophile, each column is a nucleophile
        l.append(a)

    a1 = np.hstack(l[0:6])
    a2 = np.hstack(l[6:12])
    a3 = np.hstack(l[12:18])
    a4 = np.hstack(l[18:24])
    a = np.vstack([a1, a2, a3, a4])

    if binary:
        a = a>cutoff

    fig, ax = plt.subplots()
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='white')
    text_kwargs_fs9 = dict(ha='center', va='center', fontsize=9, color='white')
    text_kwargs_fs8 = dict(ha='center', va='center', fontsize=8, color='white')
    text_kwargs_fs7 = dict(ha='center', va='center', fontsize=7, color='white')
    if binary:
        im = ax.imshow(a, cmap='inferno', vmin=0, vmax=2)
    else:
        im = ax.imshow(a, cmap='inferno', vmin=0, vmax=110)
    ii = 0
    for i in range(4):
        for j in range(6):
            ax.add_patch(Rectangle((8*j-0.5, 8*i-0.5), 8, 8, fill=False, edgecolor='white', lw=2))
            if len(ligand_names[ii])<11:
                plt.text(8 * j + 3.5, 8 * i + 2.5, ligand_names[ii], **text_kwargs)
            elif len(ligand_names[ii])<13:
                plt.text(8 * j + 3.5, 8 * i + 2.5, ligand_names[ii], **text_kwargs_fs8)
            else:
                plt.text(8 * j + 3.5, 8 * i + 2.5, ligand_names[ii], **text_kwargs_fs7)
            plt.text(8 * j + 3.5, 8 * i + 4.5, str(round(averages[ii],2)), **text_kwargs)
            ii = ii+1

    ax.set_yticks(np.arange(8), labels=['1', '2', '3', '4', '5', '7', '9', '10'], fontsize=8)
    ax_t = ax.secondary_xaxis('top')
    ax_t.set_xticks(np.arange(8), labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I'], fontsize=8)
    ax.set_xticks([], [])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax_t.spines['top'].set_visible(False)

    if not binary:
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Yield (%)', rotation=270)
    plt.rcParams['savefig.dpi'] = 300
    plt.show()
    return None


# plot set 2 by products (64 products, 24 ligands)
# split 8 nucleophiles and 8 electrophiles
def plot_bar_box_substrates(df, plotwhat):

    df = df[['electrophile_id', 'nucleophile_id', 'yield']].copy()

    # create two dict for electrophiles and nucleophiles: letter/num->chemical name
    # elec = df[['electrophile_pci_name', 'electrophile']]
    # elec = elec.drop_duplicates(ignore_index=True)
    # nuc = df[['nucleophile_pci_name', 'nucleophile']]
    # nuc = nuc.drop_duplicates(ignore_index=True)
    # df.drop(['electrophile', 'nucleophile'], axis=1, inplace=True)  # elec labels, nuc labels, yield

    if plotwhat in {'electrophile', 'nucleophile'}:  # plot a box plot and a stacked bar chart

        name = plotwhat + '_id'
        yields = df[[name, 'yield']]
        # avg = yields.groupby([name]).mean()
        # names = yields[name].drop_duplicates().to_list()  # CAREFUL! names will get sorted
        # names = [str(name) for name in names]  # make sure labels are passed as string

        # box plot
        results = yields.groupby([name]).apply(lambda x: x.values[:, 1].tolist())
        names = [str(name) for name in results.index]
        _box_plot(list(results), names, plotwhat)  # from mpl docs, list is more efficient than np array

        # stacked bar
        bins = [-1, 1, 20, 40, 60, 80, 200]  # 100%+ yield exist
        categories = ['0%', '0-20%', '20%-40%', '40%-60%', '60%-80%', '80%-100%+']
        binned = yields.groupby([name, pd.cut(yields['yield'], bins)]).size().unstack()  # groupby yield bins and ligand
        names = [str(name) for name in binned.index]
        count = binned.values  # for each ligand, count number of yields in each yield bin
        _categorical_bar(names, count, categories, ylabel=plotwhat)  # I split these early for generalizabiliy, make sure they have the same ligand sequence

        plt.show()

    elif plotwhat == 'both':
        groups = df.groupby(['electrophile_id', 'nucleophile_id'])
        avgs = groups.mean().unstack()
        medians = groups.median().unstack()  # after unstacking row name is elec, col name is nuc
        elec_names = list(medians.index.values)  # might be sorted, get names this way
        nuc_names = list(list(zip(*medians.columns.values))[1])
        _heatmap(nuc_names, elec_names, np.array(medians.values),
                 title='Median yield', xlabel='nucleophile (imidazole)', ylabel='electrophile (aryl bromide)')
        _heatmap(nuc_names, elec_names, np.array(avgs.values),
                 title='Average yield', xlabel='nucleophile (imidazole)', ylabel='electrophile (aryl bromide)')
        plt.show()

    return None


def plot_bar_box_ligand(whichplot):

    fd = df.copy()
    yields = fd[['ligand_name', 'yield']]
    #avgs = fd.groupby(['ligand_name']).mean()

    if whichplot == 'boxplot':  # box plot #TODO
        yields = yields.groupby(['ligand_name']).apply(lambda x: x.values[:, 1].tolist())
        names = [str(name) for name in yields.index]
        _box_plot(list(yields), names, 'ligand_name')

    # TODO: not very necessary
    # elif whichplot == 'average':  # bar plot average yield
    #     print(avgs)
    #     plt.clf()
    #     ys = np.linspace()
    #     plt.barh(names, width=avgs.values)
    #     plt.show()

    elif whichplot == 'categories':  # 20% interval categories
        # try to bin the yields
        bins = [-1, 1, 20, 40, 60, 80, 200]  # 100%+ yield exist
        categories = ['0%', '0-20%', '20%-40%', '40%-60%', '60%-80%', '80%-100%+']
        binned = yields.groupby(['ligand_name', pd.cut(yields['yield'], bins)]).size().unstack()  # groupby yield bins and ligand
        ligands = list(binned.index)  # list of ligand names
        count = binned.values  # for each ligand, count number of yields in each yield bin
        _categorical_bar(ligands, count, categories)  # I split these early for generalizabiliy, make sure they have the same ligand sequence

    plt.rcParams['savefig.dpi'] = 300
    plt.show()
    return None


def plot_one_ligand_result():  # heatmap for one ligand, with numerical yield
    l = []

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


def plot_best_ligand_with_diff_metric(n_largest=5, preset_color_dict=None):  # 6 bar plots, each with top 5 ligands, and their performance wrt metric

    with open('colors.yml', 'r') as file:
        COLORS = yaml.safe_load(file)

    stats = df.groupby(by=['ligand_name']).describe()
    twentyfive = stats.loc[:, ('yield', '25%')].nlargest(n_largest)  # 1st quantile top 5
    median = stats.loc[:, ('yield', '50%')].nlargest(n_largest)  # 2nd quantile
    seventyfive = stats.loc[:, ('yield', '75%')].nlargest(n_largest)  # 3rd quantile
    mean = stats.loc[:, ('yield', 'mean')].nlargest(n_largest)  # average

    overtwenty = df.loc[df['yield'] > 20].groupby(by='ligand_name').size().nlargest(n_largest)  # top 5, over 20%, count
    overeighty = df.loc[df['yield'] > 80].groupby(by='ligand_name').size().nlargest(n_largest)  # top 5, over 80%, count

    # make color dictionary, one color for one ligand
    all_top_ligands = []
    for li in [twentyfive, median, seventyfive, mean, overtwenty, overeighty]:
        all_top_ligands = all_top_ligands + list(li.index)
    all_top_ligands = list(set(all_top_ligands))
    # colors = {}
    # colormap = plt.cm.tab10.colors
    # for i in range(len(all_top_ligands)):
    #     colors[all_top_ligands[i]] = colormap[i]

    color_list = [COLORS['coral_essence'], COLORS['cornhusk'], COLORS['stucco'], COLORS['peach_quartz'],
                  COLORS['baby_blue'], COLORS['monument'], COLORS['provence'], COLORS['pink_tint'],
                  COLORS['classic_blue'], COLORS['lime_punch'], COLORS['pirate_black'], COLORS['jasmine_green'],
                  COLORS['red_violet']]
    colors = {}

    if preset_color_dict is not None:  # provide an color dictionary {ligand_name: color}
        color_list_count = 0  # this index is used in case the supplied color dict does not contain the ligand that need to be plotted
        # then colors will be incrementally selected from color_list
        for i in range(len(all_top_ligands)):
            ligand = all_top_ligands[i]
            try:
                colors[ligand] = preset_color_dict[ligand]
            except KeyError:
                colors[ligand] = 'gray'
                color_list_count += 1
    else:
        if len(all_top_ligands) > len(color_list):
            raise RuntimeError('not enough colors for all top ligands. {0} colors, {1} ligands'.format(len(color_list),
                                                                                                       len(all_top_ligands)))
        for i in range(len(all_top_ligands)):
            colors[all_top_ligands[i]] = color_list[i]

    def get_colors(ll):  # for a list of names, get their color from overall color dict
        out = []
        for l in ll:
            out.append(colors[l])
        return out

    def trim(ll):  # trim the long ligand names
        return [s[:10] for s in ll]

    figsize = (10,6)
    kwargs = {'aa': True, 'width': 0.5}
    plt.rcParams['savefig.dpi'] = 300
    figs, axs = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    def ax_plot(ax_x, ax_y, df, title, y_label=None):
        x = trim(list(df.index))
        y = list(df.values)
        axs[ax_x, ax_y].bar(x, y, color=get_colors(list(df.index)), **kwargs)
        for i in range(len(x)):  # plot value
            axs[ax_x, ax_y].text(i, y[i]+0.5, round(y[i], 2), ha='center')
        axs[ax_x, ax_y].set_title(title)  # title
        if y_label:  # y label
            axs[ax_x, ax_y].set_ylabel(y_label)
        axs[ax_x, ax_y].set_ylim(top=axs[ax_x, ax_y].get_ylim()[1] + 5)  # adjust ylim top so value text fits

    ax_plot(0, 0, twentyfive, title='1st quantile (Q1)', y_label='yield (%)')
    ax_plot(0, 1, median, title='median')
    ax_plot(1, 0, seventyfive, title='3rd quantile (Q3)', y_label='yield (%)')
    ax_plot(1, 1, mean, title='average')
    ax_plot(2, 0, overtwenty, title='yield >20%', y_label='count')
    ax_plot(2, 1, overeighty, title='yield >80%')

    plt.show()


def plot_results_with_model_substrates(cutoff=50, select=False):
    """
    a heatmap, with each substrate pair as model system, highest yielding ligand is identified

    Parameters
    ----------
    cutoff: yield cutoff. If the highest yielding ligand gives a yield lower than cutoff, it's considered not optimized
    select: plot only selected few ligands for better visualization

    Returns
    -------

    """
    fd = df.copy()
    fd['combo'] = fd['electrophile_id'].astype('str') + fd['nucleophile_id'].astype('str')
    #fd = fd.sort_values(by=['combo', 'ligand_name'])
    max = fd.loc[fd.groupby(by=['combo'])['yield'].idxmax()]
    #print(list(max['ligand_name'].unique()))
    #print(max.loc[max['plot']!=0]['ligand_name'].value_counts())

    def color_select(x):  # to assign colors
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

    # new way to assign colors for all ligands that give above cutoff yields
    ligands_to_color = max.loc[max['yield']>cutoff]['ligand_name'].unique()

    def color(x):
        vals = np.arange(len(ligands_to_color)) + 1
        d = dict(zip(ligands_to_color, vals))
        if x not in d:
            return 0
        else:
            return d[x]

    max['valid'] = df['yield'].apply(lambda x: 0 if x<cutoff else 1)  # 0 for plotting, if highest yield < 75%
    if select:
        max['plot'] = df['ligand_name'].apply(color_select)
    else:
        max['plot'] = df['ligand_name'].apply(color)
    max['plot'] = max['plot']*max['valid']
    max = max.pivot(index='nucleophile_id', columns='electrophile_id', values='plot')
    max[max==0] = -1  # set all zeros to -1, this helps with plotting with a cmap, i can set the color for -1

    fig, ax = plt.subplots()
    cmap = mpl.cm.get_cmap('Paired').copy()
    cmap.set_under('k')
    im = ax.imshow(max, cmap=cmap, vmin=1)

    # grid line
    for i in range(8):
        for j in range(8):
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='white', lw=1))

    ax.set_xticks(np.arange(8), labels=list(max.columns))
    ax.set_yticks(np.arange(8), labels=list(max.index))
    ax.set_xlabel('electrophile (aryl bromide)')
    ax.set_ylabel('nucleophile (imidazole)')
    if select:
        values = list(np.arange(7))
        ligand_color = [f'Not optimized (<{cutoff}%)', 'CgMe-PPh', 'tBPh-CPhos', 'Cy-BippyPhos', 'Et-PhenCar-Phos', 'PPh3', 'other ligands']
    else:
        values = list(np.arange(len(ligands_to_color)+1))
        ligand_color = list(ligands_to_color)
        ligand_color.insert(0, f'Not optimized (<{cutoff}%)')
    colors = [im.cmap(im.norm(value)) for value in values]

    patches = [mpatches.Patch(color=colors[i], label=ligand_color[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)

    ax.spines['top'].set_visible(False)  # remove boundaries
    ax.spines['right'].set_visible(False)

    plt.rcParams['savefig.dpi'] = 600
    plt.show()
    return None


def plot_results_with_model_substrates_color_match_publication(cutoff=75, preset_color_dict=None):
    """
    a heatmap, with each substrate pair as model system, highest yielding ligand is identified

    Parameters
    ----------
    cutoff: yield cutoff. If the highest yielding ligand gives a yield lower than cutoff, it's considered not optimized
    select: plot only selected few ligands for better visualization

    Returns
    -------

    """
    fd = df.copy()
    fd['combo'] = fd['electrophile_id'].astype('str') + fd['nucleophile_id'].astype('str')
    #fd = fd.sort_values(by=['combo', 'ligand_name'])
    max = fd.loc[fd.groupby(by=['combo'])['yield'].idxmax()]
    #print(list(max['ligand_name'].unique()))
    #print(max.loc[max['plot']!=0]['ligand_name'].value_counts())

    # new way to assign colors for all ligands that give above cutoff yields
    ligands_to_color = max.loc[max['yield']>cutoff]['ligand_name'].unique()

    val_to_rgb = {}  # {value: rgb}
    def color(x):
        # x: ligand name
        vals = np.arange(len(ligands_to_color)) + 1
        d = dict(zip(ligands_to_color, vals))  # {name: value}
        if x not in d:
            return 0
        else:
            if preset_color_dict is not None:
                val_to_rgb[d[x]] = preset_color_dict[x]
            return d[x]

    max['valid'] = df['yield'].apply(lambda x: 0 if x<cutoff else 1)  # 0 for plotting, if highest yield < 75%
    max['plot'] = df['ligand_name'].apply(color)
    max['plot'] = max['plot']*max['valid']
    max = max.pivot(index='nucleophile_id', columns='electrophile_id', values='plot')
    max[max==0] = -1  # set all zeros to -1, this helps with plotting with a cmap, i can set the color for -1

    fig, ax = plt.subplots()
    if preset_color_dict is not None:
        # val_to_rgb is unordered dict, have to call one by one with a np.arange() list
        listedcolors = [val_to_rgb[ii] for ii in np.arange(len(ligands_to_color))+1]
        cmap = mpl.colors.ListedColormap(listedcolors)
    else:
        cmap = mpl.cm.get_cmap('Paired').copy()
    cmap.set_under('k')
    im = ax.imshow(max, cmap=cmap, vmin=1)

    # grid line
    for i in range(8):
        for j in range(8):
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='white', lw=1))

    ax.set_xticks(np.arange(8), labels=list(max.columns))
    ax.set_yticks(np.arange(8), labels=list(max.index))
    ax.set_xlabel('Electrophile (aryl bromide)')
    ax.set_ylabel('Nucleophile (imidazole)')

    values = list(np.arange(len(ligands_to_color)+1))
    ligand_color = list(ligands_to_color)
    ligand_color.insert(0, f'Not optimized (<{cutoff}%)')
    colors = [im.cmap(im.norm(value)) for value in values]

    patches = [mpatches.Patch(color=colors[i], label=ligand_color[i]) for i in range(len(values))]
    plt.legend(title='Optimal ligand', handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)
    #plt.title('Optimal ligand from model substrate approach')

    ax.spines['top'].set_visible(False)  # remove boundaries
    ax.spines['right'].set_visible(False)

    plt.rcParams['savefig.dpi'] = 600
    plt.show()
    return None

# random sampling baseline, for each ligand, sample n experiments, plot the top 5 ligands as bar plots
# parameters: identify the ligand of interest, and how many experiments per ligand
def plot_simulations_random_sampling(ligand='Cy-BippyPhos', n_exp_per_ligand=3, n_simulations=5000):
    best = []
    gb = df.groupby(by=['ligand_name'])
    for i in tqdm(range(n_simulations)):
        sample = gb.sample(n_exp_per_ligand).groupby('ligand_name').mean(numeric_only=True)
        best.append(sample['yield'].idxmax())

    c = Counter(best).most_common(6)  # outputs a list of tuples
    labels, values = zip(*c)
    percentage = np.array(values)/n_simulations*100
    percentage = [str(round(p, 1)) + '%' for p in percentage]
    return dict(zip(labels, values))

    fig, ax = plt.subplots()

    index = labels.index(ligand)  # find Cy-BippyPhos and assign a diff color
    colors = [classic_blue_hex]*len(labels) # classic blue
    colors[index] = coral_essence_hex  # coralessence

    b = ax.bar(np.arange(len(labels)), values, color=colors, width=0.5)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel('ligand')
    ax.set_ylabel('N times identified as optimal')
    ax.tick_params(axis='x', labelrotation=25)
    ax.bar_label(b, percentage, padding=3)

    ax.spines['top'].set_visible(False)  # remove boundaries
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
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
    ax.scatter(n_exps[-1], probs[-1], c=coral_essence_hex, s=100, zorder=5)  # full HTE
    ax.scatter(0, 1/24*100, c=jasmine_green_hex, s=100, zorder=2)  # random guess with zero experiments
    ax.scatter(n_exps[:-1], probs[:-1], c=classic_blue_hex, zorder=3)  # all random sampling points

    # interpolate with spline
    sample_position = [0, 10, 20, 30, 40, 50, 55, 58, 60, 63]
    spline_x = [n_exps[i] for i in sample_position]
    spline_y = [probs[i] for i in sample_position]
    cs = interpolate.CubicSpline(spline_x, spline_y)
    ax.plot(n_exps, cs(n_exps), c=pirate_black_hex, zorder=4)

    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('number of experiments (24 x sample_size)')
    ax.set_ylabel('probability of finding Cy-BippyPhos (%)')

    plt.rcParams['savefig.dpi'] = 300
    plt.show()
    return None


def _calculate_random_sampling_1():

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

    l = []

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


def calculate_random_sampling_n(num=-1):

    # same as sampling 1
    # except need to construct a new array with the average of selections (select n from 64)
    # this array will get really big after the first few n's

    # # test array
    # X = np.array(np.linspace(0,23,24))
    # np.random.shuffle(X)
    # X = X.reshape(4,6)
    # print(X)

    def select_n_and_average_list(l, n):
        # l: the list
        # n: choose n from list
        combos = list(itertools.combinations(l, n))
        avgs = list(map(lambda x: np.sum(x)/len(x), combos))  # x: iterable; lambda function calcs average
        return avgs

    l = []
    names = []

    for ligand in ligands:
        names.append(ligand)
        tempdf = df.loc[df['ligand_name'] == ligand]
        tempdf = tempdf.drop(['ligand_name'], axis=1)
        a = tempdf.groupby(['electrophile_id'], sort=True)['yield'].apply(list).to_list()
        a = list(itertools.chain(*a))  # raw reaction data, list of list flattened
        l.append(select_n_and_average_list(a, n=num))  # exhaustively select n, average

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

    # # Calculate for Cy-BippyPhos
    # roi = ligands.index('Cy-BippyPhos')  # row number for the row of interest
    # ll = X[roi, :]
    # combos = [count_and_multiply(n) for n in tqdm(ll)]
    # probability = np.sum(combos)  # probability = n of (roi value bigger than all other rows)/ total number of combinations
    # print(probability)

    # test on all ligands, should sum to 1
    probs = []
    for i in tqdm(range(X.shape[0]), desc='1st loop'):
        # roi = ligands.index(ligand)
        roi = i
        ll = X[roi, :]
        combos = [count_and_multiply(n) for n in tqdm(ll, desc='2nd loop', leave=False)]
        probability = np.sum(combos)  # probability = n of (roi value bigger than all other rows)/ total number of combinations
        probs.append(probability)

    print(np.sum(probs))
    return dict(zip(names, probs))


def plot_calculated_sampling_results():
    # calcualted random sampling probablity for CyBippyPhos
    # for each ligand from 64 reactions sample n
    n_samples = [1, 2, 3]
    probs = [0.09605319776941937,
             0.11949538313442956,
             0.13790994055621056]

    # results for all ligands
    names = ['A-caPhos', 'BrettPhos', 'CX-FBu', 'CX-PICy', 'Cy-DavePhos', 'Cy-vBRIDP', 'Kwong', 'MeO-KITPHOS',
             'P(o-Anis)3', 'Et-PhenCar-Phos', 'JackiePhos', 'P(3,5-CF3-Ph)3', 'P(fur)3', 'Ph-PhenCar-Phos', 'PPh2Me',
             'PPhtBu2', 'tBPh-CPhos', 'PPhMe2', 'Cy-BippyPhos', 'PCy3 HBF4', 'CgMe-PPh', 'PPh3', 'X-Phos', 'PnBu3 HBF4']
    all_probs_sample_1 = [0.02235491355695742, 0.012889775708775555, 7.69627179056963e-05, 0.0417756749696991,
                          0.039048385468827634, 0.04161961728746584, 0.025744260436573345, 0.029184853696674062,
                          7.474721952528571e-06, 0.09093318053882313, 0.05868876608747421, 0.039463485950751956,
                          0.053031149196008404, 0.0011331080147418521, 0.03471303570275247, 0.04656681946906384,
                          0.10308541622876125, 0.043843141716261885, 0.09605319776941937, 0.009207834493359874,
                          0.0920188142733526, 0.04741189696643618, 0.041802988426598, 0.029707348822057736]
    all_probs_sample_2 = [0.014633092868738654, 0.0020509255588031486, 1.6411882399193023e-05, 0.04143143651725963,
                          0.0547999354261401, 0.05628769652509228, 0.031787762902610706, 0.030707503764305664,
                          3.4471558154505643e-07, 0.10309969108938917, 0.07786748470698612, 0.03508674590269098,
                          0.03928706485136098, 0.0016859142447752493, 0.013663736468258083, 0.037730440082642966,
                          0.10576717560365459, 0.028648936158248404, 0.11949538313442956, 0.008169847326890035,
                          0.09568413870350644, 0.04059732312888409, 0.05113478725239504, 0.010523714288516273]
    all_probs_sample_3 = [0.00960862456644954, 0.0007124769217796899, 1.3215942435787734e-07, 0.040798121574155764,
                          0.05537203296386926, 0.057351145049020155, 0.028280809061158832, 0.02736811584272358,
                          2.341633018533184e-08, 0.11483637851475693, 0.08447609292719994, 0.03130507451956868,
                          0.03386758832855324, 0.0007010045603914646, 0.008300543205232724, 0.030930537648562902,
                          0.11563449702770918, 0.021802964270785142, 0.13790994055621056, 0.004757889795311808,
                          0.10450518674980756, 0.034899801119376954, 0.05065730597056208, 0.0060425728091676395]

    X = np.arange(len(names))
    fig, ax = plt.subplots()
    ax.barh(X + 0.0, all_probs_sample_1, height=0.25, color=classic_blue_hex, zorder=10)
    ax.barh(X + 0.25, all_probs_sample_2, height=0.25, color=provence_hex, zorder=11)
    ax.barh(X + 0.5, all_probs_sample_3, height=0.25, color=baby_blue_hex, zorder=12)
    ax.set_yticks(X + 0.25, names)
    ax.set_xticks(np.linspace(0, 0.14, 15))
    ax.set_xlabel('Probability')
    ax.legend(labels=['sample 1 exp', 'sample 2 exp', 'sample 3 exp'])
    ax.grid(axis='x', alpha=0.5)
    ax.set_title('Calculated probability for random sampling')

    plt.rcParams['savefig.dpi'] = 300
    plt.show()


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


# categorical heatmap
def _heatmap(xs, ys, data, title=None, xlabel=None, ylabel=None):  # TODO: update heatmap
    """
        Parameters
        ----------
        xs : list
            A list of names
        ys : list
            A list of names
        data : ndarray
            data
        title: str
            title of heatmap
    """

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=110)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xs)))
    ax.set_yticks(np.arange(len(ys)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xs)
    ax.set_yticklabels(ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ys)):
        for j in range(len(xs)):
            text = ax.text(j, i, round(data[i, j], 2),
                           ha="center", va="center", color="w")

    ax.set_title(str(title))
    fig.tight_layout()
    return plt


# draw a horizontal box plot
def _box_plot(data, labels, name):
    """
        Parameters
        ----------
        labels : list
            A list of names (individual reactions, ligands, chemicals...)
        data : list of arrays
            array of results for each label, grouped into a list
        name : str
            general name of the variable (electrophile, nucleophile, substrate...)
    """

    # colors
    la_gold = '#FDB927'
    la_purple = '#552583'

    plt.subplots(figsize=(9.2, 5))
    plt.boxplot(data,
                notch=False,
                labels=labels,
                vert=False,
                patch_artist=True,
                boxprops=dict(facecolor=la_gold, color=la_gold),
                capprops=dict(color=la_purple, linewidth=1.5),
                whiskerprops=dict(color=la_purple, linewidth=1.5),
                medianprops=dict(color=la_purple, linewidth=1.5),
                flierprops=dict(markerfacecolor=la_purple, markeredgecolor=la_purple, marker='.'))
    plt.grid(color=la_purple, axis='x', linestyle='-', linewidth=0.75, alpha=0.2)
    plt.ylabel(name)
    plt.xlabel('yield')
    plt.title(str('Yield grouped by ' + name))
    return plt


# draw a horizontal bar chart for discrete categorical values
def _categorical_bar(labels, data, category_names, title=None, ylabel=None):
    """
        Parameters
        ----------
        labels : list
            A list of entries (individual reactions, ligands, chemicals...)
        data : numpy array
            Data as numpy array; for each label count each category
        category_names : list of str
            The category labels
    """
    # labels = list(results.keys())
    # data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('Spectral')(
        np.linspace(0.1, 0.9, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        #text_color = 'white' if r * g * b < 0.5 else 'black'  # auto adjust text color based on color
        text_color = 'black'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if int(c) != 0:
                ax.text(x, y, str(int(c)), ha='center', va='center', color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='medium')
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    return fig, ax


def simulate_etc(top=1, max_sample=3, n_simulations=10000):
    top1 = ['Cy-BippyPhos']
    top5 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos']
    top9 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos',
            'Cy-vBRIDP', 'Cy-DavePhos', 'X-Phos', 'CX-PICy']

    if top == 1:
        top = top1
    elif top == 5:
        top = top5
    elif top == 9:
        top = top9
    else:
        exit()

    # fetch ground truth data
    df = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')

    percentages = []
    #avg_cumu_rewards = []
    gb = df.groupby(by=['ligand_name'])
    for n_sample in tqdm(range(max_sample), desc='1st loop'):
        count = 0
        reward = 0
        for _ in tqdm(range(n_simulations), desc='2nd loop', leave=False):
            sample = gb.sample(n_sample+1).groupby('ligand_name')
            sample_mean = sample.mean(numeric_only=True)
            sample_sum = sample.sum(numeric_only=True).sum().values[0]
            reward = reward+sample_sum
            # if sample['yield'].idxmax() in top_six:  # no tie breaking when sampling 1 with yield cutoff
            #     count = count + 1
            maxs = sample_mean.loc[sample_mean['yield']==sample_mean['yield'].max()]
            random_one = random.choice(list(maxs.index))
            if random_one in top:
                count = count+1
        percentages.append(count/n_simulations)
        #avg_cumu_rewards.append(reward/n_simulations)

    print(percentages)
    #print(avg_cumu_rewards)
    # with yield: [0.5971, 0.66, 0.7173]
    # 60% cutoff binary, no max tie breaking: [0.388, 0.5382, 0.6154]
    # 60% cutoff binary, with max tie breaking: [0.4301, 0.5488, 0.6136] (helps with sample 1 case, more ties)
    # 60% cutoff binary, cumulative reward: [7.1552, 14.3058, 21.4805]

    # 50% cutoff binary top three: accuracy: [0.2263, 0.3055, 0.3833]; cumu reward [9.7952, 19.6476, 29.4682]
    # 50% cutoff binary top eight: accur: [0.5371, 0.6623, 0.7558]  cumu: [9.8194, 19.6467, 29.4898]
    return None


def plot_ligand_perf_expansion(scenario=1, nlargest=5, preset_color_dict=None):
    # preset_color_dict is used to ensure consistent colors for ligands throughout different plots
    # one set of color is saved in arylation_colors.json

    with open('colors.yml', 'r') as file:
        COLORS = yaml.safe_load(file)

    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
    if scenario == 1:
        nlist = ['nA', 'nB', 'nC', 'nD']
        elist = ['e1', 'e2', 'e3', 'e4']  # these two are the initial lists of nucleophiles and electrophiles
        # then expand nucleophiles first, then electrophiles
        list_1 = df.loc[(df['nucleophile_id'].isin(nlist)) & (df['electrophile_id'].isin(elist))].groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
        list_2 = df.loc[df['electrophile_id'].isin(elist)].groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
        list_3 = df.groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
    elif scenario == 2:
        nlist = ['nE', 'nF', 'nG', 'nI']
        elist = ['e5', 'e7', 'e9', 'e10']  # these two are the initial lists of nucleophiles and electrophiles
        # then expand nucleophiles first, then electrophiles
        list_1 = df.loc[(df['nucleophile_id'].isin(nlist)) & (df['electrophile_id'].isin(elist))].groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
        list_2 = df.loc[df['electrophile_id'].isin(elist)].groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
        list_3 = df.groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
    elif scenario == 3:
        nlist = ['nE', 'nF', 'nG', 'nI']
        elist = ['e1', 'e2', 'e3', 'e4']  # these two are the initial lists of nucleophiles and electrophiles
        # then expand nucleophiles first, then electrophiles
        list_1 = df.loc[(df['nucleophile_id'].isin(nlist)) & (df['electrophile_id'].isin(elist))].groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
        list_2 = df.loc[df['electrophile_id'].isin(elist)].groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])
        list_3 = df.groupby(['ligand_name']).mean(numeric_only=True).nlargest(nlargest, ['yield'])


    all_top_ligands = list(list_1.index) + list(list_2.index) + list(list_3.index)
    all_top_ligands = list(set(all_top_ligands))
    color_list = [COLORS['coral_essence'], COLORS['cornhusk'], COLORS['stucco'], COLORS['peach_quartz'],
                  COLORS['baby_blue'], COLORS['monument'], COLORS['provence'], COLORS['pink_tint'],
                  COLORS['classic_blue'], COLORS['lime_punch'], COLORS['pirate_black'], COLORS['jasmine_green'],
                  COLORS['red_violet']]
    colors = {}

    if preset_color_dict is not None:  # provide an color dictionary {ligand_name: color}
        print('am i execting')
        color_list_count = 0  # this index is used in case the supplied color dict does not contain the ligand that need to be plotted
        # then colors will be incrementally selected from color_list
        for i in range(len(all_top_ligands)):
            ligand = all_top_ligands[i]
            try:
                colors[ligand] = preset_color_dict[ligand]
            except KeyError:
                colors[ligand] = 'gray'
                color_list_count += 1
    else:
        if len(all_top_ligands) > len(color_list):
            raise RuntimeError('not enough colors for all top ligands. {0} colors, {1} ligands'.format(len(color_list),
                                                                                                       len(all_top_ligands)))
        for i in range(len(all_top_ligands)):
            colors[all_top_ligands[i]] = color_list[i]

    # colors here match the accuracy plot
    # colors = {
    #     'Cy-BippyPhos': '#1f77b4',
    #     'Et-PhenCar-Phos': '#ff7f0e',
    #     'tBPh-CPhos': '#2ca02c',
    #     'CgMe-PPh': '#d62728',
    #     'JackiePhos': '#9467bd',
    # }

    figsize = (10,6)
    kwargs = {'aa': True, 'width': 0.5}
    plt.rcParams['savefig.dpi'] = 300
    figs, axs = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)

    def trim(ll):  # trim the long ligand names
        return [s[:10] for s in ll]

    def get_colors(ll):  # for a list of names, get their color from overall color dict
        out = []
        for l in ll:
            out.append(colors[l])
        return out

    def ax_plot(ax_x, df, title, y_label=None):
        x = trim(list(df.index))
        y = list(df['yield'].values)
        axs[ax_x].bar(x, y, color=get_colors(list(df.index)), **kwargs)
        for i in range(len(x)):  # plot value
            axs[ax_x].text(i, y[i]+0.5, round(y[i], 2), ha='center')
        axs[ax_x].set_title(title)  # title
        if y_label:  # y label
            axs[ax_x].set_ylabel(y_label)
        axs[ax_x].set_ylim(top=axs[ax_x].get_ylim()[1] + 5)  # adjust ylim top so value text fits

    ax_plot(0, list_1, title='Phase I (16 products)', y_label='Yield (%)')
    ax_plot(1, list_2, title='Phase II (32 products)', y_label='Yield (%)')
    ax_plot(2, list_3, title='Phase III (64 products)', y_label='Yield (%)')

    plt.show()


if __name__ == '__main__':
    import json
    with open('arylation_colors.json', 'r') as f:
        c_dict = json.load(f)

    # plot_results_with_model_substrates_color_match_publication(preset_color_dict=c_dict)

    plot_ligand_perf_expansion(preset_color_dict=c_dict)

    # plot_ligand_perf_expansion(preset_color_dict=c_dict)
    # # calculate random sampling accuracy
    # for n in range(4):
    #     d = calculate_random_sampling_n(num=n+1)
    #     of_interest = [d[na] for na in names]
    #     sum = np.sum(of_interest)
    #     print(f'sample {n+1}, prob={sum}')
    #     # prob=0.4407793748978305
    #     # prob = 0.5019138732379659
    #     # prob=0.5573620957756842


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
