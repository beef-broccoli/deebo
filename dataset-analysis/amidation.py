import pandas as pd
import numpy as np
import itertools
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import random

def plot_all_results(single_component='activator'):
    """

    Parameters
    ----------
    single_component: str
        three components to plot here, one of which can be plotted on its own

    Returns
    -------

    """

    # processing dataset
    # use shorter name for bases
    # use labels for nucleophiles
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')

    short_name_dict = {
        '1-Methylimidazole': 'MeIm',
        '2,6-Lutidine': 'lutidine',
        'N-methylmorpholine': 'MeMorph',
        'Diisopropylethylamine': 'DIPEA'
    }
    nuc_id_dict = dict(zip(df['nucleophile_name'].unique(), [f'n{num}' for num in np.arange(len(df['nucleophile_name'].unique()))+1]))
    print(nuc_id_dict)

    df['base_name_long'] = df['base_name']
    df['base_name'] = df['base_name_long'].apply(lambda x: short_name_dict[x])
    df['nucleophile_id'] = df['nucleophile_name'].apply(lambda x: nuc_id_dict[x])

    # depending on which component is plotted on its own, group the other two and do all groupby's
    allthree = ['activator', 'base', 'solvent']
    allthree.remove(single_component)
    df['combo'] = df[f'{allthree[0]}_name'] + '/' + df[f'{allthree[1]}_name']
    df = df[['nucleophile_id', 'combo', f'{single_component}_name', 'yield']]
    df = df.sort_values(by=['nucleophile_id', 'combo', f'{single_component}_name'])
    combo_labels = df['combo'].unique()
    single_component_labels = df[f'{single_component}_name'].unique()

    gb = df.groupby(by=['nucleophile_id', 'combo'])['yield'].apply(list)
    gb2 = gb.groupby(by=['nucleophile_id']).apply(list)

    nuc_ids_list = gb2.index
    to_stack = []
    for n in nuc_ids_list[:5]:
        to_stack.append(np.array(gb2.loc[n]))
    first_five = np.hstack(to_stack)
    to_stack = []
    for n in nuc_ids_list[5:]:
        to_stack.append(np.array(gb2.loc[n]))
    second_five = np.hstack(to_stack)
    data = np.vstack((first_five, second_five))

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=110)
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='white')
    ii = 0
    #
    for i in range(2):
        for j in range(5):
            ax.add_patch(Rectangle((len(single_component_labels) * j - 0.5, len(combo_labels) * i - 0.5),
                                   len(single_component_labels), len(combo_labels), fill=False, edgecolor='white', lw=2))
            plt.text(len(single_component_labels) * (j+0.5)-0.5, len(combo_labels) * (i+0.5)-0.5, str(nuc_ids_list[ii]), **text_kwargs)
            ii = ii + 1

    #ax.set_xticks(np.arange(8), activator_labels, rotation=90)
    ax_t = ax.secondary_xaxis('top')
    ax_t.set_xticks(np.arange(len(single_component_labels)*5), labels=np.tile(single_component_labels, 5), rotation=90)
    ax.set_yticks(np.arange(len(combo_labels)*2), labels=np.tile(combo_labels, 2))
    ax.set_xticks([], labels=[])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax_t.spines['top'].set_visible(False)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('yield (%)', rotation=270, fontsize=14, labelpad=25)

    plt.rcParams['savefig.dpi'] = 300

    plt.show()


def plot_two_dimensions(other_dim):
    """

    Parameters
    ----------
    other_dim: str
        plot yield with activator and other_dim

    Returns
    -------

    """

    # processing dataset
    # use shorter name for bases
    # use labels for nucleophiles
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')

    short_name_dict = {
        '1-Methylimidazole': 'MeIm',
        '2,6-Lutidine': 'lutidine',
        'N-methylmorpholine': 'MeMorph',
        'Diisopropylethylamine': 'DIPEA'
    }
    nuc_id_dict = dict(zip(df['nucleophile_name'].unique(), [f'n{num}' for num in np.arange(len(df['nucleophile_name'].unique()))+1]))

    df['base_name_long'] = df['base_name']
    df['base_name'] = df['base_name_long'].apply(lambda x: short_name_dict[x])
    df['nucleophile_id'] = df['nucleophile_name'].apply(lambda x: nuc_id_dict[x])

    df = df[['nucleophile_id', 'activator_name', 'base_name', 'solvent_name', 'yield']]
    df = df.sort_values(by=['nucleophile_id', 'activator_name', f'{other_dim}_name'])
    activator_labels = df['activator_name'].unique()
    other_labels = df[f'{other_dim}_name'].unique()

    gb1 = df.groupby(by=['nucleophile_id', f'{other_dim}_name', 'activator_name'])['yield'].mean()
    gb2 = gb1.groupby(by=['nucleophile_id', f'{other_dim}_name']).apply(list)
    gb = gb2.groupby(by=['nucleophile_id']).apply(list)

    nuc_ids_list = gb.index
    to_stack = []
    for n in nuc_ids_list[:5]:
        to_stack.append(np.array(gb.loc[n]))
    first_five = np.hstack(to_stack)
    to_stack = []
    for n in nuc_ids_list[5:]:
        to_stack.append(np.array(gb.loc[n]))
    second_five = np.hstack(to_stack)
    data = np.vstack((first_five, second_five))

    fig, ax = plt.subplots()
    #im = ax.imshow(data, cmap='inferno', vmin=0, vmax=110)
    im = ax.imshow(data, cmap='inferno')
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='white')
    ii = 0
    #
    for i in range(2):
        for j in range(5):
            ax.add_patch(Rectangle((len(activator_labels)* j - 0.5, len(other_labels) * i - 0.5),
                                   len(activator_labels), len(other_labels), fill=False, edgecolor='white', lw=2))
            plt.text(len(activator_labels) * (j+0.5)-0.5, len(other_labels) * (i+0.5)-0.5, str(nuc_ids_list[ii]), **text_kwargs)
            ii = ii + 1

    #ax.set_xticks(np.arange(8), activator_labels, rotation=90)
    ax_t = ax.secondary_xaxis('top')
    ax_t.set_xticks(np.arange(len(activator_labels)*5), labels=np.tile(activator_labels, 5), rotation=90)
    ax.set_yticks(np.arange(len(other_labels)*2), labels=np.tile(other_labels, 2))
    ax.set_xticks([], labels=[])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax_t.spines['top'].set_visible(False)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('yield (%)', rotation=270, fontsize=14, labelpad=25)

    plt.rcParams['savefig.dpi'] = 300

    plt.show()


def plot_activator():
    """

    Parameters
    ----------
    other_dim: str
        plot yield with activator and other_dim

    Returns
    -------

    """

    # processing dataset
    # use shorter name for bases
    # use labels for nucleophiles
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')

    short_name_dict = {
        '1-Methylimidazole': 'MeIm',
        '2,6-Lutidine': 'lutidine',
        'N-methylmorpholine': 'MeMorph',
        'Diisopropylethylamine': 'DIPEA'
    }
    nuc_id_dict = dict(zip(df['nucleophile_name'].unique(), [f'n{num}' for num in np.arange(len(df['nucleophile_name'].unique()))+1]))

    df['base_name_long'] = df['base_name']
    df['base_name'] = df['base_name_long'].apply(lambda x: short_name_dict[x])
    df['nucleophile_id'] = df['nucleophile_name'].apply(lambda x: nuc_id_dict[x])

    df = df[['nucleophile_id', 'activator_name', 'base_name', 'solvent_name', 'yield']]
    df = df.sort_values(by=['nucleophile_id', 'activator_name'])
    activator_labels = df['activator_name'].unique()

    gb1 = df.groupby(by=['nucleophile_id', 'activator_name'])['yield'].mean()
    gb = gb1.groupby(by=['nucleophile_id']).apply(list)
    data = np.array(list(gb.values))
    nuc_ids_list = gb.index

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=110)
    #im = ax.imshow(data, cmap='inferno')
    text_kwargs = dict(ha='center', va='center', fontsize=12, color='white')

    # plot the numbers
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, str(round(data[i,j],1)), **text_kwargs)

    ax.set_xticks(np.arange(data.shape[1]), activator_labels, rotation=90)
    ax.set_yticks(np.arange(data.shape[0]), nuc_ids_list)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('yield (%)', rotation=270, fontsize=14, labelpad=25)

    plt.rcParams['savefig.dpi'] = 300

    plt.show()


def change_later():

    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')

    short_name_dict = {
        '1-Methylimidazole': 'MeIm',
        '2,6-Lutidine': 'lutidine',
        'N-methylmorpholine': 'MeMorph',
        'Diisopropylethylamine': 'DIPEA'
    }
    nuc_id_dict = dict(zip(df['nucleophile_name'].unique(), [f'n{num}' for num in np.arange(len(df['nucleophile_name'].unique()))+1]))

    df['base_name_long'] = df['base_name']
    df['base_name'] = df['base_name_long'].apply(lambda x: short_name_dict[x])
    df['nucleophile_id'] = df['nucleophile_name'].apply(lambda x: nuc_id_dict[x])

    gb = df.groupby(by=['activator_name', 'base_name'])['yield'].mean().sort_values(ascending=False)
    gb.to_csv('test.csv')


def plot_average_yield_by_substrate(single_component):
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')

    short_name_dict = {
        '1-Methylimidazole': 'MeIm',
        '2,6-Lutidine': 'lutidine',
        'N-methylmorpholine': 'MeMorph',
        'Diisopropylethylamine': 'DIPEA'
    }
    nuc_id_dict = dict(zip(df['nucleophile_name'].unique(),
                           [f'n{num}' for num in np.arange(len(df['nucleophile_name'].unique())) + 1]))

    df['base_name_long'] = df['base_name']
    df['base_name'] = df['base_name_long'].apply(lambda x: short_name_dict[x])
    df['nucleophile_id'] = df['nucleophile_name'].apply(lambda x: nuc_id_dict[x])

    # depending on which component is plotted on its own, group the other two and do all groupby's
    df = df[['nucleophile_id', 'base_name', 'activator_name', 'solvent_name','yield']]
    df = df.sort_values(by=['nucleophile_id', f'{single_component}_name'])
    single_component_labels = df[f'{single_component}_name'].unique()
    df = df.groupby(by=['nucleophile_id', f'{single_component}_name'])['yield'].mean()
    df = df.groupby(by='nucleophile_id').apply(list)
    nuc_labels = df.index
    data = np.array(df.values.tolist())  # data shape: n_nucleophiles x n_<single_component>

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=110)
    text_kwargs = dict(ha='center', va='center', color='white')

    #
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, str(round(data[i, j], 1)), **text_kwargs)

    # ax.set_xticks(np.arange(8), activator_labels, rotation=90)
    ax.set_xticks(np.arange(len(single_component_labels)), labels=single_component_labels, rotation=90)
    ax.set_yticks(np.arange(len(nuc_labels)), labels=nuc_labels)
    ax.set_xlabel(f'{single_component}')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('yield (%)', rotation=270, labelpad=25)

    plt.title(f'Average yield')
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 300

    plt.show()


def plot_best_with_diff_metric(nlargest=8, which_dimension='activator'):  # 6 bar plots, each with top 5 ligands, and their performance wrt metric
    # activator only, or combo

    with open('colors.yml', 'r') as file:
        COLORS = yaml.safe_load(file)

    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')
    short_name_dict = {
        '1-Methylimidazole': 'MeIm',
        '2,6-Lutidine': 'lutidine',
        'N-methylmorpholine': 'MeMorph',
        'Diisopropylethylamine': 'DIPEA'
    }
    nuc_id_dict = dict(zip(df['nucleophile_name'].unique(),
                           [f'n{num}' for num in np.arange(len(df['nucleophile_name'].unique())) + 1]))

    df['base_name_long'] = df['base_name']
    df['base_name'] = df['base_name_long'].apply(lambda x: short_name_dict[x])
    df['nucleophile_id'] = df['nucleophile_name'].apply(lambda x: nuc_id_dict[x])
    df = df[['nucleophile_id', 'base_name', 'activator_name', 'solvent_name','yield']]

    df['combo'] = df['activator_name'].astype('str') + '/' + df['base_name'].astype('str')
    if which_dimension == 'activator':
        which_dimension = 'activator_name'
    stats = df.groupby(by=[which_dimension]).describe()
    twentyfive = stats.loc[:, ('yield', '25%')].nlargest(nlargest)  # 1st quantile top 5
    median = stats.loc[:, ('yield', '50%')].nlargest(nlargest)  # 2nd quantile
    seventyfive = stats.loc[:, ('yield', '75%')].nlargest(nlargest)  # 3rd quantile
    mean = stats.loc[:, ('yield', 'mean')].nlargest(nlargest)  # average

    overtwenty = df.loc[df['yield'] > 20].groupby(by=which_dimension).size().nlargest(nlargest)
    overeighty = df.loc[df['yield'] > 60].groupby(by=which_dimension).size().nlargest(nlargest)

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
    if len(all_top_ligands) > len(color_list):
        raise RuntimeError('not enough colors for all top options. {0} colors, {1} options'.format(len(color_list), len(all_top_ligands)))
    for i in range(len(all_top_ligands)):
        colors[all_top_ligands[i]] = color_list[i]

    def get_colors(ll):  # for a list of names, get their color from overall color dict
        out = []
        for l in ll:
            out.append(colors[l])
        return out

    def trim(ll):  # trim the long ligand names
        return [s[:20] for s in ll]

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
    ax_plot(2, 1, overeighty, title='yield >60%')


    plt.show()


def simulate_etc_activator(top=1, max_sample=3, n_simulations=10000):
    top1 = ['DPPCl']
    top3 = ['DPPCl', 'BOP-Cl', 'TCFH']

    if top == 1:
        top = top1
    elif top == 3:
        top = top3
    else:
        exit()

    # fetch ground truth data
    df = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')

    percentages = []
    #avg_cumu_rewards = []
    gb = df.groupby(by=['activator_name'])
    for n_sample in tqdm(range(max_sample), desc='1st loop'):
        count = 0
        reward = 0
        for _ in tqdm(range(n_simulations), desc='2nd loop', leave=False):
            sample = gb.sample(n_sample+1).groupby('activator_name')
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
    # top1 [0.0, 0.2334, 0.2772, 0.3188, 0.3513, 0.3749, 0.4077, 0.4324, 0.4465, 0.4732, 0.4998, 0.5069, 0.5252]
    # top3 [0.0, 0.5395, 0.6302, 0.6601, 0.6981, 0.7379, 0.7682, 0.7814, 0.8033, 0.8163, 0.8403, 0.8482, 0.862]


def simulate_etc_combo(top=1, max_sample=3, n_simulations=10000):
    top1 = [('DPPCl', 'N-methylmorpholine')]
    top2 = [('DPPCl', 'N-methylmorpholine'),
            ('DPPCl', 'Diisopropylethylamine')]

    if top == 1:
        top = top1
    elif top == 2:
        top = top2
    else:
        exit()

    # fetch ground truth data
    df = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv')

    percentages = []
    #avg_cumu_rewards = []
    gb = df.groupby(by=['activator_name', 'base_name'])
    for n_sample in tqdm(range(max_sample), desc='1st loop'):
        count = 0
        reward = 0
        for _ in tqdm(range(n_simulations), desc='2nd loop', leave=False):
            sample = gb.sample(n_sample+1).groupby(by=['activator_name', 'base_name'])
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
    # top1 [0.0, 0.1214, 0.1757, 0.2109]
    # top2 [0.0, 0.2031, 0.2934, 0.3476]


if __name__ == '__main__':
    pass