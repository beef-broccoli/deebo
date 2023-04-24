import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import itertools
import yaml
import gif


def plot_all_results():
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/cn-processed.csv')
    df = df[['base_name', 'ligand_name', 'substrate_id', 'additive_id', 'yield']]

    LS = df['ligand_name'].unique()
    BS = df['base_name'].unique()

    ds = []
    averages = []
    for l, b in itertools.product(LS, BS):
        tempdf = df.loc[(df['ligand_name'] == l) & (df['base_name'] == b)]
        tempdf = tempdf.drop(['ligand_name', 'base_name'], axis=1)
        a = np.array(tempdf.groupby(['substrate_id'], sort=True)['yield'].apply(list).to_list())
        # each row of a is a substrate, each column of a is an additive
        ds.append(a)
        averages.append(round(np.average(a), 2))

    data = np.vstack([np.hstack(ds[0:3]),
                      np.hstack(ds[3:6]),
                      np.hstack(ds[6:9]),
                      np.hstack(ds[9:12])])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno')
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='white')
    ii = 0
    for i in range(4):
        for j in range(3):
            ax.add_patch(Rectangle((20 * j - 0.5, 15 * i - 0.5), 20, 15, fill=False, edgecolor='white', lw=2))
            plt.text(20 * j + 9.5, 15 * i + 7, averages[ii], **text_kwargs)
            ii = ii + 1

    #plt.axis('off')
    ax.set_xticks([9.5, 29.5, 49.5], BS, fontsize=14)
    ax.set_yticks([7, 22, 37, 52], LS, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_ylabel('yield (%)', rotation=270, fontsize=14, labelpad=25)
    plt.rcParams['savefig.dpi'] = 300

    plt.show()


def plot_one_combination():
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/cn-processed.csv')
    df = df[['base_name', 'ligand_name', 'substrate_id', 'additive_id', 'yield']]

    LS = df['ligand_name'].unique()
    BS = df['base_name'].unique()

    ds = []
    averages = []
    tempdf = df.loc[(df['ligand_name'] == 'AdBrettPhos') & (df['base_name'] == 'BTMG')]
    tempdf = tempdf.drop(['ligand_name', 'base_name'], axis=1)
    a = np.array(tempdf.groupby(['substrate_id'], sort=False)['yield'].apply(list).to_list())
    # each row of a is a substrate, each column of a is an additive

    fig, ax = plt.subplots()
    im = ax.imshow(a, cmap='inferno', vmin=min(df['yield']), vmax=max(df['yield']))
    text_kwargs = dict(ha='center', va='center', fontsize=11, color='white')
    ii = 0
    for i in range(20):
        for j in range(15):
            plt.text(i, j, round(a[j, i], 1), **text_kwargs)

    #plt.axis('off')
    xticks = np.arange(20)
    yticks = np.arange(15)
    ax.set_xticks(xticks, tempdf['additive_id'].unique())
    ax.set_xlabel('additive')
    ax.set_yticks(yticks, tempdf['substrate_id'].unique())
    ax.set_ylabel('substrate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_ylabel('yield (%)', rotation=270, fontsize=14, labelpad=25)
    plt.title('yield for (AdBrettPhos, BTMG)')
    plt.rcParams['savefig.dpi'] = 300
    plt.tight_layout()

    plt.show()

@gif.frame
def plot_acquisition_history_heatmap_cn(history_fp='./test/history.csv', sim=0, roun=0, binary=False,
                                                     cutoff=80):
    """
    plots snapshots of acquisition history

    Parameters
    ----------
    history_fp: str
        file path of history.csv
    roun: list-like
        snapshot of heatmap at this round
    sim: int
        which simulation to plot
    binary: bool
        plot heatmap with binary cutoff or not
    cutoff: int or float
        the cutoff yield for binary

    Returns
    -------

    """
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/cn-processed.csv')
    df = df[['base_name', 'ligand_name', 'substrate_id', 'additive_id', 'yield']]

    LS = df['ligand_name'].unique()
    BS = df['base_name'].unique()

    ground_truth = df[['base_name', 'ligand_name', 'substrate_id', 'additive_id']].to_numpy()

    # from acquisition history, fetch the reactions that was run, find them in ground_truth to get the indexes (to get yield later)
    history = pd.read_csv(history_fp)
    history = history.loc[(history['round'] <= roun) & (history['num_sims'] == sim)][
        ['base_name', 'ligand_name', 'substrate_id', 'additive_id']]
    history = history.to_numpy()

    # get the indexes for the experiments run, keep the yield, and set the rest of the yields to -1 to signal no rxns run
    indexes = []
    for row in range(history.shape[0]):
        indexes.append(np.argwhere(np.isin(ground_truth, history[row, :]).all(axis=1))[0, 0])
    fd = df.reset_index()
    idx_to_set = fd.index.difference(indexes)
    fd.loc[idx_to_set, 'yield'] = -1

    # sort data into matrixes
    ds = []
    averages = []
    for l, b in itertools.product(LS, BS):

        # get all data
        tempdf = fd.loc[(fd['ligand_name'] == l) & (fd['base_name'] == b)]
        tempdf = tempdf.drop(['ligand_name', 'base_name'], axis=1)
        a = np.array(tempdf.groupby(['substrate_id'], sort=True)['yield'].apply(list).to_list())
        # each row of a is a substrate, each column of a is an additive
        ds.append(a)

        # do average
        to_average = fd.loc[(fd['ligand_name'] == l) & (fd['base_name'] == b) & (fd['yield'] != -1)]['yield'].to_numpy()
        if len(to_average) == 0:  # catch the np.average warning for empty array
            averages.append('n/a')
        else:
            averages.append(round(np.average(to_average), 1))

    data = np.vstack([np.hstack(ds[0:3]),
                      np.hstack(ds[3:6]),
                      np.hstack(ds[6:9]),
                      np.hstack(ds[9:12])])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=110)
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='white')
    ii = 0
    for i in range(4):
        for j in range(3):
            ax.add_patch(Rectangle((20 * j - 0.5, 15 * i - 0.5), 20, 15, fill=False, edgecolor='white', lw=2))
            plt.text(20 * j + 9.5, 15 * i + 7, averages[ii], **text_kwargs)
            ii = ii + 1

    #plt.axis('off')
    for i in range(1):
        for j in range(1):
            ax.set_xticks([9.5, 29.5, 49.5], BS, fontsize=14)
            ax.set_yticks([7, 22, 37, 52], LS, fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_ylabel('yield (%)', rotation=270, fontsize=14, labelpad=25)
    plt.title(f't={roun}', fontsize=14)
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 300
    #plt.show()


def make_heatmap_gif(plot_func, n_sim=0, max_n_round=100, binary=False, history_fp='', save_fp=''):
    frames = []
    for ii in range(max_n_round):
        frames.append(
            plot_func(sim=n_sim,
                      roun=ii,
                      binary=binary,
                      history_fp=history_fp))

    assert save_fp.endswith('.gif'), 'file suffix needs to be .gif'
    gif.save(frames, save_fp, duration=100)

    return None


def plot_best_with_diff_metric(nlargest=5, which_dimension='combo'):  # 6 bar plots, each with top 5 ligands, and their performance wrt metric

    with open('colors.yml', 'r') as file:
        COLORS = yaml.safe_load(file)

    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/cn-processed.csv')
    df = df[['base_name', 'ligand_name', 'substrate_id', 'additive_id', 'yield']]

    LS = df['ligand_name'].unique()
    BS = df['base_name'].unique()

    df['ligand_name'] = df['ligand_name'].apply(lambda x: str(x).rstrip('hos'))  # trim the phosphine names

    df['combo'] = df['ligand_name'].astype('str') + '/' + df['base_name'].astype('str')
    stats = df.groupby(by=[which_dimension]).describe()
    twentyfive = stats.loc[:, ('yield', '25%')].nlargest(nlargest)  # 1st quantile top 5
    median = stats.loc[:, ('yield', '50%')].nlargest(nlargest)  # 2nd quantile
    seventyfive = stats.loc[:, ('yield', '75%')].nlargest(nlargest)  # 3rd quantile
    mean = stats.loc[:, ('yield', 'mean')].nlargest(nlargest)  # average

    overtwenty = df.loc[df['yield'] > 20].groupby(by=which_dimension).size().nlargest(nlargest)  # top 5, over 20%, count
    overeighty = df.loc[df['yield'] > 80].groupby(by=which_dimension).size().nlargest(nlargest)  # top 5, over 80%, count

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
                  COLORS['baby_blue'], COLORS['monument'], COLORS['provence'], COLORS['pink_tint']]
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
    ax_plot(2, 1, overeighty, title='yield >80%')

    plt.show()


if __name__ == '__main__':
    # make_heatmap_gif(plot_acquisition_history_heatmap_cn,
    #                  0,
    #                  100,
    #                  history_fp='/Users/mac/Desktop/project deebo/deebo/deebo/dataset_logs/cn/bayes_ucb_gaussian_c=2_assumed_sd=0.25-500s-100r-1e/history.csv',
    #                  save_fp='test.gif')

    plot_best_with_diff_metric()