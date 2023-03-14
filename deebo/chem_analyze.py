import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import gif
import itertools
from utils import scaler

la_gold = '#FDB927'
la_purple = '#552583'


# TODO: somehow ETC provides a more accurace estimation of average. Need to investigate.
# TODO: try lower number of simulations
# TODO: try to enforce 2 experiments per arm
# TODO: if predicted yield is low, sample again. could be risky

# arm elimination


def plot_arm_counts(d='',
                    top_n=5,
                    title='',
                    bar_errbar=False,
                    bar_color='#1f77b4',
                    plot='bar'):
    """
    Plot the average sampling counts for each arm over all simulations

    Parameters
    ----------
    d: str
        directory path for acquisition, should have an arms.pkl file and a history.csv file
    top_n: int
        plot top n most sampled arms
    title: str
        title of the plot
    bar_errbar: bool
        plot a 1 std error for bar plot
    bar_color: str
        color for bar plot
    plot: str
        box plot or bar plot

    Returns
    -------
    None
    """

    import os
    import pickle

    # load files
    if not os.path.isfile(f'{d}/arms.pkl'):
        exit('arms.pkl does not exist in this directory')
    if not os.path.isfile(f'{d}/history.csv'):
        exit('history.csv does not exist in this directory')
    with open(f'{d}/arms.pkl', 'rb') as f:
        arms_dict = pickle.load(f)
    df = pd.read_csv(f'{d}/log.csv')

    # grab some info from log
    num_sims = max(df['num_sims']) + 1  # number of simulations done
    max_horizon = max(df['horizon']) + 1  # max time horizon

    # calculate average number of selection per simulation for top arms
    allcounts = df[['num_sims', 'chosen_arm', 'reward']].groupby(['chosen_arm', 'num_sims']).count()

    # for bar plot, calculate average and std
    sorted_means = allcounts.groupby('chosen_arm').agg({'reward': ['mean', 'std']}).sort_values(by=('reward', 'mean'),
                                                                                                ascending=False)
    average_counts = list(sorted_means.values[:top_n, 0].flatten())
    average_counts_errs = list(sorted_means.values[:top_n, 1].flatten())
    arms_indexes = sorted_means.index.to_numpy()[:top_n]  # corresponding arm index of top n arms
    arm_names = ['/'.join(arms_dict[ii]) for ii in arms_indexes]  # arm names come as tuple, join all elements in tuple

    # for box plot, get all results
    x = allcounts.groupby('chosen_arm')['reward'].apply(np.array)
    x = x.loc[x.index[[int(a) for a in arms_indexes]]]  # use the arms_indexes got from sorted array above, make it int
    indexes = x.index.to_list()  # substrate pairs names
    values = x.to_list()  # list of arrays, since dimensions might not match

    # calculate baseline (time horizon evenly divided by number of arms)
    baseline = max_horizon / len(arms_dict)

    # start plotting
    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    if plot == 'bar':
        if bar_errbar:
            ax.barh(arm_names, average_counts, height=0.5, xerr=average_counts_errs, capsize=4, color=bar_color)
        else:
            ax.barh(arm_names, average_counts, height=0.5, color=bar_color)
        plt.axvline(x=baseline, ymin=0, ymax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)
        ax.set_xlabel('average number of times sampled')
        ax.set_ylabel('arms')
        ax.set_xticks(np.arange(max(average_counts) + max(average_counts_errs)))
    elif plot == 'box':
        plt.axvline(x=baseline, ymin=0, ymax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)
        ax.boxplot(values,
                   notch=False,
                   labels=arm_names,
                   vert=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=la_gold, color=la_gold),
                   capprops=dict(color=la_purple, linewidth=1.5),
                   whiskerprops=dict(color=la_purple, linewidth=1.5),
                   medianprops=dict(color=la_purple, linewidth=1.5),
                   flierprops=dict(markerfacecolor=la_purple, markeredgecolor=la_purple, marker='.'),
                   showmeans=True,
                   meanprops=dict(marker='x', markeredgecolor=la_purple, markerfacecolor=la_purple))
        ax.set_xlabel('number of times sampled')
        ax.set_ylabel('arms')
        ax.set_xticks(np.arange(max([max(v) for v in values]) + 1))
    else:
        pass

    ax.set_title(title)
    plt.show()

    return None


def plot_arm_rewards(ground_truth_loc,
                     d='',
                     top_n=5,
                     title='',
                     errbar=False):
    """

    Parameters
    ----------
    ground_truth_loc: str
        location for ground truth file
    d: str
        directory path for acquisition, should have an arms.pkl file and a history.csv file
    top_n: int
        plot top n most sampled arms
    title: str
        title of the plot
    errbar: bool
        plot error bar or not

    Returns
    -------
    None

    """

    import pickle
    import os

    if not os.path.isfile(f'{d}/arms.pkl'):
        exit('arms.pkl does not exist in this directory')
    if not os.path.isfile(f'{d}/history.csv'):
        exit('history.csv does not exist in this directory')
    with open(f'{d}/arms.pkl', 'rb') as f:
        arms_dict = pickle.load(f)
    df = pd.read_csv(f'{d}/log.csv')

    max_horizon = max(df['horizon']) + 1  # max time horizon
    n_arms = len(arms_dict)

    # for each arm, average yield across each simulation first
    # then calculate the average yield
    # the simulations where a particular arm is not sampled is ignored here
    gb = df[['num_sims', 'chosen_arm', 'reward']].groupby(['chosen_arm', 'num_sims']).mean().groupby('chosen_arm')
    sorted_means = gb.agg({'reward': ['mean', 'std']}).sort_values(by=('reward', 'mean'),
                                                                   ascending=False)  # sorted mean values and pick top n
    sim_average_vals = list(sorted_means.values[:top_n, 0].flatten())
    sim_average_errs = list(sorted_means.values[:top_n, 1].flatten())
    arms_indexes = sorted_means.index.to_numpy()[:top_n]  # corresponding arm index of top n arms
    arms_names = ['/'.join(arms_dict[ii]) for ii in arms_indexes]

    true_averages, etc_averages, etc_errs = calculate_true_and_etc_average(arms_dict,
                                                                           arms_indexes,
                                                                           ground_truth=pd.read_csv(ground_truth_loc),
                                                                           n_sim=100,
                                                                           n_sample=int(max_horizon // n_arms))

    # it's a horizontal bar plot, but use vertical bar terminology here
    width = 0.3  # actually is height
    xs = np.arange(len(arms_names))  # actually is ys

    plt.rcParams['savefig.dpi'] = 300
    if errbar:
        plt.barh(xs - width / 2, sim_average_vals, color=la_gold, height=width, label='experimental average',
                 xerr=sim_average_errs, capsize=4)
        plt.barh(xs + width / 2, etc_averages, color=la_purple, height=width, label='explore-then-commit baseline',
                 xerr=etc_errs, capsize=4)
    else:
        plt.barh(xs - width / 2, sim_average_vals, color=la_gold, height=width, label='experimental average')
        plt.barh(xs + width / 2, etc_averages, color=la_purple, height=width, label='explore-then-commit baseline')
    plt.yticks(xs, arms_names)
    plt.xlabel('yield')
    for ii in range(len(true_averages)):
        plt.vlines(true_averages[ii], ymin=xs[ii] - width - 0.1, ymax=xs[ii] + width + 0.1, linestyles='dotted',
                   colors='k')
    plt.title(title)
    plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left', fontsize='medium')
    plt.tight_layout()
    plt.show()

    return None


def calculate_true_and_etc_average(arms_dict,
                                   arms_indexes,
                                   ground_truth,
                                   n_sim=-1,
                                   n_sample=-1):
    """
    Helper function to calculate true average and explore-then-commit average

    Parameters
    ----------
    arms_dict: dict
        dictionary from arms.pkl, stores arm indexes and corresponding names
    arms_indexes: list like
        the indexes for arms of interest
    ground_truth: pd.DataFrame
        data frame with experimental results
    n_sim: int
        number of simulations for explore then commit, good to match the actual acquisition n_sim
    n_sample: int
        number of samples drawn per arm. This is calculated (# of available experiments // # of available arms)

    Returns
    -------

    """

    if ground_truth['yield'].max() > 2:
        ground_truth['yield'] = ground_truth['yield'].apply(scaler)

    arms = [arms_dict[ii] for ii in arms_indexes]  # get all relevant arms names as a list of tuples
    inverse_arms_dict = {v: k for k, v in arms_dict.items()}  # inverse arms_dict {arm_name: arm_index}

    # figure out which columns are involved in arms
    example = arms[0]
    cols = []
    for e in example:
        l = ground_truth.columns[(ground_truth == e).any()].to_list()
        assert (len(l) == 1)
        cols.append(l[0])
    ground_truth['to_query'] = list(zip(*[ground_truth[c] for c in cols]))  # select these cols and make into tuple
    ground_truth = ground_truth[['to_query', 'yield']]
    filtered = ground_truth[
        ground_truth['to_query'].isin(arms)]  # filter, only use arms of interest supplied by indexes

    # calculate average and generate a dict of results
    means = filtered.groupby(['to_query']).mean()['yield'].to_dict()
    true_averages = {}
    for arm in arms:
        true_averages[inverse_arms_dict[arm]] = means[arm]
    true_averages = [true_averages[ii] for ii in arms_indexes]  # make into a list based on arms_indexes

    # do explore-then-commit
    means = np.zeros((n_sim, len(arms)))
    for n in range(n_sim):
        for ii in range(len(arms)):
            df = filtered.loc[filtered['to_query'] == arms[ii]]
            y = df['yield'].sample(n_sample)
            means[n, ii] = y.mean()
    etc_averages = np.average(means, axis=0)  # arms is already sorted with arms_indexes, can directly use here
    etc_errs = np.std(means, axis=0)

    return true_averages, etc_averages, etc_errs


def calculate_etc_accuracy(arms_dict,
                           explore_limit,
                           arms_indexes,
                           n_sim,
                           ground_truth):
    if ground_truth['yield'].max() > 2:
        ground_truth['yield'] = ground_truth['yield'].apply(scaler)

    arms_of_interest = [arms_dict[ii] for ii in arms_indexes]  # get all relevant arms names as a list of tuples
    arms_all = list(arms_dict.values())  # all arm names
    inverse_arms_dict = {v: k for k, v in arms_dict.items()}  # inverse arms_dict {arm_name: arm_index}

    # figure out which columns are involved in arms
    example = arms_of_interest[0]
    cols = []
    for e in example:
        l = ground_truth.columns[(ground_truth == e).any()].to_list()
        assert (len(l) == 1)
        cols.append(l[0])
    ground_truth['to_query'] = list(zip(*[ground_truth[c] for c in cols]))  # select these cols and make into tuple
    ground_truth = ground_truth[['to_query', 'yield']]

    # do explore then commit
    means = np.zeros((n_sim, len(arms_all)))

    for e in explore_limit:
        for n in range(n_sim):
            for ii in range(len(arms_all)):
                pass
            # TODO: write a generic version of this

    return None


def plot_probs_choosing_best_arm(best_arm_indexes,
                                 fn_list,
                                 legend_list,
                                 fp='',
                                 hline=0,
                                 vline=0,
                                 etc_baseline=False,
                                 etc_fp='',
                                 title='',
                                 legend_title='',
                                 long_legend=False,
                                 ignore_first_rounds=0):
    """
    The probability of choosing the best arm(s) at each time point across all simulations

    Parameters
    ----------
    best_arm_indexes: list like
        list of indexes for optimal arms
    fn_list: Collection
        list of data file names
    legend_list: Collection
        list of labels for legend
    hline: int/float
        value for plotting horizontal baseline
    vline: int/float
        value for plotting a vertical baseline
    etc_baseline: bool
        display explore-then-commit baseline or not
    etc_fp: str
        file path for calculated etc baseline at each time point, a numpy array object
    fp: str
        the deepest common directory for where the data files are stored
    title: str
        title for the plot
    legend_title: str
        title for the legend
    long_legend: bool
        if true, legend will be plotted outside the plot; if false mpl finds the best position within plot
    ignore_first_rounds: int
        when plotting, ignore the first n rounds. Useful for algos that require running one pass of all arms

    Returns
    -------
    matplotlib.pyplot plt object

    """

    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    if hline != 0:
        plt.axhline(y=hline, xmin=0, xmax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)
    if vline != 0:
        plt.axvline(x=vline, ymin=0, ymax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)

    if etc_baseline:
        base = np.load(etc_fp)
        plt.plot(np.arange(len(base))[ignore_first_rounds:], base[ignore_first_rounds:], color='black',
                 label='explore-then-commit', lw=2)

    for i in range(len(fps)):
        fp = fps[i]
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'chosen_arm']]

        n_simulations = int(np.max(df['num_sims'])) + 1
        time_horizon = int(np.max(df['horizon'])) + 1
        all_arms = np.zeros((n_simulations, time_horizon))

        for ii in range(int(n_simulations)):
            all_arms[ii, :] = list(df.loc[df['num_sims'] == ii]['chosen_arm'])

        counts = np.count_nonzero(np.isin(all_arms, best_arm_indexes),
                                  axis=0)  # average across simulations. shape: (1, time_horizon)
        probs = counts / n_simulations
        ax.plot(np.arange(time_horizon)[ignore_first_rounds:], probs[ignore_first_rounds:], label=str(legend_list[i]))

    ax.set_xlabel('time horizon')
    ax.set_ylabel(f'probability of finding best arm: {best_arm_indexes}')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    if long_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
    else:
        ax.legend(title=legend_title)

    plt.show()


def plot_accuracy_best_arm(best_arm_indexes,
                           fn_list,
                           legend_list,
                           fp='',
                           hlines=None,
                           vlines=None,
                           etc_baseline=False,
                           etc_fp='',
                           title='',
                           legend_title='',
                           long_legend=False,
                           ignore_first_rounds=0):
    """
    Accuracy at each time point.
    At each time point, consider all past experiments until this point, and pick the arm with the highest number of samples
    Accuracy = (# of times best empirical arm is in best_arm_indexes) / (# of simulations)

    Parameters
    ----------
    best_arm_indexes

    Returns
    -------

    """

    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    if hlines is not None:
        for hline in hlines:
            plt.axhline(y=hline, xmin=0, xmax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)
    if vlines is not None:
        for vline in vlines:
            plt.axvline(x=vline, ymin=0, ymax=1, linestyle='dashed', color='black', alpha=0.5)

    if etc_baseline:
        base = np.load(etc_fp)
        plt.plot(np.arange(len(base))[ignore_first_rounds:], base[ignore_first_rounds:], color='black',
                 label='explore-then-commit', lw=2)

    for i in range(len(fps)):
        fp = fps[i]
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'chosen_arm']]

        n_simulations = int(np.max(df['num_sims'])) + 1
        time_horizon = int(np.max(df['horizon'])) + 1

        best_arms = np.zeros((n_simulations, time_horizon))  # each time point will have a best arm up to that point

        for n in range(int(n_simulations)):
            data = np.array(list(df.loc[df['num_sims'] == n]['chosen_arm']))
            for t in range(len(data)):
                u, counts = np.unique(data[:t+1], return_counts=True)
                best_arms[n, t] = u[np.random.choice(np.flatnonzero(counts == max(counts)))]

        isinfunc = lambda x: x in best_arm_indexes
        visinfunc = np.vectorize(isinfunc)
        boo = visinfunc(best_arms)
        probs = boo.sum(axis=0)/n_simulations

        ax.plot(np.arange(time_horizon)[ignore_first_rounds:], probs[ignore_first_rounds:], label=str(legend_list[i]))

    ax.set_xlabel('time horizon')
    ax.set_ylabel(f'Accuracy of identifying best arm: {best_arm_indexes}')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    if long_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
    else:
        ax.legend(title=legend_title)
    plt.show()

    return None


@gif.frame
def plot_acquisition_history_heatmap_arylation_scope(history_fp='./test/history.csv', roun=0, sim=0, binary=False,
                                                     cutoff=80):
    """
    plot heatmap for acquisition history at specific time point. Decorator is used to make gifs

    Parameters
    ----------
    history_fp: str
        file path of the history.csv file from acquisition
    roun: int
        avoid built-in func round(); plot the heatmap until this round
    sim: int
        which simulation
    binary
    cutoff

    Returns
    -------

    """

    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
    df = df[['ligand_name', 'electrophile_id', 'nucleophile_id', 'yield']]
    df['electrophile_id'] = df['electrophile_id'].apply(lambda x: x.lstrip('e')).astype(
        'int')  # change to 'int' for sorting purposes, so 10 is not immediately after 1
    df['nucleophile_id'] = df['nucleophile_id'].apply(lambda x: x.lstrip('n'))
    ligands = list(df['ligand_name'].unique())

    # heatmap is divided into 4x6 grid based on ligands. Each ligand is represented by a 8x8 block, overall 32x48
    df = df.sort_values(by=['ligand_name', 'electrophile_id', 'nucleophile_id'])
    ligand_names = list(df['ligand_name'].unique())
    nuc_names = list(df['nucleophile_id'].unique())
    elec_names = list(df['electrophile_id'].unique())

    ground_truth = df[['electrophile_id', 'nucleophile_id', 'ligand_name']].to_numpy()

    # from acquisition history, fetch the reactions that was run, find them in ground_truth to get the indexes (to get yield later)
    history = pd.read_csv(history_fp)
    history = history.loc[(history['round'] <= roun) & (history['num_sims'] == sim)][
        ['electrophile_id', 'nucleophile_id', 'ligand_name']]
    history['electrophile_id'] = history['electrophile_id'].apply(lambda x: x.lstrip('e')).astype('int')
    history['nucleophile_id'] = history['nucleophile_id'].apply(lambda x: x.lstrip('n'))
    history = history.to_numpy()

    # get the indexes for the experiments run, keep the yield, and set the rest of the yields to -1 to signal no rxns run
    indexes = []
    for row in range(history.shape[0]):
        indexes.append(np.argwhere(np.isin(ground_truth, history[row, :]).all(axis=1))[0, 0])
    df = df.reset_index()
    idx_to_set = df.index.difference(indexes)
    df.loc[idx_to_set, 'yield'] = -1

    # divide yields by ligand and into 8x8 stacks
    l = []
    for ligand in ligand_names:
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

    # if binary mode is active, set 0/1 values based on cutoff and keep the -1's
    if binary:
        def set_value(x):
            if 0 <= x < cutoff:
                return 0
            elif x >= cutoff:
                return 1
            else:
                return x

        f = np.vectorize(set_value)
        a = f(a)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap_custom = mpl.colormaps['inferno']
    cmap_custom.set_under('silver')  # for the unrun reactions with yield set to -1
    im = ax.imshow(a, cmap=cmap_custom, vmin=0, vmax=110)
    if binary:
        im = ax.imshow(a, cmap=cmap_custom, vmin=0, vmax=2)
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='white')
    ii = 0
    for i in range(4):
        for j in range(6):
            ax.add_patch(Rectangle((8 * j - 0.5, 8 * i - 0.5), 8, 8, fill=False, edgecolor='white', lw=2))
            plt.text(8 * j + 3.5, 8 * i + 3.5, ligand_names[ii], **text_kwargs)
            ii = ii + 1
    plt.axis('off')
    if not binary:
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('yield (%)', rotation=270)
    plt.rcParams['savefig.dpi'] = 300
    # plt.show()
    return None


@gif.frame
def plot_acquisition_history_heatmap_deoxyf(history_fp='./test/history.csv', roun=0, sim=0, binary=False,
                                                     cutoff=80):
    """

    Parameters
    ----------
    history_fp: str
        file path of history.csv
    roun: int
        avoid built-in func round(); plot up until this round
    sim: int
        which simulation to plot
    binary:
    cutoff:

    Returns
    -------

    """
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv')

    df = df[['base_name', 'fluoride_name', 'substrate_name', 'yield']]
    fd = df.copy()
    df = df.loc[df['substrate_name'] != 's37']
    fs = list(df['fluoride_name'].unique())
    bs = list(df['base_name'].unique())
    ss = list(df['substrate_name'].unique())

    ground_truth = df[['base_name', 'fluoride_name', 'substrate_name']].to_numpy()

    # from acquisition history, fetch the reactions that was run, find them in ground_truth to get the indexes (to get yield later)
    history = pd.read_csv(history_fp)
    history = history.loc[(history['round'] <= roun) & (history['num_sims'] == sim)][
        ['base_name', 'fluoride_name', 'substrate_name']]
    history = history.loc[history['substrate_name'] != 's37']
    history = history.to_numpy()

    # get the indexes for the experiments run, keep the yield, and set the rest of the yields to -1 to signal no rxns run
    indexes = []
    for row in range(history.shape[0]):
        indexes.append(np.argwhere(np.isin(ground_truth, history[row, :]).all(axis=1))[0, 0])
    df = df.reset_index()
    idx_to_set = df.index.difference(indexes)
    df.loc[idx_to_set, 'yield'] = -1

    ds = []
    averages = []
    for f, b in itertools.product(fs, bs):
        ds.append(df.loc[(df['fluoride_name'] == f) & (df['base_name'] == b)]['yield'].to_numpy().reshape(6,6))
        to_average = df.loc[(df['fluoride_name'] == f) & (df['base_name'] == b) & (df['yield']!=-1)]['yield'].to_numpy()
        if len(to_average) == 0:  # catch the np.average warning for empty array
            averages.append('n/a')
        else:
            averages.append(round(np.average(to_average), 1))

    data = np.hstack([np.vstack(ds[0:4]),
                      np.vstack(ds[4:8]),
                      np.vstack(ds[8:12]),
                      np.vstack(ds[12:16]),
                      np.vstack(ds[16:20])])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=110)
    text_kwargs = dict(ha='center', va='center', fontsize=12, color='white')
    ii = 0
    for i in range(5):
        for j in range(4):
            ax.add_patch(Rectangle((6 * i - 0.5, 6 * j - 0.5), 6, 6, fill=False, edgecolor='white', lw=2))
            plt.text(6 * i + 2.5, 6 * j + 2.5, averages[ii], **text_kwargs)
            ii = ii + 1
    #plt.axis('off')
    ax.set_xticks([2.5, 8.5, 14.5, 20.5, 26.5], fs)
    ax.set_yticks([2.5, 8.5, 14.5, 20.5], bs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('yield (%)', rotation=270)
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


if __name__ == '__main__':
    import pickle

    dd = 'dataset_logs/nib/etoh-60cutoff/'
    num_sims = 500
    num_round = 75
    num_exp = 1
    fn_list = [f'{dd}{n}/log.csv' for n in
               [f'ts_gaussian-{num_sims}s-{num_round}r-{num_exp}e',
                f'ts_beta-{num_sims}s-{num_round}r-{num_exp}e',
                f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                f'ucb1-{num_sims}s-{num_round}r-{num_exp}e',
                f'bayes_ucb_gaussian-{num_sims}s-{num_round}r-{num_exp}e',
                f'bayes_ucb_beta-{num_sims}s-{num_round}r-{num_exp}e',
                f'random-{num_sims}s-{num_round}r-{num_exp}e',
                ]]
    fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/nib-etoh.csv'
    with open(f'{dd}ts_gaussian-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
        arms_dict = pickle.load(f)

    reverse_arms_dict = {v: k for k, v in arms_dict.items()}
    # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
    # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
    #ligands = [(b,) for b in bs]
    ligands = ['PPh2Cy', 'CX-PCy', 'PPh3', 'P(p-F-Ph)3', 'P(p-Anis)3', 'Cy-JohnPhos']
    ligands = [(l,) for l in ligands]
    indexes = [reverse_arms_dict[l] for l in ligands]

    plot_accuracy_best_arm(best_arm_indexes=indexes, fn_list=fn_list,
                           legend_list=['TS Gaussian',
                                        'TS Beta',
                                        'ucb1-tuned',
                                        'ucb1',
                                        'Bayes ucb gaussian',
                                        'Bayes ucb beta',
                                        'random'],
                           etc_baseline=False, etc_fp=f'{dd}etc.npy',
                           ignore_first_rounds=0, title=f'Accuracy of identifying as optimal',
                           legend_title='algorithm')

    # plot_arm_counts('dataset_logs/aryl-scope-ligand/BayesUCBGaussian-400s-200r-1e', top_n=10, bar_errbar=True, plot='box', title='Average # of samples')

    # plot_arm_rewards(fp, d='dataset_logs/aryl-scope-ligand/BayesUCBGaussian-400s-200r-1e', top_n=10)

    # make_heatmap_gif(plot_acquisition_history_heatmap_deoxyf,
    #                  n_sim=0,
    #                  max_n_round=100,
    #                  binary=False,
    #                  history_fp=f'{dd}etc-1s-73r-1e/history.csv',
    #                  save_fp=f'test/test.gif')
