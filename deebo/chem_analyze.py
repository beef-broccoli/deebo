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
                           xlabel='time horizon (number of experiments)',
                           legend_title='',
                           long_legend=False,
                           ignore_first_rounds=0,
                           shade_first_rounds=0,
                           max_horizon_plot=0):
    """
    Accuracy up to each time point
    At each time point, consider all past experiments until this point, and pick the arm with the highest number of samples
    Accuracy = (# of times best empirical arm is in best_arm_indexes) / (# of simulations)

    Parameters
    ----------
    best_arm_indexes: Collection
        list of indexes for the best arms
    fn_list: Collection
        list of data file names
    legend_list: Collection
        list of labels used in legend
    fp: str
        file path of the deepest common dir, used with fn_list
    hlines: Collection
        list of y axis locations to draw horizontal lines
    vlines: Collection
        list of x axis locations to draw horizontal lines
    etc_baseline: bool
        draw the explore-then-commit baseline or not
    etc_fp: str
        filepath of .npy file that stores numpy array with etc baseline
    title: str
        title of plot
    legend_title: str
        title of legend
    long_legend: bool
        if the legend is long, will move the legend outside of the plot
    ignore_first_rounds: int
        ignore the first n data points in the plot for all traces
    shade_first_rounds: int
        plot all data points, but vertically shade until x=n. And mark this area as "exploration"
    max_horizon_plot: int
        plot until this maximum time horizon

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

    max_time_horizon = 0  # etc baseline cuts off at this time horizon
    for i in range(len(fps)):
        fp = fps[i]
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'chosen_arm']]

        n_simulations = int(np.max(df['num_sims'])) + 1
        time_horizon = int(np.max(df['horizon'])) + 1
        if time_horizon > max_time_horizon:
            max_time_horizon = time_horizon

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

        if max_horizon_plot == 0:
            ax.plot(np.arange(time_horizon)[ignore_first_rounds:], probs[ignore_first_rounds:], label=str(legend_list[i]))
        else:
            ax.plot(np.arange(time_horizon)[ignore_first_rounds:max_horizon_plot], probs[ignore_first_rounds:max_horizon_plot],
                    label=str(legend_list[i]))
    if max_horizon_plot != 0:
        max_time_horizon = max_horizon_plot  # adjust max_time_horizon again before plotting ETC

    if etc_baseline:
        base = np.load(etc_fp)
        plt.plot(np.arange(len(base))[ignore_first_rounds:max_time_horizon], base[ignore_first_rounds:max_time_horizon],
                 color='black',
                 label='explore-then-commit',
                 lw=2,
                 zorder=-100)

    if shade_first_rounds != 0:
        ax.axvspan(0, shade_first_rounds, facecolor='lightgray', alpha=0.5, zorder=100)
        _, ymax = ax.get_ylim()
        ax.text(shade_first_rounds/2, ymax*0.75, 'exploration', verticalalignment='center',
                horizontalalignment='center', zorder=101, rotation=90)

    # ##custom text area
    # ax.text(37, 0.91, '1% of data', c='black', backgroundcolor='whitesmoke', fontstyle='italic', fontweight='semibold', ha='center', va='center')
    # ax.text(70, 0.96, '2% of data', c='black', backgroundcolor='whitesmoke', fontstyle='italic', fontweight='semibold', ha='center', va='center')
    # ##

    ax.set_xlabel(xlabel)
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


def plot_accuracy_best_arm_scope_expansion(arm_indexes,
                                           legend_list,
                                           baseline_arm_indexes,
                                           fp='',
                                           baseline_fp='',
                                           hlines=None,
                                           vlines=None,
                                           etc_baseline=False,
                                           etc_fp='',
                                           title='',
                                           legend_title='',
                                           long_legend=False,
                                           ignore_first_rounds=0,
                                           shade_first_rounds=0,
                                           max_horizon_plot=0):
    """
    Accuracy up to each time point
    At each time point, consider all past experiments until this point, and pick the arm with the highest number of samples
    Accuracy = (# of times best empirical arm is in best_arm_indexes) / (# of simulations)
    Modified for scope expansion results with arylation data, top-1 accuracy for different ligands from the same data

    Parameters
    ----------
    arm_indexes: Collection
        list of indexes for all the arms to be plotted
    legend_list: Collection
        list of labels used in legend, list of ligands in this case
    baseline_arm_indexes: Collection
        list of indexes for arms to be plotted from baseline file
    fp: str
        file path of the data file
    baseline_fp: str
        file path of the data file where baseline is plotted. In this case it's optimization with the full scope
    hlines: Collection
        list of y axis locations to draw horizontal lines
    vlines: Collection
        list of x axis locations to draw horizontal lines
    etc_baseline: bool
        draw the explore-then-commit baseline or not
    etc_fp: str
        filepath of .npy file that stores numpy array with etc baseline
    title: str
        title of plot
    legend_title: str
        title of legend
    long_legend: bool
        if the legend is long, will move the legend outside of the plot
    ignore_first_rounds: int
        ignore the first n data points in the plot for all traces
    shade_first_rounds: int
        plot all data points, but vertically shade until x=n. And mark this area as "exploration"
    max_horizon_plot: int
        plot until this maximum time horizon

    Returns
    -------

    """

    assert len(arm_indexes) == len(legend_list)

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    if hlines is not None:
        for hline in hlines:
            plt.axhline(y=hline, xmin=0, xmax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)
    if vlines is not None:
        for vline in vlines:
            plt.axvline(x=vline, ymin=0, ymax=1, linestyle='dashed', color='black', alpha=0.5)

    max_time_horizon = 0  # etc baseline cuts off at this time horizon

    # this plots accuracies for multiple arms from the data log with scope expansion
    df = pd.read_csv(fp)
    df = df[['num_sims', 'horizon', 'chosen_arm']]

    n_simulations = int(np.max(df['num_sims'])) + 1
    time_horizon = int(np.max(df['horizon'])) + 1
    if time_horizon > max_time_horizon:
        max_time_horizon = time_horizon

    best_arms = np.zeros((n_simulations, time_horizon))  # each time point will have a best arm up to that point

    # calculate best arms for all time points
    for n in range(int(n_simulations)):
        data = np.array(list(df.loc[df['num_sims'] == n]['chosen_arm']))
        for t in range(len(data)):
            u, counts = np.unique(data[:t + 1], return_counts=True)
            best_arms[n, t] = u[np.random.choice(np.flatnonzero(counts == max(counts)))]

    for i in range(len(arm_indexes)):
        isinfunc = lambda x: x == arm_indexes[i]
        visinfunc = np.vectorize(isinfunc)
        boo = visinfunc(best_arms)
        probs = boo.sum(axis=0) / n_simulations

        if max_horizon_plot == 0:
            ax.plot(np.arange(time_horizon)[ignore_first_rounds:], probs[ignore_first_rounds:],
                    label=str(legend_list[i]))
        else:
            ax.plot(np.arange(time_horizon)[ignore_first_rounds:max_horizon_plot],
                    probs[ignore_first_rounds:max_horizon_plot],
                    label=str(legend_list[i]))

    # this plots a baseline, where the same algorithm picks the overall top-1 arm with all the data
    df = pd.read_csv(baseline_fp)
    df = df[['num_sims', 'horizon', 'chosen_arm']]

    n_simulations = int(np.max(df['num_sims'])) + 1
    time_horizon = int(np.max(df['horizon'])) + 1
    if time_horizon > max_time_horizon:
        max_time_horizon = time_horizon

    best_arms = np.zeros((n_simulations, time_horizon))  # each time point will have a best arm up to that point

    # calculate best arms for all time points
    for n in range(int(n_simulations)):
        data = np.array(list(df.loc[df['num_sims'] == n]['chosen_arm']))
        for t in range(len(data)):
            u, counts = np.unique(data[:t + 1], return_counts=True)
            best_arms[n, t] = u[np.random.choice(np.flatnonzero(counts == max(counts)))]

    for i in range(len(baseline_arm_indexes)):
        isinfunc = lambda x: x == baseline_arm_indexes[i]
        visinfunc = np.vectorize(isinfunc)
        boo = visinfunc(best_arms)
        probs = boo.sum(axis=0) / n_simulations

        if max_horizon_plot == 0:
            ax.plot(np.arange(time_horizon)[ignore_first_rounds:], probs[ignore_first_rounds:],
                    label=f'{legend_list[0]} (full scope)', alpha=0.5, lw=2, c='#1f77b4')
        else:
            ax.plot(np.arange(time_horizon)[ignore_first_rounds:max_horizon_plot],
                    probs[ignore_first_rounds:max_horizon_plot],
                    label=f'{legend_list[0]} (full scope)', alpha=0.5, lw=2, c='#1f77b4')

    if max_horizon_plot != 0:
        max_time_horizon = max_horizon_plot  # adjust max_time_horizon again before plotting ETC

    if etc_baseline:
        base = np.load(etc_fp)
        plt.plot(np.arange(len(base))[ignore_first_rounds:max_time_horizon], base[ignore_first_rounds:max_time_horizon],
                 color='black',
                 label='explore-then-commit',
                 lw=2,
                 zorder=-100)

    if shade_first_rounds != 0:
        ax.axvspan(0, shade_first_rounds, facecolor='lightgray', alpha=0.5, zorder=100)
        _, ymax = ax.get_ylim()
        ax.text(shade_first_rounds / 2, ymax * 0.75, 'exploration', verticalalignment='center',
                horizontalalignment='center', zorder=101, rotation=90)

    # ##custom text area
    # ax.text(37, 0.91, '1% of data', c='black', backgroundcolor='whitesmoke', fontstyle='italic', fontweight='semibold', ha='center', va='center')
    # ax.text(70, 0.96, '2% of data', c='black', backgroundcolor='whitesmoke', fontstyle='italic', fontweight='semibold', ha='center', va='center')
    # ##

    ax.set_xlabel('time horizon (number of experiments)')
    ax.set_ylabel(f'Accuracy of identifying best arm')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    if long_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
    else:
        ax.legend(title=legend_title)

    plt.show()
    return None


def plot_cumulative_reward(fn_list,
                           legend_list,
                           fp='',
                           title='Cumulative reward',
                           legend_title='',
                           ignore_first_rounds=0,
                           shade_first_rounds=0,
                           long_legend=False,
                           etc_baseline=False,
                           etc_fp=''):

    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    max_time_horizon=0
    for i in range(len(fps)):
        fp = fps[i]
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'cumulative_reward']]

        def get_rewards(df):  # for one simulation, calculate reward (average or cumulative) at each time horizon t
            rewards = df['cumulative_reward'].to_numpy()
            return rewards

        n_simulations = int(np.max(df['num_sims']))+1
        time_horizon = int(np.max(df['horizon']))+1
        all_rewards = np.zeros((n_simulations, time_horizon))
        if time_horizon > max_time_horizon:
            max_time_horizon = time_horizon

        for ii in range(int(n_simulations)):
            rewards = df.loc[df['num_sims'] == ii]['cumulative_reward'].to_numpy()
            all_rewards[ii, :] = rewards

        avg_reward = np.average(all_rewards, axis=0)  # average across simulations. shape: (1, time_horizon)
        ax.plot(np.arange(time_horizon)[ignore_first_rounds:], avg_reward[ignore_first_rounds:], label=str(legend_list[i]))

    if shade_first_rounds != 0:
        ax.axvspan(0, shade_first_rounds, facecolor='lightgray', alpha=0.5, zorder=100)
        _, ymax = ax.get_ylim()
        ax.text(shade_first_rounds/2, ymax*0.75, 'exploration', verticalalignment='center', horizontalalignment='center', zorder=101)

    if etc_baseline:
        base = np.load(etc_fp)
        plt.plot(np.arange(len(base))[ignore_first_rounds:max_time_horizon], base[ignore_first_rounds:max_time_horizon],
                 color='black',
                 label='explore-then-commit',
                 lw=2,
                 zorder=-100)

    ax.set_xlabel('time horizon')
    ax.set_ylabel('cumulative reward')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    if long_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
    else:
        ax.legend(title=legend_title)
    plt.show()



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


def figure_dimensionality():
    nib = [0.0, 0.0, 0.0, 0.0, 0.19, 0.214, 0.136, 0.108, 0.118, 0.102, 0.104, 0.078, 0.074, 0.062, 0.052, 0.06, 0.114, 0.088, 0.164, 0.172, 0.148, 0.138, 0.136, 0.052, 0.094, 0.166, 0.2, 0.236, 0.284, 0.32, 0.332, 0.366, 0.398, 0.392, 0.392, 0.4, 0.43, 0.446, 0.452, 0.452, 0.456, 0.474, 0.496, 0.496, 0.506, 0.522, 0.532, 0.54, 0.536, 0.548, 0.548, 0.56, 0.572, 0.566, 0.582, 0.578, 0.592, 0.608, 0.616, 0.624, 0.63, 0.636, 0.63, 0.638, 0.632, 0.638, 0.64, 0.642, 0.65, 0.646, 0.648, 0.654, 0.656, 0.676, 0.678, 0.682, 0.676, 0.682, 0.682, 0.688, 0.708, 0.704, 0.702, 0.702, 0.708, 0.716, 0.722, 0.724, 0.724, 0.736, 0.728, 0.734, 0.734, 0.738, 0.736, 0.752, 0.758, 0.758, 0.75, 0.766]
    deoxyf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0575, 0.055, 0.055, 0.055, 0.055, 0.0975, 0.21, 0.2125, 0.1775, 0.2275, 0.26, 0.2375, 0.26, 0.2625, 0.295, 0.31, 0.3, 0.3275, 0.325, 0.34, 0.375, 0.395, 0.4025, 0.4075, 0.4175, 0.4225, 0.42, 0.435, 0.4375, 0.4225, 0.4375, 0.4375, 0.4475, 0.45, 0.465, 0.4625, 0.4825, 0.495, 0.505, 0.5, 0.5175, 0.51, 0.53, 0.5175, 0.525, 0.53, 0.54, 0.5475, 0.55, 0.5525, 0.545, 0.54, 0.545, 0.5575, 0.5875, 0.585, 0.6025, 0.5975, 0.6025, 0.59, 0.6075, 0.5975, 0.615, 0.62, 0.6175, 0.635, 0.6275, 0.635, 0.6375, 0.635, 0.645, 0.6475, 0.655, 0.655, 0.6575, 0.6625, 0.665, 0.675, 0.68, 0.675, 0.685, 0.685, 0.7125, 0.715, 0.715, 0.7275]
    maldi = [0.0, 0.0, 0.338, 0.248, 0.486, 0.336, 0.302, 0.338, 0.512, 0.442, 0.47, 0.512, 0.552, 0.508, 0.556, 0.592, 0.59, 0.61, 0.606, 0.658, 0.664, 0.636, 0.658, 0.684, 0.7, 0.71, 0.73, 0.732, 0.74, 0.742, 0.76, 0.754, 0.768, 0.774, 0.774, 0.786, 0.79, 0.816, 0.8, 0.81, 0.8, 0.804, 0.814, 0.824, 0.832, 0.832, 0.85, 0.844, 0.862, 0.86, 0.864, 0.878, 0.88, 0.89, 0.898, 0.886, 0.9, 0.91, 0.906, 0.91, 0.924, 0.914, 0.9, 0.922, 0.936, 0.93, 0.926, 0.928, 0.93, 0.944, 0.938, 0.944, 0.948, 0.952, 0.96, 0.956, 0.952, 0.946, 0.944, 0.956, 0.95, 0.946, 0.954, 0.954, 0.956, 0.952, 0.95, 0.942, 0.952, 0.95, 0.954, 0.956, 0.958, 0.956, 0.96, 0.956, 0.962, 0.966, 0.966, 0.97, 0.964, 0.972, 0.968, 0.97, 0.97, 0.972, 0.972, 0.976, 0.974, 0.972, 0.972, 0.974, 0.972, 0.974, 0.976, 0.974, 0.974, 0.974, 0.976, 0.978, 0.98, 0.978, 0.98, 0.98, 0.98, 0.98, 0.978, 0.982, 0.984, 0.98, 0.984, 0.984, 0.982, 0.984, 0.986, 0.982, 0.986, 0.986, 0.986, 0.986, 0.988, 0.988, 0.99, 0.99, 0.99, 0.99, 0.992, 0.99, 0.992, 0.992, 0.992, 0.992, 0.992, 0.994, 0.994, 0.994, 0.994, 0.992, 0.992, 0.994, 0.992, 0.992, 0.994, 0.992, 0.994, 0.996, 0.996, 0.994, 0.994, 0.992, 0.994, 0.994, 0.994, 0.994, 0.996, 0.996, 0.996, 0.996, 0.996, 0.996, 0.996, 0.996, 0.996, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994]

    legends = [
        'Nickel borylation, top-3 accuracy, Bayes UCB (beta prior)',
        'Deoxyfluorination, top-2 accuracy, Bayes UCB (gaussian prior)',
        'C-N cross-coupling, top-1 accuracy, UCB1'
    ]

    plt.rcParams['savefig.dpi'] = 300
    for p, l in zip([nib, deoxyf, maldi], legends):
        plt.plot(p[:100], label=l)
    plt.grid(visible=True, which='both', alpha=0.5)
    plt.legend(title='dataset, metric, algorithm', bbox_to_anchor=(0.5, -0.2), loc="upper center", fancybox=True)
    plt.xlabel('number of experiments')
    plt.ylabel('top-n accuracy')
    plt.title('Accuracy of identifying top general conditions for 3 datasets')
    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    import pickle

    def nib():
        dd = 'dataset_logs/nib/etoh-60cutoff/'
        num_sims = 500
        num_round = 75
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'ts_gaussian_squared-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ts_beta-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_squared_c=2-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_beta-{num_sims}s-{num_round}r-{num_exp}e',
                    f'new_bayes_ucb_beta-{num_sims}s-{num_round}r-{num_exp}e',
                    f'eps_greedy_annealing-{num_sims}s-{num_round}r-{num_exp}e',
                    f'random-{num_sims}s-{num_round}r-{num_exp}e',
                    ]]
        legend_list = ['TS (normal prior)',
                       'TS (beta prior)',
                       'ucb1-tuned',
                       'ucb1',
                       'Bayes ucb (normal prior)',
                       'Bayes ucb (beta prior, 2SD)',
                       'Bayes ucb (beta prior, ppf)',
                       'Annealing ε-greedy',
                       'pure exploration',]
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/nib-etoh.csv'
        with open(f'{dd}random-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        topsix = ['PPh2Cy', 'CX-PCy', 'PPh3', 'P(p-F-Ph)3', 'P(p-Anis)3', 'Cy-JohnPhos']
        ligands = [(l,) for l in topsix]
        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=True,
                               etc_fp=f'{dd}etc.npy',
                               shade_first_rounds=23,
                               title=f'Accuracy of identifying optimal ligands',
                               legend_title='algorithm',
                               long_legend=True)

        # plot_cumulative_reward(fn_list=fn_list,
        #                        legend_list=legend_list,
        #                        title='Cumulative reward',
        #                        legend_title='algorithm',
        #                        shade_first_rounds=23,
        #                        long_legend=True,
        #                        etc_baseline=True,
        #                        etc_fp=f'{dd}etc_cumu_reward.npy')
        return

    def nib_50(top=3):
        dd = 'dataset_logs/nib/etoh-50cutoff/'
        num_sims = 500
        num_round = 75
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'ts_gaussian_squared-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ts_beta-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_squared_c=2-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_beta_c=2-{num_sims}s-{num_round}r-{num_exp}e',
                    f'new_bayes_ucb_beta-{num_sims}s-{num_round}r-{num_exp}e',
                    f'eps_greedy_annealing-{num_sims}s-{num_round}r-{num_exp}e',
                    f'random-{num_sims}s-{num_round}r-{num_exp}e',
                    ]]
        legend_list = ['TS (normal prior)',
                       'TS (beta prior)',
                       'ucb1-tuned',
                       'ucb1',
                       'Bayes ucb (normal prior)',
                       'Bayes ucb (beta prior, 2SD)',
                       'Bayes ucb (beta prior, ppf)',
                       'Annealing ε-greedy',
                       'pure exploration',]
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/nib-etoh.csv'
        with open(f'{dd}random-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]

        top_three = ['Cy-JohnPhos', 'P(p-Anis)3', 'PPh2Cy']
        top_eight = ['PPh2Cy', 'CX-PCy', 'PPh3', 'P(p-F-Ph)3', 'P(p-Anis)3', 'Cy-JohnPhos', 'A-paPhos',
                     'Cy-PhenCar-Phos']
        if top == 3:
            ligands = [(l,) for l in top_three]
        elif top == 8:
            ligands = [(l,) for l in top_eight]
        else:
            exit()
        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=True,
                               etc_fp=f'{dd}top{top}.npy',
                               shade_first_rounds=23,
                               title=f'Accuracy of identifying top{top} optimal ligands',
                               legend_title='algorithm',
                               long_legend=True)

        # plot_cumulative_reward(fn_list=fn_list,
        #                        legend_list=legend_list,
        #                        title='Cumulative reward',
        #                        legend_title='algorithm',
        #                        shade_first_rounds=23,
        #                        long_legend=True,
        #                        etc_baseline=True,
        #                        etc_fp=f'{dd}etc_cumu_reward.npy')
        return

    def deoxyf():
        dd = 'dataset_logs/deoxyf/combo/'
        num_sims = 400
        num_round = 150
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'ts_gaussian_squared-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ts_gaussian_assumed_sd_0.25-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_squared_c=2-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-100r-{num_exp}e',
                    f'eps_greedy_annealing-{num_sims}s-{num_round}r-{num_exp}e',
                    ]]
        legend_list = ['TS (squared)',
                       'TS (fixed sd 0.25)',
                       'ucb1-tuned',
                       'Bayes ucb (2SD, squared)',
                       'Bayes ucb (2SD, 0.25)',
                       'ε-greedy',]
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv'
        with open(f'{dd}random-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        ligands = [('BTMG', 'PBSF'), ('BTPP', 'PBSF')]
        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=True,
                               etc_fp=f'{dd}etc/pbsf_btpp_btmg.npy',
                               shade_first_rounds=20,
                               title=f'Accuracy of identifying {ligands} as optimal',
                               legend_title='algorithm',
                               long_legend=False,
                               max_horizon_plot=100)

        # plot_cumulative_reward(fn_list=fn_list,
        #                        legend_list=legend_list,
        #                        title='Cumulative reward',
        #                        legend_title='algorithm',
        #                        shade_first_rounds=23,
        #                        long_legend=True,
        #                        etc_baseline=True,
        #                        etc_fp=f'{dd}etc_cumu_reward.npy')
        return

    def deoxyf_batch_interpolation():
        dd = 'dataset_logs/deoxyf/combo/interpolation/'
        num_sims = 200
        num_round = 100
        batch_sizes = [2,3,4,5,6]
        fn_list = [f'{dd}ucb1tuned-{num_sims}s-{num_round}r-{b}b/log.csv' for b in batch_sizes]
        fn_list = [f'{dd}/../ucb1tuned-400s-150r-1e/log.csv'] + fn_list
        legend_list = [1]+batch_sizes
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv'
        with open(f'/Users/mac/Desktop/project deebo/deebo/deebo/dataset_logs/deoxyf/combo/ucb1tuned-400s-150r-1e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        ligands = [('BTMG', 'PBSF'), ('BTPP', 'PBSF')]
        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               shade_first_rounds=20,
                               title=f'Accuracy of identifying {ligands} as optimal',
                               legend_title='batch size',
                               xlabel='number of experiments',
                               long_legend=False,
                               max_horizon_plot=100)

        # plot_cumulative_reward(fn_list=fn_list,
        #                        legend_list=legend_list,
        #                        shade_first_rounds=20,
        #                        title=f'Cumu reward of identifying {ligands} as optimal',
        #                        legend_title='algorithm',
        #                        long_legend=False,)

    def deoxyf_batch_real_time():
        # being lazy. data here are extracted when the original accuracy plotting function was used.
        one = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.0675, 0.0675, 0.0375, 0.06,
         0.1075, 0.2275, 0.2, 0.23, 0.2325, 0.2275, 0.2225, 0.255, 0.235, 0.26, 0.2525, 0.28, 0.28, 0.3025, 0.325, 0.33,
         0.3275, 0.305, 0.325, 0.3725, 0.325, 0.3525, 0.345, 0.3475, 0.3575, 0.365, 0.3475, 0.3775, 0.35, 0.385, 0.365,
         0.3875, 0.3825, 0.3975, 0.41, 0.4125, 0.4075, 0.42, 0.41, 0.4225, 0.4225, 0.415, 0.44, 0.43, 0.4475, 0.46,
         0.46, 0.4425, 0.46, 0.455, 0.475, 0.4525, 0.465, 0.48, 0.495, 0.4875, 0.5125, 0.495, 0.495, 0.52, 0.53, 0.545,
         0.5325, 0.535, 0.51, 0.5525, 0.545, 0.55, 0.5625, 0.55, 0.5625, 0.5575, 0.565, 0.565, 0.58, 0.5825, 0.58,
         0.6075, 0.58, 0.585, 0.585, 0.5975, 0.615, 0.6125, 0.6075, 0.6175, 0.62, 0.6125, 0.6375, 0.63, 0.64, 0.63,
         0.63, 0.65, 0.6375, 0.65, 0.6425, 0.65, 0.6475, 0.6375, 0.6475, 0.6475, 0.655, 0.66, 0.6575, 0.6575, 0.6675,
         0.6825, 0.675, 0.6675, 0.685, 0.6775, 0.6725, 0.6675, 0.685, 0.68, 0.6875, 0.6725, 0.6975, 0.7025, 0.6925, 0.7,
         0.7075, 0.71, 0.7225, 0.72, 0.7075, 0.7175, 0.705, 0.725, 0.7175]
        one = one[:100]

        two = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.055, 0.065, 0.08, 0.065, 0.08, 0.095,
         0.165, 0.195, 0.185, 0.2, 0.17, 0.205, 0.17, 0.22, 0.235, 0.265, 0.26, 0.25, 0.265, 0.265, 0.28, 0.27, 0.27,
         0.305, 0.305, 0.305, 0.3, 0.305, 0.315, 0.31, 0.36, 0.325, 0.335, 0.33, 0.38, 0.395, 0.305, 0.355, 0.32, 0.38,
         0.415, 0.38, 0.385, 0.39, 0.405, 0.43, 0.41, 0.41, 0.4, 0.385, 0.4, 0.39, 0.41, 0.38, 0.4, 0.415, 0.41, 0.425,
         0.44, 0.415, 0.445, 0.435, 0.44, 0.435, 0.445, 0.47, 0.47, 0.46, 0.485, 0.485, 0.475, 0.5, 0.495, 0.525, 0.51,
         0.53, 0.52, 0.51, 0.53, 0.535, 0.52, 0.55, 0.53, 0.55, 0.55, 0.595]

        three = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.06, 0.065, 0.04, 0.07, 0.125,
         0.12, 0.195, 0.21, 0.24, 0.225, 0.23, 0.235, 0.265, 0.26, 0.275, 0.31, 0.31, 0.34, 0.33, 0.295, 0.32, 0.33,
         0.295, 0.335, 0.365, 0.335, 0.385, 0.355, 0.34, 0.34, 0.365, 0.375, 0.39, 0.395, 0.43, 0.42, 0.48, 0.43, 0.445,
         0.43, 0.47, 0.47, 0.44, 0.465, 0.465, 0.485, 0.5, 0.47, 0.495, 0.485, 0.495, 0.5, 0.49, 0.515, 0.53, 0.545,
         0.5, 0.565, 0.535, 0.555, 0.535, 0.54, 0.545, 0.57, 0.53, 0.565, 0.57, 0.59, 0.595, 0.585, 0.57, 0.585, 0.595,
         0.6, 0.605, 0.625, 0.62, 0.59, 0.61, 0.62, 0.6, 0.61, 0.62, 0.645, 0.6]

        four = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.065, 0.095, 0.08, 0.03, 0.06, 0.09,
         0.18, 0.165, 0.16, 0.215, 0.17, 0.24, 0.235, 0.295, 0.27, 0.295, 0.3, 0.305, 0.225, 0.265, 0.265, 0.295, 0.27,
         0.27, 0.305, 0.27, 0.29, 0.275, 0.305, 0.345, 0.325, 0.34, 0.32, 0.315, 0.34, 0.365, 0.345, 0.355, 0.37, 0.39,
         0.41, 0.415, 0.4, 0.415, 0.405, 0.395, 0.42, 0.405, 0.415, 0.4, 0.415, 0.4, 0.4, 0.385, 0.415, 0.41, 0.415,
         0.43, 0.42, 0.43, 0.415, 0.42, 0.465, 0.445, 0.43, 0.43, 0.445, 0.44, 0.435, 0.43, 0.455, 0.465, 0.46, 0.44,
         0.47, 0.48, 0.48, 0.505, 0.485, 0.485, 0.52, 0.5, 0.52, 0.53, 0.51, 0.54]

        five = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.055, 0.045, 0.035, 0.04, 0.115,
         0.245, 0.215, 0.195, 0.22, 0.205, 0.235, 0.26, 0.255, 0.3, 0.265, 0.285, 0.29, 0.34, 0.34, 0.325, 0.34, 0.37,
         0.36, 0.4, 0.37, 0.375, 0.385, 0.425, 0.46, 0.44, 0.46, 0.475, 0.48, 0.475, 0.47, 0.41, 0.45, 0.43, 0.465,
         0.45, 0.465, 0.475, 0.465, 0.475, 0.445, 0.49, 0.485, 0.5, 0.51, 0.49, 0.495, 0.49, 0.52, 0.48, 0.52, 0.55,
         0.505, 0.55, 0.56, 0.52, 0.525, 0.54, 0.53, 0.535, 0.515, 0.52, 0.55, 0.585, 0.57, 0.57, 0.585, 0.565, 0.56,
         0.56, 0.585, 0.62, 0.6, 0.62, 0.605, 0.6, 0.625, 0.635, 0.635, 0.63, 0.645]

        six = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045, 0.045, 0.04, 0.045, 0.03, 0.12,
         0.105, 0.16, 0.125, 0.115, 0.17, 0.22, 0.235, 0.25, 0.255, 0.265, 0.3, 0.345, 0.365, 0.35, 0.335, 0.34, 0.355,
         0.33, 0.365, 0.385, 0.38, 0.355, 0.375, 0.37, 0.405, 0.395, 0.41, 0.39, 0.375, 0.455, 0.45, 0.435, 0.45, 0.45,
         0.425, 0.46, 0.475, 0.455, 0.435, 0.45, 0.45, 0.46, 0.435, 0.46, 0.46, 0.445, 0.47, 0.465, 0.465, 0.465, 0.465,
         0.475, 0.475, 0.47, 0.485, 0.495, 0.49, 0.47, 0.48, 0.515, 0.5, 0.505, 0.51, 0.505, 0.515, 0.535, 0.53, 0.57,
         0.575, 0.55, 0.59, 0.59, 0.585, 0.6, 0.575, 0.61, 0.605, 0.6, 0.62, 0.615]

        def highest_every_n_element(n, arr):
            # for an array, find the highest value every n element
            n_round = len(arr) // n
            n_residual = len(arr) % n
            maxes = []
            for ii in range(n_round):
                maxes.append(max(arr[ii*n:(ii+1)*n]))
            if n_residual != 0:
                maxes.append(max(arr[-n_residual:]))
            return maxes

        batch_sizes = [1,2,3,4,5,6]
        data = [one, two, three, four, five, six]
        d = dict(zip(batch_sizes, data))
        for b in batch_sizes:
            plt.plot(highest_every_n_element(b, d[b]))
        plt.legend(batch_sizes, title='batch size')
        plt.grid(visible=True, which='both', alpha=0.5)
        plt.title('Accuracy of identifying [(\'BTMG\', \'PBSF\'), (\'BTPP\', \'PBSF\')] as optimal')
        plt.ylabel('Accuracy of identifying best arm: [14, 19]')
        plt.xlabel('Actual time horizon (number of rounds)')
        plt.show()

    def cn():
        dd = 'dataset_logs/cn/'
        num_sims = 500
        num_round = 100
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'ts_gaussian_squared-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ts_gaussian_assumed_sd_0.25-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_squared_c=2-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-100r-{num_exp}e',
                    f'eps_greedy_annealing-{num_sims}s-{num_round}r-{num_exp}e',
                    ]]
        legend_list = ['TS (squared)',
                       'TS (fixed sd 0.25)',
                       'ucb1-tuned',
                       'Bayes ucb (2SD, squared)',
                       'Bayes ucb (2SD, 0.25)',
                       'ε-greedy',]
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/cn-processed.csv'
        with open(f'{dd}ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        ligands = [('MTBD', 'tBuXPhos'),
                   ('MTBD', 'tBuBrettPhos'),
                   ('MTBD', 'AdBrettPhos')]
        # ligands = [('MTBD', 'tBuXPhos'),
        #            ('MTBD', 'tBuBrettPhos')]
        # ligands = [('MTBD', 'tBuXPhos'),]
        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=False,
                               etc_fp=f'{dd}/etc/top3.npy',
                               shade_first_rounds=12,
                               ignore_first_rounds=12,
                               title=f'Accuracy of identifying {ligands} as optimal',
                               legend_title='algorithm',
                               long_legend=False,
                               max_horizon_plot=100,
                               vlines=[36, 72],
                               hlines=[0.96])

    def aryl(top=1):
        dd = 'dataset_logs/aryl-scope-ligand/'
        num_sims = 500
        num_round = 200
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'ts_gaussian_squared-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ts_gaussian_assumed_sd_0.25-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_squared_c=2-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-150r-{num_exp}e',
                    f'eps_greedy_annealing-{num_sims}s-{num_round}r-{num_exp}e',
                    f'random-{num_sims}s-{num_round}r-{num_exp}e'
                    ]]
        legend_list = ['TS (squared)',
                       'TS (fixed sd 0.25)',
                       'ucb1-tuned',
                       'ucb1',
                       'Bayes ucb (2SD, squared)',
                       'Bayes ucb (2SD, 0.25)',
                       'ε-greedy',
                       'pure exploration']
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv'
        with open(f'{dd}ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        top1 = ['Cy-BippyPhos']
        top5 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos']
        top9 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos',
                'Cy-vBRIDP', 'Cy-DavePhos', 'X-Phos', 'CX-PICy']
        if top == 1:
            ligands = [(l,) for l in top1]
        elif top == 5:
            ligands = [(l,) for l in top5]
        elif top == 9:
            ligands = [(l,) for l in top9]
        else:
            exit()

        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=True,
                               etc_fp=f'{dd}/etc/top{top}.npy',
                               shade_first_rounds=24,
                               ignore_first_rounds=0,
                               title=f'Accuracy of identifying top {top} optimal ligands',
                               legend_title='algorithm',
                               long_legend=True,
                               max_horizon_plot=150,)

    def amidation(top=1, combo=True):
        if combo:
            dd = 'dataset_logs/amidation/combo/'
            shade_n = 32
            what = 'activator/base'  # for title of the plot
        else:
            dd = 'dataset_logs/amidation/activator/'
            shade_n = 8
            what = 'activator'
        num_sims = 500
        num_round = 96
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'ts_gaussian_squared-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ts_gaussian_assumed_sd_0.25-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_squared_c=2-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
                    f'eps_greedy_annealing-{num_sims}s-{num_round}r-{num_exp}e',
                    f'random-{num_sims}s-{num_round}r-{num_exp}e'
                    ]]
        legend_list = ['TS (squared)',
                       'TS (fixed sd 0.25)',
                       'ucb1-tuned',
                       'ucb1',
                       'Bayes ucb (2SD, squared)',
                       'Bayes ucb (2SD, 0.25)',
                       'ε-greedy',
                       'pure exploration']
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/ami.csv'
        with open(f'{dd}ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        if not combo:  # just activators
            top1 = ['DPPCl']
            top3 = ['DPPCl', 'BOP-Cl', 'TCFH']
            if top == 1:
                ligands = [(l,) for l in top1]
            elif top == 3:
                ligands = [(l,) for l in top3]
            else:
                exit()
        else:  # base + activators
            top1 = [('DPPCl', 'N-methylmorpholine')]
            top2 = [('DPPCl', 'N-methylmorpholine'),
                    ('DPPCl', 'Diisopropylethylamine')]
            if top == 1:
                ligands = top1
            elif top == 2:
                ligands = top2
            else:
                exit()

        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=True,
                               etc_fp=f'{dd}/etc/top{top}.npy',
                               shade_first_rounds=shade_n,
                               ignore_first_rounds=0,
                               title=f'Accuracy of identifying top {top} {what}',
                               legend_title='algorithm',
                               long_legend=True,)

    def aryl_sampling_mode_from_legacy(top=1):
        dd = 'dataset_logs/aryl-scope-ligand-legacy/'
        num_sims = 400
        num_round = 100
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'BayesUCBGaussian-{num_sims}s-{num_round}r-{num_exp}e',
                    f'sampling mode test/BayesUCBGaussian-{num_sims}s-{num_round}r-{num_exp}e-highest',
                    f'sampling mode test/BayesUCBGaussian-{num_sims}s-{num_round}r-{num_exp}e-random-highest',
                    ]]
        legend_list = ['random', 'highest', 'random from top 5 highest']
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv'
        with open(f'{dd}BayesUCBGaussian-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        top1 = ['Cy-BippyPhos']
        top5 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos']
        top9 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos',
                'Cy-vBRIDP', 'Cy-DavePhos', 'X-Phos', 'CX-PICy']
        if top == 1:
            ligands = [(l,) for l in top1]
        elif top == 5:
            ligands = [(l,) for l in top5]
        elif top == 9:
            ligands = [(l,) for l in top9]
        else:
            exit()

        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=False,
                               etc_fp=f'{dd}/etc/top{top}.npy',
                               shade_first_rounds=24,
                               ignore_first_rounds=0,
                               title=f'Accuracy of identifying top {top} optimal ligands',
                               legend_title='algorithm',
                               long_legend=True,
                               max_horizon_plot=150,)

    def aryl_scope_expansion():
        num_sims = 500
        num_round = 300
        num_exp = 1
        dd = f'./dataset_logs/aryl-scope-ligand/expansion/scenario1/ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e/'
        log_fp = f'{dd}log.csv'
        legend_list = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos']
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv'
        with open(f'{dd}arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        top1 = ['Cy-BippyPhos']
        top5 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos']
        top9 = ['Cy-BippyPhos', 'Et-PhenCar-Phos', 'tBPh-CPhos', 'CgMe-PPh', 'JackiePhos',
                'Cy-vBRIDP', 'Cy-DavePhos', 'X-Phos', 'CX-PICy']

        indexes = [reverse_arms_dict[(l,)] for l in legend_list]
        baseline_indexes = [reverse_arms_dict[('Cy-BippyPhos',)]]
        baseline_fp = '/Users/mac/Desktop/project deebo/deebo/deebo/dataset_logs/aryl-scope-ligand/ucb1tuned-500s-200r-1e/log.csv'

        plot_accuracy_best_arm_scope_expansion(arm_indexes=indexes,
                                               fp=log_fp,
                                               baseline_fp=baseline_fp,
                                               baseline_arm_indexes=baseline_indexes,
                                               legend_list=legend_list,
                                               shade_first_rounds=24,
                                               ignore_first_rounds=0,
                                               title=f'Accuracy of identifying ligand as optimal',
                                               legend_title='ligand',
                                               long_legend=True,
                                               max_horizon_plot=200,
                                               vlines=[50, 100])

    def maldi():
        name = 'amine'
        dd = f'dataset_logs/merck-maldi/{name}/'
        num_sims = 500
        num_round = 200
        num_exp = 1
        fn_list = [f'{dd}{n}/log.csv' for n in
                   [f'ts_gaussian_squared-{num_sims}s-190r-{num_exp}e',
                    f'ts_gaussian_assumed_sd_0.25-{num_sims}s-190r-{num_exp}e',
                    f'ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e',
                    f'ucb1-{num_sims}s-{num_round}r-{num_exp}e',
                    f'bayes_ucb_gaussian_squared_c=2-{num_sims}s-190r-{num_exp}e',
                    f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-190r-{num_exp}e',
                    f'eps_greedy_annealing-{num_sims}s-{num_round}r-{num_exp}e',
                    ]]
        legend_list = ['TS (squared)',
                       'TS (fixed sd 0.25)',
                       'ucb1-tuned',
                       'ucb1',
                       'Bayes ucb (2SD, squared)',
                       'Bayes ucb (2SD, 0.25)',
                       'ε-greedy',]
        #f'bayes_ucb_gaussian_c=2_assumed_sd=0.25-{num_sims}s-{num_round}r-{num_exp}e',
        # 'Bayes ucb (2SD, 0.25)',
        fp = f'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/maldi-{name}.csv'
        with open(f'{dd}ucb1tuned-{num_sims}s-{num_round}r-{num_exp}e/arms.pkl', 'rb') as f:
            arms_dict = pickle.load(f)

        reverse_arms_dict = {v: k for k, v in arms_dict.items()}
        # ligands = ['Cy-BippyPhos', 'CgMe-PPh', 'Et-PhenCar-Phos', 'JackiePhos', 'tBPh-CPhos']
        # ligands = ['Et-PhenCar-Phos', 'JackiePhos']
        #ligands = [(b,) for b in bs]
        if name == 'bromide':
            ligands = [('Cu',)]
        elif name == 'amine':
            ligands = [('Pd',)]
        else:
            exit()
        # ligands = [('MTBD', 'tBuXPhos'),
        #            ('MTBD', 'tBuBrettPhos')]
        # ligands = [('MTBD', 'tBuXPhos'),]
        indexes = [reverse_arms_dict[l] for l in ligands]

        plot_accuracy_best_arm(best_arm_indexes=indexes,
                               fn_list=fn_list,
                               legend_list=legend_list,
                               etc_baseline=False,
                               etc_fp=f'{dd}/etc/top3.npy',
                               ignore_first_rounds=4,
                               title=f'Accuracy of identifying optimal conditions for {name}',
                               legend_title='algorithm',
                               long_legend=False,
                               max_horizon_plot=200,
                               vlines=None,
                               hlines=None)

    deoxyf_batch_real_time()
    # plot_arm_counts('dataset_logs/aryl-scope-ligand/BayesUCBGaussian-400s-200r-1e', top_n=10, bar_errbar=True, plot='box', title='Average # of samples')

    # plot_arm_rewards(fp, d='dataset_logs/aryl-scope-ligand/BayesUCBGaussian-400s-200r-1e', top_n=10)

    # make_heatmap_gif(plot_acquisition_history_heatmap_deoxyf,
    #                  n_sim=0,
    #                  max_n_round=100,
    #                  binary=False,
    #                  history_fp=f'{dd}etc-1s-73r-1e/history.csv',
    #                  save_fp=f'test/test.gif')
