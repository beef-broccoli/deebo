import sys

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from glob import glob
from utils import plot_info_file_path_match

# check chem_analyze for more up-to-date function with more functionalities


def plot_probs_choosing_best_arm(fn_list,
                                 legend_list,
                                 hline=0,
                                 vline=0,
                                 etc_baseline=False,
                                 etc_fp='',
                                 best_arm_index=0,
                                 fp='',
                                 title='',
                                 legend_title='',
                                 long_legend=False,
                                 ignore_first_rounds=0):
    """
    plot the probabilities of choosing the optimal arm for a list of algorithms using acquisition logs.

    Parameters
    ----------
    fn_list: Collection of str
        list of data file names
    legend_list: Collection of str
        list of labels for legend
    hline: int/float
        value for plotting horizontal baseline
    vline: int/float
        value for plotting a vertical baseline
    etc_baseline: bool
        display explore-then-commit baseline or not
    etc_fp: str
        file path for calculated etc baseline at each time point, a numpy array object
    best_arm_index: int or list-like
        a single index for best arm (needed for calculation), or a list of indexes if best arms are different
    fp: str
        deepest common directory for where the data files are stored
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
    None

    """

    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    if hline != 0:
        plt.axhline(y=hline, xmin=0, xmax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)
    if vline !=0:
        plt.axvline(x=vline, ymin=0, ymax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)

    if etc_baseline:
        base = np.load(etc_fp)
        plt.plot(np.arange(len(base))[ignore_first_rounds:], base[ignore_first_rounds:], color='black', label='explore-then-commit', lw=2)

    if isinstance(best_arm_index, int):
        best_arm_index = [best_arm_index]*len(fps)  # best_arm_index supplied as a single number, duplicate into list
    else:
        assert len(best_arm_index) == len(fps), \
            'if best_arm_index is supplied as a list, its length needs to match the number of files'

    for i in range(len(fps)):
        fp = fps[i]
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'chosen_arm']]

        n_simulations = int(np.max(df['num_sims'])) + 1
        time_horizon = int(np.max(df['horizon'])) + 1
        all_arms = np.zeros((n_simulations, time_horizon))

        for ii in range(int(n_simulations)):
            all_arms[ii, :] = list(df.loc[df['num_sims'] == ii]['chosen_arm'])

        counts = np.count_nonzero(all_arms == best_arm_index[i], axis=0)  # average across simulations. shape: (1, time_horizon)
        probs = counts / n_simulations
        ax.plot(np.arange(time_horizon)[ignore_first_rounds:], probs[ignore_first_rounds:], label=str(legend_list[i]))

    ax.set_xlabel('time horizon')
    ax.set_ylabel('probability of finding best arm')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    if long_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
    else:
        ax.legend(title=legend_title)

    plt.show()
    return None


def plot_average_reward(fn_list,
                        legend_list,
                        baseline=0,
                        show_se=False,
                        fp='',
                        title='',
                        legend_title='',
                        long_legend=False):

    """
    plot the average reward at each time point for a list of algorithms using acquisition logs.

    Parameters
    ----------
    fn_list: Collection of str
        list of data file names
    legend_list: Collection of str
        list of labels for legend
    baseline: int or float
        horizontal baseline
    show_se: bool
        show the standard error interval or not
    fp: str
        the deepest common directory, this is just a convenience to be used with fn_list
    title: str
        title for the plot
    legend_title: str
        title for the legend
    long_legend: bool
        if true, legend will be plotted outside the plot; if false mpl finds the best position within plot

    Returns
    -------
    None

    """


    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    if baseline != 0:
        plt.axhline(y=baseline, xmin=0, xmax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)

    for i in range(len(fps)):
        fp = fps[i]
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'reward']]

        n_simulations = int(np.max(df['num_sims']))+1
        time_horizon = int(np.max(df['horizon']))+1
        all_rewards = np.zeros((n_simulations, time_horizon))

        for ii in range(int(n_simulations)):
            all_rewards[ii, :] = list(df.loc[df['num_sims'] == ii]['reward'])

        avg_reward = np.average(all_rewards, axis=0)  # average across simulations. shape: (1, time_horizon)
        interval = stats.sem(all_rewards, axis=0)  # standard error
        lower_bound = avg_reward - interval
        upper_bound = avg_reward + interval
        xs = np.arange(time_horizon)
        ax.plot(xs, avg_reward, label=str(legend_list[i]))
        if show_se:  # makes me dizzy; se too small
            ax.fill_between(xs, lower_bound, upper_bound, alpha=0.3)

    ax.set_xlabel('time horizon')
    ax.set_ylabel('average reward')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    if long_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
    else:
        ax.legend(title=legend_title)

    plt.show()
    return None


# maybe: baseline cumu reward with ETC
def plot_cumulative_reward(fn_list,
                           legend_list,
                           fp='',
                           title='',
                           legend_title=''):

    """
    plot the cumulative reward up to each time point for a list of algorithms using acquisition logs.

    Parameters
    ----------
    fn_list: Collection of str
        list of data file names
    legend_list: Collection of str
        list of labels for legend
    fp: str
        the deepest common directory, this is just a convenience to be used with fn_list
    title: str
        title for the plot
    legend_title: str
        title for the legend

    Returns
    -------
    None

    """

    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

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

        for ii in range(int(n_simulations)):
            rewards = df.loc[df['num_sims'] == ii]['cumulative_reward'].to_numpy()
            all_rewards[ii, :] = rewards

        probs = np.average(all_rewards, axis=0)  # average across simulations. shape: (1, time_horizon)
        ax.plot(np.arange(time_horizon), probs, label=str(legend_list[i]))

    ax.set_xlabel('time horizon')
    ax.set_ylabel('cumulative reward')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.legend(title=legend_title, loc='upper left')

    plt.show()
    return None


def _plot_etc_baseline(explore_times,
                       fn_list,
                       legend_list,
                       best_arm_index=0,
                       fp='',
                       title='',
                       legend_title='',
                       long_legend=False,
                       ):
    """
    Deprecated. Used to analyze the ETC logs and plot ETC baseline.
    There is better ways to get the numbers directly during simulation.

    Parameters
    ----------
    explore_times: a list of total number of exploration rounds
    fn_list
    legend_list
    best_arm_index
    fp
    title
    legend_title
    long_legend

    Returns
    -------

    """

    # file name needs to be in a sequence where small # is first

    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    last_counts = np.array([])
    n_simulations = 0
    time_horizon = 0
    for i in range(len(fps)):
        df = pd.read_csv(fps[i])
        df = df[['num_sims', 'horizon', 'chosen_arm', 'reward']]
        n_simulations = int(np.max(df['num_sims'])) + 1
        time_horizon = int(np.max(df['horizon'])) + 1
        all_arms = np.zeros((n_simulations, time_horizon))
        for ii in range(int(n_simulations)):
            all_arms[ii, :] = list(df.loc[df['num_sims'] == ii]['chosen_arm'])
        counts = np.count_nonzero(all_arms == best_arm_index, axis=0)  # average across simulations. shape: (1, time_horizon)

        # set counts from the exploration counts to 0
        counts[:explore_times[i]] = 0

        # combine counts with
        if i != 0:
            last_counts[counts.astype('bool')] = 0
            last_counts = last_counts + counts
        else:
            last_counts = counts

    probs = last_counts / n_simulations
    ax.plot(np.arange(time_horizon), probs)
    np.save(fp+'baseline.npy', probs)

    ax.set_xlabel('time horizon')
    ax.set_ylabel('probability of finding best arm')
    ax.set_title(title)
    ax.grid(visible=True, which='both', alpha=0.5)
    if long_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
    else:
        ax.legend(title=legend_title)

    plt.show()

    return None


def plot_probs_choosing_best_arm_all(folder_path=None):
    """
    Function that can more efficiently plot results for each algo with all parameters
    This is written for results for specific test scenario, where results for all algorithms are in the same folder
    This checks the folder name and its file path, and fetches the pre-set parameter and plots all results in that folder
    Only works with preset Bernoulli testings done (scenario 1-5)

    Parameters
    ----------
    folder_path: file path for the folder where all results are stored.

    Returns
    -------
    title (str), legend title (str) and best arm index (int)

    """
    if not folder_path:
        sys.exit()

    if not folder_path.endswith('/'):
        folder_path = folder_path + '/'

    fn_list = sorted(glob(f'{folder_path}*.csv'))
    legend_list = [fn[len(folder_path):-len('.csv')] for fn in fn_list]

    title, legend_title, best_arm_index = plot_info_file_path_match(folder_path)

    plot_probs_choosing_best_arm(fn_list,
                                 legend_list,
                                 best_arm_index=best_arm_index,
                                 fp='./',
                                 title=f'Accuracy of {title}',
                                 legend_title=f'{legend_title}',
                                 long_legend=True)

    return None


def _test_plot():

    # example on using plot function
    plt.rcParams['savefig.dpi'] = 300

    eps = [0.1, 0.2, 0.3, 0.4, 0.5]
    fn_list = ['epsilon_' + str(e) + '.csv' for e in eps]
    fn_list.append('annealing_epsilon_greedy.csv')
    legend_list = [str(e) for e in eps]
    legend_list.append('jdk')

    plot_cumulative_reward(fn_list, legend_list, fp='./logs/epsilon_greedy_test/', title='ss', legend_title='dd')


def _plot_boltzmann():
    # example on using plot function
    plt.rcParams['savefig.dpi'] = 300

    taus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    fn_list = ['tau_' + str(t) + '.csv' for t in taus]
    fn_list.append('annealing_boltzmann_test.csv')
    legend_list = [str(t) for t in taus]
    legend_list.append('annealing')

    plot_probs_choosing_best_arm(fn_list, legend_list, best_arm_index=4, fp='./logs/Boltzmann_test/', title='accuracy of softmax', legend_title='tau')


def _deprecated_calculate_baseline(chemarms: list):
    # TODO: take chem arms, calculate a baseline for probability in a traditional reaction optimziation way
    from deebo.legacy.chem_arms_legacy import ChemArmRandomDraw

    # type check; check chem arms are from same dataset
    url = chemarms[0].data_url
    name = chemarms[0].name
    for arm in chemarms:
        assert isinstance(arm, ChemArmRandomDraw), "required argument: a list of ChemArm objects"
        assert arm.name == name, "ChemArmSim objects should describe same reaction components"
        assert arm.data_url == url, "ChemArmSim objects should come from the same dataset"

    df = pd.read_csv(url)

    temp = df[list(name)]

    return


def _deprecated_cal_baseline():

    from deebo.legacy.chem_arms_legacy import ChemArmRandomDraw
    import itertools

    # build chem arms
    dataset_url = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-conditions.csv'
    names = ('base_smiles', 'solvent_smiles')  # same names with column name in df
    base = ['O=C([O-])C.[K+]', 'O=C([O-])C(C)(C)C.[K+]']
    solvent = ['CC(N(C)C)=O', 'CCCC#N']
    vals = list(itertools.product(base, solvent))  # sequence has to match what's in "names"
    arms = list(map(lambda x: ChemArmRandomDraw(x, names, dataset_url), vals))

    # test basline
    #calculate_baseline(arms)


if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 300
    import itertools

    def scenario1_best_perfomers():
        prefix = 'logs/scenario1/'
        n_list = ['eps_greedy/annealing',
                  'softmax/tau_0.2',
                  'pursuit/lr_0.05',
                  'optim/ucb1_tuned',
                  'TS/TS_beta',
                  'TS/TS_gaussian_squared',
                  'optim/bayes_ucb_beta_c=2',
                  'optim/new_bayes_ucb_beta',
                  'optim/bayes_ucb_gaussian_c=2_squared',
                  'optim/bayes_ucb_gaussian_c=2_assumed_sd=0.25',
                  ]
        fn_list = [f'{prefix}{n}.csv' for n in n_list]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['eps greedy(annealing)',
                         'softmax (tau=0.2)',
                         'pursuit (lr=0.05)',
                         'ucb1-tuned',
                         'thompson sampling (beta prior)',
                         'thompson sampling (normal prior, squared)',
                         'bayes ucb (beta prior, 2SD)',
                         'bayes ucb (beta prior, ppf)',
                         'bayes ucb (normal prior, 2SD, squared)',
                         'bayes ucb (normal prior, 2SD, 0.25)',
                         ],
            etc_baseline=True,
            etc_fp=f'{prefix}baseline.npy',
            title='Accuracy of scenario 1 best performers',
            legend_title='algorithms',
            best_arm_index=4,
            long_legend=True,
            ignore_first_rounds=5
        )
        return None

    def scenario2_best_perfomers():
        prefix = 'logs/scenario2/'
        n_list = ['optim/ucb1_tuned',
                  'TS/TS_beta',
                  'TS/TS_gaussian_squared',
                  'optim/bayes_ucb_beta_c=1',
                  'optim/new_bayes_ucb_beta',
                  'optim/bayes_ucb_gaussian_c=1_squared',
                  'optim/bayes_ucb_gaussian_c=2_assumed_sd=0.25',
                  ]
        fn_list = [f'{prefix}{n}.csv' for n in n_list]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['ucb1-tuned',
                         'TS (beta prior)',
                         'TS (normal prior, squared)',
                         'bayes ucb (beta prior, 1SD)',
                         'bayes ucb (beta prior, ppf)',
                         'bayes ucb (normal prior, 1SD, squared)',
                         'bayes ucb (normal prior, 2SD, 0.25)',
                         ],
            etc_baseline=True,
            etc_fp=f'{prefix}baseline.npy',
            title='Accuracy of scenario 2 best performers',
            legend_title='algorithms',
            best_arm_index=4,
            long_legend=False,
            ignore_first_rounds=5
        )
        return None

    def scenario3_best_perfomers():
        prefix = 'logs/scenario3/'
        n_list = ['optim/ucb1_tuned',
                  'TS/TS_beta',
                  'TS/TS_gaussian_squared',
                  'optim/bayes_ucb_beta_c=2',
                  'optim/new_bayes_ucb_beta',
                  'optim/bayes_ucb_gaussian_c=2_squared',
                  'optim/bayes_ucb_gaussian_c=2_assumed_sd=0.25',
                  ]
        fn_list = [f'{prefix}{n}.csv' for n in n_list]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['ucb1-tuned',
                         'TS (beta prior)',
                         'TS (normal prior, squared)',
                         'bayes ucb (beta prior, 2SD)',
                         'bayes ucb (beta prior, ppf)',
                         'bayes ucb (normal prior, 2SD, squared)',
                         'bayes ucb (normal prior, 2SD, 0.25)',
                         ],
            etc_baseline=True,
            etc_fp=f'{prefix}baseline.npy',
            title='Accuracy of scenario 3 best performers',
            legend_title='algorithms',
            best_arm_index=4,
            long_legend=False,
            ignore_first_rounds=5
        )
        return None

    def scenario4_best_perfomers():
        prefix = 'logs/scenario4/'
        n_list = ['eps_greedy/annealing',
                  'pursuit/lr_0.025',
                  'pursuit/lr_0.05',
                  'optim/ucb1_tuned',
                  'optim/TS'
                  ]
        fn_list = [f'{prefix}{n}.csv' for n in n_list]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['eps greedy(annealing)',
                         'pursuit (lr=0.025)',
                         'pursuit (lr=0.05)',
                         'ucb1-tuned',
                         'thompson sampling (beta prior)'],
            etc_baseline=True,
            etc_fp=f'{prefix}baseline.npy',
            title='Accuracy of scenario 4 best performers',
            legend_title='algorithms',
            best_arm_index=8,
            long_legend=False,
            ignore_first_rounds=9
        )
        return None

    def scenario5_best_perfomers():
        prefix = 'logs/scenario5/'
        n_list = ['eps_greedy/annealing',
                  'pursuit/lr_0.025',
                  'optim/TS'
                  ]
        fn_list = [f'{prefix}{n}.csv' for n in n_list]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['eps greedy(annealing)',
                         'pursuit (lr=0.025)',
                         'thompson sampling (beta prior)'],
            etc_baseline=True,
            etc_fp=f'{prefix}baseline.npy',
            title='Accuracy of scenario 5 best performers',
            legend_title='algorithms',
            best_arm_index=18,
            long_legend=False,
            ignore_first_rounds=19,
        )
        return None

    def scalability():
        # # Average reward
        # fn_list = [f'logs/scalability/scenario{n}/optim/TS.csv' for n in [11, 12, 13, 14, 15]]
        # plot_average_reward(
        #     fn_list=fn_list,
        #     legend_list=['20', '50', '100', '500', '1000'],
        #     title='Average reward with TS (beta prior)',
        #     legend_title='# of arms',
        #     show_se=True,
        #     long_legend=False,
        # )

        # Accuracy
        fn_list = ['logs/scalability/scenario11/optim/TS-1000s-10000r.csv',
                   'logs/scalability/scenario12/optim/TS-1000s-10000r.csv',
                   'logs/scalability/scenario13/optim/TS-1000s-10000r.csv',
                   'logs/scalability/scenario14/optim/TS-1000s-15000r.csv',
                   'logs/scalability/scenario15/optim/TS-500s-15000r.csv',]
        fn_list = ['logs/scalability/scenario11/optim/ucb1_tuned-1000s-10000r.csv',
                   'logs/scalability/scenario12/optim/ucb1_tuned-1000s-10000r.csv',
                   'logs/scalability/scenario13/optim/ucb1_tuned-1000s-10000r.csv',]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['20', '50', '100', '500', '1000'],
            title='Accuracy with TS (beta prior)',
            legend_title='# of arms',
            best_arm_index=[19, 49, 99, 499, 999],
            long_legend=False,
        )
        plot_average_reward(
            fn_list=fn_list,
            legend_list=['20', '50', '100'],
            title='Accuracy with TS (beta prior)',
            legend_title='# of arms',
            long_legend=False,
        )
        return None

    def normal_scenario1_best_performers(sd=0.5):
        prefix = 'logs/normal arm/scenario1/'
        n_list = [f'eps_greedy_annealing_real_sd_{sd}',
                  f'ucb1tuned_real_sd_{sd}',
                  f'TS/real_sd_{sd}/assume_sd_0.25',
                  f'TS/TS_squared/realsd_{sd}',
                  f'BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_c=2_assumed_sd=0.25',
                  f'BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_squared_c=2',
                  ]
        fn_list = [f'{prefix}{n}.csv' for n in n_list]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['eps greedy (annealing)',
                         'ucb1-tuned',
                         'TS (fixed sd 0.25)',
                         'TS (squared)',
                         'bayes ucb (normal prior, 2SD, 0.25)',
                         'bayes ucb (normal prior, 2SD, squared)',
                         ],
            title=f'Accuracy of normal reward testing best performers, scenario 1 means, sd={sd}',
            legend_title='algorithms',
            best_arm_index=4,
            long_legend=False,
            ignore_first_rounds=5
        )

    scalability()


    # s = 1
    # sd = 0.25
    # plot_probs_choosing_best_arm(
    #     fn_list=[f'./logs/normal arm/scenario1/BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_c=2_assumed_sd=0.1.csv',
    #              f'./logs/normal arm/scenario1/BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_c=2_assumed_sd=0.25.csv',
    #              f'./logs/normal arm/scenario1/BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_c=2_assumed_sd=0.5.csv',
    #              f'./logs/normal arm/scenario1/BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_c=2_assumed_sd=0.75.csv',
    #              f'./logs/normal arm/scenario1/BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_c=2_assumed_sd=1.csv',
    #              f'./logs/normal arm/scenario1/BayesUCBGaussian/real_sd_{sd}/bayes_ucb_gaussian_squared_c=2.csv',
    #              f'./logs/normal arm/scenario1/BayesUCBGaussian/real_sd_{sd}/new_bayes_ucb_gaussian.csv',
    #              ],
    #     legend_list=['apporach 1, assume sd 0.1',
    #                  'apporach 1, assume sd 0.25',
    #                  'apporach 1, assume sd 0.5',
    #                  'apporach 1, assume sd 0.75',
    #                  'apporach 1, assume sd 1',
    #                  '"squared"',
    #                  'apporach 2 (ppf)'
    #     ],
    #     legend_title='confidence bound',
    #     title=f'Actual SD={sd}',
    #     best_arm_index=4,
    #     ignore_first_rounds=5,
    # )