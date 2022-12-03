import sys

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from glob import glob
from utils import plot_info_file_path_match


def calculate_baseline(chemarms: list):
    # TODO: take chem arms, calculate a baseline for probability in a traditional reaction optimziation way
    from arms_chem import ChemArmSim

    # type check; check chem arms are from same dataset
    url = chemarms[0].data_url
    name = chemarms[0].name
    for arm in chemarms:
        assert isinstance(arm, ChemArmSim), "required argument: a list of ChemArm objects"
        assert arm.name == name, "ChemArmSim objects should describe same reaction components"
        assert arm.data_url == url, "ChemArmSim objects should come from the same dataset"

    df = pd.read_csv(url)

    temp = df[list(name)]

    return


def plot_probs_choosing_best_arm_all(folder_path=None):
    """
    Func that can more efficiently plot results for each algo with all parameters

    Parameters
    ----------
    folder_path
    best_arm

    Returns
    -------

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

    return



def plot_probs_choosing_best_arm(fn_list,
                                 legend_list,
                                 manual_baseline=0,
                                 etc_baseline=False,
                                 etc_fp='',
                                 best_arm_index=0,
                                 fp='',
                                 title='',
                                 legend_title='',
                                 long_legend=False,
                                 ignore_first_rounds=0):
    """

    Parameters
    ----------
    fn_list: list of data file names
    legend_list: list of legend names
    baseline: a horizontal baseline
    best_arm_index: the index for best arm (needed for calculation)
    fp: directory for where the data files are stored
    title: title for the plot
    legend_title: title for the legend
    long_legend: if true, legend will be plotted outside the plot; if false mpl finds the best position within plot

    Returns
    -------
    matplotlib.pyplot plt object

    """

    assert len(fn_list) == len(legend_list)

    fps = [fp + fn for fn in fn_list]

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    if manual_baseline != 0:
        plt.axhline(y=manual_baseline, xmin=0, xmax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)

    if etc_baseline:
        base = np.load(etc_fp)
        plt.plot(np.arange(len(base))[ignore_first_rounds:], base[ignore_first_rounds:], color='black', label='explore-then-commit', lw=2)

    for i in range(len(fps)):
        fp = fps[i]
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'chosen_arm']]

        n_simulations = int(np.max(df['num_sims'])) + 1
        time_horizon = int(np.max(df['horizon'])) + 1
        all_arms = np.zeros((n_simulations, time_horizon))

        for ii in range(int(n_simulations)):
            all_arms[ii, :] = list(df.loc[df['num_sims'] == ii]['chosen_arm'])

        counts = np.count_nonzero(all_arms == best_arm_index, axis=0)  # average across simulations. shape: (1, time_horizon)
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


def plot_average_reward(fn_list,
                        legend_list,
                        baseline=0,
                        show_se=False,
                        fp='',
                        title='',
                        legend_title='',
                        long_legend=False):

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


def plot_cumulative_reward(fn_list,
                           legend_list,
                           fp='',
                           title='',
                           legend_title=''):

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


def plot_regret():
    return


def plot_etc_baseline(explore_times,
                      fn_list,
                      legend_list,
                      best_arm_index=0,
                      fp='',
                      title='',
                      legend_title='',
                      long_legend=False,
                      ):
    """

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


    return


def _test_plot():

    # example on using plot function
    plt.rcParams['savefig.dpi'] = 300

    eps = [0.1, 0.2, 0.3, 0.4, 0.5]
    fn_list = ['epsilon_' + str(e) + '.csv' for e in eps]
    fn_list.append('annealing_epsilon_greedy.csv')
    legend_list = [str(e) for e in eps]
    legend_list.append('jdk')

    plot_cumulative_reward(fn_list, legend_list, fp='./logs/epsilon_greedy_test/', title='ss', legend_title='dd')


def _test_cal_baseline():

    from arms_chem import ChemArmSim
    import itertools

    # build chem arms
    dataset_url = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-conditions.csv'
    names = ('base_smiles', 'solvent_smiles')  # same names with column name in df
    base = ['O=C([O-])C.[K+]', 'O=C([O-])C(C)(C)C.[K+]']
    solvent = ['CC(N(C)C)=O', 'CCCC#N']
    vals = list(itertools.product(base, solvent))  # sequence has to match what's in "names"
    arms = list(map(lambda x: ChemArmSim(x, names, dataset_url), vals))

    # test basline
    calculate_baseline(arms)


def _plot_boltzmann():
    # example on using plot function
    plt.rcParams['savefig.dpi'] = 300

    taus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    fn_list = ['tau_' + str(t) + '.csv' for t in taus]
    fn_list.append('annealing_boltzmann_test.csv')
    legend_list = [str(t) for t in taus]
    legend_list.append('annealing')

    plot_probs_choosing_best_arm(fn_list, legend_list, best_arm_index=4, fp='./logs/Boltzmann_test/', title='accuracy of softmax', legend_title='tau')


if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 300
    import itertools

    def scenario1_best_perfomers():
        prefix = 'logs/scenario1/'
        n_list = ['eps_greedy/annealing',
                  'softmax/tau_0.2',
                  'pursuit/lr_0.05',
                  'optim/ucb1_tuned',
                  'optim/TS'
                  ]
        fn_list = [f'{prefix}{n}.csv' for n in n_list]
        plot_probs_choosing_best_arm(
            fn_list=fn_list,
            legend_list=['eps greedy(annealing)',
                         'softmax (tau=0.2)',
                         'pursuit (lr=0.05)',
                         'ucb1-tuned',
                         'thompson sampling (beta prior)'],
            etc_baseline=True,
            etc_fp=f'{prefix}baseline.npy',
            title='Accuracy of scenario 1 best performers',
            legend_title='algorithms',
            best_arm_index=4,
            long_legend=False,
            ignore_first_rounds=5
        )
        return None

    scenario1_best_perfomers()
    #plot_probs_choosing_best_arm_all(folder_path='logs/scenario1/exp3')

    # ns = list(np.arange(27)+1)
    # fn_list = [f'{n}_exp_per_arm.csv' for n in ns]
    # plot_etc_baseline(explore_times=[n*9 for n in ns], fn_list=fn_list, legend_list=ns,
    #                   best_arm_index=8, fp='./baseline_logs/scenario4/etc/', long_legend=True)

    # names = ['ucb1', 'ucb1_tuned', 'TS']

    # fn_list = [
    #     'TS',
    #     'ucb1',
    #     'ucb1_tuned',
    #     'ucbv',
    #     'moss',
    #     'ucb2_0.5'
    # ]
    # fn_list = [f+'.csv' for f in fn_list]
    # legend_list = [
    #     'TS',
    #     'UCB1',
    #     'UCB1-tuned',
    #     'UCB-V',
    #     'MOSS',
    #     'UCB2 (alpha=0.5)'
    # ]
    #
    # fn_list = ['dmed.csv', 'dmed_modified.csv']
    # legend_list = ['DMED', 'DMED (modified)']
    #
    # fn_list = [
    #     'optim/TS.csv',
    #     'optim/ucb1_tuned.csv',
    #     'eps_greedy/annealing_epsilon_greedy.csv',
    #     'eps_greedy/epsilon_0.1.csv',
    #     'softmax/tau_0.1.csv',
    #     'softmax/tau_0.2.csv',
    #     'pursuit/pursuit_lr_0.05.csv',
    #     'reinforcement_comparison/rc_alpha_0.05_beta_0.4.csv',
    #     'dmed.csv'
    # ]
    # legend_list = [
    #     'TS (beta prior)',
    #     'UCB1-Tuned',
    #     'eps-greedy (annealing)',
    #     'eps-greedy (0.1)',
    #     'softmax (0.1)',
    #     'softmax (0.2)',
    #     'pursuit (0.05)',
    #     'RC (0.05, 0.4)',
    #     'DMED'
    # ]
    #
    # plot_probs_choosing_best_arm(fn_list, legend_list, best_arm_index=4, fp='./logs/scenario1/',
    #                              title='Accuracy of best algorithms in scenario 1', legend_title='algorithms',
    #                              long_legend=True, etc_baseline=True, etc_fp='./logs/scenario1/ETC/baseline.npy')
    #
    # gammas = [0.1, 0.2, 0.3, 0.4, 0.5]
    # fn_list = ['ucb2_'+str(a)+'.csv' for a in gammas]
    # #fn_list.append('annealing.csv')
    # legend_list = gammas
    # #legend_list.append('annealing')
    # plot_cumulative_reward(fn_list, legend_list, fp='./logs/scenario1/optim/',
    #                              title='Cumulative reward of UCB2 algorithm', legend_title='alpha')

    # es = np.arange(15)
    # es = es+1
    # fn_list = [f'etc_{e}.csv' for e in es]
    # legend_list = [str(e) for e in es]
    # plot_etc_baseline([e*5 for e in es], fn_list, legend_list, fp='./logs/scenario3/ETC/', best_arm_index=4, title='', legend_title='')

    # plot_probs_choosing_best_arm(['dmed.csv', 'dmed_modified.csv', 'optim/ucb1_tuned.csv', 'optim/ucb1.csv'],
    #                              ['dmed', 'dmed mod', 'ucb1 tuned', 'ucb1'],
    #                              best_arm_index=4,
    #                              fp='./logs/scenario1/')

