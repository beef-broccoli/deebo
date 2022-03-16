import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def eps_greedy_probs_choosing_best_arm():

    best_arm = 4
    # reward = [0.1, 0.2, 0.3, 0.4, 0.9]
    eps = [0.1, 0.2, 0.3, 0.4, 0.5]  # all epsilons tested

    fig, ax = plt.subplots()

    for e in eps:
        fp = './logs/epsilon_greedy_test/epsilon_' + str(e)
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'chosen_arm']]

        n_simulations = int(np.max(df['num_sims']))+1
        time_horizon = int(np.max(df['horizon']))+1
        all_arms = np.zeros((n_simulations, time_horizon))

        for ii in range(int(n_simulations)):
            all_arms[ii, :] = list(df.loc[df['num_sims'] == ii]['chosen_arm'])

        counts = np.count_nonzero(all_arms == best_arm, axis=0)  # average across simulations. shape: (1, time_horizon)
        probs = counts/n_simulations
        ax.plot(np.arange(time_horizon), probs, label=str(e))

    ax.set_xlabel('time horizon')
    ax.set_ylabel('probability of finding best arm')
    ax.set_title('accuracy of epsilon greedy algorithm')
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.legend(title='epsilon', loc='lower right')

    plt.show()


def eps_greedy_average_reward():

    best_arm = 4
    # reward = [0.1, 0.2, 0.3, 0.4, 0.9]
    eps = [0.1, 0.2, 0.3, 0.4, 0.5]  # all epsilons tested

    fig, ax = plt.subplots()

    for e in eps:
        fp = './logs/epsilon_greedy_test/epsilon_' + str(e)
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'reward']]

        n_simulations = int(np.max(df['num_sims']))+1
        time_horizon = int(np.max(df['horizon']))+1
        all_rewards = np.zeros((n_simulations, time_horizon))

        for ii in range(int(n_simulations)):
            all_rewards[ii, :] = list(df.loc[df['num_sims'] == ii]['reward'])

        avg_reward = np.average(all_rewards, axis=0)  # average across simulations. shape: (1, time_horizon)
        ax.plot(np.arange(time_horizon), avg_reward, label=str(e))

    ax.set_xlabel('time horizon')
    ax.set_ylabel('average reward')
    ax.set_title('average reward of epsilon greedy algorithm')
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.legend(title='epsilon', loc='lower right')

    plt.show()


def eps_greedy_cumulative_reward():

    best_arm = 4
    # reward = [0.1, 0.2, 0.3, 0.4, 0.9]
    eps = [0.1, 0.2, 0.3, 0.4, 0.5]  # all epsilons tested

    fig, ax = plt.subplots()

    for e in eps:
        fp = './logs/epsilon_greedy_test/epsilon_' + str(e)
        df = pd.read_csv(fp)
        df = df[['num_sims', 'horizon', 'cumulative_reward']]

        def get_rewards(df):  # for one simulation, calculate reward (average or cumulative) at each time horizon t
            rewards = df['cumulative_reward'].to_numpy()
            return rewards

        n_simulations = int(np.max(df['num_sims']))+1
        time_horizon = int(np.max(df['horizon']))+1
        all_rewards = np.zeros((n_simulations, time_horizon))

        for i in range(int(n_simulations)):
            rewards = df.loc[df['num_sims'] == i]['cumulative_reward'].to_numpy()
            all_rewards[i, :] = rewards

        probs = np.average(all_rewards, axis=0)  # average across simulations. shape: (1, time_horizon)
        ax.plot(np.arange(time_horizon), probs, label=str(e))

    ax.set_xlabel('time horizon')
    ax.set_ylabel('cumulative reward')
    ax.set_title('cumulative reward of epsilon greedy algorithm')
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.legend(title='epsilon', loc='upper left')

    plt.show()


if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 300
    eps_greedy_cumulative_reward()