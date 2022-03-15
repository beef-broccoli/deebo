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

        def get_probs(df):  # for one simulation, calculate probability at each time horizon t
            chosen_arms = list(df['chosen_arm'])
            best_arm_count = 0
            probs = []
            for i in range(len(chosen_arms)):
                if chosen_arms[i] == best_arm:  # arm 4
                    best_arm_count = best_arm_count + 1
                probs.append(best_arm_count/(i+1))  # probability of selecting best arm at time horizon t
            return probs

        n_simulations = int(np.max(df['num_sims']))+1
        time_horizon = int(np.max(df['horizon']))+1
        all_probs = np.zeros((n_simulations, time_horizon))

        for i in range(int(n_simulations)):
            probs = get_probs(df.loc[df['num_sims'] == i])
            all_probs[i, :] = probs

        probs = np.average(all_probs, axis=0)  # average across simulations. shape: (1, time_horizon)
        ax.plot(np.arange(time_horizon), probs, label=str(e))

    ax.set_xlabel('time horizon')
    ax.set_ylabel('probability of finding best arm')
    ax.set_title('accuracy of epsilon greedy algorithm')
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.legend(title='epsilon', loc='upper left')

    plt.show()


def eps_greedy_reward(average=False):

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
            if average:
                a = np.arange(len(rewards)) + 1
                rewards = rewards/a
            return rewards

        n_simulations = int(np.max(df['num_sims']))+1
        time_horizon = int(np.max(df['horizon']))+1
        all_rewards = np.zeros((n_simulations, time_horizon))

        for i in range(int(n_simulations)):
            rewards = get_rewards(df.loc[df['num_sims'] == i])
            all_rewards[i, :] = rewards

        probs = np.average(all_rewards, axis=0)  # average across simulations. shape: (1, time_horizon)
        ax.plot(np.arange(time_horizon), probs, label=str(e))

    ax.set_xlabel('time horizon')
    if average:
        ax.set_ylabel('average reward')
        ax.set_title('average reward of epsilon greedy algorithm')
    else:
        ax.set_ylabel('cumulative reward')
        ax.set_title('cumulative reward of epsilon greedy algorithm')
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.legend(title='epsilon', loc='upper left')

    plt.show()



if __name__ == '__main__':
    reward(average=False)