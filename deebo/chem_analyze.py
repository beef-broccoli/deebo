import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_arm_stats(arms_dict,
                   top_n=5,
                   fp='log.csv',
                   title='',
                   legend_title='arms',
                   long_legend=False):

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()

    df = pd.read_csv(fp)
    num_sims = max(df['num_sims']) + 1  # number of simulations done
    df = df[['chosen_arm', 'reward']]
    gb = df.groupby(['chosen_arm'])
    avg_counts = gb.count().apply(lambda x: x/num_sims)['reward'].sort_values(ascending=False)
    avg_counts_val = avg_counts.values[:top_n]  # sorted count values
    arm_indexes = avg_counts.index.to_numpy()[:top_n]   # corresponding arm index
    arm_names = ['/'.join(arms_dict[ii]) for ii in arm_indexes]

    xs = np.arange(len(avg_counts_val))
    ax.bar(arm_names, avg_counts_val)
    plt.show()

    return None


if __name__ == '__main__':
    import pickle

    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
    ligands = ground_truth['ligand_name'].unique()
    with open(f'./test/arms.pkl', 'rb') as f:
        arms_dict = pickle.load(f)

    plot_arm_stats(arms_dict, fp='./test/log.csv')