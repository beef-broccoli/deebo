import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


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
    max_horizon = max(df['horizon']) + 1  # max time horizon
    df = df[['chosen_arm', 'reward']]
    gb = df.groupby(['chosen_arm'])
    avg_counts = gb.count().apply(lambda x: x/num_sims)['reward'].sort_values(ascending=False)
    avg_counts_val = avg_counts.values[:top_n]  # sorted count values
    arm_indexes = avg_counts.index.to_numpy()[:top_n]   # corresponding arm index
    arm_names = ['/'.join(arms_dict[ii]) for ii in arm_indexes]

    # calculate baseline (time horizon evenly divided by number of arms)
    baseline = max_horizon/len(arms_dict)

    xs = np.arange(len(avg_counts_val))
    ax.barh(arm_names, avg_counts_val)
    plt.axvline(x=baseline, ymin=0, ymax=1, linestyle='dashed', color='black', label='baseline', alpha=0.5)
    ax.set_xlabel('average number of times sampled')
    ax.set_ylabel('arms')
    ax.set_title(title)
    plt.show()

    return None


def plot_acquisition_history_heatmap_arylation_scope(history_fp='./test/history.csv', round=0, sim=0, binary=False, cutoff=80):

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

    ground_truth = df[['electrophile_id', 'nucleophile_id', 'ligand_name']].to_numpy()

    history = pd.read_csv(history_fp)
    history = history.loc[(history['round']<=round) & (history['num_sims']==sim)][['electrophile_id', 'nucleophile_id', 'ligand_name']]
    history['electrophile_id'] = history['electrophile_id'].apply(lambda x: x.lstrip('e')).astype('int')
    history['nucleophile_id'] = history['nucleophile_id'].apply(lambda x: x.lstrip('n'))
    history = history.to_numpy()
    print(history)

    indexes = []
    for row in range(history.shape[0]):
        indexes.append(np.argwhere(np.isin(ground_truth, history[row,:]).all(axis=1))[0,0])
    print(indexes)
    print(df.iloc[indexes])
    df = df.reset_index()
    idx_to_set = df.index.difference(indexes)
    df.loc[idx_to_set, 'yield'] = -1
    print(df.iloc[indexes])

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

    def set_value(x):
        if 0<=x<cutoff:
            return 0
        elif x>=cutoff:
            return 1
        else:
            return x
    if binary:
        f = np.vectorize(set_value)
        a = f(a)

    fig, ax = plt.subplots()
    cmap_custom = mpl.colormaps['inferno']
    cmap_custom.set_under('silver')
    im = ax.imshow(a, cmap=cmap_custom, vmin=0, vmax=110)
    if binary:
        im = ax.imshow(a, cmap=cmap_custom, vmin=0, vmax=2)
    if not binary:
        text_kwargs = dict(ha='center', va='center', fontsize=10, color='white')
    else:
        text_kwargs = dict(ha='center', va='center', fontsize=10, color='white')
    ii = 0
    for i in range(4):
        for j in range(6):
            ax.add_patch(Rectangle((8*j-0.5, 8*i-0.5), 8, 8, fill=False, edgecolor='white', lw=2))
            plt.text(8*j+3.5, 8*i+3.5, ligand_names[ii], **text_kwargs)
            ii = ii+1
    plt.axis('off')
    if not binary:
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('yield (%)', rotation=270)
    plt.rcParams['savefig.dpi'] = 300
    plt.show()
    return None


if __name__ == '__main__':
    import pickle

    # dir = 'dataset_logs/aryl-scope-ligand/eps_greedy_annealing/'
    # with open(f'{dir}arms.pkl', 'rb') as f:
    #     arms_dict = pickle.load(f)
    # plot_arm_stats(arms_dict, top_n=10, fp=f'{dir}/log.csv')

    plot_acquisition_history_heatmap_arylation_scope(sim=0,
                                                     round=1,
                                                     binary=False,
        history_fp='./dataset_logs/aryl-scope-ligand/eps_greedy_annealing/history.csv')