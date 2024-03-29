import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import random


def plot_all_results():
    DF = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/NiB/rxns/inchi_23l.csv')

    DF = DF[['electrophile_id', 'ligand_name', 'solvent_name', 'yield']]
    FD = DF.copy()
    es = list(DF['electrophile_id'].unique())  # 33
    ls = list(DF['ligand_name'].unique())  # 23
    ss = list(DF['solvent_name'].unique())  # 2

    df = DF.copy()
    fd = DF.copy()
    ds = []
    averages = []

    # group by all three, sorted on all three levels; get sorted solvent names
    first = df.groupby(by=['electrophile_id', 'ligand_name', 'solvent_name'])['yield'].apply(list)
    solvents = first.index.levels[-1].to_list()
    # group by two, now yield is [<ethanol_yield>, <methanol_yield>]; get sorted ligand names
    second = first.groupby(by=['electrophile_id', 'ligand_name']).apply(lambda x: list(itertools.chain(*x)))
    ligands = second.index.levels[-1].to_list()
    # group by one, now yield is [<A-paPhos_ethanol_yield>, <A-paPhos_methanol_yield>, <CX-FBu_ethanol_yield>, <CX-FBu_methanol_yield>...];
    # get sorted substrate names
    third = second.groupby(by='electrophile_id').apply(lambda x: list(itertools.chain(*x)))
    substrates = third.index.to_list()

    # # plot all results with EtOH/MeOH comparison
    # data = np.array(list(third.values))  # np.array constructor only works with list of lists; series values created above were defaulted to array of lists
    # fig, ax = plt.subplots()
    # im = ax.imshow(data, cmap='inferno')
    # for ii in range(len(ligands)):
    #     for jj in range(len(substrates)):
    #         ax.add_patch(
    #             Rectangle((2 * ii - 0.5, 1 * jj - 0.5), 2, 1, fill=False, edgecolor='white', lw=1)
    #         )
    #
    # x_pos = np.arange(len(ligands))*2+0.5
    # ax.set_xticks(x_pos, labels=ligands, rotation=90)
    # ax.set_yticks(np.arange(len(substrates)), labels=substrates)
    #
    # ax_t = ax.secondary_xaxis('top')
    # ax_t.set_xticks(np.arange(2), labels=solvents, rotation=45)
    #
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('yield (%)', rotation=270)
    #
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax_t.spines['top'].set_visible(False)
    #
    # plt.rcParams['savefig.dpi'] = 300
    # plt.tight_layout()
    # plt.show()

    # plot difference in yield
    difference = second.apply(lambda x: x[0]-x[1])  # apply to list; EtOH-MeOH
    difference = difference.groupby(by='ligand_name').apply(lambda x: list(itertools.chain(x)))
    difference = np.array(list(difference.values))

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()
    abso = np.absolute(difference)
    # denoise, set diff to 0 if +- 5%
    #difference = np.where(abso>5, difference, 0)

    im = ax.imshow(difference, cmap='RdBu', vmin=-np.max(abso), vmax=np.max(abso))

    # this one substrate is x axis
    x_pos = np.arange(len(substrates))
    ax.set_xticks(x_pos, labels=substrates, rotation=90)
    ax.set_yticks(np.arange(len(ligands)), labels=ligands)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('yield (EtOH-MeOH) (%)', rotation=270)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return None


def plot_cutoff_heatmap(cutoff=60, solvent='EtOH', topn=8):
    """
    
    Parameters
    ----------
    cutoff: float or int
        cutoff for yield, 0-100
    solvent: str
        {'MeOH', 'EtOH', 'both'}
        if both, cutoff threshold has to be satisfied for both solvents
    topn: int
        label the top n ligands names in red

    Returns
    -------

    """
    
    DF = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/NiB/rxns/inchi_23l.csv')

    DF = DF[['electrophile_id', 'ligand_name', 'solvent_name', 'yield']]
    df = DF.copy()
    es = list(DF['electrophile_id'].unique())  # 33
    ls = list(DF['ligand_name'].unique())  # 23
    ss = list(DF['solvent_name'].unique())  # 2
    
    # methanol data
    df_me = df.loc[df['solvent_name']=='MeOH']
    df_me = df_me.drop(columns=['solvent_name'])
    df_me = df_me.groupby(by=['electrophile_id', 'ligand_name'], group_keys=True)['yield'].apply(list)
    substrates = df_me.index.levels[0].to_list()
    df_me = df_me.groupby(by='ligand_name').apply(lambda x: list(itertools.chain(*x)))
    ligands = df_me.index.to_list()
    data_me = np.array(list(df_me.values))
    data_me = np.where(data_me < cutoff, 0, 1)

    # ethanol data
    df_et = df.loc[df['solvent_name']=='EtOH']
    df_et = df_et.drop(columns=['solvent_name'])
    df_et = df_et.groupby(by=['electrophile_id', 'ligand_name'], group_keys=True)['yield'].apply(list)
    substrates = df_et.index.levels[0].to_list()
    df_et = df_et.groupby(by='ligand_name').apply(lambda x: list(itertools.chain(*x)))
    ligands = df_et.index.to_list()
    data_et = np.array(list(df_et.values))
    data_et = np.where(data_et < cutoff, 0, 1)

    if solvent == 'MeOH':
        data = data_me
    elif solvent == 'EtOH':
        data = data_et
    elif solvent == 'both':
        data = np.multiply(data_me, data_et)
    else:
        exit()

    # print out ligands based on counts that exceed threshold
    newlist = []
    for n in np.argsort(data.sum(axis=1)):
        newlist.append(ligands[n])

    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=1.5)

    # this one substrate is x axis
    x_pos = np.arange(len(substrates))
    ax.set_xticks(x_pos, labels=substrates, rotation=90)
    ylabels = [f'{l} ({str(c)})' for l, c in zip(ligands, data.sum(axis=1))]  # this adds counts to ligand
    ax.set_yticks(np.arange(len(ligands)), labels=ylabels)
    for x in np.argsort(data.sum(axis=1))[-topn:]:  # red the top n ligands
        ax.get_yticklabels()[x].set_color('navy')
        ax.get_yticklabels()[x].set_fontweight('bold')

    for ii in range(len(substrates)):
        for jj in range(len(ligands)):
            ax.add_patch(
                Rectangle((ii - 0.5, jj - 0.5), 1, 1, fill=False, edgecolor='white', lw=0.5)
            )


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    zero_patch = mpatches.Patch(color=(0, 0, 2/255), label='<50%')
    one_patch = mpatches.Patch(color=(228/255, 131/255, 68/255), label='>=50%')
    plt.legend(handles=[one_patch, zero_patch], title='yield threshold', bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xlabel('substrates')
    plt.ylabel('ligands')
    plt.title(f'{solvent}, {cutoff}% cutoff, top{topn}')
    plt.tight_layout()
    plt.show()

    return None


def simulate_etc(max_sample=4, n_simulations=10000):

    """
    Functions to simulate explore-then-commit baseline

    Parameters
    ----------
    max_sample: int
        maximum number of samples for each condition. Usually (# of budget experiments) % (# of conditions).
    n_simulations: int
        how many times explore-then-commit is simulated on the same dataset.

    Returns
    -------

    """

    top_six = ['PPh2Cy', 'CX-PCy', 'PPh3', 'P(p-F-Ph)3', 'P(p-Anis)3', 'Cy-JohnPhos']
    top_three = ['Cy-JohnPhos', 'P(p-Anis)3', 'PPh2Cy']
    top_eight = ['PPh2Cy', 'CX-PCy', 'PPh3', 'P(p-F-Ph)3', 'P(p-Anis)3', 'Cy-JohnPhos', 'A-paPhos', 'Cy-PhenCar-Phos']

    # fetch ground truth data
    df = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/nib-etoh.csv', index_col=0)
    df['yield'] = df['yield'].apply(lambda x: 0 if x<50 else 1)

    percentages = []
    avg_cumu_rewards = []
    gb = df.groupby(by=['ligand_name'])
    for n_sample in tqdm(range(max_sample), desc='1st loop'):
        count = 0
        reward = 0
        for i in tqdm(range(n_simulations), desc='2nd loop', leave=False):
            sample = gb.sample(n_sample+1).groupby('ligand_name')
            sample_mean = sample.mean(numeric_only=True)
            sample_sum = sample.sum(numeric_only=True).sum().values[0]
            reward = reward+sample_sum
            # if sample['yield'].idxmax() in top_six:  # no tie breaking when sampling 1 with yield cutoff
            #     count = count + 1
            maxs = sample_mean.loc[sample_mean['yield']==sample_mean['yield'].max()]
            random_one = random.choice(list(maxs.index))
            if random_one in top_three:
                count = count+1
        percentages.append(count/n_simulations)
        avg_cumu_rewards.append(reward/n_simulations)

    print(percentages)
    print(avg_cumu_rewards)
    # with yield: [0.5971, 0.66, 0.7173]
    # 60% cutoff binary, no max tie breaking: [0.388, 0.5382, 0.6154]
    # 60% cutoff binary, with max tie breaking: [0.4301, 0.5488, 0.6136] (helps with sample 1 case, more ties)
    # 60% cutoff binary, cumulative reward: [7.1552, 14.3058, 21.4805]

    # 50% cutoff binary top three: accuracy: [0.2263, 0.3055, 0.3833, 0.5027]; cumu reward [9.7952, 19.6476, 29.4682, 49.117]
    # 50% cutoff binary top eight: accur: [0.5371, 0.6623, 0.7558, 0.848]  cumu: [9.8194, 19.6467, 29.4898, 49.0107]
    return None


if __name__ == '__main__':

    #plot_cutoff_heatmap(cutoff=60)

    #simulate_etc()

    a = np.array([0.0, 9.7952, 19.6476, 29.4682, 49.117])
    a = np.array(a).repeat(23)
    np.save('top3_cumu.npy', a)
    # np.save('top8.npy', b)