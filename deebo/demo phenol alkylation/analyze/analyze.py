import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools


def plot_acquisition_history_heatmap(scope_fp='../backups/round5-18/cache/scope.pkl', sim=0, roun=0, binary=False, cutoff=80):
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

    # import pickle
    #
    # with open(scope_fp, 'rb') as f:
    #     s = pickle.load(f)
    #
    # print(s.data)

    df = pd.read_csv('scope.csv', index_col=0)  # from a scope object, exp history
    df['yield'] = df['yield'].apply(lambda x: 100*x)
    df['temperature'] = df['temperature'].apply(lambda x: x[1:]+'°C')
    df['TS'] = df['temperature'] + ', ' + df['solvent_name']
    df = df.sort_values(by=['base_name', 'TS', 'mesylate_id', 'phenol_id'])
    df['yield'] = df['yield'].fillna(-20)

    ts = df['TS'].unique()
    bs = df['base_name'].unique()
    ms = df['mesylate_id'].unique()
    ps = df['phenol_id'].unique()
    df = df.drop(['temperature', 'solvent_name', 'prediction'], axis=1)

    # sort data into matrixes
    ds = []
    averages = []
    for t, b in itertools.product(ts, bs):

        # get all data
        tempdf = df.loc[(df['TS'] == t) & (df['base_name'] == b)]
        tempdf = tempdf.drop(['TS', 'base_name'], axis=1)
        a = np.array(tempdf.groupby(by=['mesylate_id'], sort=True)['yield'].apply(list).to_list())
        # each row of a is a mesylate, each column of a is an phenol
        ds.append(a)

        # do average
        to_average = df.loc[(df['TS'] == t) & (df['base_name'] == b) & (df['yield'] != -20)]['yield'].to_numpy()
        if len(to_average) == 0:  # catch the np.average warning for empty array
            averages.append('n/a')
        else:
            averages.append(round(np.average(to_average), 1))

    # could've written some recursive loop here, but not worth it
    data = np.vstack([np.hstack(ds[0:6]),
                      np.hstack(ds[6:12]),
                      np.hstack(ds[12:18]),
                      np.hstack(ds[18:24]),
                      np.hstack(ds[24:30]),
                      np.hstack(ds[30:36]),
                      ])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', vmin=-15, vmax=80)
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='white')
    ii = 0
    for i in range(6):
        for j in range(6):
            ax.add_patch(Rectangle((5 * j - 0.5, 5 * i - 0.5), 5, 5, fill=False, edgecolor='white', lw=2))
            plt.text(5 * j + 2, 5 * i + 2, averages[ii], **text_kwargs)
            ii = ii + 1

    # set x axis (base name), y axis (temp, solvent) labels
    tick_pos = [2, 7, 12, 17, 22, 27]
    ax.set_xticks(tick_pos, bs, fontsize=13)
    ax.set_yticks(tick_pos, ts, fontsize=13)

    ax_t = ax.secondary_xaxis('top')
    ax_t.set_xticks(np.arange(5)+25, labels=ps, rotation=45)
    ax_r = ax.secondary_yaxis('right')
    ax_r.set_yticks(np.arange(5), labels=ms)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax_t.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_ylabel('Yield (%)', rotation=270, fontsize=14, labelpad=25)
    plt.title(f'Experimental history (90/900 reactions)', fontsize=14)
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 300
    plt.show()

    return


def bar_plot_average_yield():

    df = pd.read_csv('history.csv')
    df['yield'] = df['yield']*100
    df['temperature'] = df['temperature'].apply(lambda x: x[1:]+'°C')

    figs, axes = plt.subplots(2, 2)

    # all conditions
    df['all'] = df['base_name']+('\n')+df['solvent_name']+'\n'+df['temperature']
    fd = df.groupby('all')['yield'].mean().apply(lambda x: round(x, 1)).sort_values(ascending=False)[0:6]
    #axes[0, 0].bar(['A', 'B', 'C', 'D', 'E'], fd.values)
    p = axes[0, 0].bar(fd.index, fd.values)
    b, t = axes[0,0].get_ylim()
    axes[0, 0].set_ylim(b, t+2)
    axes[0, 0].bar_label(p, label_type='edge', c='k', fontsize=8)
    axes[0, 0].set_xticklabels(fd.index, fontsize=8)
    axes[0, 0].set_title('Base-solvent-temperature (top 6)')
    axes[0, 0].set_ylabel('Yield (%)')
    #print(f'Base-solvent-temperature condition details: {fd.index.map(str)}')

    # base
    fd = df.groupby('base_name')['yield'].mean().apply(lambda x: round(x, 1)).sort_values(ascending=False)
    p = axes[1, 0].bar(fd.index, fd.values)
    b, t = axes[1,0].get_ylim()
    axes[1, 0].set_ylim(b, t+2)
    axes[1, 0].bar_label(p, label_type='edge', c='k', fontsize=8)
    axes[1, 0].set_xticklabels(fd.index, fontsize=8)
    axes[1, 0].set_title('Base')
    axes[1, 0].set_ylabel('Yield (%)')

    # solvent
    fd = df.groupby('solvent_name')['yield'].mean().apply(lambda x: round(x, 1)).sort_values(ascending=False)
    p = axes[0, 1].bar(fd.index, fd.values)
    b, t = axes[0,1].get_ylim()
    axes[0, 1].set_ylim(b, t+2)
    axes[0, 1].bar_label(p, label_type='edge', c='k', fontsize=8)
    axes[0, 1].set_xticklabels(fd.index)
    axes[0, 1].set_title('Solvent')

    # temperature
    fd = df.groupby('temperature')['yield'].mean().apply(lambda x: round(x, 1)).sort_values(ascending=False)
    p = axes[1, 1].bar(fd.index, fd.values)
    b, t = axes[1,1].get_ylim()
    axes[1, 1].set_ylim(b, t+2)
    axes[1, 1].bar_label(p, label_type='edge', c='k', fontsize=8)
    axes[1, 1].set_xticklabels(fd.index)
    axes[1, 1].set_title('Temperature')

    figs.suptitle('Average yield (90/900 reactions)')
    plt.rcParams['savefig.dpi']=300
    plt.tight_layout()
    plt.show()

    return


def bar_plot_counts():

    df = pd.read_csv('history.csv')
    df['yield'] = df['yield']*100
    df['temperature'] = df['temperature'].apply(lambda x: x[1:]+'°C')

    figs, axes = plt.subplots(2, 2)

    # all conditions
    df['all'] = df['base_name']+('\n')+df['solvent_name']+'\n'+df['temperature']
    fd = df.groupby('all')['yield'].count().sort_values(ascending=False)[0:5]

    p = axes[0, 0].bar(fd.index, fd.values)
    b, t = axes[0,0].get_ylim()
    axes[0, 0].set_ylim(b, t+2)
    axes[0, 0].bar_label(p, label_type='edge', c='k')
    axes[0, 0].set_xticklabels(fd.index, fontsize=8)
    axes[0, 0].set_title('Base-solvent-temperature')
    axes[0, 0].set_ylabel('n_samples')

    # base
    fd = df.groupby('base_name')['yield'].count().sort_values(ascending=False)
    p = axes[1, 0].bar(fd.index, fd.values)
    b, t = axes[1, 0].get_ylim()
    axes[1, 0].set_ylim(b, t+2)
    axes[1, 0].bar_label(p, label_type='edge', c='k')
    axes[1, 0].set_xticklabels(fd.index, fontsize=8)
    axes[1, 0].set_title('Base')
    axes[1, 0].set_ylabel('n_samples')

    # solvent
    fd = df.groupby('solvent_name')['yield'].count().sort_values(ascending=False)
    p = axes[0, 1].bar(fd.index, fd.values)
    b, t = axes[0,1].get_ylim()
    axes[0, 1].set_ylim(b, t+2)
    axes[0, 1].bar_label(p, label_type='edge', c='k')
    axes[0, 1].set_xticklabels(fd.index)
    axes[0, 1].set_title('Solvent')

    # temperature
    fd = df.groupby('temperature')['yield'].count().sort_values(ascending=False)
    p = axes[1, 1].bar(fd.index, fd.values)
    b, t = axes[1,1].get_ylim()
    axes[1, 1].set_ylim(b, t+2)
    axes[1, 1].bar_label(p, label_type='edge', c='k')
    axes[1, 1].set_xticklabels(fd.index)
    axes[1, 1].set_title('Temperature')

    figs.suptitle('Number of samples (90/900 reactions)')
    plt.rcParams['savefig.dpi']=300
    plt.tight_layout()
    plt.show()

    return


def compare_conditions():
    # compare condition performance for each substrate, average of three solvents
    df = pd.read_csv('validation data.csv')
    df['base-solvent-temp'] = df['base'] + '-' + df['solvent'] + '-' + df['temperature']
    df['phenol-mesylate'] = df['phenol'] + '-' + df['mesylate']
    df = df.sort_values(by=['phenol-mesylate', 'base-solvent-temp'])
    data = df.groupby(by=['phenol-mesylate'])['yield'].apply(list)
    ids = data.index

    conditions = ['BTMG-tAmOH-60°C', 'Cs2CO3-DMF-60°C', 'K3PO4-DMF-60°C']

    plt.rcParams['savefig.dpi'] = 300
    fig, axs = plt.subplots(3, 4, figsize=(6,4), constrained_layout=True)

    def ax_plot(x, y, data, title, y_ticks=False):
        colors = ['#f26b5b', '#0f4c81', '#373838']
        conditions = ['BTMG-tAmOH-60°C', 'Cs2CO3-DMF-60°C', 'K3PO4-DMF-60°C',]
        axs[x, y].barh(np.arange(len(data)), data, height=0.5, color=colors)
        for p, v in zip(np.arange(len(data)), data):
            axs[x, y].text(v+1, p, str(round(v,1)), ha='left', va='center', c='black')
        axs[x, y].set_title(title, fontweight='bold')  # title
        if y_ticks:  # y label
            axs[x, y].set_yticks(np.arange(len(data)), conditions)
        else:
            axs[x, y].set_yticks([])
        axs[x, y].set_xlim([0,50])

    id_count = 0
    for row in range(3):
        for column in range(4):
            if id_count in [0, 4, 8]:
                ax_plot(row, column, data.loc[ids[id_count]], title=ids[id_count], y_ticks=True)
            else:
                ax_plot(row, column, data.loc[ids[id_count]], title=ids[id_count])
            id_count += 1
            if id_count == 11:
                break

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    #fig.tight_layout()
    #plt.tight_layout()
    plt.show()


def bms_history_stripplot():

    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['savefig.dpi'] = 300

    df = pd.read_csv('BMS history data.csv')
    df['Base'] = df['Base'].apply(lambda x: 'DIPEA' if x == 'Diisopropylethylamine' else x)
    df['Base'] = df['Base'].apply(lambda x: 'Pempidine' if x == 'Pentamethylipiperidine' else x)

    # sort based on counts for bases
    counts = df.groupby(by='Base')['Z_Score_AP'].max().sort_values(ascending=False)
    order = list(counts.index)

    df['Z_Score_AP'] = df['Z_Score_AP'].apply(lambda x: x + np.random.rand()/10)

    sns.stripplot(data=df, x='Z_Score_AP', y='Base',
                  hue='Solvent',
                  alpha=0.8,
                  size=7,
                  palette='tab20',
                  order=order)
    plt.show()

    return None


if __name__ == '__main__':
    bar_plot_average_yield()