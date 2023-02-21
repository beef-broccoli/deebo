import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import itertools
import yaml

df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv')

df = df[['base_name', 'fluoride_name', 'substrate_name', 'yield']]
fd = df.copy()
df = df.loc[df['substrate_name'] != 's37']
fs = list(df['fluoride_name'].unique())
bs = list(df['base_name'].unique())
ss = list(df['substrate_name'].unique())

with open('colors.yml', 'r') as file:
    COLORS = yaml.safe_load(file)


def plot_all_results():
    ds = []
    averages = []
    for f, b in itertools.product(fs, bs):
        ds.append(df.loc[(df['fluoride_name'] == f) & (df['base_name'] == b)]['yield'].to_numpy().reshape(6,6))
        averages.append(round(np.average(fd.loc[(fd['fluoride_name'] == f) & (fd['base_name'] == b)]['yield'].to_numpy()),1))

    data = np.hstack([np.vstack(ds[0:4]),
                      np.vstack(ds[4:8]),
                      np.vstack(ds[8:12]),
                      np.vstack(ds[12:16]),
                      np.vstack(ds[16:20])])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno')
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

    plt.show()


def plot_results_with_model_substrates(cutoff=75, select=False, which_dimension='combo'):
    """
    a heatmap, with each substrate pair as model system, highest yielding ligand is identified

    Parameters
    ----------
    cutoff: int
        yield cutoff from 0-100. If the highest yielding ligand gives a yield lower than cutoff, it's considered not optimized
    select: bool
        plot only selected few ligands for better visualization. **need to modify function
    which_dimension: str
        choose from {'fluoride_name, 'base_name', 'combo'}. Which dimension to plot

    Returns
    -------

    """
    fd = df.copy()

    #fd = fd.sort_values(by=['combo', 'ligand_name'])
    fd['combo'] = fd['fluoride_name'].astype('str') + '/' + fd['base_name'].astype('str')
    max = fd.loc[fd.groupby(by=['substrate_name'])['yield'].idxmax()]

    #print(list(max['ligand_name'].unique()))
    #print(max.loc[max['plot']!=0]['ligand_name'].value_counts())

    def color_select(x):  # to assign colors
        if x == 'CgMe-PPh':
            return 1
        elif x == 'tBPh-CPhos':
            return 2
        elif x == 'Cy-BippyPhos':
            return 3
        elif x == 'Et-PhenCar-Phos':
            return 4
        elif x == 'PPh3':
            return 5
        else:
            return 6

    # new way to assign colors for all ligands that give above cutoff yields
    to_color = max.loc[max['yield']>cutoff][which_dimension].unique()

    def color(x):
        vals = np.arange(len(to_color)) + 1
        d = dict(zip(to_color, vals))
        if x not in d:
            return 0
        else:
            return d[x]

    max['valid'] = fd['yield'].apply(lambda x: 0 if x<cutoff else 1)  # 0 for plotting, if highest yield < 75%
    if select:
        max['plot'] = fd[which_dimension].apply(color_select)
    else:
        max['plot'] = fd[which_dimension].apply(color)
    max['plot'] = max['plot']*max['valid']

    max_color = max['plot'].to_numpy().reshape(6,6)  # plot color code
    max['text'] = max['substrate_name'] + ' (' + max['yield'].astype(str) + '%)'
    max_text = max['text'].to_numpy().reshape(6,6)
    # max_highest_yield = max['yield'].to_numpy().reshape(6,6)  # plot highest yield
    # max_sub_name = max['substrate_name'].to_numpy().reshape(6,6)  # substrate name

    fig, ax = plt.subplots()
    im = ax.imshow(max_color, cmap='turbo')

    # grid line
    for i in range(6):
        for j in range(6):
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='white', lw=1))
            ax.text(j, i, max_text[i, j], ha="center", va="center", color="w")

    # ax.set_xticks(np.arange(6), labels=list(max.columns))
    # ax.set_yticks(np.arange(6), labels=list(max.index))
    # ax.set_xlabel('electrophile (aryl bromide)')
    # ax.set_ylabel('nucleophile (imidazole)')
    if select:
        values = list(np.arange(7))
        ligand_color = [f'Not optimized (<{cutoff}%)', 'CgMe-PPh', 'tBPh-CPhos', 'Cy-BippyPhos', 'Et-PhenCar-Phos', 'PPh3', 'other ligands']
    else:
        values = list(np.arange(len(to_color)+1))
        ligand_color = list(to_color)
        ligand_color.insert(0, f'Not optimized (<{cutoff}%)')
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label=ligand_color[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.spines['top'].set_visible(False)  # remove boundaries
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.axis('off')
    plt.rcParams['savefig.dpi'] = 600
    plt.show()


def plot_best_with_diff_metric(which_dimension='combo'):  # 6 bar plots, each with top 5 ligands, and their performance wrt metric

    df['combo'] = df['fluoride_name'].astype('str') + '/' + df['base_name'].astype('str')
    stats = df.groupby(by=[which_dimension]).describe()
    twentyfive = stats.loc[:, ('yield', '25%')].nlargest(5)  # 1st quantile top 5
    median = stats.loc[:, ('yield', '50%')].nlargest(5)  # 2nd quantile
    seventyfive = stats.loc[:, ('yield', '75%')].nlargest(5)  # 3rd quantile
    mean = stats.loc[:, ('yield', 'mean')].nlargest(5)  # average

    overtwenty = df.loc[df['yield'] > 20].groupby(by=which_dimension).size().nlargest(5)  # top 5, over 20%, count
    overeighty = df.loc[df['yield'] > 80].groupby(by=which_dimension).size().nlargest(5)  # top 5, over 80%, count

    # make color dictionary, one color for one ligand
    all_top_ligands = []
    for li in [twentyfive, median, seventyfive, mean, overtwenty, overeighty]:
        all_top_ligands = all_top_ligands + list(li.index)
    all_top_ligands = list(set(all_top_ligands))
    # colors = {}
    # colormap = plt.cm.tab10.colors
    # for i in range(len(all_top_ligands)):
    #     colors[all_top_ligands[i]] = colormap[i]

    color_list = [COLORS['coral_essence'], COLORS['cornhusk'], COLORS['stucco'], COLORS['peach_quartz'],
                  COLORS['baby_blue'], COLORS['monument'], COLORS['provence'], COLORS['pink_tint']]
    colors = {}
    if len(all_top_ligands) > len(color_list):
        raise RuntimeError('not enough colors for all top options. {0} colors, {1} options'.format(len(color_list), len(all_top_ligands)))
    for i in range(len(all_top_ligands)):
        colors[all_top_ligands[i]] = color_list[i]

    def get_colors(ll):  # for a list of names, get their color from overall color dict
        out = []
        for l in ll:
            out.append(colors[l])
        return out

    def trim(ll):  # trim the long ligand names
        return [s[:10] for s in ll]

    figsize = (10,6)
    kwargs = {'aa': True, 'width': 0.5}
    plt.rcParams['savefig.dpi'] = 300
    figs, axs = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    def ax_plot(ax_x, ax_y, df, title, y_label=None):
        x = trim(list(df.index))
        y = list(df.values)
        axs[ax_x, ax_y].bar(x, y, color=get_colors(list(df.index)), **kwargs)
        for i in range(len(x)):  # plot value
            axs[ax_x, ax_y].text(i, y[i]+0.5, round(y[i], 2), ha='center')
        axs[ax_x, ax_y].set_title(title)  # title
        if y_label:  # y label
            axs[ax_x, ax_y].set_ylabel(y_label)
        axs[ax_x, ax_y].set_ylim(top=axs[ax_x, ax_y].get_ylim()[1] + 5)  # adjust ylim top so value text fits

    ax_plot(0, 0, twentyfive, title='1st quantile (Q1)', y_label='yield (%)')
    ax_plot(0, 1, median, title='median')
    ax_plot(1, 0, seventyfive, title='3rd quantile (Q3)', y_label='yield (%)')
    ax_plot(1, 1, mean, title='average')
    ax_plot(2, 0, overtwenty, title='yield >20%', y_label='count')
    ax_plot(2, 1, overeighty, title='yield >80%')

    plt.show()


if __name__ == '__main__':

    plot_best_with_diff_metric(which_dimension='fluoride_name')

