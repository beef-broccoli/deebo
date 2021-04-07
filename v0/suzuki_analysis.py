# process the suzuki dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from plot_utils import draw_heatmap, annotate_heatmap


def main():
    ligands, average, results = get_results(components=['base', 'ligand'], short=False, cutoff=0.7)
    print(np.count_nonzero(results))
    exit()
    plt.boxplot(results.T, vert=False, labels=ligands)
    plt.show()
    # two_component_visualization('substrate1', 'ligand')
    return None


def get_results(components, short=False, cutoff=None):
    """short only chooses normal reactants (reactant 1: aryl halide; reactant 2: bronic acid and boronate)

    :param components: TODO
    :param short:
    :param cutoff:
    :return:
    """
    dic = {
        'ligand': 'Ligand_Short_Hand',
        'reactant1': 'Reactant_1_Name',
        'reactant2': 'Reactant_2_Name',
        'base': 'Reagent_1_Short_Hand',
        'solvent': 'Solvent_1_Short_Hand',
        'yield': 'Product_Yield_PCT_Area_UV',
    }
    small_dic = {
        'reactant1': ['6-chloroquinoline', '6-Bromoquinoline', '6-triflatequinoline', '6-Iodoquinoline'],
        'reactant2': ['2a, Boronic Acid', '2b, Boronic Ester']
    }
    raw = pd.read_csv('../data/raw.csv')

    if short:
        raw = raw[raw[dic['reactant1']].isin(small_dic['reactant1'])]
        raw = raw[raw[dic['reactant2']].isin(small_dic['reactant2'])]

    if cutoff is not None:
        if cutoff >= 1 or cutoff <= 0:
            raise ValueError('Yield cutoff out of range (0,1)')
        y = raw[dic['yield']]
        y_max = 100.0 if y.max() < 100.0 else y.max()
        raw[dic['yield']] = (y/y_max) >= cutoff

    for component in components:
        if component == 'solvent':  # clean up mismatched solvent names
            raw.loc[raw[dic['solvent']] == 'MeOH/H2O_V2 9:1', dic['solvent']] = 'MeOH'
            raw.loc[raw[dic['solvent']] == 'THF_V2', dic['solvent']] = 'THF'
        if component not in dic.keys():
            raise ValueError('Reaction component not found.')

    col_names = [dic[component] for component in components]
    cols = col_names + [dic['yield']]  # column name with yield name
    w_yield = raw[cols]
    gb = w_yield.groupby(col_names)
    average = gb.mean()
    results = np.array(gb.apply(lambda x: x.values.tolist()).tolist())[:, :, -1]

    # clean up np array dtype
    if cutoff is not None:
        results = results == 'True'
    else:
        results = results.astype(np.float32)
    keys = list(gb.groups.keys())

    return keys, average, results


def two_component_visualization(component1, component2):
    raw = pd.read_csv('../data/raw.csv')
    if {component1, component2} == {'ligand', 'substrate1'}:
        short = raw[['Reactant_1_Name', 'Ligand_Short_Hand', 'Product_Yield_PCT_Area_UV']]
        gb = short.groupby(['Reactant_1_Name', 'Ligand_Short_Hand'])
        reactant_label = list(dict.fromkeys(list(zip(*gb.groups.keys()))[0]))
        ligand_label = list(dict.fromkeys(list(zip(*gb.groups.keys()))[1]))
        mean = np.array(gb.mean().unstack().values)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = np.arange(mean.shape[1])
        ys = np.arange(mean.shape[0])
        xs, ys = np.meshgrid(xs, ys)
        xs, ys = xs.ravel(), ys.ravel()
        depth = 0.1
        width = 0.75
        ax.bar3d(xs, ys, np.zeros_like(mean.ravel()), width, depth, mean.ravel(), shade=True)
        ax.set_xlabel('ligand')
        ax.set_ylabel('substrate')
        plt.tight_layout()
        plt.show()
        # print(mean.loc[('6-Bromoquinoline', 'AmPhos'), 'Product_Yield_PCT_Area_UV'])

    return None


if __name__ == '__main__':
    main()

