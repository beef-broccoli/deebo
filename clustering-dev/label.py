# label substrates based on conditions that achieve highest yield

import pandas as pd
from pathlib import Path
import numpy as np


NO_REACTIVITY_YIELD = 30  # yield cutoff for hit/no hit


def data_loader(fp):
    df = pd.read_csv(fp)
    return df


# # identify substrates with low yield in all conditions
# def filter_low_yield(df):
#     grouped = df.groupby('substrate_SMILES')
#     no_reactivity_list = []
#     for name, group in grouped:
#         yields = list(group['yield'])
#         if all(y <= NO_REACTIVITY_YIELD for y in yields):
#             no_reactivity_list.append(name)
#     return no_reactivity_list


# for each substrate, for one reaction component, identify the best candidate
# **need to rewrite the way component is passed into function at some point
def label_single_component(df, substrates, component, mode='avg'):
    best_component = 'best_' + component + '_SMILES'
    component = component + '_SMILES'
    substrates[best_component] = len(substrates)*['none']  # initialize labels

    grouped = df.groupby('substrate_SMILES')

    dic = {}  # substrates with all yields below threshold will not be in this dict
    for name, group in grouped:  # group by substrate

        yields = list(group['yield'])
        if all(y <= NO_REACTIVITY_YIELD for y in yields):
            continue  # substrates with all yields below threshold

        grouped_component = group.groupby(component)
        scores = {}
        for n, g in grouped_component:  # group by component; TODO: handle ties; close ones
            yields = list(g['yield'])
            score = score_func(yields, mode)
            scores[score] = n
        max = np.max(list(scores.keys()))
        dic[name] = scores[max]

    for s in list(dic.keys()):
        substrates.loc[substrates['substrate_SMILES'] == s, best_component] = dic[s]

    return substrates


# for one substrate with one condition, score with different modes
# l: a list of yields
# mode:
#   avg: average
#   max: maximum yield
#   pts: points based on categories (0-20% 1 point, 21-40% 2 points, ..., >80% 5 points)
def score_func(l, mode='avg'):
    if mode == 'avg':
        return np.mean(l)
    elif mode == 'max':
        return np.max(l)
    elif mode == 'pts':  # TODO: too many ties
        score = 0
        for ll in l:
            if ll <= 20:
                score = score+1
            elif 20 < ll <= 40:
                score = score+2
            elif 40 < ll <= 60:
                score = score+3
            elif 60 < ll <= 80:
                score = score+4
            elif ll > 80:
                score = score+5
            else:
                pass
        return score
    else:
        return None


if __name__ == '__main__':
    fp = Path.cwd().parent / 'data/deoxy/experiment_index.csv'
    sub = Path.cwd().parent / 'data/deoxy/substrate.csv'
    print(
        label_single_component(data_loader(fp), data_loader(sub), 'fluoride', mode='avg')
    )
