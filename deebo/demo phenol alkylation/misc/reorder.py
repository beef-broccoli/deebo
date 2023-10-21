# this function makes sure that the proposed_experiments.csv files with and without yields are in the same order

import pandas as pd
import numpy as np

df_og = pd.read_csv('proposed_experiments.csv', index_col=0)
df_new = pd.read_csv('proposed_experiments_unordered.csv', index_col=0)

keys = df_og.index.values
vals = np.arange(len(keys))
custom_dict = dict(zip(keys, vals))

df_new = df_new.sort_index(key=lambda x: x.map(custom_dict))
df_new.to_csv('proposed_experiments_ordered.csv')
