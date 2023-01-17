# the idea is to get rid of outliers during optimization
# but doesn't look like it works great here, especially since it detects high yield reactions as outliers also

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

plt.rcParams['savefig.dpi'] = 300

df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')
coi = ['nucleophile_id', 'electrophile_id', 'ligand_name', 'yield']
df = df[coi]
df['electrophile_id'] = df['electrophile_id'].apply(lambda x: x.lstrip('e')).astype('int')  # for sorting purposes, so 10 is not immediately after 1
df['nucleophile_id'] = df['nucleophile_id'].apply(lambda x: x.lstrip('n'))
df = df.sort_values(by=['nucleophile_id', 'electrophile_id', 'ligand_name'])
g = df.groupby(['nucleophile_id', 'electrophile_id'])['yield'].apply(np.array)

group_names = g.index.to_list()  # substrate pairs names
values = np.stack(g.to_list(), axis=0)  # yield values 64x24

clf = IsolationForest()
#clf = LocalOutlierFactor(n_neighbors=2)
ys = clf.fit_predict(values)

results = pd.DataFrame(group_names, columns=['nucleophiles', 'electrophiles'])
results['mask'] = ys
results = results.pivot(index='nucleophiles', columns='electrophiles', values='mask')
print(results)

fig, ax = plt.subplots()
cmap = mpl.colors.ListedColormap(['white', 'black'])
im = ax.imshow(results, cmap=cmap)

# grid line
for i in range(len(results.index)):
    for j in range(len(results.columns)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='white', lw=1))

ax.set_xticks(np.arange(8), labels=list(results.columns))
ax.set_yticks(np.arange(8), labels=list(results.index))
ax.set_xlabel('electrophile (aryl bromide)')
ax.set_ylabel('nucleophile (imidazole)')
plt.title('isolation forest')
plt.show()