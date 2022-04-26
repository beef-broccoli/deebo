import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import itertools

df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv')

df = df[['base_name', 'fluoride_name', 'substrate_name', 'yield']]
fd = df.copy()
df = df.loc[df['substrate_name'] != 's37']
fs = list(df['fluoride_name'].unique())
bs = list(df['base_name'].unique())
ss = list(df['substrate_name'].unique())

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

