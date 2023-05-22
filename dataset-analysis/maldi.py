import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/maldi-bromide.csv')

eic_max = df['EIC(+)[M+H] Product Area'].max()
df['EIC(+)[M+H] Product Area'] = df['EIC(+)[M+H] Product Area']/eic_max
gb = df.groupby(by='condition')['EIC(+)[M+H] Product Area'].mean()
vals = gb.values
index = gb.index.values
plt.bar([0,1,2,3], vals)
plt.xticks([0,1,2,3], index)
for ii in [0,1,2,3]:
    plt.text(ii, vals[ii]+0.005, str(round(vals[ii], 3)), ha='center', va='center')
plt.title('Average UPLC-MS ion counts (normalized) for four different catalytic methods')
plt.ylabel('Average UPLC-MS ion counts (normalized)')
plt.xlabel('Catalytic methods')
plt.show()
