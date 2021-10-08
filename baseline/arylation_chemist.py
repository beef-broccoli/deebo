# arylation scope dataset, chemist baseline
#
# From 64 combinations, randomly pick 1 and find the best ligand out of 24
# Then apply the same ligand to all 64 combinations, calculate percentage of cases where this ligand is the best ligand
# Calculate the average percentage for all 64 combinations
#
# Notes:
# - 4E gives no yield for all ligands, so I'm only simulating through 63 combinations


import pandas as pd
from collections import Counter

df = pd.read_csv('../data/arylation/scope_ligand.csv')
df = df[['ligand', 'electrophile_pci_name', 'nucleophile_pci_name', 'yield']]
df['combinations'] = df['electrophile_pci_name'].astype(str) + df['nucleophile_pci_name'].astype(str)
df = df[['ligand', 'combinations', 'yield']]

idx = df.groupby(['combinations'])['yield'].transform(max) == df['yield']
df = df[idx]
df = df.loc[df['combinations'] != '4E']

counter = Counter(list(df['ligand']))
df['percentage_all_combinations'] = df['ligand'].apply(lambda x: counter[x] / 64)

print(
    df.to_string()
      )