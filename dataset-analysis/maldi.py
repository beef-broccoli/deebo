import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def plot_average_bar():
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


def simulate_etc(max_sample=50, n_simulations=10000):

    # fetch ground truth data
    df = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/maldi-amine.csv', index_col=0)
    eic_max = df['EIC(+)[M+H] Product Area'].max()
    df['EIC(+)[M+H] Product Area'] = df['EIC(+)[M+H] Product Area']/eic_max

    percentages = []
    avg_cumu_rewards = []
    gb = df.groupby(by=['condition'])
    for n_sample in tqdm(range(max_sample), desc='1st loop'):
        count = 0
        reward = 0
        for i in tqdm(range(n_simulations), desc='2nd loop', leave=False):
            sample = gb.sample(n_sample+1).groupby('condition')
            sample_mean = sample.mean(numeric_only=True)
            sample_sum = sample.sum(numeric_only=True).sum().values[0]
            reward = reward+sample_sum
            # if sample['yield'].idxmax() in top_six:  # no tie breaking when sampling 1 with yield cutoff
            #     count = count + 1
            maxs = sample_mean.loc[sample_mean['EIC(+)[M+H] Product Area']==sample_mean['EIC(+)[M+H] Product Area'].max()]
            random_one = random.choice(list(maxs.index))
            if random_one == 'Pd':
                count = count+1
        percentages.append(count/n_simulations)
        avg_cumu_rewards.append(reward/n_simulations)

    print(percentages)
    print(avg_cumu_rewards)
    return None


if __name__ == '__main__':
    import numpy as np

    plot_average_bar()
    #simulate_etc()

    # accuracy = [0.5004, 0.5595, 0.6095, 0.6582, 0.6867, 0.732, 0.75, 0.7705, 0.7833, 0.8047, 0.8279, 0.8377, 0.8496, 0.8664,
    #  0.8713, 0.8794, 0.8935, 0.8981, 0.9081, 0.9162, 0.9226, 0.931, 0.9285, 0.9368, 0.9407, 0.9465, 0.9507, 0.9505,
    #  0.9547, 0.9586, 0.9619, 0.9627, 0.9694, 0.972, 0.9755, 0.975, 0.9756, 0.978, 0.98, 0.9788, 0.9816, 0.9852, 0.9873,
    #  0.9874, 0.9884, 0.9903, 0.9907, 0.9923, 0.9918, 0.9921]
    #
    # a = np.array(accuracy).repeat(4)
    # np.save('etc.npy', a)
