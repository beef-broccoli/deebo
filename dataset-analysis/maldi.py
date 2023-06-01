import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/maldi-bromide.csv')


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


def simulate_etc(max_sample=25, n_simulations=10000):

    # fetch ground truth data
    df = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/maldi-bromide.csv', index_col=0)
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
            if random_one == 'Cu':
                count = count+1
        percentages.append(count/n_simulations)
        print(percentages)
        avg_cumu_rewards.append(reward/n_simulations)

    print(percentages)
    print(avg_cumu_rewards)
    return None
    # accuracy: [0.4408, 0.4798, 0.5338, 0.5712, 0.598, 0.631, 0.6518, 0.6839, 0.6921, 0.7109, 0.7204, 0.748, 0.7537, 0.757, 0.7723, 0.7845, 0.7929, 0.8038, 0.8061, 0.8111, 0.8246, 0.8204, 0.8349, 0.8363, 0.846]
    # cumu reward: [1.1996480000000043, 2.438325499999996, 3.639613999999986, 4.897659500000015, 6.122077500000004, 7.265374999999997, 8.524333500000003, 9.743947500000035, 10.893117499999963, 12.12595950000006, 13.378394000000032, 14.487321, 15.66271149999999, 16.8917145, 18.20148749999996, 19.46042899999994, 20.750283999999997, 21.702172000000004, 23.16975500000006, 24.273904999999957, 25.34884049999992, 26.842558499999956, 27.91789749999997, 29.295134499999968, 30.53761949999994]

if __name__ == '__main__':
    import numpy as np

    a = [0.0, 0.4408, 0.4798, 0.5338, 0.5712, 0.598, 0.631, 0.6518, 0.6839, 0.6921, 0.7109, 0.7204, 0.748, 0.7537, 0.757, 0.7723, 0.7845, 0.7929, 0.8038, 0.8061, 0.8111, 0.8246, 0.8204, 0.8349, 0.8363, 0.846]
    a = np.array(a).repeat(4)
    np.save('baseline.npy', a)
