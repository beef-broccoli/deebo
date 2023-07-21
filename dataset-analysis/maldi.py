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


def simulate_etc(max_sample=50, n_simulations=10000):

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
        avg_cumu_rewards.append(reward/n_simulations)

    print(percentages)
    print(avg_cumu_rewards)
    return None
    # accuracy: [0.4408, 0.4798, 0.5338, 0.5712, 0.598, 0.631, 0.6518, 0.6839, 0.6921, 0.7109, 0.7204, 0.748, 0.7537, 0.757, 0.7723, 0.7845, 0.7929, 0.8038, 0.8061, 0.8111, 0.8246, 0.8204, 0.8349, 0.8363, 0.846]
    # cumu reward: [1.1996480000000043, 2.438325499999996, 3.639613999999986, 4.897659500000015, 6.122077500000004, 7.265374999999997, 8.524333500000003, 9.743947500000035, 10.893117499999963, 12.12595950000006, 13.378394000000032, 14.487321, 15.66271149999999, 16.8917145, 18.20148749999996, 19.46042899999994, 20.750283999999997, 21.702172000000004, 23.16975500000006, 24.273904999999957, 25.34884049999992, 26.842558499999956, 27.91789749999997, 29.295134499999968, 30.53761949999994]
    #cumu_reward = [1.2204375000000043, 2.439783000000002, 3.6033550000000134, 4.81396549999999, 6.03998950000002, 7.276120000000062, 8.5215295, 9.684210499999962, 10.967089499999991, 12.228079499999982, 13.329161500000037, 14.49245400000003, 15.811102000000023, 16.873830999999964, 18.08134300000003, 19.405756999999905, 20.629049499999883, 21.746651500000006, 23.311561999999977, 24.20063699999989, 25.458129499999973, 26.727208500000007, 27.932370500000044, 29.02257800000013, 30.372234499999973, 31.50999949999999, 32.79040350000015, 33.92846099999991, 35.10167350000006, 36.3433510000001, 37.756402500000135, 38.701805000000085, 40.04339299999994, 41.34455500000004, 42.595835499999986, 43.52444199999986, 44.88117000000003, 46.011245500000115, 47.470463500000086, 48.500259500000205, 49.67786800000008, 51.07926549999986, 52.02494150000027, 53.23033449999989, 54.59021750000014, 55.87807199999996, 56.922660500000056, 58.268032999999754, 59.312976999999975, 60.70482300000011]
    #accuracy = [0.4395, 0.488, 0.5366, 0.5692, 0.6145, 0.6331, 0.651, 0.6773, 0.6886, 0.7098, 0.7145, 0.7325, 0.7577, 0.7647, 0.7741, 0.7754, 0.7925, 0.8043, 0.8069, 0.8164, 0.8204, 0.83, 0.8366, 0.8424, 0.8464, 0.8552, 0.8591, 0.8582, 0.8702, 0.872, 0.8759, 0.884, 0.8856, 0.8867, 0.8881, 0.8966, 0.8984, 0.9024, 0.9058, 0.9167, 0.9109, 0.9154, 0.9246, 0.9225, 0.9253, 0.9305, 0.933, 0.933, 0.9363, 0.9458]

if __name__ == '__main__':
    import numpy as np

    #simulate_etc()
    accuracy = [0.4395, 0.488, 0.5366, 0.5692, 0.6145, 0.6331, 0.651, 0.6773, 0.6886, 0.7098, 0.7145, 0.7325, 0.7577, 0.7647, 0.7741, 0.7754, 0.7925, 0.8043, 0.8069, 0.8164, 0.8204, 0.83, 0.8366, 0.8424, 0.8464, 0.8552, 0.8591, 0.8582, 0.8702, 0.872, 0.8759, 0.884, 0.8856, 0.8867, 0.8881, 0.8966, 0.8984, 0.9024, 0.9058, 0.9167, 0.9109, 0.9154, 0.9246, 0.9225, 0.9253, 0.9305, 0.933, 0.933, 0.9363, 0.9458]
    # a = [0.0, 0.4408, 0.4798, 0.5338, 0.5712, 0.598, 0.631, 0.6518, 0.6839, 0.6921, 0.7109, 0.7204, 0.748, 0.7537, 0.757, 0.7723, 0.7845, 0.7929, 0.8038, 0.8061, 0.8111, 0.8246, 0.8204, 0.8349, 0.8363, 0.846]
    a = np.array(accuracy).repeat(4)
    np.save('baseline-200.npy', a)
