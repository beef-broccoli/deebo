import pickle
import os
import pandas as pd

from chem_arms import Scope
from chem_arms import propose_initial_experiments_interpolation, update_and_propose_interpolation
import algos_regret
import utils

"""Some analysis functions for the single optimization run logs in (single_run_logs)"""


def deoxyf(dir='./single_run_logs/deoxyf/'):
    # check file path, some are hard coded
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv')
    df['yield'] = df['yield'].apply(utils.scaler)

    # make dictionary for querying yield
    df['combined'] = df[['base_name', 'fluoride_name', 'substrate_name']].apply(lambda x: frozenset(x), axis=1)
    ys_lookup_dict = pd.Series(df['yield'].values, index=df['combined']).to_dict()

    substrate_smiles_to_id = dict(zip(df['substrate_smiles'].unique(),
                                      df['substrate_name'].unique()))

    df = df[['base_name', 'fluoride_name', 'substrate_name', 'yield']]
    bases = df['base_name'].unique()
    fluorides = df['fluoride_name'].unique()
    substrates = df['substrate_name'].unique()

    # make encodings for substrates
    encodings = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deoxyF/mols/ECFP/substrate.csv'
    )
    encodings.set_index('substrate_SMILES', inplace=True)
    ordered_ids = [substrate_smiles_to_id[s] for s in encodings.index]  # smiles to id, maintain order
    encodings = dict(zip(ordered_ids, encodings.to_numpy().tolist()))  # {id: [ECFP1, ECFP2, ...]}
    encodings = {'substrate_name': encodings}  # one more level to specify it's substrates

    scope_dict = {'base_name': bases,
                  'fluoride_name': fluorides,
                  'substrate_name': substrates}
    arms_dict = {'base_name': bases}
    arms_dict_2 = {'fluoride_name': fluorides}
    arms_dict_3 = {'base_name': bases,
                   'fluoride_name': fluorides}
    algo = algos_regret.BayesUCBGaussian(len(bases), assumed_sd=0.25, c=2)

    def query_and_fill(lookup_dict, workdir):
        proposed = pd.read_csv(f'{workdir}proposed_experiments.csv', index_col=0)
        keys_to_look_up = proposed[list(scope_dict.keys())].apply(lambda x: frozenset(x), axis=1)
        ys = [lookup_dict[x] for x in keys_to_look_up]
        proposed['yield'] = ys
        proposed.to_csv(f'{workdir}proposed_experiments.csv')
        return None

    # propose initial experiments
    propose_initial_experiments_interpolation(scope_dict, arms_dict, algo, dir=dir, num_exp=3)

    rounds = 10
    num_exp = 3
    for i in range(rounds):
        query_and_fill(ys_lookup_dict, dir)
        if i == rounds-1:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings, update_only=True)
        else:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings)

    with open('./single_run_logs/deoxyf/cache/scope.pkl', 'rb') as f:
        scope = pickle.load(f)
    with open('./single_run_logs/deoxyf/cache/algo.pkl', 'rb') as f:
        algo = pickle.load(f)

    ranks = algo.ranking
    old_arms = scope.arms
    old_arm_labels = scope.arm_labels
    print(f'rankings for {old_arm_labels}:')
    print(f'{[old_arms[r] for r in ranks]}')
    print(f'counts: {[algo.counts[r] for r in ranks]}\n')

    # now trying to change arms, change scope, initialize new algo and update, and save
    scope.clear_arms()
    scope.build_arms(arms_dict_3)
    new_algo = algos_regret.BayesUCBGaussian(len(scope.arms), assumed_sd=0.25, c=2)
    results_for_new_arms = scope.sort_results_with_arms()

    for ii in range(len(results_for_new_arms)):
        results = results_for_new_arms[ii]
        for jj in range(len(results)):
            new_algo.update(ii, results[jj])

    with open('./single_run_logs/deoxyf/cache/scope.pkl', 'wb') as f:
        pickle.dump(scope, f)
    with open('./single_run_logs/deoxyf/cache/algo.pkl', 'wb') as f:
        pickle.dump(new_algo, f)

    # now run optimization.
    # the only not so great thing here is the log. arm index will mean different things after pivoting arms
    rounds = 25
    num_exp = 3
    update_and_propose_interpolation(dir, num_exp, encoding_dict=encodings, propose_only=True)
    for i in range(rounds):
        query_and_fill(ys_lookup_dict, dir)
        if i == rounds-1:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings, update_only=True)
        else:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings)

    with open('./single_run_logs/deoxyf/cache/scope.pkl', 'rb') as f:
        scope = pickle.load(f)
    with open('./single_run_logs/deoxyf/cache/algo.pkl', 'rb') as f:
        algo = pickle.load(f)

    ranks = algo.ranking
    old_arms = scope.arms
    old_arm_labels = scope.arm_labels
    print(f'rankings for {old_arm_labels}:')
    print(f'{[old_arms[r] for r in ranks]}')
    print(f'counts: {[algo.counts[r] for r in ranks]}\n')


def amidation_phase1(dir='./single_run_logs/amidation/phase1/'):

    def query_and_fill(scope_dict, lookup_dict, workdir):
        proposed = pd.read_csv(f'{workdir}proposed_experiments.csv', index_col=0)
        keys_to_look_up = proposed[list(scope_dict.keys())].apply(lambda x: frozenset(x), axis=1)
        ys = [lookup_dict[x] for x in keys_to_look_up]
        proposed['yield'] = ys
        proposed.to_csv(f'{workdir}proposed_experiments.csv')
        return None

    # load data
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/amidation.csv')
    df['yield'] = df['yield'].apply(utils.scaler)
    # make dictionary for querying yield
    df['combined'] = df[['activator_name', 'solvent_name', 'base_name', 'nucleophile_id']].apply(lambda x: frozenset(x), axis=1)
    ys_lookup_dict = pd.Series(df['yield'].values, index=df['combined']).to_dict()
    # for the encoding dict
    smiles_to_id = dict(zip(df['nucleophile_smiles'].unique(), df['nucleophile_id'].unique()))

    df = df[['activator_name', 'solvent_name', 'base_name', 'nucleophile_id', 'yield']]
    bases = df['base_name'].unique()
    activators = df['activator_name'].unique()
    solvents = df['solvent_name'].unique()
    nucleophiles = df['nucleophile_id'].unique()

    # grab encodings for substrates
    encodings = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/amidation/mols/morganFP/nucleophile.csv'
    )
    encodings.set_index('nucleophile_smiles', inplace=True)
    ordered_ids = [smiles_to_id[s] for s in encodings.index]  # smiles to id, maintain order
    encodings = dict(zip(ordered_ids, encodings.to_numpy().tolist()))  # {id: [ECFP1, ECFP2, ...]}
    encodings = {'nucleophile_id': encodings}  # one more level to specify it's substrates

    # set parameters for optimization
    scope_dict = {'base_name': bases,
                  'activator_name': activators,
                  'solvent_name': solvents,
                  'nucleophile_id': nucleophiles}
    arms_dict = {'activator_name': activators}
    #algo = algos_regret.BayesUCBGaussian(len(activators), assumed_sd=0.25, c=2)
    algo = algos_regret.UCB1Tuned(len(activators))

    # propose initial experiments
    propose_initial_experiments_interpolation(scope_dict, arms_dict, algo, dir=dir, num_exp=5)

    # iteratively query
    rounds = 8
    num_exp = 5
    for i in range(rounds):
        query_and_fill(scope_dict, ys_lookup_dict, dir)
        if i == rounds-1:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings, update_only=True)
        else:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings)

    # output some quick stats from this optimization
    with open(f'{dir}cache/scope.pkl', 'rb') as f:
        scope = pickle.load(f)
    with open(f'{dir}cache/algo.pkl', 'rb') as f:
        algo = pickle.load(f)

    ranks = algo.ranking
    old_arms = scope.arms
    old_arm_labels = scope.arm_labels
    print(f'rankings for {old_arm_labels}:')
    print(f'{[old_arms[r] for r in ranks]}')
    print(f'counts: {[algo.counts[r] for r in ranks]}\n')
    print(f'means: {[round(algo.emp_means[r],2) for r in ranks]}\n')

    return None


# copy all phase 1 files into phase2 folder. Clumsy but works for now
def amidation_phase2(dir='./single_run_logs/amidation/phase2/'):

    def query_and_fill(scope_dict, lookup_dict, workdir):
        proposed = pd.read_csv(f'{workdir}proposed_experiments.csv', index_col=0)
        keys_to_look_up = proposed[list(scope_dict.keys())].apply(lambda x: frozenset(x), axis=1)
        ys = [lookup_dict[x] for x in keys_to_look_up]
        proposed['yield'] = ys
        proposed.to_csv(f'{workdir}proposed_experiments.csv')
        return None

    # load data
    df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/amidation.csv')
    df['yield'] = df['yield'].apply(utils.scaler)
    # make dictionary for querying yield
    df['combined'] = df[['activator_name', 'solvent_name', 'base_name', 'nucleophile_id']].apply(lambda x: frozenset(x), axis=1)
    ys_lookup_dict = pd.Series(df['yield'].values, index=df['combined']).to_dict()
    # for the encoding dict
    smiles_to_id = dict(zip(df['nucleophile_smiles'].unique(), df['nucleophile_id'].unique()))

    df = df[['activator_name', 'solvent_name', 'base_name', 'nucleophile_id', 'yield']]
    bases = df['base_name'].unique()
    activators = df['activator_name'].unique()
    solvents = df['solvent_name'].unique()
    nucleophiles = df['nucleophile_id'].unique()

    # grab encodings for substrates
    encodings = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/amidation/mols/morganFP/nucleophile.csv'
    )
    encodings.set_index('nucleophile_smiles', inplace=True)
    ordered_ids = [smiles_to_id[s] for s in encodings.index]  # smiles to id, maintain order
    encodings = dict(zip(ordered_ids, encodings.to_numpy().tolist()))  # {id: [ECFP1, ECFP2, ...]}
    encodings = {'nucleophile_id': encodings}  # one more level to specify it's substrates

    # load all phase 1 stuff
    with open(f'{dir}cache/scope.pkl', 'rb') as f:
        scope = pickle.load(f)
    with open(f'{dir}cache/algo.pkl', 'rb') as f:
        algo = pickle.load(f)

    # now trying to change arms, change scope, initialize new algo and update, and save
    scope_dict = {'base_name': bases,
                  'activator_name': activators,
                  'solvent_name': solvents,
                  'nucleophile_id': nucleophiles}
    arms_dict = {'activator_name': ['BOP-Cl', 'TFFH', 'DPPCl', 'TCFH'],
                 'base_name': bases}
    scope.clear_arms()
    scope.build_arms(arms_dict)

    new_algo = algos_regret.UCB1Tuned(len(scope.arms))
    results_for_new_arms = scope.sort_results_with_arms()

    count = 0
    for ii in range(len(results_for_new_arms)):
        results = results_for_new_arms[ii]
        for jj in range(len(results)):
            new_algo.update(ii, results[jj])
            count+=1

    with open(f'{dir}cache/scope.pkl', 'wb') as f:
        pickle.dump(scope, f)
    with open(f'{dir}cache/algo.pkl', 'wb') as f:
        pickle.dump(new_algo, f)

    # now run optimization.
    # the only not so great thing here is the log. arm index will mean different things after pivoting arms
    rounds = 16
    num_exp = 5
    update_and_propose_interpolation(dir, num_exp, encoding_dict=encodings, propose_only=True)
    for i in range(rounds):
        query_and_fill(scope_dict, ys_lookup_dict, dir)
        if i == rounds-1:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings, update_only=True)
        else:
            update_and_propose_interpolation(dir=dir, num_exp=num_exp, encoding_dict=encodings)

    with open(f'{dir}cache/scope.pkl', 'rb') as f:
        scope = pickle.load(f)
    with open(f'{dir}cache/algo.pkl', 'rb') as f:
        algo = pickle.load(f)

    ranks = algo.ranking
    old_arms = scope.arms
    old_arm_labels = scope.arm_labels
    print(f'rankings for {old_arm_labels}:')
    print(f'{[old_arms[r] for r in ranks]}')
    print(f'counts: {[algo.counts[r] for r in ranks]}\n')
    print(f'means: {[round(algo.emp_means[r],2) for r in ranks]}\n')


def plot_prediction_model_accuracy():
    """
    Final prediction model, plot some accuracies
    Divided up into three groups:
    - Reactions in training data
    - Reactions with activators included in the activator/base round
    - Reactions with activators eliminated

    Returns
    -------

    """
    import utils
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['savefig.dpi']=600
    import scipy
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    with open('./single_run_logs/amidation/phase2/cache/scope.pkl', 'rb') as f:
        s = pickle.load(f)

    pred = s.data
    pred = pred.sort_values(by=['activator_name', 'base_name', 'nucleophile_id', 'solvent_name'])
    ground_truth = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/amidation.csv')
    ground_truth = ground_truth.sort_values(by=['activator_name', 'base_name', 'nucleophile_id', 'solvent_name'])
    pred['true_yield'] = ground_truth['yield'].apply(utils.scaler).values
    # CAREFUL: need to strip index off the series above. pandas seems to sort by index when you assign series to a col

    training_data = pred.loc[pred['yield'].notna()]
    test_data = pred.loc[pred['yield'].isna()]
    eliminated_activators = test_data.loc[test_data['activator_name'].isin(['PFTU', 'HOTU', 'HATU', 'PyBOP'])]
    included_activators = test_data.loc[test_data['activator_name'].isin(['DPPCl', 'BOP-Cl', 'TCFH', 'TFFH'])]
    #print(len(training_data), len(eliminated_activators), len(included_activators))

    plt.scatter(training_data['prediction']*100, training_data['true_yield']*100, label='Reactions explored', alpha=0.7)
    plt.scatter(eliminated_activators['prediction']*100, eliminated_activators['true_yield']*100,
                label='Reactions unexplored, eliminated activators', marker='v', alpha=0.5)
    plt.scatter(included_activators['prediction']*100, included_activators['true_yield']*100,
                label='Reactions unexplored, retained activators', marker='s', alpha=0.4)
    plt.xlabel('Predicted yield (%)')
    plt.ylabel('Experimental yield (%)')
    plt.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100), color='k', ls='dashed', alpha=0.5)
    #plt.plot(np.linspace(0, 0.8, 100), np.linspace(0, 0.8, 100)+0.2, color='k', ls='dashed', alpha=0.5)
    #plt.plot(np.linspace(0.2, 1, 100), np.linspace(0.2, 1, 100)-0.2, color='k', ls='dashed', alpha=0.5)

    print(f'training data')
    y_true = training_data['true_yield']
    y_pred = training_data['prediction']
    print(f'rmse: {math.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'mae: {mean_absolute_error(y_true, y_pred)}')
    print(f'r2: {r2_score(y_true, y_pred)}\n')

    print(f'unrun, eliminated activators data')
    y_true = eliminated_activators['true_yield']
    y_pred = eliminated_activators['prediction']
    print(f'rmse: {math.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'mae: {mean_absolute_error(y_true, y_pred)}')
    print(f'r2: {r2_score(y_true, y_pred)}\n')

    print(f'unrun, retained activators data')
    y_true = included_activators['true_yield']
    y_pred = included_activators['prediction']
    print(f'rmse: {math.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'mae: {mean_absolute_error(y_true, y_pred)}')
    print(f'r2: {r2_score(y_true, y_pred)}\n')

    #plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center')
    plt.legend()
    plt.show()


def retrain_model():
    # doesnt really work; stick to the final predictions from model

    import numpy as np
    import pandas as pd
    import math
    from sklearn.ensemble import RandomForestRegressor as RFR
    from sklearn.neural_network import MLPRegressor as MLPR
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    dir = './single_run_logs/amidation/ml-training-data/'
    Xs_train = np.load(f'{dir}xs_train_ohe.npy')
    ys_train = np.load(f'{dir}ys_train_ohe.npy')
    Xs = np.load(f'{dir}xs_ohe.npy')
    print(Xs_train.shape)
    ground_truth = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/amidation.csv')
    ground_truth = ground_truth.sort_values(by=['activator_name', 'base_name', 'nucleophile_id', 'solvent_name'])
    ys_true = ground_truth['yield'].apply(utils.scaler).values

    model = RFR()
    model.fit(Xs_train, ys_train)
    ys_pred = model.predict(Xs)

    # evaluate
    with open('./single_run_logs/amidation/phase2/cache/scope.pkl', 'rb') as f:
        s = pickle.load(f)

    pred = s.data
    pred['prediction'] = ys_pred  # assign new prediction results
    pred = pred.sort_values(by=['activator_name', 'base_name', 'nucleophile_id', 'solvent_name'])
    ground_truth = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/amidation.csv')
    ground_truth = ground_truth.sort_values(by=['activator_name', 'base_name', 'nucleophile_id', 'solvent_name'])
    pred['true_yield'] = ground_truth['yield'].apply(utils.scaler).values

    training_data = pred.loc[pred['yield'].notna()]
    test_data = pred.loc[pred['yield'].isna()]
    eliminated_activators = test_data.loc[test_data['activator_name'].isin(['PFTU', 'HOTU', 'HATU', 'PyBOP'])]
    included_activators = test_data.loc[test_data['activator_name'].isin(['DPPCl', 'BOP-Cl', 'TCFH', 'TFFH'])]

    print(f'training data')
    y_true = training_data['true_yield']
    y_pred = training_data['prediction']
    print(f'rmse: {math.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'mae: {mean_absolute_error(y_true, y_pred)}')

    print(f'unrun, eliminated activators data')
    y_true = eliminated_activators['true_yield']
    y_pred = eliminated_activators['prediction']
    print(f'rmse: {math.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'mae: {mean_absolute_error(y_true, y_pred)}')

    print(f'unrun, retained activators data')
    y_true = included_activators['true_yield']
    y_pred = included_activators['prediction']
    print(f'rmse: {math.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'mae: {mean_absolute_error(y_true, y_pred)}')


def experimental_round_substrates():
    df1 = pd.read_csv('./single_run_logs/amidation/phase1/history.csv')
    df2 = pd.read_csv('./single_run_logs/amidation/phase2/history.csv')

    def select_optimal_conditions(df):
        df['activator-base'] = df['activator_name'] + '-' + df['base_name']
        df = df.loc[df['activator-base'].isin(
            ['DPPCl-N-methylmorpholine', 'DPPCl-Diisopropylethylamine', 'HATU-Diisopropylethylamine'])]
        print(df.groupby(by=['activator-base', 'nucleophile_id'])['yield'].apply(list))

        return None

    select_optimal_conditions(df1)
    select_optimal_conditions(df2)


if __name__ == '__main__':

    # import utils
    # with open('./single_run_logs/amidation/phase2/cache/scope.pkl', 'rb') as f:
    #     s = pickle.load(f)
    #
    # # grab encodings for substrates
    # df = pd.read_csv('https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/amidation.csv')
    # df['yield'] = df['yield'].apply(utils.scaler)
    # # make dictionary for querying yield
    # df['combined'] = df[['activator_name', 'solvent_name', 'base_name', 'nucleophile_id']].apply(lambda x: frozenset(x), axis=1)
    # ys_lookup_dict = pd.Series(df['yield'].values, index=df['combined']).to_dict()
    # # for the encoding dict
    # smiles_to_id = dict(zip(df['nucleophile_smiles'].unique(), df['nucleophile_id'].unique()))
    # encodings = pd.read_csv(
    #     'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/amidation/mols/morganFP/nucleophile.csv'
    # )
    # encodings.set_index('nucleophile_smiles', inplace=True)
    # ordered_ids = [smiles_to_id[s] for s in encodings.index]  # smiles to id, maintain order
    # encodings = dict(zip(ordered_ids, encodings.to_numpy().tolist()))  # {id: [ECFP1, ECFP2, ...]}
    # encodings = {'nucleophile_id': encodings}  # one more level to specify it's substrates
    #
    # s.predict(encoding_dict=None)

    # dir = '/Users/mac/Desktop/project deebo/deebo/deebo/single_run_logs/amidation/phase2/'
    # # output some quick stats from this optimization
    # with open(f'{dir}cache/scope.pkl', 'rb') as f:
    #     scope = pickle.load(f)
    # with open(f'{dir}cache/algo.pkl', 'rb') as f:
    #     algo = pickle.load(f)
    #
    # ranks = algo.ranking
    # old_arms = scope.arms
    # old_arm_labels = scope.arm_labels
    # print(f'rankings for {old_arm_labels}:')
    # print(f'{[old_arms[r] for r in ranks]}')
    # print(f'counts: {[algo.counts[r] for r in ranks]}\n')
    # print(f'means: {[round(algo.emp_means[r],2) for r in ranks]}\n')

    plot_prediction_model_accuracy()

    # args = np.argsort(a.emp_means)
    # print(f'ranks: {np.array(s.arms)[args]}')
    # print(f'means: {np.array(a.emp_means)[args]}')
    # print(f'counts: {np.array(a.counts)[args]}')

    # import matplotlib.pyplot as plt
    # df = pd.read_csv('/Users/mac/Desktop/project deebo/deebo/deebo/single_run_logs/amidation/phase2/history.csv')
    #
    # # plt.scatter(df['yield'], df['prediction'])
    # # plt.xlabel('experimental yield')
    # # plt.ylabel('predicted yield')
    # # plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k', alpha=0.5)
    # # import scipy
    # # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df['yield'], df['prediction'])
    # # print(r_value**2)
    # # plt.show()
    #