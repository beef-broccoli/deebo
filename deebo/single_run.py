import pickle

import pandas as pd

from chem_arms import Scope
from chem_arms import propose_initial_experiments_interpolation, update_and_propose_interpolation
from algos_regret import UCB1Tuned
import utils


def deoxyf(dir='./single_run_logs/deoxyf/'):
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
    algo = UCB1Tuned(len(bases))

    # propose initial experiments
    propose_initial_experiments_interpolation(scope_dict, arms_dict, algo, dir=dir, num_exp=3)

    def query_and_fill(lookup_dict, workdir):
        proposed = pd.read_csv(f'{workdir}proposed_experiments.csv', index_col=0)
        keys_to_look_up = proposed[list(scope_dict.keys())].apply(lambda x: frozenset(x), axis=1)
        ys = [lookup_dict[x] for x in keys_to_look_up]
        proposed['yield'] = ys
        proposed.to_csv(f'{workdir}proposed_experiments.csv')
        return None

    rounds = 15
    for i in range(rounds):
        query_and_fill(ys_lookup_dict, dir)
        if i == rounds-1:
            update_and_propose_interpolation(dir=dir, num_exp=3, encoding_dict=encodings, update_only=True)
        else:
            update_and_propose_interpolation(dir=dir, num_exp=3, encoding_dict=encodings)

    with open('./single_run_logs/deoxyf/cache/scope.pkl', 'rb') as f:
        scope = pickle.load(f)

    # now trying to change arms
    old_arms = scope.arms
    old_arm_labels = scope.arm_labels
    scope.clear_arms()
    scope.build_arms(arms_dict_3)
    new_algo = UCB1Tuned(n_arms=len(scope.arms))
    results_for_new_arms = scope.sort_results_with_arms()
    print(results_for_new_arms)

    for ii in range(len(results_for_new_arms)):
        results = results_for_new_arms[ii]
        for jj in range(len(results)):
            new_algo.update(ii, results[jj])

    with open('./single_run_logs/deoxyf/cache/scope.pkl', 'wb') as f:
        pickle.dump(scope, f)
    with open('./single_run_logs/deoxyf/cache/algo.pkl', 'wb') as f:
        pickle.dump(new_algo, f)

    # the only not so great thing here is the log. arm index will mean different things after pivoting arms
    update_and_propose_interpolation(dir, 2, encoding_dict=encodings, propose_only=True)
    rounds = 20
    for i in range(rounds):
        query_and_fill(ys_lookup_dict, dir)
        if i == rounds-1:
            update_and_propose_interpolation(dir=dir, num_exp=2, encoding_dict=encodings, update_only=True)
        else:
            update_and_propose_interpolation(dir=dir, num_exp=2, encoding_dict=encodings)



if __name__ == '__main__':
    #deoxyf()
    with open('./single_run_logs/deoxyf/cache/algo.pkl', 'rb') as f:
        a = pickle.load(f)
    with open('./single_run_logs/deoxyf/cache/scope.pkl', 'rb') as f:
        s = pickle.load(f)
    print(s.sort_results_with_arms())
    print(sum(a.counts))
    print(a.emp_means)