from chem_arms import simulate_propose_and_update
import algos_regret
import utils

import pathlib
import pandas as pd


def deoxyf():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv')

    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    ground_truth = ground_truth[['base_name',
                                 'fluoride_name',
                                 'substrate_name',
                                 'yield']]
    bases = ground_truth['base_name'].unique()
    fluorides = ground_truth['fluoride_name'].unique()
    substrates = ground_truth['substrate_name'].unique()

#######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'base_name': bases,
                  'fluoride_name': fluorides,
                  'substrate_name': substrates}
    arms_dict = {'base_name': bases,
                 'fluoride_name': fluorides,}
    algo = algos_regret.ETC(len(bases), explore_limit=2)
    wkdir = './dataset_logs/deoxyf/combo/'
    num_sims = 1
    num_round = 73
    num_exp = 1
    propose_mode = 'random'
#######################################################################################################################

    dir_name = f'{wkdir}{algo.__str__()}-{num_sims}s-{num_round}r-{num_exp}e/'
    p = pathlib.Path(dir_name)
    p.mkdir(parents=True)

    simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo,
                                dir=dir_name, num_sims=num_sims,
                                num_round=num_round, num_exp=num_exp, propose_mode=propose_mode)

if __name__ == '__main__':
    deoxyf()