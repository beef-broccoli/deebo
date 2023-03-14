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


def deoxyf_adversarial():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv')
    seg1 = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf-seg1.csv')
    seg2 = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf-seg2.csv')
    seg3 = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf-seg3.csv')

    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    ground_truth = ground_truth[['base_name',
                                 'fluoride_name',
                                 'substrate_name',
                                 'yield']]
    bases = ground_truth['base_name'].unique()
    fluorides = ground_truth['fluoride_name'].unique()
    substrates_all = ground_truth['substrate_name'].unique()

    substrates_1 = seg1['substrate_name'].unique()
    substrates_2 = seg2['substrate_name'].unique()
    substrates_3 = seg3['substrate_name'].unique()

#######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'base_name': bases,
                  'fluoride_name': fluorides,
                  'substrate_name': substrates_1}  # substrate_1 to start
    expansion_dict = {50: {'substrate_name': substrates_2},
                      100: {'substrate_name': substrates_3}}
    arms_dict = {'base_name': bases,
                 'fluoride_name': fluorides,}
    algo = algos_regret.BayesUCBGaussian(n_arms=20)
    wkdir = './dataset_logs/deoxyf/adversarial/combo/'
    num_sims = 400
    num_round = 150
    num_exp = 1
    propose_mode = 'random'
#######################################################################################################################

    dir_name = f'{wkdir}{algo.__str__()}-{num_sims}s-{num_round}r-{num_exp}e/'
    p = pathlib.Path(dir_name)
    p.mkdir(parents=True)

    simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo,
                                dir=dir_name, num_sims=num_sims,
                                num_round=num_round, num_exp=num_exp, propose_mode=propose_mode,
                                expansion_dict=expansion_dict
                                )


def nickel_borylation():
    # 60% cutoff for yield, EtOH, use the top ligands identified by z score in paper
    # top 6 ligand is identical to the ones in the paper

    top_six = ['PPh2Cy', 'CX-PCy', 'PPh3', 'P(p-F-Ph)3', 'P(p-Anis)3', 'Cy-JohnPhos']

    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/nib-etoh.csv', index_col=0)

    ground_truth['yield'] = ground_truth['yield'].apply(utils.cutoffer, args=(60,))
    ligands = ground_truth['ligand_name'].unique()
    electrophiles = ground_truth['electrophile_id'].unique()

    #######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'electrophile_id': electrophiles,
                  'ligand_name': ligands,}
    arms_dict = {'ligand_name': ligands}
    algos = [algos_regret.BayesUCBBeta(n_arms=len(ligands)),  # keeps picking CyJohnPhos
             algos_regret.BayesUCBGaussian(n_arms=len(ligands)),
             algos_regret.ThompsonSamplingBeta(n_arms=len(ligands)),
             algos_regret.ThompsonSamplingGaussianFixedVar(n_arms=len(ligands)),
             algos_regret.UCB1Tuned(n_arms=len(ligands)),
             algos_regret.UCB1(n_arms=len(ligands)),
             algos_regret.Random(n_arms=len(ligands)),
             ]
    wkdir = './dataset_logs/nib/etoh-60cutoff/'
    num_sims = 500
    num_round = 75
    num_exp = 1
    propose_mode = 'random'
    #######################################################################################################################

    for algo in algos:
        dir_name = f'{wkdir}{algo.__str__()}-{num_sims}s-{num_round}r-{num_exp}e/'
        p = pathlib.Path(dir_name)
        p.mkdir(parents=True)

        simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo,
                                    dir=dir_name, num_sims=num_sims,
                                    num_round=num_round, num_exp=num_exp, propose_mode=propose_mode,
                                    )
    return


if __name__ == '__main__':
    nickel_borylation()