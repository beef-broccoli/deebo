from chem_arms import simulate_propose_and_update, simulate_propose_and_update_interpolation
import algos_regret
import utils

import pathlib
import pandas as pd
import numpy as np


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
                 'fluoride_name': fluorides}
    #algo = algos_regret.ThompsonSamplingGaussianFixedVar(len(bases)*len(fluorides), assumed_sd=0.25)
    algo = algos_regret.UCB1Tuned(len(bases)*len(fluorides))
    wkdir = './dataset_logs/deoxyf/combo/'
    num_sims = 400
    num_round = 200
    num_exp = 1
    propose_mode = 'random'
#######################################################################################################################

    dir_name = f'{wkdir}{algo.__str__()}-{num_sims}s-{num_round}r-{num_exp}e/'
    p = pathlib.Path(dir_name)
    p.mkdir(parents=True)

    simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo,
                                dir=dir_name, num_sims=num_sims,
                                num_round=num_round, num_exp=num_exp, propose_mode=propose_mode)


def deoxyf_adversarial(): #TODO
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


def deoxyf_interpolation():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/deoxyf.csv')
    substrate_smiles_to_id = dict(zip(ground_truth['substrate_smiles'].unique(),
                                   ground_truth['substrate_name'].unique()))

    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    ground_truth = ground_truth[['base_name',
                                 'fluoride_name',
                                 'substrate_name',
                                 'yield']]
    bases = ground_truth['base_name'].unique()
    fluorides = ground_truth['fluoride_name'].unique()
    substrates = ground_truth['substrate_name'].unique()

    # make encodings for substrates
    encodings = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deoxyF/mols/ECFP/substrate.csv'
    )
    encodings.set_index('substrate_SMILES', inplace=True)
    ordered_ids = [substrate_smiles_to_id[s] for s in encodings.index]  # smiles to id, maintain order
    encodings = dict(zip(ordered_ids, encodings.to_numpy().tolist()))  # {id: [ECFP1, ECFP2, ...]}
    encodings = {'substrate_name': encodings}  # one more level to specify it's substrates

    #######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'base_name': bases,
                  'fluoride_name': fluorides,
                  'substrate_name': substrates}
    arms_dict = {'base_name': bases,
                 'fluoride_name': fluorides}
    # algo = algos_regret.ThompsonSamplingGaussianFixedVar(len(bases)*len(fluorides), assumed_sd=0.25)
    algo = algos_regret.UCB1Tuned(len(bases) * len(fluorides))
    wkdir = './dataset_logs/deoxyf/combo/interpolation/'
    num_sims = 200
    num_round = 200
    num_exp = 1
    propose_mode = 'random'
    batch_size = 50
    #######################################################################################################################

    dir_name = f'{wkdir}{algo.__str__()}-{num_sims}s-{num_round}r-{batch_size}b/'
    p = pathlib.Path(dir_name)
    p.mkdir(parents=True)

    simulate_propose_and_update_interpolation(scope_dict, arms_dict, encodings, ground_truth, algo,
                                              dir=dir_name, n_sims=num_sims, n_horizon=num_round,
                                              batch_size=batch_size)


def nickel_borylation():
    # first version:
    # 60% cutoff for yield, EtOH, use the top ligands identified by z score in paper
    # top 6 ligand is identical to the ones in the paper

    # second version:
    # 50% cutoff for yield, EtOH, use the top ligands identified by z score in paper
    # top 3 and top 8 ligands are identical to the ones in the paper

    top_three = ['Cy-JohnPhos', 'P(p-Anis)3', 'PPh2Cy']
    top_eight = ['PPh2Cy', 'CX-PCy', 'PPh3', 'P(p-F-Ph)3', 'P(p-Anis)3', 'Cy-JohnPhos', 'A-paPhos', 'Cy-PhenCar-Phos']

    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/nib-etoh.csv', index_col=0)

    ground_truth['yield'] = ground_truth['yield'].apply(utils.cutoffer, args=(50,))
    ligands = ground_truth['ligand_name'].unique()
    electrophiles = ground_truth['electrophile_id'].unique()

    #######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'electrophile_id': electrophiles,
                  'ligand_name': ligands,}
    arms_dict = {'ligand_name': ligands}
    # bayes ucb beta keeps picking the same arm with n_exp=100, but looks to be optimal
    algos = [algos_regret.ThompsonSamplingBeta(len(ligands)),
             algos_regret.ThompsonSamplingGaussianFixedVarSquared(len(ligands)),
             algos_regret.UCB1Tuned(len(ligands)),
             algos_regret.UCB1(len(ligands)),
             algos_regret.BayesUCBGaussianSquared(len(ligands)),
             algos_regret.BayesUCBBeta(len(ligands)),
             algos_regret.BayesUCBBetaPPF(len(ligands)),
             algos_regret.AnnealingEpsilonGreedy(len(ligands)),
             algos_regret.Random(len(ligands))
             ]
    algos = [
             algos_regret.Random(len(ligands))
             ]
    wkdir = './dataset_logs/nib/etoh-50cutoff/'
    num_sims = 500
    num_round = 100
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


def cn():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/cn-processed.csv')

    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    ground_truth = ground_truth[['base_name',
                                 'ligand_name',
                                 'substrate_id',
                                 'additive_id',
                                 'yield']]

    bases = ground_truth['base_name'].unique()
    ligands = ground_truth['ligand_name'].unique()
    additives = ground_truth['additive_id'].unique()
    substrates = ground_truth['substrate_id'].unique()

    #######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'base_name': bases,
                  'ligand_name': ligands,
                  'additive_id': additives,
                  'substrate_id': substrates}
    arms_dict = {'base_name': bases,
                 'ligand_name': ligands}
    n_arms = len(bases)*len(ligands)
    algos = [algos_regret.UCB1Tuned(n_arms),
             algos_regret.UCB1(n_arms),
             algos_regret.AnnealingEpsilonGreedy(n_arms),
             algos_regret.ThompsonSamplingGaussianFixedVar(n_arms, assumed_sd=0.25),
             algos_regret.ThompsonSamplingGaussianFixedVarSquared(n_arms),
             algos_regret.BayesUCBGaussian(n_arms, assumed_sd=0.25, c=2),
             algos_regret.BayesUCBGaussianSquared(n_arms, c=2),
             algos_regret.Random(n_arms)]
    # algo = algos_regret.ThompsonSamplingGaussianFixedVar(len(bases)*len(fluorides), assumed_sd=0.25)
    wkdir = './dataset_logs/cn/'
    num_sims = 500
    num_round = 100
    num_exp = 1
    propose_mode = 'random'
    #######################################################################################################################
    for algo in algos:
        dir_name = f'{wkdir}{algo.__str__()}-{num_sims}s-{num_round}r-{num_exp}e/'
        p = pathlib.Path(dir_name)
        p.mkdir(parents=True)

        simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo,
                                    dir=dir_name, num_sims=num_sims,
                                    num_round=num_round, num_exp=num_exp, propose_mode=propose_mode)


def arylation():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')

    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    ground_truth = ground_truth[['ligand_name',
                                 'electrophile_id',
                                 'nucleophile_id',
                                 'yield']]

    ligands = ground_truth['ligand_name'].unique()
    elecs = ground_truth['electrophile_id'].unique()
    nucs = ground_truth['nucleophile_id'].unique()

    #######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'ligand_name': ligands,
                  'electrophile_id': elecs,
                  'nucleophile_id': nucs}
    arms_dict = {'ligand_name': ligands}
    n_arms = len(ligands)
    # algos = [algos_regret.UCB1Tuned(n_arms),
    #          algos_regret.UCB1(n_arms),
    #          algos_regret.AnnealingEpsilonGreedy(n_arms),
    #          algos_regret.ThompsonSamplingGaussianFixedVar(n_arms, assumed_sd=0.25),
    #          algos_regret.ThompsonSamplingGaussianFixedVarSquared(n_arms),
    #          algos_regret.BayesUCBGaussian(n_arms, assumed_sd=0.25, c=2),
    #          algos_regret.BayesUCBGaussianSquared(n_arms, c=2),
    #          algos_regret.Random(n_arms)]
    algos = [algos_regret.BayesUCBGaussianSquared(n_arms, c=2),
             algos_regret.Random(n_arms)]
    # algo = algos_regret.ThompsonSamplingGaussianFixedVar(len(bases)*len(fluorides), assumed_sd=0.25)
    wkdir = './dataset_logs/aryl-scope-ligand/'
    num_sims = 500
    num_round = 200
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
                                    predict=False)


def arylation_expansion():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv')


    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    ground_truth = ground_truth[['ligand_name',
                                 'electrophile_id',
                                 'nucleophile_id',
                                 'yield']]

    ligands = ground_truth['ligand_name'].unique()
    elecs = ground_truth['electrophile_id'].unique()
    nucs = ground_truth['nucleophile_id'].unique()

    #######################################################################################################################
    scope_dict = {'nucleophile_id': ['nA', 'nB', 'nC', 'nD'],
                  'electrophile_id': ['e1', 'e2', 'e3', 'e4'],
                  'ligand_name': ligands}
    expansion_dict = {50: {'nucleophile_id': ['nE', 'nF', 'nG', 'nI']},
                      100: {'electrophile_id': ['e5', 'e7', 'e9', 'e10']}}
    arms_dict = {'ligand_name': ligands}
    n_arms = len(ligands)
    algos = [algos_regret.UCB1Tuned(n_arms),
             algos_regret.UCB1(n_arms)]
    wkdir = './dataset_logs/aryl-scope-ligand/expansion/scenario1/'
    num_sims = 500
    num_round = 300
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
                                    expansion_dict=expansion_dict)


def amidation():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/amidation.csv')

    ground_truth['yield'] = ground_truth['yield'].apply(utils.scaler)
    short_name_dict = {
        '1-Methylimidazole': 'MeIm',
        '2,6-Lutidine': 'lutidine',
        'N-methylmorpholine': 'MeMorph',
        'Diisopropylethylamine': 'DIPEA'
    }
    nuc_id_dict = dict(zip(ground_truth['nucleophile_name'].unique(),
                           [f'n{num}' for num in np.arange(len(ground_truth['nucleophile_name'].unique())) + 1]))
    ground_truth['nucleophile_id'] = ground_truth['nucleophile_name'].apply(lambda x: nuc_id_dict[x])
    ground_truth = ground_truth[['nucleophile_id', 'base_name', 'activator_name', 'solvent_name','yield']]

    bases = ground_truth['base_name'].unique()
    activators = ground_truth['activator_name'].unique()
    solvents = ground_truth['solvent_name'].unique()
    nucs = ground_truth['nucleophile_id'].unique()

    #######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'nucleophile_id': nucs,
                  'base_name': bases,
                  'activator_name': activators,
                  'solvent_name': solvents}
    arms_dict = {'activator_name': activators,
                 'base_name': bases,}
    n_arms = len(activators)*len(bases)
    algos = [algos_regret.UCB1Tuned(n_arms),
             algos_regret.UCB1(n_arms),
             algos_regret.AnnealingEpsilonGreedy(n_arms),
             algos_regret.ThompsonSamplingGaussianFixedVar(n_arms, assumed_sd=0.25),
             algos_regret.ThompsonSamplingGaussianFixedVarSquared(n_arms),
             algos_regret.BayesUCBGaussian(n_arms, assumed_sd=0.25, c=2),
             algos_regret.BayesUCBGaussianSquared(n_arms, c=2),
             algos_regret.Random(n_arms)]
    wkdir = './dataset_logs/amidation/combo/'
    num_sims = 500
    num_round = 96
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
                                    predict=False)


def cn_maldi():
    # fetch ground truth data
    ground_truth = pd.read_csv(
        'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/maldi-amine.csv')
    eic_max = ground_truth['EIC(+)[M+H] Product Area'].max()
    ground_truth['yield'] = ground_truth['EIC(+)[M+H] Product Area'] / eic_max  # being naughty here, "yield"

    ground_truth = ground_truth[['substrate_id',
                                 'condition',
                                 'yield',]]

    conditions = ground_truth['condition'].unique()
    substrates = ground_truth['substrate_id'].unique()

    #######################################################################################################################
    # build dictionary for acquisition
    scope_dict = {'condition': conditions,
                  'substrate_id': substrates}
    arms_dict = {'condition': conditions}
    n_arms = len(conditions)
    # algos = [algos_regret.UCB1Tuned(n_arms),
    #          algos_regret.UCB1(n_arms),
    #          algos_regret.AnnealingEpsilonGreedy(n_arms),
    #          algos_regret.ThompsonSamplingGaussianFixedVar(n_arms, assumed_sd=0.25),
    #          algos_regret.ThompsonSamplingGaussianFixedVarSquared(n_arms),
    #          algos_regret.BayesUCBGaussian(n_arms, assumed_sd=0.25, c=2),
    #          algos_regret.BayesUCBGaussianSquared(n_arms, c=2),
    #          algos_regret.Random(n_arms)]
    algos = [algos_regret.Random(n_arms)]
    # algo = algos_regret.ThompsonSamplingGaussianFixedVar(len(bases)*len(fluorides), assumed_sd=0.25)
    wkdir = './test/merck-maldi/amine/'
    num_sims = 1000
    num_round = 200
    num_exp = 1
    propose_mode = 'random'
    #######################################################################################################################
    for algo in algos:
        dir_name = f'{wkdir}{algo.__str__()}-{num_sims}s-{num_round}r-{num_exp}e/'
        p = pathlib.Path(dir_name)
        p.mkdir(parents=True)

        simulate_propose_and_update(scope_dict, arms_dict, ground_truth, algo,
                                    dir=dir_name, num_sims=num_sims,
                                    num_round=num_round, num_exp=num_exp, propose_mode=propose_mode)


if __name__ == '__main__':
    cn_maldi()
