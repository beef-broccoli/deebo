import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import DataStructs


def fingerprint_csv(mols, n_bits=2048, radius=2, output_path=''):

    """
    for all mols, generate mol from SMILES, then generate 2048 bit morganFP using rdkit implementation
    write a csv file, each row is one experiment, each column is one bit of the morgan fingerprint
    df size: (# of molecules, n_bits)

    Parameters
    ----------
    mols
    n_bits
    radius
    output_path

    Returns
    -------

    """

    mol_df = pd.DataFrame(mols, columns=['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(mol_df, smilesCol='SMILES', molCol='ROMol')
    assert mol_df.isnull().sum().sum() == 0, 'some rdkit mol files fail to generate'

    # featurize with morgan FP
    mol_df['morganFP'] = mol_df.apply(lambda x: GetMorganFingerprintAsBitVect(x['ROMol'], radius=radius, nBits=n_bits), axis=1)
    mol_df = mol_df.drop(['ROMol'], axis=1)  # faster lookup
    mol_df = mol_df.set_index('SMILES')  # use SMILES as df index

    cols = ['']*n_bits
    df = pd.DataFrame(columns=cols, index=mol_df.index)
    for index, row in mol_df.iterrows():  # not ideal, but only run it once to create full set, okay
        fp = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(row['morganFP'], fp)
        df.loc[index] = list(fp)
    assert df.isnull().sum().sum() == 0

    # save to csv
    if output_path is not None:
        df.to_csv(output_path)  # with index (name)


if __name__ == '__main__':
    phenols = {
        'p1': 'OC1=CC=C(C2=CC=CC=C2)C=C1',
        'p2': 'OC1=CC(C=CN2C(OC(C)(C)C)=O)=C2C=C1',
        'p3': 'OC(C(C(F)(F)F)=C1)=CC=C1C#N',
        'p4': 'OC1=CN=CC=C1C',
        'p5': 'OC1=CC=C2N=CC=CC2=C1',
        'p6': 'OC1=CC=CC(C(OCC)=O)=C1'
    }

    mes = {
        'm1': 'CC(OC(N1CCC(OS(C)(=O)=O)CC1)=O)(C)C',
        'm2': 'CCCCCCOS(C)(=O)=O',
        'm3': 'CCC(OS(C)(=O)=O)C',
        'm4': 'CS(OC1CC2=CC=CC=C2C1)(=O)=O',
        'm5': 'O=C(O[C@@H](COS(C)(=O)=O)C1)N1C(C=C2)=CC(F)=C2N3CCOCC3',
        'm6': 'O=C(OCC1=CC=CC=C1)N2C[C@H](OS(C)(=O)=O)CC2',
    }

    import pandas as pd
    df = pd.read_csv('phenols_morganFP.csv')
    df.insert(0,'id', value=phenols.keys())
    df.to_csv('phenols_morganFP_new.csv', index=False)

