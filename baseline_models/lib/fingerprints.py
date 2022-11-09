from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np


def get_difference_fingerprint(rxn_smiles, rad, nbits):
    """
    Get difference Morgan fingerprint between reactants and products

    Args:
        rxn_smiles (str): the full reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        np.array: the difference fingerprint
    """
    reactants = rxn_smiles.split('>>')[0]
    products = rxn_smiles.split('>>')[-1]

    reactants_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactants), radius=rad, nBits=nbits))
    products_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(products), radius=rad, nBits=nbits))

    return reactants_fp - products_fp


def get_fingerprint(rxn_smiles, rad, nbits):
    """
    Get Morgan fingerprint of the reactants

    Args:
        rxn_smiles (str): the full reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        np.array: the reactant fingerprint
    """
    reactants = rxn_smiles.split('>>')[0]
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactants), radius=rad, nBits=nbits)


def get_df_fingerprints_rp(df, rad, nbits):
    """
    Get the dataframe with all the fingerprints of reactants and products

    Args:
        df (pd.DataFrame): the dataframe containing the reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        pd.DataFrame: the update dataframe
    """
    df['dipole'] = df['smiles'].apply(lambda x: [smi for smi in x.split('>')[0].split('.') if '+' in smi and '-' in smi][0])
    df['dipolarophile'] = df['smiles'].apply(lambda x: [smi for smi in x.split('>')[0].split('.') if not ('+' in smi and '-' in smi)][0])
    df['fingerprint_dipole'] = df['dipole'].apply(lambda x: 
            get_fingerprint(x, rad, nbits))
    df['fingerprint_dipolarophile'] = df['dipolarophile'].apply(lambda x: 
            get_fingerprint(x, rad, nbits))
    df['fingerprint_product'] = df['smiles'].apply(lambda x: get_fingerprint(x.split('>')[-1], rad, nbits))
    
    return df[['rxn_id', 'fingerprint_dipole', 'fingerprint_dipolarophile', 'fingerprint_product', 'DG_TS']]


def get_df_fingerprints(df, rad, nbits):
    """
    Get the dataframe with all the difference fingerprints

    Args:
        df (pd.DataFrame): the dataframe containing the reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        pd.DataFrame: the update dataframe
    """
    df['fingerprint'] = df['smiles'].apply(lambda x: 
            get_difference_fingerprint(x, rad, nbits))

    return df[['rxn_id', 'fingerprint','DG_TS']]
