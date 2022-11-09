import pandas as pd
from rdkit import Chem
from argparse import ArgumentParser
import numpy as np


parser = ArgumentParser()
parser.add_argument('--promising-reactions-file', type=str, required=True,
                    help='.csv file containing predicted reaction and activation energies for the promising reactions')


def strip_mapnum(smiles):
    """Strips map numbers from a SMILES string

    Args:
        smiles (str): SMILES string

    Returns:
        str: SMILES string with map numbers stripped
    """
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)


def construct_smiles_dict(rxn_smiles):
    """Construct a dictionary for a reaction SMILES containing dipole and dipolarophile identity

    Args:
        rxn_smiles (str): The input reaction SMILES

    Returns:
        dict: dictionary containing SMILES for dipole and dipolarophile respectively
    """
    reactants = list(map(lambda x: strip_mapnum(x), rxn_smiles.split('>')[0].split('.')))
    if '+' in reactants[0] and '-' in reactants[0]:
        dipole, dipolarophile = reactants[0], reactants[1]
    else:
        dipole, dipolarophile = reactants[1], reactants[0]
    
    return {'rxn_smiles': rxn_smiles, 'dipole': dipole, 'dipolarophile': dipolarophile}


def expand_df_pred(df_pred):
    """Expand the DataFrame with extra columns for the dipole and dipolarophile SMILES

    Args:
        df_pred (pd.DataFrame): input DataFrame

    Returns:
        pd.DataFrame: extended DataFrame
    """
    df_pred['dipole'] = df_pred.apply(lambda x: x['smiles_dict']['dipole'], axis=1)
    df_pred['dipolarophile'] = df_pred.apply(lambda x: x['smiles_dict']['dipolarophile'], axis=1) 

    return df_pred


def get_filtered_rxn_smiles_biofrag(dipole_list, predictions_biofrag_file):
    """Get the reactions for the biofragments

    Args:
        dipole_list (List[string]): list of dipole SMILES
        predictions_biofrag_file (string): path to the file containing the biofrag predictions

    Returns:
        pd.DataFrame: dataframe
    """
    df_pred_biofrag = pd.read_csv(predictions_biofrag_file)

    df_pred_biofrag['smiles_dict'] = df_pred_biofrag['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))

    df_pred_biofrag = expand_df_pred(df_pred_biofrag) #, smiles_dict)

    df_pred_biofrag['filter'] = df_pred_biofrag['dipole'].apply(lambda x: x in dipole_list)

    return df_pred_biofrag[df_pred_biofrag['filter']]


def check_cyclooctyne(rxn_smiles):
    """Check for the presence of cyclooctyne in a reaction SMILES 

    Args:
        rxn_smiles (string): reaction SMILES    

    Returns:
        bool: whether or not cyclooctyne is present
    """
    reactants = Chem.MolFromSmiles(rxn_smiles.split('>')[0])
    cyclooctyne = Chem.MolFromSmiles('C1C#CCCCCC1')

    if len(reactants.GetSubstructMatch(cyclooctyne)) > 0:
        return True
    return False


def reindex_df(df):
    """re-index a dataframe and rename old index as 'rxn_id'

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: output dataframe
    """
    df.reset_index(inplace=True)
    df = df.rename(columns= {'index':'rxn_id'})

    return df


def add_solvent_temp_column(df):
    """Add solvent and temperature columns

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: output dataframe
    """
    df['solvent'] = 'water'
    df['temp'] = 298.15

    return df


if __name__ == '__main__':
    args = parser.parse_args()
    promising_reactions = pd.read_csv(args.promising_reactions_file)
    promising_reactions['smiles_dict'] = promising_reactions['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    promising_reactions = expand_df_pred(promising_reactions)

    print(len(promising_reactions))
    print(len(promising_reactions.dipole.unique()))

    # extract dipoles already done
    df_round1 = pd.read_csv('synthetic_filter/round_1_validation.csv')
    df_round1['smiles_dict'] = df_round1['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    df_round1 = expand_df_pred(df_round1)
    dipoles_already_done = list(df_round1.dipole.unique())

    # filter dipole list
    dipoles_not_yet_done = np.array(list(set(promising_reactions.dipole.unique()) - set(dipoles_already_done)))
    print(len(dipoles_not_yet_done))
    df_reaction_list = []

    # check whether the dipolarophile is a cyclooctyne, because you already have plenty of those
    promising_reactions['cyclooctyne'] = promising_reactions['rxn_smiles'].apply(lambda x: check_cyclooctyne(x))
    for dipole in dipoles_not_yet_done:
        if len(promising_reactions[promising_reactions['dipole'] == dipole]) > 5:
            df_reaction_list.append(promising_reactions[promising_reactions['dipole'] == dipole][promising_reactions["cyclooctyne"] == False])
            n_already_sampled = len(promising_reactions[promising_reactions['dipole'] == dipole][promising_reactions["cyclooctyne"] == False])
            if n_already_sampled < 5:
                df_reaction_list.append(promising_reactions[promising_reactions['dipole'] == dipole][promising_reactions["cyclooctyne"] == True].sample(n = 5 - n_already_sampled))

    df_synthetic_validation = pd.concat(df_reaction_list) 
    df_biofrag_validation = get_filtered_rxn_smiles_biofrag(df_synthetic_validation.dipole.unique(), "bio_prediction_files/predictions_bio_round2.csv")
    df_validation_combined = add_solvent_temp_column(reindex_df(pd.concat([df_synthetic_validation, df_biofrag_validation])))
    df_validation_combined[['rxn_id', 'rxn_smiles', 'solvent', 'temp', 'predicted_activation_energy', 'predicted_reaction_energy']].to_csv('round_2_validation.csv')
