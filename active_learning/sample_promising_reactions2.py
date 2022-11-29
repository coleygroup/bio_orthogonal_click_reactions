import pandas as pd
from rdkit import Chem
from argparse import ArgumentParser
import os
import numpy as np


parser = ArgumentParser()
parser.add_argument('--predictions-file', type=str, required=True,
                    help='input .csv file containing predicted reaction and activation energies')


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
    """re-index a dataframe and reset 'rxn_id'

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: output dataframe
    """
    df = df[[column for column in df.columns if column != 'rxn_id']]
    df.reset_index(inplace=True)
    df = df[[column for column in df.columns if column != 'index']]
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
    promising_reactions = pd.read_csv(args.predictions_file)
    promising_reactions['smiles_dict'] = promising_reactions['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    promising_reactions = expand_df_pred(promising_reactions)

    print(len(promising_reactions))
    print(len(promising_reactions.dipole.unique()))

    # extract dipoles already done
    df_round1 = pd.read_csv('../screening_iteration0/validation_iteration0/iteration_0_validation.csv')
    df_round1['smiles_dict'] = df_round1['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    df_round1 = expand_df_pred(df_round1)
    dipoles_already_done1 = list(df_round1.dipole.unique())
    dipolarophiles_already_done1 = list(df_round1.dipolarophile.unique())  

    df_round2 = pd.read_csv('../screening_iteration1/validation_iteration1/iteration_1_validation.csv')
    df_round2['smiles_dict'] = df_round2['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    df_round2 = expand_df_pred(df_round2)
    dipoles_already_done2 = list(df_round2.dipole.unique())
    dipolarophiles_already_done2 = list(df_round2.dipolarophile.unique())    

    dipoles_already_done = dipoles_already_done1 + dipoles_already_done2
    dipolarophiles_already_done = dipolarophiles_already_done1 + dipolarophiles_already_done2

    # filter dipole list
    dipoles_not_yet_done = np.array(list(set(promising_reactions.dipole.unique()) - set(dipoles_already_done)))
    dipolarophiles_not_yet_done = np.array(list(set(promising_reactions.dipolarophile.unique()) - set(dipolarophiles_already_done)))
    print(len(dipoles_not_yet_done), len(dipolarophiles_not_yet_done))
    df_reaction_list = []

    # check whether the dipolarophile is a cyclooctyne, because you already have plenty of those
    promising_reactions['cyclooctyne'] = promising_reactions['rxn_smiles'].apply(lambda x: check_cyclooctyne(x))
    for dipole in dipoles_already_done:
        print(f'{dipole}: {len(promising_reactions[promising_reactions["dipole"] == dipole])}')
    for dipole in dipoles_not_yet_done:
        # if less then 10 synthetic reactions, sample all of them
        if len(promising_reactions[promising_reactions['dipole'] == dipole]) <= 10:
            df_reaction_list.append(promising_reactions[promising_reactions['dipole'] == dipole])
            n_already_sampled = len(promising_reactions[promising_reactions['dipole'] == dipole])
        else:
            # else, start by sampling all synthetic reactions that do not involve a cyclooctyne-based dipolarophile
            df_reaction_list.append(promising_reactions[promising_reactions['dipole'] == dipole][promising_reactions["cyclooctyne"] == False])
            n_already_sampled = len(promising_reactions[promising_reactions['dipole'] == dipole][promising_reactions["cyclooctyne"] == False])
            if n_already_sampled < 10:
                # if less than 10 sampled this manner, then sample first additional reactions involving previously unseen cyclooctynes
                promising_reactions_cyclooctyne = promising_reactions[promising_reactions['dipole'] == dipole][promising_reactions["cyclooctyne"] == True]
                promising_reactions_cyclooctyne['dipolarophile_not_yet_done'] = promising_reactions_cyclooctyne['dipolarophile'].apply(lambda x: x in dipolarophiles_not_yet_done)
                df_reaction_list.append(promising_reactions_cyclooctyne[promising_reactions_cyclooctyne['dipolarophile_not_yet_done'] == True])
                n_already_sampled += len(promising_reactions_cyclooctyne[promising_reactions_cyclooctyne['dipolarophile_not_yet_done'] == True])
                if n_already_sampled < 10:
                    # if this doesn't bring you to 10 reactions, sample previously seen dipolarophiles
                    df_reaction_list.append(promising_reactions_cyclooctyne[promising_reactions_cyclooctyne['dipole'] == dipole][promising_reactions_cyclooctyne['dipolarophile_not_yet_done'] == False].sample(
                        n = 10 - n_already_sampled, random_state=2))
    
    df_synthetic_validation = pd.concat(df_reaction_list)
    df_biofrag_validation = get_filtered_rxn_smiles_biofrag(df_synthetic_validation.dipole.unique(), "bio/predictions_bio_iteration2.csv")
    df_validation_combined = add_solvent_temp_column(reindex_df(pd.concat([df_synthetic_validation, df_biofrag_validation])))
    df_validation_combined[['rxn_id', 'rxn_smiles', 'solvent', 'temp', 'predicted_activation_energy', 'predicted_reaction_energy']].to_csv('iteration_2_validation.csv')
