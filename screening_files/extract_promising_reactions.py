import pandas as pd
from rdkit import Chem
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--predictions-file', type=str, required=True,
                    help='input .csv file containing predicted reaction and activation energies')
parser.add_argument('--threshold-dipolarophiles', type=int, default=26, 
                    help='threshold for the mean activation energy of a dipolarophile to gauge intrinsic reactivity')
parser.add_argument('--threshold-reverse-barrier', type=int, default=28,
                    help='threshold for the reverse barrier (to ensure irreversibility)')
parser.add_argument('--max-g-act', type=int, default=21,
                    help='maximal G_act for a dipole-dipolarophile combination to be considered suitable')


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


def get_molecule_lists(df_pred):
    """Extract the unique dipoles and dipolarophiles 

    Args:
        df_pred (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    dipole_list = df_pred.dipole.unique()
    dipolarophile_list = df_pred.dipolarophile.unique() 

    return dipole_list, dipolarophile_list


def get_individual_stat_list(df_pred, molecule_list, type='dipole'):
    """Get reaction statistics across the entire dataset for an individual molecule of a specific type

    Args:
        df_pred (pd.DataFrame): dataframe containing all the predictions
        molecule_list (List[strings]): list of SMILES
        type (str, optional): type of molecules. Defaults to 'dipole'.

    Returns:
        List: list of reaction statistics
    """
    molecule_stat_list = []

    for molecule in molecule_list:
        df_molecule = df_pred[df_pred[type] == molecule]
        molecule_stat_list.append([molecule, df_molecule['predicted_activation_energy'].min(), df_molecule['predicted_activation_energy'].mean(),
                                df_molecule['predicted_activation_energy'].std(), df_molecule['predicted_reaction_energy'].min(), 
                                df_molecule['predicted_reaction_energy'].mean(), df_molecule['predicted_reaction_energy'].std()])

    return molecule_stat_list


def get_statistics(df_pred, dipole_list, dipolarophile_list):
    """Get reaction statistics across the entire dataset

    Args:
        df_pred (pd.DataFrame): dataframe containing all the predictions
        dipole_list (List): list of dipole SMILES
        dipolarophile_list (List): list of dipolarophile SMILES

    Returns:
        (pd.DataFrame, pd.DataFrame): tuple of dataframes with dipole and dipolarophile statistics
    """
    dipole_stat_list = get_individual_stat_list(df_pred, dipole_list, type='dipole')
    dipolarophile_stat_list = get_individual_stat_list(df_pred, dipolarophile_list, type='dipolarophile')

    df_dipole_stat = pd.DataFrame(dipole_stat_list, columns=['dipole_smiles', 'G_act_min', 'G_act_mean', 'G_act_st_dev', 'G_r_min', 'G_r_mean', 'G_r_std_dev'])
    df_dipolarophile_stat = pd.DataFrame(dipolarophile_stat_list, columns=['dipolarophile_smiles', 'G_act_min', 'G_act_mean', 'G_act_st_dev', 'G_r_min', 'G_r_mean', 'G_r_std_dev']) 

    return df_dipole_stat, df_dipolarophile_stat


def select_dipolarophiles(df_dipolarophile_stat, threshold_dipolarophiles=26):
    """Select dipolarophiles that are not too reactive

    Args:
        df_dipolarophile_stat (pd.DataFrame): dataframe containing statistics for the dipolarophile
        threshold_dipolarophiles (int, optional): threshold to apply. Defaults to 26.

    Returns:
        pd.Series: series of dipolarophile SMILES
    """
    df_dipolarophile_not_too_reactive = df_dipolarophile_stat[df_dipolarophile_stat['G_act_mean'] > threshold_dipolarophiles]

    num_dipolarophiles = {}

    for threshold in range(12, 26, 2):
        num_dipolarophiles[threshold] = len(df_dipolarophile_not_too_reactive[df_dipolarophile_not_too_reactive['G_act_min'] < threshold])

    print(f'Total number of dipolarophiles: {len(df_dipolarophile_stat)}')
    print(f'Total number of dipolarophiles with average barrier above {threshold_dipolarophiles} kcal/mol: {len(df_dipolarophile_not_too_reactive)}')

    for key in num_dipolarophiles.keys():
        print(f'Total number of dipolarophiles with lowest barrier below {key}: {num_dipolarophiles[key]}')
    
    return df_dipolarophile_not_too_reactive['dipolarophile_smiles']


def filter_function(dipolarophile, G_act, G_r, dipolarophile_list, min_reverse_barrier, max_g_act):
    """ Function to determine whether reaction SMILES need to be filtered out or not.

    Args:
        dipolarophile (str): SMILES of the dipolarophile
        G_act (float): activation energy value
        G_r (float): reaction energy value
        dipolarophile_list (List[strings]): list of dipolarophile SMILES
        min_reverse_barrier (int): minimum barrier in the reverse direction
        max_g_act (int): maximum activation energy

    Returns:
        bool: whether or not to filter out the reaction SMILES
    """
    # make sure the reaction is fast, irreversible, and the dipolarophile is not too reactive
    if  dipolarophile in dipolarophile_list and (G_act - G_r) > min_reverse_barrier and G_act < max_g_act:
        return True
    else:
        return False


def get_final_filtered_rxn_smiles_synthetic(df_pred, valid_dipolarophiles, min_reverse_barrier=28, max_g_act=20):
    """ Get final dataframe with retained reaction SMILES

    Args:
        df_pred (pd.DataFrame): dataframe containing all the predictions
        valid_dipolarophiles (List[strings]): list of dipolarophile SMILES
        min_reverse_barrier (int): minimum barrier in the reverse direction
        max_g_act (int): maximum activation energy

    Returns:
        pd.DataFrame: dataframe with retained reaction SMILES
    """
    df_pred['final_filter'] = df_pred.apply(lambda x: filter_function(x['dipolarophile'], x['predicted_activation_energy'], 
                            x['predicted_reaction_energy'], valid_dipolarophiles, min_reverse_barrier, max_g_act), axis=1)
    
    return df_pred[df_pred['final_filter']][['rxn_id', 'rxn_smiles', 'dipole', 'dipolarophile', 
                                            'predicted_activation_energy', 'predicted_reaction_energy']]


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
    df_pred = pd.read_csv(args.predictions_file)

    # save the dipole and dipolarophile separately in dataframe and then get corresponding lists
    df_pred['smiles_dict'] = df_pred['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    df_pred = expand_df_pred(df_pred)
    dipole_list, dipolarophile_list = get_molecule_lists(df_pred)
    
    # get statistics for each dipole/dipolarophile and save them to corresponding files
    df_dipole_stat, df_dipolarophile_stat = get_statistics(df_pred, dipole_list, dipolarophile_list)  

    # select reactions and save them
    df_dipolarophiles_selected = select_dipolarophiles(df_dipolarophile_stat, args.threshold_dipolarophiles)
    promising_reactions = get_final_filtered_rxn_smiles_synthetic(df_pred,
                                df_dipolarophiles_selected.values.tolist(), args.threshold_reverse_barrier, args.max_g_act)
    promising_reactions[['rxn_id', 'rxn_smiles', 'predicted_activation_energy', 'predicted_reaction_energy']].to_csv('promising_synthetic_reactions.csv')
    
    # print some summarizing statistics
    print(f'Number of promising reactions selected: {len(promising_reactions)}')
    print(f'Number of unique dipoles selected: {len(promising_reactions.dipole.unique())}')
    print(f'Number of unique dipolarophiles selected: {len(promising_reactions.dipolarophile.unique())}')
