import pandas as pd
from rdkit import Chem
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument('--predictions-file', type=str, required=True,
                    help='input .csv file containing predicted reaction and activation energies')
parser.add_argument('--threshold-lower', type=int, default=26,
                    help='threshold to decide whether a dipole is too reactive with biofragments')


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
        df_pred (pd.DataFrame): dataframe containing all the predictions

    Returns:
        List[List]: list of the statistics
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


def select_dipoles(df_dipole_stat, strongly_endothermic_dipoles, threshold_lower=26):
    """Select dipoles that are not too reactive

    Args:
        df_dipole_stat (pd.DataFrame): dataframe containing statistics for the dipoles
        strongly_endothermic_dipoles (List[string]): list of strongly endothermic dipole SMILES
        threshold_lower (int, optional): threshold to apply. Defaults to 26.

    Returns:
        pd.Series: series of dipole SMILES
    """
    num_dipoles = {}

    for threshold in range(18, 36, 2):
        num_dipoles[threshold] = len(df_dipole_stat[df_dipole_stat['G_act_min'] > threshold])
    
    print(f'Total number of dipoles: {len(df_dipole_stat)}')
    for key in num_dipoles.keys():
        print(f'Total number of dipoles with lowest barrier above {key} kcal/mol: {num_dipoles[key]}')

    retained_endothermic_dipoles = set(strongly_endothermic_dipoles) - set(df_dipole_stat['dipole_smiles'].values.tolist())

    print(f'\nNumber of strongly endothermic dipoles retained: {len(retained_endothermic_dipoles)}')

    selected_dipoles_list = list(set(df_dipole_stat[df_dipole_stat['G_act_min'] > threshold_lower]['dipole_smiles'].values.tolist() + 
                                        list(retained_endothermic_dipoles)))

    selected_dipoles = pd.DataFrame(selected_dipoles_list, columns=['dipole_smiles'])

    return selected_dipoles


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


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs('bio_filter', exist_ok=True)
    df_pred = pd.read_csv(args.predictions_file)

    # reactions that are strongly endothermic should not be taken into account when determining
    # the statistics -> remove them from the dataframe (but retain them for further analysis since dipoles which are
    # strongly endothermic with all biofragments are also promising)
    df_strongly_endothermic =  df_pred[df_pred['predicted_reaction_energy'] > 5]
    df_pred = df_pred[df_pred['predicted_reaction_energy'] < 5]

    # save the dipole and dipolarophile separately in dataframe and then get corresponding lists
    df_pred['smiles_dict'] = df_pred['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    df_pred = expand_df_pred(df_pred)
    dipole_list, dipolarophile_list = get_molecule_lists(df_pred)

    # extract the highly endothermic dipoles and store them in list
    df_strongly_endothermic['smiles_dict'] = df_strongly_endothermic['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    df_strongly_endothermic = expand_df_pred(df_strongly_endothermic)
    strongly_endothermic_dipoles = df_strongly_endothermic['dipole'].unique()
    
    # get statistics for each dipole/dipolarophile and save them to corresponding files
    df_dipole_stat, df_dipolarophile_stat = get_statistics(df_pred, dipole_list, dipolarophile_list)
    df_dipole_stat.to_csv(f'bio_filter/dipole_stat_biofrag.csv')
    df_dipolarophile_stat.to_csv(f'bio_filter/dipolarophile_stat_biofrag.csv')

    # select dipoles based on criteria
    df_dipoles_selected = select_dipoles(df_dipole_stat, strongly_endothermic_dipoles, args.threshold_lower)
    
    df_dipoles_selected.to_csv('bio_filter/dipoles_above_threshold.csv')

    print(f'\n{len(df_dipoles_selected)} dipoles selected!')
