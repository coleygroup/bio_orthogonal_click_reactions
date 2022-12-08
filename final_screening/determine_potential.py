import pandas as pd
from argparse import ArgumentParser
from rdkit import Chem


parser = ArgumentParser()
parser.add_argument('--predictions-file', type=str, required=True,
                    help='input .csv file containing predicted reaction and activation energies')
parser.add_argument('--dipole-statistics-file', type=str, required=True,
                    help='.csv file containing the statistics for the individual dipoles')
parser.add_argument('--output-file', type=str, default='screening_results.csv',
                    help='.csv file to which the output of the screening will be written')


def get_dipole_dict(statistics_file):
    """Get a dictionary with the lowest barriers with a biofragment for every dipole

    Args:
        statistics_file (str): path to the statistics file

    Returns:
        dict: a dictionary with the dipole SMILES as keys and 
        the minimum activation energy with a biofragment as value
    """
    df_dipoles = pd.read_csv(statistics_file)
    dipole_smiles = df_dipoles.dipole_smiles.tolist()
    G_act_min = df_dipoles.G_act_min.tolist()
    dipole_dict = dict(zip(dipole_smiles,G_act_min))
    
    return dipole_dict


def determine_bio_orthogonal_potential(data_point):
    """Determine the bio-orthogonal potential for a data-point; return None when error

    Args:
        data_point (pd.Series): a datapoint in the reaction dataframe

    Returns:
        int: _description_
    """
    try:
        return -(data_point['predicted_activation_energy'] - dipole_dict[data_point['dipole']])
    except:
        return None


def assign_dipolarophile_scaffold(rxn_smiles, scaffold_dict):
    """Assign dipolarophile scaffold.

    Args:
        rxn_smiles (str): a reaction SMILES string
        scaffold_dict (dict): a dictionary with the dipolarophile scaffolds as keys 
        and mol objects as values

    Returns:
        str: name of the dipolarophile scaffold
    """
    reactants = Chem.MolFromSmiles(rxn_smiles.split('>')[0])
    if len(reactants.GetSubstructMatch(scaffold_dict['cyclooctyne'])) > 0:
        return 'cyclooctyne'
    elif len(reactants.GetSubstructMatch(scaffold_dict['oxonorbornadiene'])) > 0:
        return 'oxo-norbornadiene'
    elif len(reactants.GetSubstructMatch(scaffold_dict['norbornene'])) > 0:
        return 'norbornene'
    else:
        return 'non-strained'


def assign_dipole_scaffold(rxn_smiles, co2_pattern):
    """Assign dipole scaffold.

    Args:
        rxn_smiles (_type_): _description_
        co2_pattern (RDKit.Mol): an rdkit mol object corresponding to the CO2-pattern

    Returns:
        str: name of the dipole scaffold
    """
    dipole = [smi for smi in rxn_smiles.split('>')[0].split('.') if (smi.count('+:') == 1 and smi.count('-:')==1)]
    try:
        dipole = dipole[0]
    except:
        print(rxn_smiles)
    dipole_mol = Chem.MolFromSmiles(dipole)
    if dipole.count("#") - dipole.count("C#N") - dipole.count("N#C") == 1:
        return 'propargyl'
    elif len(dipole_mol.GetSubstructMatch(co2_pattern)) > 0:
        return 'cyclic'
    else:
        return 'allyl'


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


def deduplicate_dataframe(df_pred):
    """Retain only a single datapoint per dipole-dipolarophile combination

    Args:
        df_pred (pd.DataFrame): input DataFrame

    Returns:
        df_pred_deduplicated: deduplicated DataFrame
    """
    df_pred = df_pred.sort_values(by='bio_orthogonal_potential', ascending=False)
    df_pred_deduplicated = df_pred.drop_duplicates(subset = ['dipole','dipolarophile'], keep= 'first')
    df_pred_deduplicated = df_pred_deduplicated.sort_values(by='rxn_id')

    return df_pred_deduplicated


if __name__ == '__main__':
    args = parser.parse_args()
    df_pred = pd.read_csv(args.predictions_file)

    # set up dipole_dict and get scaffold_dict + co2_pattern to facilitate dipole and dipolarophile assignment
    dipole_dict = get_dipole_dict(args.dipole_statistics_file)
    scaffold_dict = {'cyclooctyne': Chem.MolFromSmiles('C1C#CCCCCC1'), 
                    'oxonorbornadiene': Chem.MolFromSmiles('C1=CC2OC1C=C2'), 
                    'norbornene': Chem.MolFromSmiles('C1=CC2CC1CC2')}
    
    co2_pattern = Chem.MolFromSmarts('c(=O)o')

    # expand the dataframe with dipole+dipolarophile columns
    df_pred['smiles_dict'] = df_pred['rxn_smiles'].apply(lambda x: construct_smiles_dict(x))
    df_pred = expand_df_pred(df_pred)

    # expand the dataframe with lowest dipole barrier and determine bio-orthogonal potential
    df_pred['lowest_dipole_barrier'] = df_pred.apply(lambda x: dipole_dict[x['dipole']], axis=1)
    df_pred['bio_orthogonal_potential'] = df_pred.apply(
        lambda x: determine_bio_orthogonal_potential(x), axis=1)

    # remove datapoints for which you were unable to determine bio-orthogonal potential
    df_pred = df_pred.dropna(subset='bio_orthogonal_potential')

    # determine the dipole and dipolarophile scaffolds
    df_pred['dipolarophile_scaffold'] = df_pred['rxn_smiles'].apply(lambda x: assign_dipolarophile_scaffold(x, scaffold_dict))
    df_pred['dipole_scaffold'] = df_pred['rxn_smiles'].apply(lambda x: assign_dipole_scaffold(x, co2_pattern))

    # deduplicate df_pred and write final dataframe
    df_pred_deduplicated = deduplicate_dataframe(df_pred)
    
    print(len(df_pred_deduplicated),len(df_pred_deduplicated[df_pred_deduplicated['bio_orthogonal_potential'] < 0]))

    df_pred_deduplicated[['rxn_id', 'rxn_smiles', 'predicted_activation_energy',
                'predicted_reaction_energy','dipole','dipolarophile',
                'lowest_dipole_barrier','bio_orthogonal_potential',
                'dipolarophile_scaffold','dipole_scaffold']].to_csv(f'{args.output_file}')
    