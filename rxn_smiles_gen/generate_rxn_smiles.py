#!/usr/bin/python
from rdkit import Chem
import argparse, logging
import time
import csv
import logging

from rdkit import Chem
from itertools import combinations_with_replacement
from rdchiral.main import rdchiralRunText
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from math import isclose
import pandas as pd

from pebble import ProcessPool

parser = argparse.ArgumentParser(description="rxn-smiles generator")
parser.add_argument(
    "--file-name-dipoles", required=True, help="csv input file for dipoles"
)
parser.add_argument(
    "--file-name-dipolarophiles",
    required=True,
    help="csv input file for dipolarophiles",
)
parser.add_argument(
    "--output-base-name", required=True, help="base name for the output files"
)
parser.add_argument("--num-cores", required=False, default=48, help="number of cores")


def get_smarts_list():
    """Get the SMARTS lists in a list

    Returns:
        List[str]: a list of SMARTS strings
    """
    smarts_list = []

    # 'aromatic' dipoles
    smarts_list.append(
        "([*:1]:[*+:2][*-:3].[C,N:4]=[C:5])>>[*:1]1[*;+0:2][*;-0:3][C,N:4][C:5]1"
    )
    smarts_list.append(
        "([*:1]:[*+:2][*-:3].[C,N:4]=[C:5])>>[*:1]1[*;+0:2][*;-0:3][C:5][C,N:4]1"
    )
    smarts_list.append(
        "([*:1]:[*+:2][*-:3].[C:4]#[C:5])>>[*:1]1[*;+0:2][*;-0:3][C:4]=[C:5]1"
    )
    smarts_list.append(
        "([*:1]:[*+:2][*-:3].[C:4]#[C:5])>>[*:1]1[*;+0:2][*;-0:3][C:5]=[C:4]1"
    )

    # 0 triple bonds
    smarts_list.append(
        "([*:1]=[*+:2][*-:3].[C,N:4]=[C:5])>>[*:1]1[*;+0:2][*;-0:3][C,N:4][C:5]1"
    )
    smarts_list.append(
        "([*:1]=[*+:2][*-:3].[C,N:4]=[C:5])>>[*:1]1[*;+0:2][*;-0:3][C:5][C,N:4]1"
    )

    # 1 triple bond
    smarts_list.append(
        "([*:1]#[*+:2][*-:3].[C,N:4]=[C:5])>>[*:1]1=[*;+0:2][*;-0:3][C,N:4][C:5]1"
    )
    smarts_list.append(
        "([*:1]#[*+:2][*-:3].[C,N:4]=[C:5])>>[*:1]1=[*;+0:2][*;-0:3][C:5][C,N:4]1"
    )
    smarts_list.append(
        "([*:1]=[*+:2][*-:3].[C:4]#[C:5])>>[*:1]1[*;+0:2][*;-0:3][C:4]=[C:5]1"
    )
    smarts_list.append(
        "([*:1]=[*+:2][*-:3].[C:4]#[C:5])>>[*:1]1[*;+0:2][*;-0:3][C:5]=[C:4]1"
    )

    # 2 triple bonds
    smarts_list.append(
        "([*:1]#[*+:2][*-:3].[C:4]#[C:5])>>[*:1]1=[*;+0:2][*;-0:3][C:4]=[C:5]1"
    )
    smarts_list.append(
        "([*:1]#[*+:2][*-:3].[C:4]#[C:5])>>[*:1]1=[*;+0:2][*;-0:3][C:5]=[C:4]1"
    )

    return smarts_list


num_elec_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Cl': 17, 'Br': 35}


def unmap_unstereo_smiles(smiles):
    ''' Remove atom mapping and stereo-assignments from SMILES simultaneously '''
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)), isomericSmiles=False)


def map_smiles(smiles):
    ''' Map atoms of SMILES based on index '''
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def unmap_smiles(smiles):
    ''' Unmap atoms of SMILES '''
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)


def get_conformer(mol):
    ''' Get a single MMFF optimized conformer of mol-object '''
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        return mol.GetConformers()[0]
    except:
        return None


def get_chiral_centers_product(mol, map_num_dict, active_idx):
    ''' Obtain a list of map numbers, corresponding to the stereocenters of the product '''
    mol_copy = Chem.Mol(mol)
    [atom.SetAtomMapNum(0) for atom in mol_copy.GetAtoms()]
    stereocenters_map_num = []
    si = Chem.FindPotentialStereo(mol_copy)

    for element in si:
        if str(element.type) == 'Atom_Tetrahedral' and element.centeredOn in active_idx:
            stereocenters_map_num.append(map_num_dict[element.centeredOn])
    
    return stereocenters_map_num


def get_product_isomers(stereocenters_map_num, p_mol):
    ''' Make combinatorial combinations of the different stereoassignments possible for the sites in the stereocenters_map_num list '''
    product_isomers = set()
    stereocenters = [atom.GetIdx() for atom in p_mol.GetAtoms() if atom.GetAtomMapNum() in stereocenters_map_num]
    for center in stereocenters:
        p_mol.GetAtomWithIdx(center).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    product_isomers.add(Chem.MolToSmiles(p_mol))

    for l in range(len(stereocenters)):
        for subset in combinations_with_replacement(stereocenters, l):
            for center in subset:
                p_mol.GetAtomWithIdx(center).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            product_isomers.add(Chem.MolToSmiles(p_mol)) 

    return product_isomers


def get_dicts_and_idx_list(p_mol, active_map_nums):
    ''' Get a list of active indices, i.e., atoms undergoing a change in bond order, as well as a dictionary of the map nums '''
    active_idx = [atom.GetIdx() for atom in p_mol.GetAtoms() if atom.GetAtomMapNum() in active_map_nums]
    map_num_dict = {atom.GetIdx(): atom.GetAtomMapNum() for atom in p_mol.GetAtoms()}

    return map_num_dict, active_idx


def check_isomers(isomer_list, map_num_for_dihedrals, angle_reactants):
    ''' 
    Generate conformers for the product isomers and compare the angle for the selected dihedral with the corresponding angle in the reactants 
    In case of failure to generate isomer: assign arbitrarily high value to give lowest priority 
    Select isomer if difference angle between reactants and isomer is < 15°
    If none of the isomers are sufficiently close in dihedral angle to the reactants, select the one with the lowest deviation
    '''
    angle_dev_isomers = []
    for isomer in isomer_list:
        isomer_mol = Chem.MolFromSmiles(isomer)
        product_idx_for_dihedral = [atom.GetIdx() for atom in isomer_mol.GetAtoms() if atom.GetAtomMapNum() in map_num_for_dihedrals]
        conf = get_conformer(isomer_mol)
        if conf is not None:
            angle_isomer = rdMolTransforms.GetDihedralDeg(conf, product_idx_for_dihedral[0],  product_idx_for_dihedral[1], product_idx_for_dihedral[2], product_idx_for_dihedral[3])
            # angle_reactants is either +/-0 or +/-180; approach the angle from both the positive or negative side is OK (returned angles are in range [-180,180])
            difference_angle = abs(angle_reactants) - abs(angle_isomer)
        else:
            # if angle isomer could not be determined, set the difference angle artifically high to give it lowest priority
            difference_angle = 1000
        if abs(difference_angle) < 15:
            return isomer
        else:
           angle_dev_isomers.append(abs(difference_angle)) 

    min_dev = min(angle_dev_isomers)
    # select the isomer with the lowest dihedral angle deviation (but make sure the isomers are sufficiently distinct in the first place)
    if min_dev < 1000 and (max(angle_dev_isomers) - min(angle_dev_isomers)) > 10:
        for idx, value in enumerate(angle_dev_isomers):
            if isclose(value, min_dev, abs_tol = 1):
                return isomer_list[idx]
    else:
        raise ValueError


def get_dipolarophile_map_num_range(reactant_smiles):
    """Get the range of the atom map numbers corresponding to the dipolarophiles"""
    reactant_smiles = map_smiles(reactant_smiles)
    reactant_smiles_list = reactant_smiles.split(".")
    for smi in reactant_smiles_list:
        if "+" in smi and "-" in smi:
            continue
        else:
            map_num_list = [
                atom.GetAtomMapNum() for atom in Chem.MolFromSmiles(smi).GetAtoms()
            ]

    return range(min(map_num_list), max(map_num_list) + 1)


def get_stereo_corr_products(reactant_smiles, reaction_smarts):
    ''' 
    Main function which generates products with RDChiral for a given reactant SMILES and SMARTS string 
    and then returns a stereocompatible stereo-isomer SMILES 
    '''
    stereo_corr_products = []
    outcomes = rdchiralRunText(reaction_smarts, reactant_smiles, keep_mapnums=False, combine_enantiomers=True, return_mapped=True) 
    r_mol = Chem.MolFromSmiles(reactant_smiles)
    # Determine range of mapnum of dipolarophile so that you can determine whether two steroecenters are 
    # in dipolarophile or not
    dipolarophile_range = get_dipolarophile_map_num_range(reactant_smiles)
    
    # Set atom map numbering in r_mol
    [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in r_mol.GetAtoms()]

    for product in outcomes[0]:
        # Identify all the active atoms, i.e., all those which undergo change in bonding
        active_map_nums = list(set(outcomes[1][product][1]))
        p_mol = Chem.MolFromSmiles(outcomes[1][product][0])
        map_num_dict, active_idx = get_dicts_and_idx_list(p_mol, active_map_nums)
        # Identify all the active stereocenters in the product
        stereocenters_product_map_num = get_chiral_centers_product(p_mol, map_num_dict, active_idx)

        stereocenters_from_dipolarophile_map_num = [map_num for map_num in stereocenters_product_map_num if map_num in dipolarophile_range]
        stereocenters_dipolarophile = [atom.GetIdx() for atom in r_mol.GetAtoms() if atom.GetAtomMapNum() in stereocenters_from_dipolarophile_map_num]

        # Get norbornene and oxo-norbornadiene patterns since stereochemistry for these types of reactions does not need to be constrained
        mol_norbornene = Chem.MolFromSmiles('C1=CC2CCC1C2')
        mol_oxo_norbornadiene =  Chem.MolFromSmiles('C1=CC2C=CC1O2')

        # If only a single stereocenter is present in the dipolarophile, or if dipolarophile is rigid (norbornene and oxo-norbornadiene),
        # then this doesn't matter and no further checks are needed, lowest product conformer can be taken without restrictions
        if len(stereocenters_from_dipolarophile_map_num) <= 1 or len(r_mol.GetSubstructMatch(mol_norbornene)) != 0 or len(r_mol.GetSubstructMatch(mol_oxo_norbornadiene)) != 0:
            stereo_corr_products.append(Chem.MolToSmiles(p_mol))
        # If two stereocenters in dipolarophile, select compatible isomer (all other stereocenters can be set independently by autodE)
        elif len(stereocenters_from_dipolarophile_map_num) == 2:
            conf_r = get_conformer(r_mol)
            neighbors_cistrans = []
            # First try to select a neighbor that is not in ring (both neighbors in ring is problematic for dihedral determination), 
            # if this doesn't work, then settle for a ring-atom 
            for center in stereocenters_dipolarophile:
                try:
                    neighbors_cistrans.append([atom for atom in r_mol.GetAtomWithIdx(center).GetNeighbors() \
                        if (atom.GetIdx() not in stereocenters_dipolarophile and atom.IsInRing() == False)][0])
                except IndexError:
                    neighbors_cistrans.append([atom for atom in r_mol.GetAtomWithIdx(center).GetNeighbors() \
                        if atom.GetIdx() not in stereocenters_dipolarophile][0])

            map_num_for_dihedral = list(map(lambda x: x.GetAtomMapNum(), [neighbors_cistrans[0], r_mol.GetAtomWithIdx(stereocenters_dipolarophile[0]), 
                r_mol.GetAtomWithIdx(stereocenters_dipolarophile[1]), neighbors_cistrans[1]]))
                
            reactant_idx_for_dihedral = [atom.GetIdx() for atom in r_mol.GetAtoms() if atom.GetAtomMapNum() in map_num_for_dihedral]              
            dihedral_r = rdMolTransforms.GetDihedralDeg(conf_r, reactant_idx_for_dihedral[0],  reactant_idx_for_dihedral[1], reactant_idx_for_dihedral[2], reactant_idx_for_dihedral[3])
            
            product_isomers = get_product_isomers(stereocenters_from_dipolarophile_map_num, p_mol)

            try:
                stereo_corr_product = check_isomers(list(product_isomers), map_num_for_dihedral, dihedral_r)
                stereo_corr_products.append(stereo_corr_product)
            except ValueError:
                print(f'Error for {reactant_smiles}')               
                continue

    return stereo_corr_products


def map_smiles(smiles):
    """ map a SMILES string """
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def get_rxn_smiles(reaction_tuple):
    """ get reaction SMILES list based on an ID, reactant SMILES and a SMARTS list to apply to the latter """
    id, reactant_smiles, smarts_list = reaction_tuple
    unmapped_products_no_stereo = []
    rxn_smiles_list_tmp = []
    for smarts in smarts_list:
        try:
            corr_prods = get_stereo_corr_products(reactant_smiles, smarts)
            idx = 0
            for corr_prod in corr_prods:
                if unmap_unstereo_smiles(corr_prod) not in unmapped_products_no_stereo:
                    rxn_smiles_list_tmp.append(
                        [
                            str(id) + str(idx),
                            f"{map_smiles(reactant_smiles)}>>{corr_prod}",
                        ]
                    )
                    unmapped_products_no_stereo.append(unmap_unstereo_smiles(corr_prod))
                    idx += 1
            else:
                continue
        except Exception:
            continue

    return rxn_smiles_list_tmp


def get_all_rxn_smiles(
    file_name_dipoles, file_name_dipolarophiles, smarts_list, output_name, num_cores=9
):
    """ Get all the reaction SMILES based on a dipole + dipolarophile list and a SMARTS list """
    start = time.time()
    reaction_list = []

    # read in dipoles, skip header
    with open(file_name_dipoles, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        dipoles = [row[1] for row in csv_reader]

    # read in dipolarophiles, skip header
    with open(file_name_dipolarophiles, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        dipolarophiles = [row[1] for row in csv_reader]

    print(dipoles, dipolarophiles)
    # make combinations and select corresponding subset of smarts_list
    for idx, dipole in enumerate(dipoles):
        for idx2, dipolarophile in enumerate(dipolarophiles):
            id = idx * len(dipolarophiles) + idx2
            reactants = f"{dipolarophile}.{dipole}"
            if "n" in reactants or "o" in reactants:
                smarts = smarts_list[:4]
            elif (
                reactants.count("#") - reactants.count("C#N") - reactants.count("N#C")
                == 0
            ):
                smarts = smarts_list[4:6]
            elif (
                reactants.count("#") - reactants.count("C#N") - reactants.count("N#C")
                == 1
            ):
                smarts = smarts_list[6:10]
            elif (
                reactants.count("#") - reactants.count("C#N") - reactants.count("N#C")
                == 2
            ):
                smarts = smarts_list[10:]

            reaction_list.append([id, reactants, smarts])

    intermediate = time.time()
    logging.info(f"Parallelizing over {num_cores} cores")

    invalid_temp = 0
    output_list = []
    ofn = f"rxn_smiles_{output_name}.txt"

    # apply function in parallel and get output
    with ProcessPool(max_workers=num_cores) as pool, open(
        ofn, "w", encoding="utf-8"
    ) as of:
        future = pool.map(get_rxn_smiles, reaction_list, timeout=100)

        iterator = future.result()

        while True:
            try:
                result = next(iterator)
                for id, rxn_smiles in result:
                    if rxn_smiles is not None:
                        of.write(f"{id},{rxn_smiles}\n")
                        output_list.append([id, rxn_smiles])

            except StopIteration:
                break
            except TimeoutError as error:
                logging.info(f"get_rxn_smiles call took more than {error.args} seconds")
                invalid_temp += 1
                raise
            except ValueError as error:
                logging.info(f"get_rxn_smiles failed due to ValueError: {error.args}")
                invalid_temp += 1
                raise
            except:
                logging.info(
                    f"reaction smiles extraction failed due to an undefined error."
                )
                pass

        pool.close()
        pool.join()

        logging.info(
            f"No of reactions where rxn_smiles generation failed: {invalid_temp}"
        )

    end = time.time()
    logging.info(
        f" Time passed building combinations : {int(intermediate - start)} sec"
    )
    logging.info(f" Time passed generating rxn_smiles : {int(end - intermediate)} sec")

    # sort and write to csv-file
    output_list.sort(key=lambda x: int(x[0]))
    output_df = pd.DataFrame(output_list, columns=["rxn_id", "rxn_smiles"])
    output_df["rxn_id"] = output_df.index
    output_df.to_csv(f"rxn_smiles_{output_name}.csv")


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(filename=f"log_rxn_smiles_generator_{args.output_base_name}.log", level=logging.INFO)

    smarts_list = get_smarts_list()

    get_all_rxn_smiles(
        args.file_name_dipoles,
        args.file_name_dipolarophiles,
        smarts_list,
        args.output_base_name,
        int(args.num_of_cores),
    )
