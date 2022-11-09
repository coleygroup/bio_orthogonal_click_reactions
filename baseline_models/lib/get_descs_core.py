import rdkit.Chem as Chem
import numpy as np
import pandas as pd


elem_list = ['C', 'O', 'N', 'F', 'Br', 'Cl', 'S',
             'Si', 'B', 'I', 'K', 'Na', 'P', 'Mg', 'Li', 'Al', 'H']


def get_reactive_core_descriptors(qm_descriptors, rs, selected_descriptors, core=[]):
    atom_fdim_qm = len(selected_descriptors) + 1

    mol_rs = Chem.MolFromSmiles(rs)
    if not mol_rs:
        raise ValueError("Could not parse smiles string:", smiles)

    fatom_index = {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in mol_rs.GetAtoms()}

    n_atoms = mol_rs.GetNumAtoms()
    fatoms_qm = np.zeros((n_atoms, atom_fdim_qm))
    core_mask = np.zeros((n_atoms,), dtype=np.int32)

    for _, smiles in enumerate(rs.split('.')):

        mol = Chem.MolFromSmiles(smiles)

        if '+' in smiles and '-' in smiles:
            dipole_indices = [fatom_index[atom.GetIntProp('molAtomMapNumber') - 1] for atom in mol.GetAtoms()]

        fatom_index_mol = {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in mol.GetAtoms()}

        qm_series = qm_descriptors.loc[smiles]

        partial_charge = qm_series['partial_charge'].reshape(-1, 1)

        fukui_elec = qm_series['fukui_elec'].reshape(-1, 1)

        fukui_neu = qm_series['fukui_neu'].reshape(-1, 1)

        nmr = qm_series['NMR'].reshape(-1, 1)

        selected_descriptors = set(selected_descriptors)

        atom_qm_descriptor = None

        # start from partial charge or fukui_elec or orbitals
        if "partial_charge" in selected_descriptors:
            atom_qm_descriptor = partial_charge

        if "fukui_elec" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, fukui_elec], axis=-1)
            else:
                atom_qm_descriptor = fukui_elec

        if "fukui_neu" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, fukui_neu], axis=-1)
            else:
                atom_qm_descriptor = fukui_neu

        if "nmr" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, nmr], axis=-1)
            else:
                atom_qm_descriptor = nmr

        atom_qm_descriptor = np.concatenate([atom_qm_descriptor, np.array([[elem_list.index(a.GetSymbol())] for a in Chem.AddHs(mol).GetAtoms()])], axis=-1)

        for map_idx in fatom_index_mol:
            fatoms_qm[fatom_index[map_idx], :] = atom_qm_descriptor[fatom_index_mol[map_idx], :]
            if fatom_index[map_idx] in core:
                core_mask[fatom_index[map_idx]] = 1

    reactive_core_descriptors = [0]*5
    for i in range(len(core_mask)):
        if core_mask[i] != 0:
            if i in dipole_indices:
                if mol_rs.GetAtoms()[i].GetFormalCharge() == 1:
                    reactive_core_descriptors[0] = fatoms_qm[i]
                elif mol_rs.GetAtoms()[i].GetFormalCharge() == -1:
                    reactive_core_descriptors[1] = fatoms_qm[i]
                elif any([neighbor.GetFormalCharge() == 1 for neighbor in mol_rs.GetAtoms()[i].GetNeighbors()]):
                    reactive_core_descriptors[2] = fatoms_qm[i]
            else:
                # fatoms_qm is an array, so if 3 is already set (i.e, the element is no longer int 0), then fill index 4
                if isinstance(reactive_core_descriptors[3], int):
                    reactive_core_descriptors[3] = fatoms_qm[i]
                else:
                    reactive_core_descriptors[4] = fatoms_qm[i]       

    for atom_descs in reactive_core_descriptors:
        if isinstance(atom_descs, int):
            print(rs)
            return None

    return reactive_core_descriptors


def get_descs_core(qm_descriptors, r_smiles, p_smiles, selected_descriptors=["partial_charge", "fukui_elec",
                                                                            "fukui_neu", "nmr"], core_buffer=0):
    rs, rs_core, p_core = _get_reacting_core(r_smiles, p_smiles, core_buffer)
    rs_features = get_reactive_core_descriptors(qm_descriptors, r_smiles, selected_descriptors, core=rs_core)
    
    if rs_features != None:
        return rs_features, r_smiles
    else:
        return None


def _get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be considered as reacting center
    return: atomidx of reacting core
    '''
    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetIntProp('molAtomMapNumber'): a for a in r_mols.GetAtoms()}
    p_dict = {a.GetIntProp('molAtomMapNumber'): a for a in p_mol.GetAtoms()}

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetIntProp('molAtomMapNumber') in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    for a_map in p_dict:
        # FIXME chiral change
        # if str(p_dict[a_map].GetChiralTag()) != str(rs_dict[a_map].GetChiralTag()):
        #    core_mapnum.add(a_map)

        a_neighbor_in_p = set([a.GetIntProp('molAtomMapNumber') for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetIntProp('molAtomMapNumber') for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx() for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx() for a in core_mapnum], buffer)

    fatom_index = \
        {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in Chem.MolFromSmiles(rs_reactants).GetAtoms()}

    core_rs = [fatom_index[x] for x in core_rs]
    core_p = [fatom_index[x] for x in core_p]

    return rs_reactants, core_rs, core_p


def _get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be considered as reacting center
    return: atomidx of reacting core
    '''
    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetIntProp('molAtomMapNumber'): a for a in r_mols.GetAtoms()}
    p_dict = {a.GetIntProp('molAtomMapNumber'): a for a in p_mol.GetAtoms()}

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetIntProp('molAtomMapNumber') in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    for a_map in p_dict:
        # FIXME chiral change
        # if str(p_dict[a_map].GetChiralTag()) != str(rs_dict[a_map].GetChiralTag()):
        #    core_mapnum.add(a_map)

        a_neighbor_in_p = set([a.GetIntProp('molAtomMapNumber') for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetIntProp('molAtomMapNumber') for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx() for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx() for a in core_mapnum], buffer)

    fatom_index = \
        {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in Chem.MolFromSmiles(rs_reactants).GetAtoms()}

    core_rs = [fatom_index[x] for x in core_rs]
    core_p = [fatom_index[x] for x in core_p]

    return rs_reactants, core_rs, core_p


def _get_buffer(m, cores, buffer):
    neighbors = set(cores)

    for i in range(buffer):
        neighbors_temp = list(neighbors)
        for c in neighbors_temp:
            neighbors.update([n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors()])

    neighbors = [m.GetAtomWithIdx(x).GetIntProp('molAtomMapNumber') - 1 for x in neighbors]

    return neighbors
