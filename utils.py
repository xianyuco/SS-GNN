from rdkit import Chem
import torch
import numpy as np
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def set_data_device(examples, device):
    if type(examples) == list or type(examples) == tuple:
        return [set_data_device(e, device) for e in examples]
    else:
        return examples.to(device)

def get_atom_features(atom, is_protein=False):
    ATOM_CODES = {}
    metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
              + list(range(37, 51)) + list(range(55, 84))
              + list(range(87, 104)))
    atom_classes = [(5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'), (16, 'S'), (34, 'Se'),
                    ([9, 17, 35, 53], 'halogen'), (metals, 'metal')]
    for code, (atomidx, name) in enumerate(atom_classes):
        if type(atomidx) is list:
            for a in atomidx:
                ATOM_CODES[a] = code
        else:
            ATOM_CODES[atomidx] = code
    try:
        classes = ATOM_CODES[atom.GetAtomicNum()]
    except:
        classes = 9

    possible_chirality_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    chirality = possible_chirality_list.index(atom.GetChiralTag())

    possible_formal_charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    try:
        charge = possible_formal_charge_list.index(atom.GetFormalCharge())
    except:
        charge = 11

    possible_hybridization_list = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    try:
        hyb = possible_hybridization_list.index(atom.GetHybridization())
    except:
        hyb = 6

    possible_numH_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    try:
        numH = possible_numH_list.index(atom.GetTotalNumHs())
    except:
        numH = 9

    possible_implicit_valence_list = [0, 1, 2, 3, 4, 5, 6, 7]
    try:
        valence = possible_implicit_valence_list.index(atom.GetTotalValence())
    except:
        valence = 8

    possible_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    try:
        degree = possible_degree_list.index(atom.GetTotalDegree())
    except:
        degree = 11

    is_aromatic = [False, True]
    aromatic = is_aromatic.index(atom.GetIsAromatic())

    is_protein = int(is_protein)

    mass = atom.GetMass() / 100

    return [classes, chirality, charge, hyb, numH, valence, degree, aromatic, is_protein, mass]


def get_bonds_features(bond, is_protein=False):
    possible_bonds_type = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                           Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.ZERO,
                           Chem.rdchem.BondType.OTHER]
    try:
        bond_type = possible_bonds_type.index(bond.GetBondType())
    except:
        bond_type = 6

    possible_bond_dirs = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT,
                          Chem.rdchem.BondDir.EITHERDOUBLE, Chem.rdchem.BondDir.UNKNOWN]
    try:
        bond_dirs = possible_bond_dirs.index(bond.GetBondDir())
    except:
        bond_dirs = 4

    stereo = int(bond.GetStereo())

    is_ring = int(bond.IsInRing())

    is_protein = int(is_protein)

    return [bond_type, bond_dirs, stereo, is_ring, is_protein]

def get_gnn_features(protein, ligand, threshhold=5):

    ligand_conf = ligand.GetConformer()
    ligand_positions = ligand_conf.GetPositions()
    protein_conf = protein.GetConformer()
    protein_positions = protein_conf.GetPositions()

    dis = ligand_positions[:, np.newaxis, :] - protein_positions[np.newaxis, :, :]
    dis = np.sqrt((dis * dis).sum(-1))
    idx = np.where(dis < threshhold)
    idx = [[i, j] for i, j in zip(idx[0], idx[1])]

    innerdis = ligand_positions[:, np.newaxis, :] - ligand_positions[np.newaxis, :, :]
    innerdis = np.sqrt((innerdis * innerdis).sum(-1))

    atom_features_list = []
    pidx = [i[1] for i in idx]
    nligand_atoms = len(ligand.GetAtoms())
    pidx = sorted(list(set(pidx)))

    pidx = [i for i in pidx if protein.GetAtomWithIdx(int(i)).GetPDBResidueInfo().GetResidueName() != 'HOH']
    pidx2tidx = {pidx[i]: i + nligand_atoms for i in range(len(pidx))}
    for i in range(len(ligand.GetAtoms())):
        atom = ligand.GetAtomWithIdx(int(i))
        atom_feature = list(get_atom_features(atom, is_protein=False)) + ligand_positions[i].tolist()
        atom_features_list.append(atom_feature)

    for i in pidx:
        atom = protein.GetAtomWithIdx(int(i))
        atom_feature = list(get_atom_features(atom, is_protein=True)) + protein_positions[i].tolist()
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

    edges_list = []
    edge_features_list = []
    if len(ligand.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in ligand.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = list(get_bonds_features(bond, is_protein=False)) + [innerdis[i, j]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

    for b in idx:
        try:
            edges_list.append((b[0], pidx2tidx[b[1]]))
            edge_feature = [6, 4, 0, 0, 1] + [dis[b[0], b[1]]]
            edge_features_list.append(edge_feature)
            edges_list.append((pidx2tidx[b[1]], b[0]))
        except:
            continue
        edge_features_list.append(edge_feature)

    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)
    return x, edge_index, edge_attr
