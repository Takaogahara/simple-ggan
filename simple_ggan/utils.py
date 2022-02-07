from rdkit import Chem, RDLogger
import numpy as np

import tensorflow as tf

RDLogger.DisableLog("rdApp.*")


def get_QM9_dataset():

    qm9 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    csv_path = tf.keras.utils.get_file(".qm9.csv", qm9)

    data = []
    with open(csv_path, "r") as f:
        for line in f.readlines()[1:]:
            data.append(line.split(",")[1])

    return data


def process_QM9_graph(data, num_atoms, atom_dim, bond_dim,
                      atom_mapping, bond_mapping, amount=None):

    if amount is None:
        amount = 1

    adjacency_tensor, feature_tensor = [], []
    for smiles in data[::amount]:
        adjacency, features = smiles_to_graph(smiles, num_atoms, atom_dim,
                                              bond_dim, atom_mapping,
                                              bond_mapping)
        adjacency_tensor.append(adjacency)
        feature_tensor.append(features)

    adjacency_tensor = np.array(adjacency_tensor)
    feature_tensor = np.array(feature_tensor)

    return (adjacency_tensor, feature_tensor)


def smiles_to_graph(smiles, num_atoms: int, atom_dim: int, bond_dim: int,
                    atom_mapping: dict, bond_mapping: dict):
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    adjacency = np.zeros((bond_dim, num_atoms, num_atoms), "float32")
    features = np.zeros((num_atoms, atom_dim), "float32")

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(atom_dim)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features


def graph_to_molecule(graph, atom_dim: int, bond_dim: int,
                      atom_mapping: dict, bond_mapping: dict):
    # Unpack graph
    adjacency, features = graph

    # RWMol is a molecule object intended to be edited
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis=1) != atom_dim - 1)
        & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == bond_dim - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule
