import pickle
import nglview as ng
from endstate_rew.system import generate_molecule
import mdtraj as md


def visualize_mol(smiles: str, forcefield: str, pickle_file: str):
    """Takes a smiles string and the path to a trajectory pickle file and returns
    a nglview instance to visualize the MD simulation.

    Args:
        smiles (str): smiles sting defining the molecule
        pickle_file (str): file path to pickled trajectorie

    Returns:
        _type_: nglview view instance
    """
    # generate mol from file
    m = generate_molecule(forcefield=forcefield, smiles=smiles)
    # write mol as pdb
    m.to_file("m.pdb", file_format="pdb")
    # load traj
    f = pickle.load(open(pickle_file, "rb"))
    # load topology from pdb file
    top = md.load("m.pdb").topology
    # NOTE: the reason why this function needs a smiles string is because it
    # has to generate a pdb file from which mdtraj reads the topology
    # this is not very elegant # FIXME: try to load topology directly

    # generate trajectory instance
    traj = md.Trajectory(f, topology=top)
    # align traj
    traj.superpose(traj)
    view = ng.show_mdtraj(traj)
    return view
