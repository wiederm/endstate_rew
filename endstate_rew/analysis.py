from curses import keyname
import glob
import pickle
import os
from os import path
from collections import namedtuple
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns
import seaborn_image as isns
from openmm import unit
from pymbar import BAR, EXP
from tqdm import tqdm
import mdtraj as md

from endstate_rew.constant import kBT, zinc_systems
from endstate_rew.system import generate_molecule


def _collect_equ_samples(
    path: str, name: str, lambda_scheme: list, every_nth_frame: int = 2
) -> Tuple[list, np.array]:

    """
    Collect equilibrium samples

    Args:
        path (str): path to the location where the samples are stored
        name (str): name of the system (used in the sample files)
        lambda_scheme (list): list of lambda states as floats
        every_nth_frame (int, optional): prune the samples further by taking only every nth sample. Defaults to 2.

    Raises:
        RuntimeError: if multuple sample files are present we can not decide which is the correct one.

    Returns:
        Tuple(coordinates, N_k)
    """

    nr_of_samples = 5_000
    nr_of_steps = 1_000
    coordinates = []
    N_k = np.zeros(len(lambda_scheme))
    # loop over lambda scheme and collect samples in nanometer
    for idx, lamb in enumerate(lambda_scheme):
        file = glob.glob(
            f"{path}/{name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.pickle"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        elif len(file) == 0:
            print("WARNING! Incomplete equ sampling. Proceed with cautions.")
        else:
            coords_ = pickle.load(open(file[0], "rb"))
            coords_ = coords_[1_000:]  # remove the first 1k samples
            coords_ = coords_[::every_nth_frame]  # take only every second sample
            N_k[idx] = len(coords_)
            coordinates.extend([c_.value_in_unit(unit.nanometer) for c_ in coords_])

    number_of_samples = len(coordinates)
    print(f"Number of samples loaded: {number_of_samples}")
    return coordinates * unit.nanometer, N_k


def calculate_u_kn(
    smiles: str,
    forcefield: str,
    path: str,
    name: str,
    every_nth_frame: int = 2,
    reload: bool = True,
) -> np.ndarray:

    """
    Calculate the u_kn matrix to be used by the mbar estimator

    Args:
        smiles (str): smiles string describing the system
        forcefield (str): which force field is used (allowed options are `openff` or `charmmmff`)
        path (str): path to location where samples are stored
        name (str): name of the system (used in the sample files)
        every_nth_frame (int, optional): prune the samples further by taking only every nth sample. Defaults to 2.
        reload (bool, optional): do you want to reload a previously saved mbar pickle file if present (every time the free energy is calculated the mbar pickle file is saved --- only loading is optional)

    Returns:
        Tuple(np.array, np.ndarray): (N_k, u_kn)
    """

    from endstate_rew.system import generate_molecule, initialize_simulation_with_openff

    try:
        # if already generated reuse
        if reload == False:
            raise FileNotFoundError
        print(f"trying to load: {path}/mbar_{every_nth_frame}.pickle")
        N_k, u_kn = pickle.load(open(f"{path}/mbar_{every_nth_frame}.pickle", "rb"))
        print(f"Reusing pregenerated mbar object: {path}/mbar_{every_nth_frame}.pickle")
    except FileNotFoundError:

        # generate molecule
        m = generate_molecule(smiles=smiles, forcefield=forcefield)
        # initialize simulation
        # first, modify path to point to openff molecule object
        w_dir = path.split("/")
        w_dir = "/".join(w_dir[:-3])
        # initialize simualtion and reload if already generated
        sim = initialize_simulation_with_openff(m, w_dir=w_dir)

        lambda_scheme = np.linspace(0, 1, 11)
        samples, N_k = _collect_equ_samples(
            path, name, lambda_scheme, every_nth_frame=every_nth_frame
        )
        samples = np.array(
            samples.value_in_unit(unit.nanometer)
        )  # positions in nanometer
        u_kn = np.zeros(
            (len(N_k), int(N_k[0] * len(N_k))), dtype=np.float64
        )  # NOTE: assuming that N_k[0] is the maximum number of samples drawn from any state k
        for k, lamb in enumerate(lambda_scheme):
            sim.context.setParameter("lambda", lamb)
            us = []
            for x in tqdm(range(len(samples))):
                sim.context.setPositions(samples[x])
                u_ = sim.context.getState(getEnergy=True).getPotentialEnergy()
                us.append(u_)
            us = np.array([u / kBT for u in us], dtype=np.float64)
            u_kn[k] = us
        pickle.dump((N_k, u_kn), open(f"{path}/mbar_{every_nth_frame}.pickle", "wb+"))

    return (N_k, u_kn)


def plot_overlap_for_equilibrium_free_energy(
    N_k: np.array, u_kn: np.ndarray, name: str
):
    """
    Calculate the overlap for each state with each other state. THe overlap is normalized to be 1 for each row.

    Args:
        N_k (np.array): numnber of samples for each state k
        u_kn (np.ndarray): each of the potential energy functions `u` describing a state `k` are applied to each sample `n` from each of the states `k`
        name (str): name of the system in the plot
    """
    from pymbar import MBAR

    # initialize the MBAR maximum likelihood estimate

    mbar = MBAR(u_kn, N_k)
    plt.figure(figsize=[8, 8], dpi=300)
    overlap = mbar.computeOverlap()["matrix"]
    sns.heatmap(
        overlap,
        cmap="Blues",
        linewidth=0.5,
        annot=True,
        fmt="0.2f",
        annot_kws={"size": "small"},
    )
    plt.title(f"Free energy estimate for {name}", fontsize=15)
    plt.savefig(f"{name}_equilibrium_free_energy_overlap.png")
    plt.show()
    plt.close()


def plot_results_for_equilibrium_free_energy(
    N_k: np.array, u_kn: np.ndarray, name: str
):
    """
    Calculate the accumulated free energy along the mutation progress.


    Args:
        N_k (np.array): numnber of samples for each state k
        u_kn (np.ndarray): each of the potential energy functions `u` describing a state `k` are applied to each sample `n` from each of the states `k`
        name (str): name of the system in the plot
    """
    from pymbar import MBAR

    # initialize the MBAR maximum likelihood estimate

    mbar = MBAR(u_kn, N_k)
    print(
        f'ddG = {mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"][0][-1]} +- {mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"][0][-1]}'
    )

    plt.figure(figsize=[8, 8], dpi=300)
    r = mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"]

    x = [a for a in np.linspace(0, 1, len(r[0]))]
    y = r[0]
    y_error = mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"][0]
    print()
    plt.errorbar(x, y, yerr=y_error, label="ddG +- stddev [kT]")
    plt.legend()
    plt.title(f"Free energy estimate for {name}", fontsize=15)
    plt.ylabel("Free energy estimate in kT", fontsize=15)
    plt.xlabel("lambda state (0 to 1)", fontsize=15)
    plt.savefig(f"{name}_equilibrium_free_energy.png")
    plt.show()
    plt.close()


def _collect_neq_samples(
    path: str, name: str, switching_length: int, direction: str = "mm_to_qml"
) -> list:
    files = glob.glob(f"{path}/{name}*{direction}*{switching_length}*.pickle")
    ws = []
    for f in files:
        w_ = pickle.load(open(f, "rb")).value_in_unit(unit.kilojoule_per_mole)
        ws.extend(w_)
    number_of_samples = len(ws)
    print(f"Number of samples used: {number_of_samples}")
    return ws * unit.kilojoule_per_mole


def collect_results(
    w_dir: str, switching_length: int, run_id: int, name: str, smiles: str
) -> NamedTuple:
    from endstate_rew.neq import perform_switching
    from endstate_rew.system import generate_molecule, initialize_simulation

    # load samples
    mm_samples = pickle.load(
        open(
            f"{w_dir}/{name}/sampling/{run_id}/{name}_mm_samples_5000_2000.pickle", "rb"
        )
    )
    qml_samples = pickle.load(
        open(
            f"{w_dir}/{name}/sampling/{run_id}/{name}_qml_samples_5000_2000.pickle",
            "rb",
        )
    )

    # get pregenerated work values
    ws_from_mm_to_qml = np.array(
        _collect_neq_samples(
            f"{w_dir}/{name}/switching/{run_id}/", name, switching_length, "mm_to_qml"
        )
        / kBT
    )
    ws_from_qml_to_mm = np.array(
        _collect_neq_samples(
            f"{w_dir}/{name}/switching/{run_id}/", name, switching_length, "qml_to_mm"
        )
        / kBT
    )

    # perform instantenious swichting (FEP) to get dE values
    switching_length = 2
    nr_of_switches = 500
    # create molecule
    # openff
    molecule = generate_molecule(forcefield="openff", smiles=smiles)

    sim = initialize_simulation(molecule, w_dir=f"{w_dir}/{name}")
    lambs = np.linspace(0, 1, switching_length)
    dEs_from_mm_to_qml = np.array(
        perform_switching(sim, lambs, samples=mm_samples, nr_of_switches=nr_of_switches)
        / kBT
    )
    lambs = np.linspace(1, 0, switching_length)
    dEs_from_qml_to_mm = np.array(
        perform_switching(
            sim, lambs, samples=qml_samples, nr_of_switches=nr_of_switches
        )
        / kBT
    )

    # pack everything in a namedtuple
    Results = namedtuple(
        "Results",
        "dWs_from_mm_to_qml dWs_from_qml_to_mm dEs_from_mm_to_qml dEs_from_qml_to_mm",
    )
    results = Results(
        ws_from_mm_to_qml, ws_from_qml_to_mm, dEs_from_mm_to_qml, dEs_from_qml_to_mm
    )
    return results


def plot_resutls_of_switching_experiments(name: str, results: NamedTuple):

    print("################################")
    print(
        f"Crooks' equation: {BAR(results.dWs_from_mm_to_qml, results.dWs_from_qml_to_mm)}"
    )
    print(f"Jarzynski's equation: {EXP(results.dWs_from_mm_to_qml)}")
    print(f"Zwanzig's equation: {EXP(results.dEs_from_mm_to_qml)}")
    print(
        f"Zwanzig's equation bidirectional: {BAR(results.dEs_from_mm_to_qml, results.dEs_from_qml_to_mm)}"
    )
    print("################################")

    sns.set_context("talk")
    fig, axs = plt.subplots(3, 1, figsize=(11.0, 9), dpi=600)
    # plot distribution of dE and dW
    #########################################
    axs[0].set_title(rf"{name} - distribution of $\Delta$W and $\Delta$E")
    palett = sns.color_palette(n_colors=8)
    palett_as_hex = palett.as_hex()
    c1, c2, c3, c4 = (
        palett_as_hex[0],
        palett_as_hex[1],
        palett_as_hex[2],
        palett_as_hex[3],
    )
    axs[0].ticklabel_format(axis="x", style="sci", useOffset=True, scilimits=(0, 0))
    # axs[1].ticklabel_format(axis='x', style='sci', useOffset=False,scilimits=(0,0))

    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dWs_from_mm_to_qml * -1,
        kde=True,
        stat="density",
        label=r"$\Delta$W(MM$\rightarrow$QML)",
        color=c1,
    )
    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dEs_from_mm_to_qml * -1,
        kde=True,
        stat="density",
        label=r"$\Delta$E(MM$\rightarrow$QML)",
        color=c2,
    )
    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dWs_from_qml_to_mm,
        kde=True,
        stat="density",
        label=r"$\Delta$W(QML$\rightarrow$MM)",
        color=c3,
    )
    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dEs_from_qml_to_mm,
        kde=True,
        stat="density",
        label=r"$\Delta$E(QML$\rightarrow$MM)",
        color=c4,
    )
    axs[0].legend()

    # plot results
    #########################################
    axs[1].set_title(rf"{name} - offset $\Delta$G(MM$\rightarrow$QML)")
    # Crooks' equation
    ddG_list, dddG_list = [], []
    ddG, dddG = BAR(results.dWs_from_mm_to_qml, results.dWs_from_qml_to_mm)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # Jarzynski's equation
    ddG, dddG = EXP(results.dWs_from_mm_to_qml)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # FEP
    ddG, dddG = EXP(results.dEs_from_mm_to_qml)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # FEP + BAR
    ddG, dddG = BAR(results.dEs_from_mm_to_qml, results.dEs_from_qml_to_mm)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)

    axs[1].errorbar(
        [i for i in range(len(ddG_list))],
        ddG_list - np.min(ddG_list),
        dddG_list,
        fmt="o",
    )
    axs[1].set_xticklabels(["", "Crooks", "", "Jazynski", "", "FEP+EXP", "", "FEP+BAR"])
    axs[1].set_ylabel("kT")
    # axs[1].legend()

    # plot cummulative stddev of dE and dW
    #########################################
    axs[2].set_title(rf"{name} - cummulative stddev of $\Delta$W and $\Delta$E")

    cum_stddev_ws_from_mm_to_qml = [
        results.dWs_from_mm_to_qml[:x].std()
        for x in range(1, len(results.dWs_from_mm_to_qml) + 1)
    ]
    cum_stddev_ws_from_qml_to_mm = [
        results.dWs_from_qml_to_mm[:x].std()
        for x in range(1, len(results.dWs_from_qml_to_mm) + 1)
    ]

    cum_stddev_dEs_from_mm_to_qml = [
        results.dEs_from_mm_to_qml[:x].std()
        for x in range(1, len(results.dEs_from_mm_to_qml) + 1)
    ]
    cum_stddev_dEs_from_qml_to_mm = [
        results.dEs_from_qml_to_mm[:x].std()
        for x in range(1, len(results.dEs_from_qml_to_mm) + 1)
    ]
    axs[2].plot(
        cum_stddev_ws_from_mm_to_qml, label=r"stddev $\Delta$W(MM$\rightarrow$QML)"
    )
    axs[2].plot(
        cum_stddev_ws_from_qml_to_mm, label=r"stddev $\Delta$W(QML$\rightarrow$MM)"
    )
    axs[2].plot(
        cum_stddev_dEs_from_mm_to_qml, label=r"stddev $\Delta$E(MM$\rightarrow$QML)"
    )
    axs[2].plot(
        cum_stddev_dEs_from_qml_to_mm, label=r"stddev $\Delta$E(QML$\rightarrow$MM)"
    )
    # plot 1 kT limit
    axs[2].axhline(y=1.0, color="yellow", linestyle=":")
    axs[2].axhline(y=2.0, color="orange", linestyle=":")

    axs[2].set_ylabel("kT")

    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"{name}_r_10ps.png")
    plt.show()


############################################################### TORSION PROFILES #####################################################################################

# save a png file of the molecule with atom indices
def save_mol_pic(zinc_id: str, ff: str):

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    IPythonConsole.drawOptions.addAtomIndices = True

    name, smiles = zinc_systems[zinc_id]
    # generate openff Molecule
    mol = generate_molecule(name=name, forcefield=ff, base="../data/hipen_data")
    # convert openff object to rdkit mol object
    mol_rd = mol.to_rdkit()

    # remove explicit H atoms
    if zinc_id == 4:
        # NOTE: FIXME: this is a temporary workaround to fix the wrong indexing in rdkit
        # when using the RemoveHs() function
        mol_draw = Chem.RWMol(mol_rd)
        # remove all explicit H atoms, except the ones on the ring (for correct indexing)
        for run in range(1, 13):
            n_atoms = mol_draw.GetNumAtoms()
            mol_draw.RemoveAtom(n_atoms - 1)
    else:
        # remove explicit H atoms
        mol_draw = Chem.RemoveHs(mol_rd)

    # get 2D representation
    AllChem.Compute2DCoords(mol_draw)
    # formatting
    d = rdMolDraw2D.MolDraw2DCairo(1500, 1000)
    d.drawOptions().fixedFontSize = 90
    d.drawOptions().fixedBondLength = 110
    d.drawOptions().annotationFontScale = 0.7
    d.drawOptions().addAtomIndices = True

    d.DrawMolecule(mol_draw)
    d.FinishDrawing()
    d.WriteDrawingText(f"{name}_{ff}.png")


# get trajectory instance
def get_traj(samples: str, name: str, ff: str):
    from glob import glob

    # depending on endstate, get correct label
    if samples == "mm":
        endstate = "0.0000"
    elif samples == "qml":
        endstate = "1.0000"

    # get pickle files for traj
    globals()["pickle_file_%s" % samples] = glob(
        f"/data/shared/projects/endstate_rew/{name}/sampling_{ff}/run*/{name}_samples_5000_steps_1000_lamb_{endstate}.pickle"
    )

    # list for collecting sampling data
    coordinates = []

    # generate traj instance only if at least one pickle file exists
    if not len(globals()["pickle_file_%s" % samples]) == 0:
        for run in globals()["pickle_file_%s" % samples]:

            # load pickle file
            coord = pickle.load(open(run, "rb"))
            # check, if sampling data is complete (MODIFY IF NR OF SAMPLING STEPS != 5000)
            if len(coord) == 5000:

                # remove first 1k samples
                coordinates.extend(coord[1000:])

                # load topology from pdb file
                top = md.load("mol.pdb").topology

                # NOTE: the reason why this function needs a smiles string is because it
                # has to generate a pdb file from which mdtraj reads the topology
                # this is not very elegant # FIXME: try to load topology directly

                # generate trajectory instance
                globals()["traj_%s" % samples] = md.Trajectory(
                    xyz=coordinates, topology=top
                )
                return globals()["traj_%s" % samples]
            else:
                print(f"{run} file contains incomplete sampling data")


# get indices of four atoms defining the torsion
def get_indices(rot_bond, rot_bond_list: list, bonds: list):

    print(f"---------- Investigating bond nr {rot_bond} ----------")

    # get indices of both atoms forming an rotatable bond
    atom_1_idx = (rot_bond_list[rot_bond]).atom1_index
    atom_2_idx = (rot_bond_list[rot_bond]).atom2_index

    # create lists to collect neighbors of atom_1 and atom_2
    neighbors1 = []
    neighbors2 = []

    # find neighbors of atoms forming the rotatable bond and add to index list (if heavy atom torsion)
    for bond in bonds:

        # get neighbors of atom_1 (of rotatable bond)
        # check, if atom_1 (of rotatable bond) is the first atom in the current bond
        if bond.atom1_index == atom_1_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_2
            if (
                not bond.atom2.element.name == "hydrogen"
                and not bond.atom2_index == atom_2_idx
            ):
                neighbors1.append(bond.atom2_index)

        # check, if atom_1 (of rotatable bond) is the second atom in the current bond
        elif bond.atom2_index == atom_1_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_2
            if (
                not bond.atom1.element.name == "hydrogen"
                and not bond.atom1_index == atom_2_idx
            ):
                neighbors1.append(bond.atom1_index)

        # get neighbors of atom_2 (of rotatable bond)
        # check, if atom_2 (of rotatable bond) is the first atom in the current bond
        if bond.atom1_index == atom_2_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_1
            if (
                not bond.atom2.element.name == "hydrogen"
                and not bond.atom2_index == atom_1_idx
            ):
                neighbors2.append(bond.atom2_index)

        # check, if atom_2 (of rotatable bond) is the second atom in the current bond
        elif bond.atom2_index == atom_2_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_1
            if (
                not bond.atom1.element.name == "hydrogen"
                and not bond.atom1_index == atom_1_idx
            ):
                neighbors2.append(bond.atom1_index)

    # check, if both atoms forming the rotatable bond have neighbors
    if len(neighbors1) > 0 and len(neighbors2) > 0:

        # list for final atom indices defining torsion
        indices = [[neighbors1[0], atom_1_idx, atom_2_idx, neighbors2[0]]]
        return indices

    else:

        print(f"No heavy atom torsions found for bond {rot_bond}")
        indices = []
        return indices


# generate torsion profile plots
def vis_torsions(zinc_id: int, ff: str):
    ############################################ LOAD MOLECULE AND GET BOND INFO ##########################################################################################

    # get zinc_id(name of the zinc system) and smiles
    name, smiles = zinc_systems[zinc_id]

    print(
        f"################################## SYSTEM {name} ##################################"
    )

    # generate mol from smiles string
    mol = generate_molecule(forcefield=ff, name=name, base="../data/hipen_data")

    # write mol as pdb
    mol.to_file("mol.pdb", file_format="pdb")

    # get all bonds
    bonds = mol.bonds

    # get all rotatable bonds
    rot_bond_list = mol.find_rotatable_bonds()
    print(len(rot_bond_list), "rotatable bonds found.")

    ################################################## GET HEAVY ATOM TORSIONS ##########################################################################################

    # create list for collecting bond nr, which have heavy atom torsion profile
    torsions = []
    all_indices = []
    plotting = False

    for rot_bond in range(len(rot_bond_list)):

        # get atom indices of all rotatable bonds
        indices = get_indices(
            rot_bond=rot_bond, rot_bond_list=rot_bond_list, bonds=bonds
        )

        # compute dihedrals only if heavy atom torsion was found for rotatable bond
        if len(indices) > 0:
            print(f"Dihedrals are computed for bond nr {rot_bond}")
            # add bond nr to list
            torsions.append(rot_bond)
            all_indices.extend(indices)

            # check if traj data can be retrieved
            globals()["traj_mm_%s" % rot_bond] = get_traj(
                samples="mm", name=name, ff=ff
            )
            globals()["traj_qml_%s" % rot_bond] = get_traj(
                samples="qml", name=name, ff=ff
            )

            if (
                globals()["traj_mm_%s" % rot_bond]
                and globals()["traj_qml_%s" % rot_bond]
            ):
                globals()["data_mm_%s" % rot_bond] = md.compute_dihedrals(
                    globals()["traj_mm_%s" % rot_bond], indices, periodic=True, opt=True
                )  # * 180.0 / np.pi
                globals()["data_qml_%s" % rot_bond] = md.compute_dihedrals(
                    globals()["traj_qml_%s" % rot_bond],
                    indices,
                    periodic=True,
                    opt=True,
                )  # * 180.0 / np.pi
                plotting = True
            else:
                print(f"Trajectory data cannot be found for {name}")
        else:
            print(f"No dihedrals will be computed for bond nr {rot_bond}")

    ################################################## PLOT TORSION PROFILES ##########################################################################################

    if plotting:
        # generate molecule picture
        save_mol_pic(zinc_id=zinc_id, ff=ff)

        # counter for addressing axis
        counter = 0

        # create corresponding nr of subplots
        fig, axs = plt.subplots(
            len(torsions) + 1, 1, figsize=(8, len(torsions) * 2 + 6), dpi=400
        )
        fig.suptitle(f"Torsion profile of {name} ({ff})", fontsize=13, weight="bold")

        # flip the image, so it is displayed correctly
        image = np.flipud(mpimg.imread(f"{name}_{ff}.png"))

        # plot the molecule image on the first axis
        axs[0].imshow(image)
        axs[0].axis("off")

        # set counter to 1
        counter += 1
        # counter for atom indices
        idx_counter = 0

        # iterate over all torsions and plot results
        for torsion in torsions:
            # add atom indices as plot title
            axs[counter].set_title(f"Torsion {all_indices[idx_counter]}")
            # sns.set(font_scale = 2)
            sns.histplot(
                ax=axs[counter],
                data={
                    "mm samples": globals()["data_mm_%s" % torsion].squeeze(),
                    "qml samples": globals()["data_qml_%s" % torsion].squeeze(),
                },
                bins=100,  # not sure how many bins to use
                kde=True,
                alpha=0.5,
                stat="density",
            )
            # adjust axis labelling
            unit = np.arange(-np.pi, np.pi + np.pi / 4, step=(1 / 4 * np.pi))
            axs[counter].set(xlim=(-np.pi, np.pi))
            axs[counter].set_xticks(
                unit, ["-π", "-3π/4", "-π/2", "-π/4", "0", "π/4", "π/2", "3π/4", "π"]
            )
            axs[counter].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            counter += 1
            idx_counter += 1
        axs[-1].set_xlabel("Dihedral angle")

        plt.tight_layout()

        if not path.isdir(f"torsion_profiles_{ff}"):
            os.makedirs(f"torsion_profiles_{ff}")
        plt.savefig(f"torsion_profiles_{ff}/{name}_{ff}.png")
    else:
        print(f"No torsion profile can be generated for {name}")
