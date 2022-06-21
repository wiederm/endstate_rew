import pickle
from glob import glob
from os import path

import numpy as np
import openmm as mm
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit
from openmm.app import (
    CharmmParameterSet,
    CharmmPsfFile,
    NoCutoff,
    Simulation,
)
from openmmml import MLPotential
from tqdm import tqdm

from endstate_rew.constant import (
    collision_rate,
    kBT,
    speed_unit,
    stepsize,
    temperature,
    zinc_systems,
)


def generate_molecule(
    forcefield: str,
    name: str = "",
    smiles: str = "",
    base: str = "",
    nr_of_conformations: int = 10,
) -> Molecule:

    # check which ff
    if forcefield == "openff":

        # if smiles string is not provided but zinc name is
        if not smiles and name:
            # look for zinc_id in zinc_systems list and get the correct smiles string
            for zinc_id, smiles_str in zinc_systems:
                if zinc_id == name:
                    smiles = smiles_str
            # if zinc name cannot be found in the list
            if not smiles:
                raise RuntimeError(
                    f"Smiles string for {name} cannot be found. Please add ZINC name and smiles string to the zinc_systems list."
                )

            # if zinc name is found: generate a molecule using openff
            molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
            molecule.generate_conformers(n_conformers=nr_of_conformations)
            assert molecule.n_conformers > 0  # check that confomations are generated
            return molecule

        # if smiles string is provided but zinc name is not
        elif smiles and not name:
            molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
            molecule.generate_conformers(n_conformers=nr_of_conformations)
            assert molecule.n_conformers > 0  # check that confomations are generated
            return molecule

        # if both, smiles and name are provided
        elif smiles and name:
            raise RuntimeError(
                "Please provide only one argument (either smiles string or zinc system name)."
            )

        # if neither smiles nor name are provided
        elif not smiles and not name:
            raise RuntimeError(
                "Please provide either smiles string or zinc system name."
            )

    elif forcefield == "charmmff":

        if not base:
            base = _get_hipen_data()

        # if smiles string is not provided but zinc name is
        if not smiles and name:

            # check if input directory exists
            if not path.isdir(base):
                raise RuntimeError(f"Path {base} is not a directory.")

            # check if input directory contains the proper .sdf file
            if not path.isfile(f"{base}/{name}/{name}.sdf"):
                raise RuntimeError(f"No .sdf file found for {name}")

            # generate openff molecule object from sdf file
            molecule = Molecule.from_file(f"{base}/{name}/{name}.sdf")
            molecule.generate_conformers(n_conformers=nr_of_conformations)
            assert molecule.n_conformers > 0  # check that confomations are generated
            return molecule

        # if smiles string is provided but zinc name is not
        elif smiles and not name:
            # look for smiles string in zinc_systems list and get the correct zinc_id
            for zinc_id, smiles_str in zinc_systems:
                if smiles_str == smiles:
                    name = zinc_id
            # if smiles string cannot be found in the list
            if not name:
                raise RuntimeError(
                    f"Zinc_id for smiles string {smiles} cannot be found. Please add ZINC name and smiles string to the zinc_systems list."
                )

            # check if input directory exists
            if not path.isdir(base):
                raise RuntimeError(f"Path {base} is not a directory.")

            # check if input directory contains at least one directory with the name 'ZINC'
            if len(glob(base + "/ZINC*")) == 0:
                raise RuntimeError(f"No {name} directory found.")

            # generate openff molecule object from sdf file
            molecule = Molecule.from_file(f"{base}/{name}/{name}.sdf")
            molecule.generate_conformers(n_conformers=nr_of_conformations)
            assert molecule.n_conformers > 0  # check that confomations are generated
            return molecule

        # if both, smiles and name are provided
        elif smiles and name:
            raise RuntimeError(
                "Please provide only one argument (either smiles string or zinc system name)."
            )

        # if neither smiles nor name are provided
        elif not smiles and not name:
            raise RuntimeError(
                "Please provide either smiles string or zinc system name."
            )
    else:
        raise RuntimeError("Either openff or charmmff. Abort.")


def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


def get_energy(sim):
    """get energy of system in a state"""
    return sim.context.getState(getEnergy=True).getPotentialEnergy()


def generate_samples(sim, n_samples: int = 1_000, n_steps_per_sample: int = 10_000):
    """generate samples using a defined system"""

    print(f"Generate samples with mixed System: {n_samples=}, {n_steps_per_sample=}")
    samples = []
    for _ in tqdm(range(n_samples)):
        sim.step(n_steps_per_sample)
        samples.append(get_positions(sim))
    return samples


def create_openff_system(molecule):
    """given a molecule it creates an openMM system and topology instance"""

    forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
    topology = molecule.to_topology()
    system = forcefield.create_openmm_system(topology)
    return system, topology


def _get_masses(system) -> np.array:
    return (
        np.array(
            [
                system.getParticleMass(atom_idx) / unit.dalton
                for atom_idx in range(system.getNumParticles())
            ]
        )
        * unit.daltons
    )


def _seed_velocities(masses: np.array) -> np.ndarray:

    # should only take
    # sim.context.setVelocitiesToTemperature(temperature)
    # but currently this returns a pytorch error
    # instead seed manually from boltzmann distribution

    sigma_v = np.array([unit.sqrt(kBT / m) / speed_unit for m in masses]) * speed_unit

    return np.random.randn(len(sigma_v), 3) * sigma_v[:, None]


def _initialize_simulation(
    at_endstate: str, topology, potential, molecule, conf_id: int, system
):
    # define integrator
    integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)
    from endstate_rew.constant import check_implementation

    implementation, platform = check_implementation()

    platform = mm.Platform.getPlatformByName(platform)

    # define the atoms that are calculated using both potentials
    if not at_endstate:
        ml_atoms = [atom.index for atom in topology.atoms()]
        ml_system = potential.createMixedSystem(
            topology,
            system,
            ml_atoms,
            interpolate=True,
            implementation=implementation,
        )
        sim = Simulation(topology, ml_system, integrator, platform=platform)
    elif at_endstate.upper() == "QML":
        print(
            "BEWARE! Using only enstate system. This should only be used for debugging."
        )
        system = potential.createSystem(topology)
        sim = Simulation(topology, system, integrator, platform=platform)
        print("Initializing QML system")
    elif at_endstate.upper() == "MM":
        print(
            "BEWARE! Using only enstate system. This should only be used for debugging."
        )
        sim = Simulation(topology, system, integrator, platform=platform)
        print("Initializing MM system")
    else:
        raise NotImplementedError()

    sim.context.setPositions(molecule.conformers[conf_id])
    # NOTE: FIXME: minimizing the energy of the interpolating potential leeds to very high energies,
    # for now avoiding call to minimizer
    print("Minimizing ...")
    u_1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    sim.minimizeEnergy(maxIterations=100)
    u_2 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"before min: {u_1}; after min: {u_2}")
    # NOTE: FIXME: velocities are seeded manually right now (otherwise pytorch error) --
    # this will be fiexed in the future
    # revert back to openMM velovity call
    sim.context.setVelocitiesToTemperature(temperature)
    # sim.context.setVelocities(_seed_velocities(_get_masses(system)))
    return sim


def initialize_simulation_with_openff(
    molecule: Molecule,
    at_endstate: str = "",
    w_dir="",
    conf_id: int = 0,
):
    """Initialize a simulation instance

    Args:
        molecule (Molecule): _description_
        at_endstate (str, optional): _description_. Defaults to ''.
        platform (str, optional): _description_. Defaults to 'CPU'.

    Returns:
        _type_: _description_
    """

    assert molecule.n_conformers > 0
    print("Using openff ...")
    # initialize potential
    potential = MLPotential("ani2x")
    # initialize openMM system and topology
    print(w_dir)
    if w_dir:
        mol_path = f"{w_dir}/system.openff"
        if path.isfile(mol_path):  # if already generated, load it
            print("load system ...")
            system, topology = pickle.load(open(mol_path, "rb"))
        else:  # if not generated, generate it and save it
            print("generate and save system ...")
            system, topology = create_openff_system(molecule)
            pickle.dump((system, topology), open(mol_path, "wb+"))
    else:
        system, topology = create_openff_system(molecule)

    return _initialize_simulation(
        at_endstate,
        topology.to_openmm(),
        potential,
        molecule,
        conf_id,
        system,
    )


def _get_hipen_data():
    import pathlib
    import endstate_rew as end

    path = pathlib.Path(end.__file__).resolve().parent
    return f"{path}/data/hipen_data"


# creating charmm systems from zinc data
def create_charmm_system(name: str, base=""):

    if not base:
        base = _get_hipen_data()

    # check if input directory exists
    if not path.isdir(base):
        raise RuntimeError("Path is not a directory.")

    # check if input directory contains at least one directory with the name 'ZINC'
    if len(glob(base + "/ZINC*")) < 1:
        raise RuntimeError("No ZINC directory found.")

    # get psf and prm files
    psf = CharmmPsfFile(f"{base}/{name}/{name}.psf")
    params = CharmmParameterSet(
        f"{base}/top_all36_cgenff.rtf",
        f"{base}/par_all36_cgenff.prm",
        f"{base}/{name}/{name}.str",
    )

    # define system object
    system = psf.createSystem(params, nonbondedMethod=NoCutoff)
    # return system object
    return system, psf.topology


# initialize simulation charmm system
def initialize_simulation_with_charmmff(
    molecule: Molecule,
    zinc_id: str,
    base: str = "",
    at_endstate: str = "",
    conf_id: int = 0,
):
    """Initialize a simulation instance

    Args:
        zinc_id (str): _description_
        base (str, optional): _description_.
        at_endstate (str, optional): _description_. Defaults to ''.
        platform (str, optional): _description_. Defaults to 'CPU'.

    Returns:
        _type_: _description_
    """
    assert molecule.n_conformers > 0

    if not base:
        base = _get_hipen_data()

    # initialize potential
    potential = MLPotential("ani2x")
    # generate the charmm system
    system, topology = create_charmm_system(zinc_id, base)

    return _initialize_simulation(
        at_endstate, topology, potential, molecule, conf_id, system
    )
