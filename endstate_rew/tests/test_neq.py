import pickle
from typing import Tuple

import numpy as np
import pytest
from endstate_rew.neq import perform_switching
from endstate_rew.system import (
    _get_hipen_data,
    _get_masses,
    _seed_velocities,
    create_openff_system,
    generate_molecule,
    initialize_simulation_with_charmmff,
    initialize_simulation_with_openff,
)
from openmm import unit
from openmm.app import Simulation


def load_endstate_system_and_samples_charmmff(
    molecule, name: str, path_to_samples: str, base: str = ""
) -> Tuple[Simulation, list, list]:
    # initialize simulation and load pre-generated samples
    try:
        from NNPOps import OptimizedTorchANI as _

        implementation = "NNPOps"
        platform = "CUDA"
    except ModuleNotFoundError:
        platform = "CPU"

    if not base:
        base = _get_hipen_data()

    n_samples = 5_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    sim = initialize_simulation_with_charmmff(molecule, name, base)

    samples_mm = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.pickle",
            "rb",
        )
    )
    samples_qml = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.pickle",
            "rb",
        )
    )

    return sim, samples_mm, samples_qml


def load_endstate_system_and_samples_openff(
    name: str,
    smiles: str,
    path_to_samples: str,
) -> Tuple[Simulation, list, list]:
    # initialize simulation and load pre-generated samples

    try:
        from NNPOps import OptimizedTorchANI as _

        implementation = "NNPOps"
        platform = "CUDA"
    except ModuleNotFoundError:
        platform = "CPU"

    n_samples = 5_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    molecule = generate_molecule(forcefield="openff", smiles=smiles)
    sim = initialize_simulation_with_openff(molecule)

    samples_mm = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.pickle",
            "rb",
        )
    )
    samples_qml = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.pickle",
            "rb",
        )
    )

    return sim, samples_mm, samples_qml


def test_mass_list():

    molecule = generate_molecule(forcefield="openff", smiles="ClCCOCCCl")
    system, _ = create_openff_system(molecule)

    # make sure that mass list generated from system and molecuel are the same
    m_list = _get_masses(system)
    masses = np.array([a.mass / unit.dalton for a in molecule.atoms]) * unit.daltons
    print(m_list)
    print(masses)
    assert np.allclose(
        m_list.value_in_unit(unit.daltons), masses.value_in_unit(unit.daltons)
    )


def test_seed_velocities():

    # test that manual velocity seeding works
    # openff
    molecule = generate_molecule(forcefield="openff", smiles="ClCCOCCCl")
    system, _ = create_openff_system(molecule)
    _seed_velocities(_get_masses(system))

    # charmmff
    molecule = generate_molecule(forcefield="charmmff", name="ZINC00079729")
    system, _ = create_openff_system(molecule)
    _seed_velocities(_get_masses(system))


@pytest.mark.parametrize(
    "ff, dW_for, dW_rev",
    [
        (
            "charmmff",
            -2405742.1451317305,
            2405742.1452882225,
        ),
        (
            "openff",
            -2346353.8752537486,
            2346353.8752537486,
        ),
    ],
)
def test_switching(ff, dW_for, dW_rev):

    name = "ZINC00077329"
    print(f"{name=}")
    print(f"{ff=}")
    # load simulation and samples for 2cle
    sim, samples_mm, samples_qml = load_endstate_system_and_samples_openff(
        name=name,
        smiles="Cn1cc(Cl)c(/C=N/O)n1",
        path_to_samples=f"data/{name}/sampling_{ff}/run01",
    )
    # perform instantaneous switching with predetermined coordinate set
    # here, we evaluate dU_forw = dU(x)_qml - dU(x)_mm and make sure that it is the same as
    # dU_rev = dU(x)_mm - dU(x)_qml
    lambs = np.linspace(0, 1, 2)
    print(lambs)
    dE_list = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(dE_list[0].value_in_unit(unit.kilojoule_per_mole), dW_for)
    lambs = np.linspace(1, 0, 2)
    print(lambs)
    dE_list = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(dE_list[0].value_in_unit(unit.kilojoule_per_mole), dW_rev)

    # perform NEQ switching
    lambs = np.linspace(0, 1, 21)
    dW_forw = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dW_forw)

    # perform NEQ switching
    lambs = np.linspace(0, 1, 101)
    dW_forw = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dW_forw)
