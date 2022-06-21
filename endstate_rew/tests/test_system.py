import numpy as np
from openmm import unit
import pytest
from endstate_rew.constant import check_implementation


def test_conf_selection():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule  # , remap_atoms

    for ff in ["openff", "charmmff"]:
        for zinc_name, smiles_str in zinc_systems:
            print(zinc_name)
            if (
                zinc_name == "ZINC00061095"
                or zinc_name == "ZINC00095858"
                or zinc_name == "ZINC00138607"
            ):  # skip system that has wrong topology
                continue

            # molecule generation with smiles
            m = generate_molecule(forcefield=ff, smiles=smiles_str)
            assert len(m.conformers) >= 1

            # molecule generation with name
            m = generate_molecule(forcefield=ff, name=zinc_name)
            assert len(m.conformers) >= 1


def test_openff_conformations():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule

    # we require deterministic conformations, here we check that
    # coordinate set 0 is always the same

    # test with simple smiles string
    ethane_smiles = "CC"
    m = generate_molecule(forcefield="openff", smiles=ethane_smiles)

    coordinates = [
        [-7.45523175e-01, 4.14444551e-02, 1.17060656e-02],
        [7.47339732e-01, 2.87860116e-03, 1.22331042e-03],
        [-1.12970717e00, -6.37431527e-01, 8.14421395e-01],
        [-1.18489973e00, 1.02557024e00, 1.99636470e-01],
        [-1.19987084e00, -3.34603249e-01, -9.38879499e-01],
        [1.08415333e00, -7.36520284e-01, -7.73193261e-01],
        [1.22661485e00, 9.61738002e-01, -2.68072775e-01],
        [1.20189300e00, -3.23076234e-01, 9.53158295e-01],
    ]

    assert len(m.conformers) == 1
    compare_coordinates_to = m.conformers[0].value_in_unit(unit.angstrom)
    assert np.allclose(compare_coordinates_to, coordinates)

    # test with zinc system
    zinc_id = zinc_systems[1][0]
    m = generate_molecule(forcefield="openff", name=zinc_id)

    coordinates = [
        [-2.75987995, -0.69053967, 0.36029375],
        [-1.47374083, -0.07361633, 0.09984744],
        [-1.33655965, 1.18256186, -0.37147202],
        [0.01006323, 1.38777693, -0.47988788],
        [0.80189666, 2.86426872, -1.04883321],
        [0.66095509, 0.24515119, -0.06958264],
        [2.09923885, 0.03630211, -0.02586892],
        [2.65278589, -1.05267619, 0.36663243],
        [4.01325859, -1.25211235, 0.40867711],
        [-0.311077, -0.63920528, 0.28300144],
        [-3.50614014, -0.46794238, -0.43942482],
        [-2.63891528, -1.78669473, 0.37180877],
        [-3.17350775, -0.27589788, 1.29720489],
        [-2.09406379, 1.89931201, -0.62057146],
        [2.7391263, 0.83811731, -0.33966023],
        [4.31655976, -2.21480533, 0.20783536],
    ]

    assert len(m.conformers) == 1
    compare_coordinates_to = m.conformers[0].value_in_unit(unit.angstrom)
    assert np.allclose(compare_coordinates_to, coordinates)


def test_smiles():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule

    # check that we can generate molecule and assert smiles before and after (openff)

    # test with simple smiles string
    ethane_smiles = "CC"
    m = generate_molecule(forcefield="openff", smiles=ethane_smiles)
    assert ethane_smiles == m.to_smiles(explicit_hydrogens=False)

    # test with zinc system
    zinc_id = zinc_systems[1][0]
    smiles = zinc_systems[1][1]
    m = generate_molecule(forcefield="openff", name=zinc_id)
    assert smiles == m.to_smiles(explicit_hydrogens=False)


def test_generate_molecule():
    from endstate_rew.system import generate_molecule

    smiles_zinc = "Cn1cc(Cl)c(/C=N/O)n1"
    name_zinc = "ZINC00077329"
    smiles_test = "ClCCOCCCl"

    # test molecule generation with zinc systems
    for ff in ["openff", "charmmff"]:
        generate_molecule(forcefield=ff, smiles=smiles_zinc)
        generate_molecule(forcefield=ff, name=name_zinc)

    # test molecule generation with small test system
    # (only openff possible, charmmff requires .sdf file for molecule generation)
    generate_molecule(forcefield="openff", smiles=smiles_test)


def test_atom_indices_charmmff():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule, _get_hipen_data

    # collect atom indices of psf file
    name = zinc_systems[1][0]
    mol = generate_molecule(forcefield="charmmff", name=name)
    n_atoms = mol.n_atoms

    # get psf file
    base = _get_hipen_data()
    file = open(f"{base}/{name}/{name}.psf")
    lines = file.readlines()

    atoms_psf = []
    line_count = 0
    atom_count = 0

    for line in lines:
        # atoms begin in line 8
        if line_count >= 7:
            atom_count += 1
            tmp = line.split()
            atoms_psf.append(tmp[4])
        # break when all atom indices are collected
        if atom_count == n_atoms:
            break
        line_count += 1

    # collect atom indices of pdb file
    # save a pdb file of the generated molecule
    mol.to_file("tmp.pdb", "pdb")
    file_pdb = open(f"tmp.pdb")

    lines = file_pdb.readlines()
    atoms_pdb = []
    line_count = 0
    atom_count = 0

    for line in lines:
        atom_count += 1
        tmp = line.split()
        atoms_pdb.append(tmp[2])
        if atom_count == n_atoms:
            break
        line_count += 1

    # make sure, that atom indexing after molecule generation is the same as in the psf file
    assert atoms_psf == atoms_pdb


@pytest.mark.parametrize(
    "ff",
    [("charmmff"), ("openff")],
)
def test_sampling(ff):
    from endstate_rew.system import (
        generate_samples,
        generate_molecule,
        initialize_simulation_with_charmmff,
        initialize_simulation_with_openff,
    )

    # initialize molecule
    name = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"

    if ff == "openff":
        molecule = generate_molecule(forcefield=ff, smiles=smiles)

        # initialize simulation and start sampling at MM endstate
        sim = initialize_simulation_with_openff(molecule, at_endstate="MM")
        mm_samples = generate_samples(sim, n_samples=5, n_steps_per_sample=10)
        # initialize simulation and start sampling at QML endstate
        sim = initialize_simulation_with_openff(molecule, at_endstate="QML")
        qml_samples = generate_samples(sim, n_samples=5, n_steps_per_sample=10)
    elif ff == "charmmff":
        molecule = generate_molecule(forcefield=ff, name=name)

        # initialize simulation for all thre cases
        sim = initialize_simulation_with_charmmff(molecule, name, at_endstate="mm")
        mm_samples = generate_samples(sim, n_samples=5, n_steps_per_sample=10)
        sim = initialize_simulation_with_charmmff(molecule, name, at_endstate="qml")
        qml_samples = generate_samples(sim, n_samples=5, n_steps_per_sample=10)
    else:
        raise NotImplemented


def test_generate_simulation_instances_with_openff():
    from endstate_rew.system import (
        generate_molecule,
        get_energy,
        initialize_simulation_with_openff,
    )
    from endstate_rew.constant import zinc_systems

    implementation, platform = check_implementation()

    # generate molecule
    ethane_smiles = "CC"
    m = generate_molecule(forcefield="openff", smiles=ethane_smiles)

    # initialize simulation for all three cases
    _ = initialize_simulation_with_openff(m, at_endstate="mm")
    _ = initialize_simulation_with_openff(m, at_endstate="qml")
    _ = initialize_simulation_with_openff(m)

    # check that potential that interpolates
    # returns the same values for the endstates
    # than the pure endstate implementation

    # at lambda=0.0 (mm endpoint)
    sim = initialize_simulation_with_openff(m, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_openff(m)
    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 0.0)
    else:
        sim.context.setParameter("lambda", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)

    # at lambda=1.0 (qml endpoint)
    sim = initialize_simulation_with_openff(m, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_openff(m)
    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 1.0)
    else:
        sim.context.setParameter("lambda", 1.0)
    e_sim_qml_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )

    assert np.isclose(e_sim_qml_endstate, e_sim_qml_interpolate_endstate)

    # double check that QML and MM endpoint have different energies
    sim = initialize_simulation_with_openff(m, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_openff(m, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    assert not np.isclose(e_sim_mm_endstate, e_sim_qml_endstate)


def test_charmm_system_generation():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import (
        create_charmm_system,
        generate_molecule,
        initialize_simulation_with_charmmff,
    )

    # list of all the charmm systems with the zinc id
    for zinc_name, smiles_str in zinc_systems:
        print(zinc_name)
        if (
            zinc_name == "ZINC00061095"
            or zinc_name == "ZINC00095858"
            or zinc_name == "ZINC00138607"
        ):  # skip system that has wrong topology
            continue

        molecule = generate_molecule(forcefield="charmmff", smiles=smiles_str)

        create_charmm_system(
            zinc_name,
        )
        _ = initialize_simulation_with_charmmff(molecule, zinc_name, at_endstate="mm")


def test_generate_simulation_instances_with_charmmff():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import (
        generate_molecule,
        get_energy,
        initialize_simulation_with_charmmff,
    )

    implementation, platform = check_implementation()

    # get zinc_id
    zinc_id = "ZINC00079729"
    smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    molecule = generate_molecule(forcefield="charmmff", smiles=smiles)

    # initialize simulation for all thre cases
    _ = initialize_simulation_with_charmmff(molecule, zinc_id, at_endstate="mm")
    _ = initialize_simulation_with_charmmff(molecule, zinc_id, at_endstate="qml")
    _ = initialize_simulation_with_charmmff(molecule, zinc_id)

    # check that potential that interpolates
    # returns the same values for the endstates
    # than the pure endstate implementation

    # at lambda=0.0 (mm endpoint)
    sim = initialize_simulation_with_charmmff(molecule, zinc_id, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_charmmff(molecule, zinc_id)
    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 0.0)
    else:
        sim.context.setParameter("lambda", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)

    # at lambda=1.0 (qml endpoint)
    sim = initialize_simulation_with_charmmff(molecule, zinc_id, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_charmmff(molecule, zinc_id)
    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 1.0)
    else:
        sim.context.setParameter("lambda", 1.0)
    e_sim_qml_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )

    assert np.isclose(e_sim_qml_endstate, e_sim_qml_interpolate_endstate)

    # double check that QML and MM endpoint have different energies
    sim = initialize_simulation_with_charmmff(molecule, zinc_id, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_charmmff(molecule, zinc_id, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    assert not np.isclose(e_sim_mm_endstate, e_sim_qml_endstate)
