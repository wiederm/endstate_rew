"""
Unit and regression test for the endstate_rew package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import os
import numpy as np
from openmm import unit
from endstate_rew.constant import check_implementation


def test_endstate_rew_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "endstate_rew" in sys.modules


def test_hipen_import():

    from endstate_rew.system import _get_hipen_data

    print(_get_hipen_data())


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Requires 'CUDA' and 'nnpops' enabled openmm-ml, which is currently not available on conda",  # TODO: FIXME!
)
def test_interpolation():
    from endstate_rew.system import (
        initialize_simulation_with_openff,
        generate_molecule,
        initialize_simulation_with_charmmff,
        # remap_atoms,
    )

    zinc_id = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    molecule = generate_molecule(forcefield="charmmff", smiles=smiles)
    implementation, platform = check_implementation()
    #    print('Skipping test --- only u')

    #######################################
    # initialize simulation for charmmff
    #######################################
    sim = initialize_simulation_with_charmmff(molecule, zinc_id, conf_id=0)

    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 1.0)
    else:
        sim.context.setParameter("lambda", 1.0)

    u_now = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    print(u_now)
    assert np.isclose(u_now, -2346049.500037839)
    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 0.0)
    else:
        sim.context.setParameter("lambda", 0.0)
    u_now = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    print(u_now)
    assert np.isclose(u_now, -29.702918583)

    #####################################
    # initialize simulation for openff
    #####################################
    sim = initialize_simulation_with_openff(molecule, conf_id=0)

    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 1.0)
    else:
        sim.context.setParameter("lambda", 1.0)
    u_now = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    print(u_now)
    assert np.isclose(u_now, -2346049.500037839)
    if implementation.lower() == "nnpops":
        sim.context.setParameter("scale", 0.0)
    else:
        sim.context.setParameter("lambda", 0.0)
    u_now = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    print(u_now)
    assert np.isclose(u_now, 298.240629196167)
