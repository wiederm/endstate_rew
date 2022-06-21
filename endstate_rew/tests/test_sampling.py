import pytest
from endstate_rew.constant import check_implementation


@pytest.mark.parametrize(
    "ff",
    [("charmmff"), ("openff")],
)
def test_sampling(ff):
    from endstate_rew.system import (
        generate_samples,
        generate_molecule,
        initialize_simulation_with_openff,
        initialize_simulation_with_charmmff,
    )

    # generate molecule
    name = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    m = generate_molecule(forcefield=ff, smiles=smiles)

    if ff == "charmmff":
        # initialize simulation for all three cases
        sim = initialize_simulation_with_charmmff(m, at_endstate="mm", zinc_id=name)
        generate_samples(sim, 1, 50)
        sim = initialize_simulation_with_charmmff(m, at_endstate="qml", zinc_id=name)
        generate_samples(sim, 1, 50)
        sim = initialize_simulation_with_charmmff(m, zinc_id=name)
        generate_samples(sim, 1, 50)
    else:
        sim = initialize_simulation_with_openff(m, at_endstate="mm")
        generate_samples(sim, 1, 50)
        sim = initialize_simulation_with_openff(m, at_endstate="qml")
        generate_samples(sim, 1, 50)
        sim = initialize_simulation_with_openff(m)
        generate_samples(sim, 1, 50)
