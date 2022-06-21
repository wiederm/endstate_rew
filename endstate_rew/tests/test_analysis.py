import numpy as np
import torch
import pytest
import os

### set number of CPU threads used by pytorch
num_threads = 2
torch.set_num_threads(num_threads)


@pytest.mark.parametrize(
    "ff",
    ["charmmff", "openff"],
)
def test_collect_equ_samples(ff):
    """test if we are able to collect samples as anticipated"""
    from endstate_rew.analysis import _collect_equ_samples

    lambs = np.linspace(0, 1, 11)
    name = "ZINC00077329"
    path = f"data/{name}/sampling_{ff}/run01/"
    samples, N_k = _collect_equ_samples(path, name=name, lambda_scheme=lambs)

    print(N_k)
    assert N_k[0] == 2000
    assert len(samples) == 22000

    samples, N_k = _collect_equ_samples(
        path, name=name, lambda_scheme=lambs, only_endstates=True
    )
    print(N_k)
    assert N_k[0] == 2000
    assert N_k[-1] == 2000
    assert len(samples) == 4000

    lambs = [0, 1]
    samples, N_k = _collect_equ_samples(
        path, name=name, lambda_scheme=lambs, only_endstates=True
    )

    print(N_k)
    assert N_k[0] == 2000
    assert N_k[-1] == 2000
    assert len(samples) == 4000

    mm_samples = samples[: int(N_k[0])]
    qml_samples = samples[int(N_k[0]) :]
    assert len(mm_samples) == 2_000
    assert len(qml_samples) == 2_000


@pytest.mark.parametrize(
    "ff, nr_of_switches",
    [("charmmff", 200), pytest.param("openff", 200, marks=pytest.mark.xfail)],
)
def test_collect_work_values(ff, nr_of_switches):
    """test if we are able to collect samples as anticipated"""
    from endstate_rew.analysis import _collect_work_values

    print(ff)
    path = f"data/ZINC00077329/switching_{ff}/ZINC00077329_neq_ws_from_mm_to_qml_{nr_of_switches}_5001.pickle"
    ws = _collect_work_values(path)
    assert len(ws) == nr_of_switches


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Runs out of time on MacOS",  # TODO: FIXME!
)
@pytest.mark.parametrize(
    "ff, ddG",
    [
        ("charmmff", -940544.0390218807),
        ("openff", -940689.0530839318),
    ],
)
def test_equilibrium_free_energy(ff, ddG):
    "test that u_kn can be calculated and that results are consistent whether we reload mbar pickle or regernerate it"
    from endstate_rew.analysis import calculate_u_kn
    from pymbar import MBAR

    name = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    path = f"data/{name}/sampling_{ff}/run01/"

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield=ff,
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=False,
        override=True,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], ddG, rtol=1e-06)

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield="charmmff",
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=True,
        override=False,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], ddG, rtol=1e-06)


@pytest.mark.parametrize(
    "ff",
    [
        ("charmmff"),
        ("openff"),
    ],
)
def test_plotting_equilibrium_free_energy(ff):
    "Test that plotting functions can be called"
    from endstate_rew.analysis import calculate_u_kn
    from endstate_rew.analysis import (
        plot_overlap_for_equilibrium_free_energy,
        plot_results_for_equilibrium_free_energy,
    )

    name = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    path = f"data/{name}/sampling_{ff}/run01/"

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield=ff,
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=False,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=name)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Requires input data that are not provided in the repo",
)
@pytest.mark.parametrize(
    "ff",
    [
        ("charmmff"),
        pytest.param("openff", marks=pytest.mark.xfail),
    ],
)
def test_collect_results(ff):
    from endstate_rew.analysis import (
        collect_results_from_neq_and_equ_free_energy_calculations,
    )

    name = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    path = f"data/{name}/"

    collect_results_from_neq_and_equ_free_energy_calculations(
        w_dir=path,
        forcefield=ff,
        run_id=1,
        smiles=smiles,
        name=name,
    )
