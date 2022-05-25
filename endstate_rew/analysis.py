import glob
import pickle
from collections import namedtuple
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openmm import unit
from pymbar import BAR, EXP
from tqdm import tqdm

from endstate_rew.constant import kBT


def _collect_equ_samples(path: str, name: str, lambda_scheme: list):
    """Collect equilibrium samples"""
    nr_of_samples = 5_000
    nr_of_steps = 1_000
    coordinates = []
    # loop over lambda scheme and collect samples in nanometer
    for lamb in lambda_scheme:
        file = glob.glob(
            f"{path}/{name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.pickle"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        elif len(file) == 0:
            print("WARNING! Incomplete equ sampling. Proceed with cautions.")
        else:
            coords_ = pickle.load(open(file[0], "rb"))
            coordinates.extend([c_.value_in_unit(unit.nanometer) for c_ in coords_])

    number_of_samples = len(coordinates)
    print(f"Number of samples loaded: {number_of_samples}")
    return coordinates * unit.nanometer


def calculate_u_ln(
    smiles: str,
    path: str,
    name: str,
):
    from endstate_rew.system import generate_molecule, initialize_simulation_with_openff

    # generate molecule
    m = generate_molecule(smiles)
    # initialize simulation
    # first, modify path to point to openff molecule object
    w_dir = path.split("/")
    w_dir = "/".join(w_dir[:-3])
    print(w_dir)
    sim = initialize_simulation_with_openff(m, w_dir=w_dir)

    lambda_scheme = np.linspace(0, 1, 11)
    samples = _collect_equ_samples(path, name, lambda_scheme)
    samples = np.array(samples.value_in_unit(unit.nanometer))  # positions in nanometer
    samples = samples[1_000:]  # remove the first 1k samples
    samples = samples[::4]  # take only every second sample #NOTE: for now every 4th
    N_k = len(samples) / len(lambda_scheme)
    print(f"{N_k=}")
    u_list = np.zeros((len(lambda_scheme), N_k))
    for lamb_idx in range(lambda_scheme):
        lamb = lambda_scheme[lamb_idx]
        sim.context.setParameter("lambda", lamb)
        us = []
        for x in tqdm(range(len(samples))):
            sim.context.setPositions(samples[x])
            u_ = sim.context.getState(getEnergy=True).getPotentialEnergy() / kBT
            us.append(u_)
        u_list[lamb_idx] = np.array(us)


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
    molecule = generate_molecule(smiles)

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
