# general imports
import pickle
import sys
import os
from os import path
import numpy as np
import torch
from endstate_rew.neq import perform_switching
from endstate_rew.system import (
    generate_molecule,
    initialize_simulation_with_charmmff,
    initialize_simulation_with_openff,
)
from endstate_rew.constant import zinc_systems
from glob import glob

num_threads = 2
torch.set_num_threads(num_threads)

###########################################################################################
# define run
if len(sys.argv) == 2:  # only true for ZINK systems
    print("Simulating zink system")
    zink_id = int(sys.argv[1])
    assert zink_id < 25
    name, smiles = zinc_systems[zink_id]
elif len(sys.argv) == 1:  # smiles and name must be provided inside of script
    ##############
    #
    name = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    #
    ##############

# choose ff and working directory
ff = "charmmff"  # "charmmff"  # openff
w_dir = f"/data/shared/projects/endstate_rew/{name}/"
# equilibrium samples
n_samples = 5_000
n_steps_per_sample = 1_000
#############
# NEQ
switching_length = 5_001
nr_of_switches = 200
#############

print(f"{ff=}")
print(f"{name=}")
print(f"{smiles=}")

mm_to_qml_filename = f"{w_dir}/switching_{ff}/{name}_neq_ws_from_mm_to_qml_{nr_of_switches}_{switching_length}.pickle"
qml_to_mm_filename = f"{w_dir}/switching_{ff}/{name}_neq_ws_from_qml_to_mm_{nr_of_switches}_{switching_length}.pickle"

if path.isfile(mm_to_qml_filename) and path.isfile(qml_to_mm_filename):
    print("All work values have already been calculated.")
    sys.exit()

# create folder
os.makedirs(f"{w_dir}/switching_{ff}", exist_ok=True)
print(f"Generate directory: {w_dir}/switching_{ff}")

###########################################################################################
###########################################################################################
# generate mol
if ff == "openff" and smiles:
    molecule = generate_molecule(forcefield=ff, smiles=smiles)
elif ff == "charmmff" and smiles and not name:
    raise RuntimeError("Charmff can not be used with SMILES input")
else:
    molecule = generate_molecule(forcefield=ff, name=name)

# initialize simulation depending on ff keyword
if ff == "openff":
    sim = initialize_simulation_with_openff(
        molecule=molecule,
        w_dir=f"/data/shared/projects/endstate_rew/{name}/",
    )
elif ff == "charmmff":
    sim = initialize_simulation_with_charmmff(molecule=molecule, zinc_id=name)
###########################################################################################
# load samples for lambda=0. , the mm endstate
mm_samples = []
mm_sample_files = glob(
    f"/data/shared/projects/endstate_rew/{name}/sampling_{ff}/run*/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.pickle"
)
nr_of_runs = len(mm_sample_files)

for samples in mm_sample_files:
    mm_samples.extend(pickle.load(open(samples, "rb")))

assert len(mm_samples) == nr_of_runs * n_samples

###########################################################################################
# load samples for lambda=1. , the qml endstate
qml_samples = []
qml_sample_files = glob(
    f"/data/shared/projects/endstate_rew/{name}/sampling_{ff}/run*/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.pickle"
)
nr_of_runs = len(qml_sample_files)

for samples in qml_sample_files:
    qml_samples.extend(pickle.load(open(samples, "rb")))

assert len(qml_samples) == nr_of_runs * n_samples


###########################################################################################
# MM endstate
# perform switching only if file does not already exist
if not path.isfile(mm_to_qml_filename):
    # define lambda space
    lambs = np.linspace(0, 1, switching_length)
    # perform NEQ from MM to QML
    ws_from_mm_to_qml = perform_switching(
        sim,
        lambdas=lambs,
        samples=mm_samples,
        nr_of_switches=nr_of_switches,
    )
    # dump work values
    print(ws_from_mm_to_qml)
    pickle.dump(ws_from_mm_to_qml, open(mm_to_qml_filename, "wb+"))
else:
    print(f"Already calculated: {mm_to_qml_filename}")

###########################################################################################
# QML endstate
# # perform switching only if file does not already exist
if not path.isfile(qml_to_mm_filename):
    # define lambda space
    lambs = np.linspace(1, 0, switching_length)
    # perform NEQ from QML to MM
    ws_from_qml_to_mm = perform_switching(
        sim,
        lambdas=lambs,
        samples=qml_samples,
        nr_of_switches=nr_of_switches,
    )
    # dump work values
    pickle.dump(ws_from_qml_to_mm, open(qml_to_mm_filename, "wb+"))
    print(ws_from_qml_to_mm)
else:
    print(f"Already calculated: {qml_to_mm_filename}")
