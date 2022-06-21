# general imports
import os
import pickle
import sys

import numpy as np
import torch
from endstate_rew.constant import zinc_systems
from endstate_rew.system import (
    generate_samples,
    generate_molecule,
    initialize_simulation_with_charmmff,
    initialize_simulation_with_openff,
)

### set number of CPU threads used by pytorch
num_threads = 2
torch.set_num_threads(num_threads)

###################
# parse command line arguments
if len(sys.argv) > 2:
    print("Simulating zink system")
    run_id = int(sys.argv[1])
    zink_id = int(sys.argv[2])
    name, smiles = zinc_systems[zink_id]
else:
    run_id = int(sys.argv[1])
    name = "2cle"
    smiles = "ClCCOCCCl"
###################
ff = "charmmff" #"openff" #"charmmff"  # openff
n_samples = 5_000
n_steps_per_sample = 1_000
n_lambdas = 11
platform = "CUDA"
###################
###################
print(f"{zink_id=}")
print(f"{name=}")
print(f"{smiles=}")
print(f"{run_id=}")
print(f"{ff=}")
print(f"{platform=}")
print(f"{n_lambdas=}")

assert ff == "openff" or ff == "charmmff"
lambs = np.linspace(0, 1, n_lambdas)
assert len(lambs) == n_lambdas
assert lambs[0] == 0.0
assert lambs[-1] == 1.0
###################
# generate mol
if ff == "openff" and smiles:
    molecule = generate_molecule(forcefield=ff, smiles=smiles)
elif ff == "charmmff" and smiles:
    molecule = generate_molecule(forcefield=ff, name=name)
else:
    raise RuntimeError("Only openff can be used with SMILES input")
# initialize working directory
w_dir = f"/data/shared/projects/endstate_rew/{name}/sampling_{ff}/run{run_id:0>2d}/"
os.makedirs(w_dir, exist_ok=True)
print(f"saving to: {w_dir}")
# select a random conformation
from random import randint

conf_id = randint(0, molecule.n_conformers - 1)
print(f"select conf_id: {conf_id}")
###################
# initialize simulation depending on ff keyword
if ff == "openff":
    sim = initialize_simulation_with_openff(
        molecule,
        w_dir=f"/data/shared/projects/endstate_rew/{name}/",
        conf_id=conf_id,
    )
elif ff == "charmmff":
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id=name, conf_id=conf_id
    )
else:
    raise RuntimeError("Either openff or charmmff. Abort.")
###################
# perform lambda protocoll
for lamb in lambs:
    print(f"{lamb=}")
    # set lambda
    sim.context.setParameter("scale", lamb)
    # set coordinates
    sim.context.setPositions(molecule.conformers[conf_id])
    # collect samples
    samples = generate_samples(
        sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
    )
    # save samples
    pickle.dump(
        samples,
        open(
            f"{w_dir}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}.pickle",
            "wb+",
        ),
    )
    print(
        f"traj dump to: {w_dir}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}.pickle"
    )
