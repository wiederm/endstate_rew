from openmm import unit
from openmmtools.constants import kB

### define units
distance_unit = unit.angstrom
time_unit = unit.femtoseconds
speed_unit = distance_unit / time_unit

# constants
stepsize = 1 * time_unit
collision_rate = 1 / unit.picosecond
temperature = 300 * unit.kelvin

kBT = kB * temperature

zinc_systems = [
    (
        "ZINC00061095",
        "CCOc1ccc2nc(/N=C\c3ccccc3O)sc2c1",
    ),  # NOTE: this system has a wrong topology in the psf file
    ("ZINC00077329", "Cn1cc(Cl)c(/C=N/O)n1"),  # hipen 2
    ("ZINC00079729", "S=c1cc(-c2ccc(Cl)cc2)ss1"),  # hipen 3
    ("ZINC00086442", "CN1C(=O)C/C(=N\O)N(C)C1=O"),  # hipen 4
    ("ZINC00087557", "NNC(=O)[C@H]1C(c2ccccc2)[C@@H]1C(=O)NN"),  # hipen 5
    (
        "ZINC00095858",
        "CCO/C(O)=N/S(=O)(=O)c1ccccc1Cl",
    ),  # NOTE: this system has a wrong topology in the psf file
    ("ZINC00107550", "C/C(=N\O)c1oc(C)nc1C"),  # hipen 7
    ("ZINC00107778", "O/N=C/C1=C(Cl)c2cc(Cl)ccc2OC1"),  # hipen 8
    ("ZINC00123162", "CC(=O)/C(=N/Nc1ccc(Cl)cc1)C(=O)c1ccccc1"),  # hipen 9
    ("ZINC00133435", "c1ccc(-c2nc3ccccc3nc2-c2ccccn2)nc1"),  # hipen 10
    (
        "ZINC00138607",
        "O=C(CC1=NO[C@H](c2ccccc2O)N1)N1CCCC1",
    ),  # NOTE: this system has a wrong topology in the psf file
    ("ZINC00140610", "Cc1cc(C)c2c(=O)[nH]sc2n1"),  # hipen 12
    ("ZINC00164361", "CCON1C(=O)c2ccccc2C1=O"),  # hipen 13
    ("ZINC00167648", "Cc1ccc(COn2c(-c3ccccc3)nc3ccccc3c2=O)cc1"),  # hipen 14
    ("ZINC00169358", "CC1=Cn2c(=O)c3ccccc3c(=O)n2C1"),  # hipen 15
    ("ZINC01036618", "COc1ccc(/C=C2/N=C(c3ccccc3)NNC2=O)cc1"),  # hipen n/a
    ("ZINC01755198", "CC(C)C(=O)NNC(=O)C(C)C"),  # hipen 16
    ("ZINC01867000", "c1ccc(-c2ccccc2-c2ccccc2)cc1"),  # hipen 17
    ("ZINC03127671", "O=C(CSCC(=O)Nc1ccccc1)NNC(=O)c1ccccc1"),  # hipen 18
    ("ZINC04344392", "CCOC(=O)NNC(=O)NCCCc1ccc2ccc3cccc4ccc1c2c34"),  # hipen 19
    ("ZINC04363792", "Clc1cc(Cl)cc(/N=c2\ssnc2-c2ccccc2Cl)c1"),  # hipen 20
    ("ZINC06568023", "O=C(NNC(=O)c1ccccc1)c1ccccc1"),  # hipen 21
    ("ZINC33381936", "O=S(=O)(O/N=C1/CCc2ccccc21)c1ccc(Cl)cc1"),  # hipen 22
]

from typing import Tuple


def check_implementation() -> Tuple[str, str]:
    try:
        from NNPOps import OptimizedTorchANI as _

        implementation = "nnpops"
        platform = "CUDA"
    except ModuleNotFoundError:
        platform = "CPU"
        implementation = ""

    print(implementation, platform)
    return implementation, platform
