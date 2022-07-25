from openmm.app import (
    CharmmParameterSet,
    CharmmPsfFile,
    PDBFile,
    CharmmCrdFile,
)
import endstate_correction
import pathlib

# define path to test systems
path = pathlib.Path(endstate_correction.__file__).resolve().parent
hipen_testsystem = f"{path}/data/hipen_data"

path = pathlib.Path(endstate_correction.__file__).resolve().parent
jctc_testsystem = f"{path}/data/jctc_data"


def test_sampling():
    """Test if we can sample with simulation instance in vacuum and watervox"""
    from endstate_correction.system import (
        create_charmm_system,
        read_box,
    )
    from endstate_correction.equ import generate_samples

    ########################################################
    ########################################################
    # ----------------- vacuum-- ---------------------------

    system_name = "ZINC00079729"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    crd = CharmmCrdFile(f"{hipen_testsystem}/{system_name}/{system_name}.crd")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )

    sim = create_charmm_system(psf=psf, parameters=params, env="vacuum", tlc="UNK")
    sim.context.setPositions(crd.positions)
    generate_samples(sim, 1, 50)

    ########################################################
    ########################################################
    # ----------------- waterbox ---------------------------
    # get all relevant files and initialize SIMulation

    system_name = "1_octanol"
    psf = CharmmPsfFile(
        f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/step3_input.psf"
    )
    pdb = PDBFile(f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/step3_input.pdb")
    params = CharmmParameterSet(
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.rtf",
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.prm",
        f"{jctc_testsystem}/toppar/top_all36_cgenff.rtf",
        f"{jctc_testsystem}/toppar/par_all36_cgenff.prm",
        f"{jctc_testsystem}/toppar/toppar_water_ions.str",
    )
    psf = read_box(psf, f"{jctc_testsystem}/{system_name}/charmm-gui/input.config.dat")
    sim = create_charmm_system(psf=psf, parameters=params, env="waterbox", tlc="UNK")
    sim.context.setPositions(pdb.positions)
    generate_samples(sim, 1, 50)
