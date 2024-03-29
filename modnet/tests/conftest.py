import pytest
from pathlib import Path
from modnet.preprocessing import CompositionContainer

from modnet.utils import get_hash_of_file
from pymatgen.core import Structure


_TEST_DATA_HASHES = {
    "MP_2018.6_subset.zip": (
        "d7d75e646dbde539645c8c0b065fd82cbe93f81d3500809655bd13d0acf2027c"
        "1786091a73f53985b08868c5be431a3c700f7f1776002df28ebf3a12a79ab1a1"
    ),
    "MP_2018.6_small_2020.zip": (
        "0efc2ce998faaadc9cf54a25e1db80834c5f53b1298da0e824ee2675124f47c8"
        "3fce2a86971a92eb3d0a860d29e0eb37683aa47ec80af2b6c8dee879584b1491"
    ),
    "MP_2018.6_small_2023.zip": (
        "47e3f34fe31679575b3143ea410d61d73158794018bdb09d362672fd3e048bbc"
        "b2d7967bceb664918b1af77e2e6c853f3ae642a42cd8ce47db304c5612cb3aeb"
    ),
    "MP_2018.6_small_composition_2020.zip": (
        "59f8c4e546df005799e3fb7a1e64daa0edfece48fa346ab0d2efe92aa107d0d1"
        "b14bb16f56bfe3f54e5a9020d088a268536f6ad86134e264ed7547b4fd583c79"
    ),
    "MP_2018.6_small_composition_2023.zip": (
        "519e6bc8c2f7277e8f9d9f8e99d4def3fc088de1978857bfaef2aa0ff2db873e"
        "c5c4bb3beeda58f1508d2ea06a98aa4743b80f991fe25007fd8f0bfa11d92edd"
    ),
}


def _load_moddata(filename):
    """Loads the pickled MODData from the test directory and checks it's hash."""
    from modnet.preprocessing import MODData

    data_file = Path(__file__).parent.joinpath(f"data/{filename}")
    if filename not in _TEST_DATA_HASHES:
        raise RuntimeError(
            f"Cannot verify hash of {filename} as it was not provided, will not load pickle."
        )
    # Loading pickles can be dangerous, so lets at least check that the MD5 matches
    # what it was when created
    assert get_hash_of_file(data_file) == _TEST_DATA_HASHES[filename]

    moddata = MODData.load(data_file)
    # For forwards compatibility with pymatgen, we have to patch our old test data to have the following attributes
    # to allow for depickling
    # This is hopefully only a temporary solution, and in future, we should serialize pymatgen objects
    # with Monty's `from_dict`/`to_dict` to avoid having to hack this private interface
    for ind, s in enumerate(moddata.structures):
        if isinstance(s, Structure):
            # assume all previous data was periodic
            moddata.structures[ind].lattice._pbc = [True, True, True]
            for jnd, site in enumerate(s.sites):
                # assume all of our previous data had ordered sites
                moddata.structures[ind].sites[jnd].label = str(next(iter(site.species)))
                # required for the global structure.is_ordered to work
                moddata.structures[ind].sites[jnd].species._n_atoms = 1.0
        elif isinstance(s, CompositionContainer):
            moddata.structures[ind].composition._n_atoms = s.composition._natoms

    return moddata


@pytest.fixture(scope="function")
def subset_moddata():
    """Loads the 100-structure featurized subset of MP.2018.6 for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_subset.zip")


@pytest.fixture(scope="function")
def small_moddata_2023():
    """Loads the small 5-structure featurized subset of MP.2018.6 for use
    in other tests, checking only the hash, updated for 2023.

    """
    return _load_moddata("MP_2018.6_small_2023.zip")


small_moddata = small_moddata_2023
"""Alias for new data."""


@pytest.fixture(scope="function")
def small_moddata_composition_2023():
    """Loads the small 5-structure featurized subset of MP.2018.6 composition only for use
    in other tests, checking only the hash, updated for 2023.

    """
    return _load_moddata("MP_2018.6_small_composition_2023.zip")


@pytest.fixture(scope="function")
def small_moddata_2020():
    """Loads the small 5-structure featurized subset of MP.2018.6 for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_small_2020.zip")


@pytest.fixture(scope="function")
def small_moddata_composition_2020():
    """Loads the small 5-structure featurized subset of MP.2018.6 composition only for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_small_composition_2020.zip")


@pytest.fixture(scope="module")
def tf_session():
    """This fixture can be used to sandbox tests that require tensorflow."""
    import tensorflow

    tensorflow.compat.v1.disable_eager_execution()
    with tensorflow.device("/device:CPU:0") as session:
        yield session

    tensorflow.keras.backend.clear_session()
