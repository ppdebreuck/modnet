import pytest
from pathlib import Path

from modnet.utils import get_hash_of_file


_TEST_DATA_HASHES = {
    "MP_2018.6_subset.zip": (
        "d7d75e646dbde539645c8c0b065fd82cbe93f81d3500809655bd13d0acf2027c"
        "1786091a73f53985b08868c5be431a3c700f7f1776002df28ebf3a12a79ab1a1"
    ),
    "MP_2018.6_small.zip": (
        "0efc2ce998faaadc9cf54a25e1db80834c5f53b1298da0e824ee2675124f47c8"
        "3fce2a86971a92eb3d0a860d29e0eb37683aa47ec80af2b6c8dee879584b1491"
    ),
    "MP_2018.6_small_composition.zip": (
        "59f8c4e546df005799e3fb7a1e64daa0edfece48fa346ab0d2efe92aa107d0d1"
        "b14bb16f56bfe3f54e5a9020d088a268536f6ad86134e264ed7547b4fd583c79"
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

    return MODData.load(data_file)


@pytest.fixture(scope="function")
def subset_moddata():
    """Loads the 100-structure featurized subset of MP.2018.6 for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_subset.zip")


@pytest.fixture(scope="function")
def small_moddata():
    """Loads the small 5-structure featurized subset of MP.2018.6 for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_small.zip")


@pytest.fixture(scope="function")
def small_moddata_composition():
    """Loads the small 5-structure featurized subset of MP.2018.6 composition only for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_small_composition.zip")


@pytest.fixture(scope="module")
def tf_session():
    """This fixture can be used to sandbox tests that require tensorflow."""
    import tensorflow

    tensorflow.compat.v1.disable_eager_execution()
    with tensorflow.device("/device:CPU:0") as session:
        yield session

    tensorflow.keras.backend.clear_session()
