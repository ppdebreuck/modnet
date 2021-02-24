import pytest
from pathlib import Path

from modnet.utils import get_hash_of_file


_TEST_DATA_HASHES = {
    "MP_2018.6_subset.zip": (
        "d7d75e646dbde539645c8c0b065fd82cbe93f81d3500809655bd13d0acf2027c"
        "1786091a73f53985b08868c5be431a3c700f7f1776002df28ebf3a12a79ab1a1"
    ),
    "MP_2018.6_small.zip": (
        "b7f31d066113d1ad1f4f3990250019835bad96c18eddefd4a0b3866fd23a6037"
        "d1ad90b1f4a1e08d12d7f0a7ce2ebcf4a1a4b673500e1118543b687dbd1749e6"
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


@pytest.fixture(scope="module")
def tf_session():
    """This fixture can be used to sandbox tests that require tensorflow."""
    import tensorflow

    tensorflow.compat.v1.disable_eager_execution()
    with tensorflow.device("/device:CPU:0") as session:
        yield session

    tensorflow.keras.backend.clear_session()
