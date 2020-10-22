import pytest
from pathlib import Path

from modnet.utils import get_hash_of_file


_TEST_DATA_HASHES = {
    "MP_2018.6_subset.zip": (
        "d7d75e646dbde539645c8c0b065fd82cbe93f81d3500809655bd13d0acf2027c"
        "1786091a73f53985b08868c5be431a3c700f7f1776002df28ebf3a12a79ab1a1"
    ),
    "MP_2018.6_small.zip": (
        "937a29dad32d18e47c84eb7c735ed8af09caede21d2339c379032fbd40c463d8"
        "ca377d9e3a777710b5741295765f6c12fbd7ab56f9176cc0ca11c9120283d878"
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


@pytest.fixture
def subset_moddata():
    """Loads the 100-structure featurized subset of MP.2018.6 for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_subset.zip")


@pytest.fixture
def small_moddata():
    """Loads the small 5-structure featurized subset of MP.2018.6 for use
    in other tests, checking only the hash.

    """
    return _load_moddata("MP_2018.6_small.zip")
