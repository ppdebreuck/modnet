# coding: utf-8
# Distributed under the terms of the MIT License.

import pytest
import pandas
import numpy as np

from modnet.ext_data import load_ext_dataset


def test_load_datasets():

    with pytest.raises(ValueError):
        load_ext_dataset("abcd", "MODData")

    with pytest.raises(ValueError):
        load_ext_dataset("MP_2018.6_CROSS_NMI", "MODData")

    path = load_ext_dataset("MP_2018.6_CROSS_NMI", "cross_nmi")
    df = pandas.read_pickle(path)
    assert df.shape == (1304, 1304)
    np.testing.assert_array_equal(df.columns, df.index.values)
