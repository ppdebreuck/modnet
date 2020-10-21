#!/usr/bin/env python


from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from modnet.preprocessing import get_cross_nmi, nmi_target, MODData
from .utils import get_sha512_of_file


def test_nmi_target():

    # Test with linear data (should get 1.0 mutual information, or very close due to algorithm used
    # in mutual_info_regression)
    npoints = 31
    x = np.linspace(0.5, 3.5, npoints)
    y = 2*x - 2
    z = 4*x + 2

    df_feat = pd.DataFrame({'x': x, 'y': y})
    df_target = pd.DataFrame({'z': z})

    # Here we fix the number of neighbors for the call to sklearn.feature_selection's mutual_info_regression to 2 so
    # that we get exactly 1 for the mutual information.
    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, n_neighbors=2)

    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc['x']['z'] == pytest.approx(1.0)
    assert df_nmi_target.loc['y']['z'] == pytest.approx(1.0)

    # Same data shuffled
    # Shuffle the x, y and z
    indices = np.arange(npoints)
    np.random.seed(42)
    np.random.shuffle(indices)
    xs = x.take(indices)
    ys = y.take(indices)
    zs = z.take(indices)

    df_feat = pd.DataFrame({'x': xs, 'y': ys})
    df_target = pd.DataFrame({'z': zs})

    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, n_neighbors=2)

    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc['x']['z'] == pytest.approx(1.0)
    assert df_nmi_target.loc['y']['z'] == pytest.approx(1.0)

    # Test with one constant feature
    c = np.ones(npoints) * 1.4
    df_feat = pd.DataFrame({'x': x, 'y': y, 'c': c})
    df_target = pd.DataFrame({'z': z})

    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, n_neighbors=2)
    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc['x']['z'] == pytest.approx(1.0)
    assert df_nmi_target.loc['y']['z'] == pytest.approx(1.0)

    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, drop_constant_features=False, n_neighbors=2)
    assert df_nmi_target.shape == (3, 1)
    assert df_nmi_target.loc['x']['z'] == pytest.approx(1.0)
    assert df_nmi_target.loc['y']['z'] == pytest.approx(1.0)
    assert df_nmi_target.loc['c']['z'] == pytest.approx(0.0)

    # Test with unrelated data (grid)
    x = np.linspace(start=2, stop=5, num=4)
    z = np.linspace(start=3, stop=7, num=5)
    x, z = np.meshgrid(x, z)
    x = x.flatten()
    z = z.flatten()
    df_feat = pd.DataFrame({'x': x})
    df_target = pd.DataFrame({'z': z})

    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target)
    assert df_nmi_target.shape == (1, 1)
    assert df_nmi_target.loc['x']['z'] == pytest.approx(0.0)

    # Test initial checks
    # Incompatible shapes
    x = np.linspace(start=2, stop=3, num=5)
    z = np.linspace(start=2, stop=3, num=8)
    df_feat = pd.DataFrame({'x': x})
    df_target = pd.DataFrame({'z': z})
    with pytest.raises(ValueError, match=r'The input features DataFrame and the target variable DataFrame '
                                         r'should contain the same number of data points.'):
        nmi_target(df_feat=df_feat, df_target=df_target)
    # Target DataFrame does not have exactly one column
    x = np.linspace(start=2, stop=3, num=5)
    z = np.linspace(start=2, stop=3, num=5)
    df_feat = pd.DataFrame({'x': x})
    df_target = pd.DataFrame({'z2': z, 'z': z})
    with pytest.raises(ValueError, match=r'The target DataFrame should have exactly one column.'):
        nmi_target(df_feat=df_feat, df_target=df_target)

    # Test with some more real data (for which NMI is not just 0.0 or 1.0)
    npoints = 200
    np.random.seed(42)
    x = np.random.rand(npoints)
    z = 4 * x + 1.0 * np.random.rand(npoints)

    df_feat = pd.DataFrame({'x': x})
    df_target = pd.DataFrame({'z': z})

    # Here we fix the random_state for the call to sklearn.feature_selection's mutual_info_regression so
    # that we always get the same value.
    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, random_state=42)
    assert df_nmi_target.shape == (1, 1)
    assert df_nmi_target.loc['x']['z'] == pytest.approx(0.3417665092162398)


def test_get_cross_nmi():

    # Test with linear data (should get 1.0 mutual information, or very close due to algorithm used
    # in mutual_info_regression)
    npoints = 31
    x = np.linspace(0.5, 3.5, npoints)
    y = 2*x - 2
    z = 4*x + 2

    df_feat = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # Here we fix the number of neighbors for the call to sklearn.feature_selection's mutual_info_regression to 2 so
    # that we get exactly 1 for the mutual information.
    df_cross_nmi = get_cross_nmi(df_feat=df_feat, n_neighbors=2)

    assert df_cross_nmi.shape == (3, 3)
    for idx in df_cross_nmi.index:
        for col in df_cross_nmi.columns:
            assert df_cross_nmi.loc[idx][col] == pytest.approx(1.0)

    # Same data shuffled
    # Shuffle the x, y and z
    indices = np.arange(npoints)
    np.random.seed(42)
    np.random.shuffle(indices)
    xs = x.take(indices)
    ys = y.take(indices)
    zs = z.take(indices)

    df_feat = pd.DataFrame({'x': xs, 'y': ys, 'z': zs})

    df_cross_nmi = get_cross_nmi(df_feat=df_feat, n_neighbors=2)

    assert df_cross_nmi.shape == (3, 3)
    for idx in df_cross_nmi.index:
        for col in df_cross_nmi.columns:
            assert df_cross_nmi.loc[idx][col] == pytest.approx(1.0)

    # Test with one constant feature
    c = np.ones(npoints) * 1.4
    df_feat = pd.DataFrame({'x': x, 'y': y, 'z': z, 'c': c})

    df_cross_nmi = get_cross_nmi(df_feat=df_feat, n_neighbors=2)
    assert df_cross_nmi.shape == (4, 4)
    for idx in df_cross_nmi.index:
        for col in df_cross_nmi.columns:
            expected = 0.0 if idx == 'c' or col == 'c' else 1.0
            assert df_cross_nmi.loc[idx][col] == pytest.approx(expected)

    # Test with unrelated data (grid)
    x = np.linspace(start=2, stop=5, num=4)
    y = np.linspace(start=3, stop=7, num=5)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    df_feat = pd.DataFrame({'x': x, 'y': y})

    df_cross_nmi = get_cross_nmi(df_feat=df_feat, n_neighbors=2)
    assert df_cross_nmi.shape == (2, 2)
    assert df_cross_nmi.loc['x']['y'] == pytest.approx(0.0)
    assert df_cross_nmi.loc['y']['x'] == pytest.approx(0.0)

    # Test with some more real data (for which NMI is not just 0.0 or 1.0)
    npoints = 200
    np.random.seed(42)
    x = np.random.rand(npoints)
    y = 4 * x + 1.0 * np.random.rand(npoints)

    df_feat = pd.DataFrame({'x': x, 'y': y})

    # Here we fix the random_state for the call to sklearn.feature_selection's mutual_info_regression so
    # that we always get the same value.
    df_cross_nmi = get_cross_nmi(df_feat=df_feat, random_state=42)
    assert df_cross_nmi.shape == (2, 2)
    assert df_cross_nmi.loc['x']['x'] == pytest.approx(1.0)
    assert df_cross_nmi.loc['y']['y'] == pytest.approx(1.0)
    assert df_cross_nmi.loc['x']['y'] == pytest.approx(0.3417665092162398)
    assert df_cross_nmi.loc['y']['x'] == pytest.approx(0.3417665092162398)


def test_load_moddata_zip():
    """ This test checks that older MODData objects can still be loaded. """

    data_file = Path(__file__).parent.joinpath("data/MP_2018.6_subset.zip")

    # Loading pickles can be dangerous, so lets at least check that the MD5 matches
    # what it was when created
    assert (
        get_sha512_of_file(data_file) ==
        "d7d75e646dbde539645c8c0b065fd82cbe93f81d3500809655bd13d0acf2027c"
        "1786091a73f53985b08868c5be431a3c700f7f1776002df28ebf3a12a79ab1a1"
    )

    data = MODData.load(data_file)
    assert len(data.structures) == 100
    assert len(data.mpids) == 100
    assert len(data.df_structure) == 100
    assert len(data.df_featurized) == 100
    assert len(data.df_targets) == 100
    assert len(data.df_targets) == 100


def test_small_moddata_featurization():
    """ This test creates a new MODData from the MP 2018.6 structures. """
    data_file = Path(__file__).parent.joinpath("data/MP_2018.6_small.zip")

    # Loading pickles can be dangerous, so lets at least check that the MD5 matches
    # what it was when created
    assert (
        get_sha512_of_file(data_file) ==
        "937a29dad32d18e47c84eb7c735ed8af09caede21d2339c379032"
        "fbd40c463d8ca377d9e3a777710b5741295765f6c12fbd7ab56f9176cc0ca11c9120283d878"
    )

    old = MODData.load(data_file)
    structures = old.structures
    targets = old.targets

    names = old.names
    new = MODData(structures, targets, target_names=names)
    new.featurize(fast=False)

    new_cols = sorted(new.df_featurized.columns.tolist())
    old_cols = sorted(old.df_featurized.columns.tolist())

    for i in range(len(old_cols)):
        print(new_cols[i], old_cols[i])
        assert new_cols[i] == old_cols[i]

    np.testing.assert_array_equal(
        old_cols,
        new_cols
    )

    for col in new.df_featurized.columns:
        np.testing.assert_almost_equal(
            new.df_featurized[col].to_numpy(),
            old.df_featurized[col].to_numpy(),
        )


def test_merge_ranked():
    from modnet.preprocessing import merge_ranked

    # Test lists of the same length
    test_features = [
        ["a", "b", "c"],
        ["d", "b", "e"]
    ]

    expected = ["a", "d", "b", "c", "e"]
    assert merge_ranked(test_features) == expected

    # Test lists of different length
    test_features = [
        ["a", "b", "c"],
        ["d", "b", "e", "g"]
    ]

    expected = ["a", "d", "b", "c", "e", "g"]
    assert merge_ranked(test_features) == expected

    test_features = [
        ["d", "b", "e", "g"],
        ["a", "b", "c"]
    ]

    expected = ["d", "a", "b", "e", "c", "g"]
    assert merge_ranked(test_features) == expected

    # Test lists with other hashable types
    test_features = [
        ["a", "b", "c"],
        ["d", "b", 2, "g"],
        ["c", 0, "e", "0"]
    ]

    expected = ["a", "d", "c", "b", 0, 2, "e", "g", "0"]
    assert merge_ranked(test_features) == expected
