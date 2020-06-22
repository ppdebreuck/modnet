#!/usr/bin/env python


import numpy as np
import pandas as pd
import pytest
from modnet.preprocessing import nmi_target


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
