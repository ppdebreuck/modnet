#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest
from modnet.preprocessing import MODData, get_cross_nmi, nmi_target


def check_column_values(new: MODData, reference: MODData, tolerance=0.03):
    """Utiliy function  to check that moddata values haven't
    changed between initial calculation.

    Allows for some columns to be checked more loosely (see inline comment below).

    """
    new_cols = set(new.df_featurized.columns)
    old_cols = set(reference.df_featurized.columns)

    # Check that the new df only adds new columns and is not missing anything
    assert not (old_cols - new_cols)

    error_cols = set()
    for col in old_cols:
        if not (
            np.absolute(
                (
                    new.df_featurized[col].to_numpy()
                    - reference.df_featurized[col].to_numpy()
                )
                / (reference.df_featurized[col].to_numpy() + 1e-6)
            ).max()
            < tolerance
        ):
            error_cols.add(col)

    # Unfortuanately there are some columns we cannot control, e.g.,
    # default symmetry tolerance changing in pymatgen leading to a
    # different number of symm ops being detected.

    # We need a mechanism to allow these discrepancies through in certain cases:
    allowed_bad_columns = [
        "GlobalSymmetryFeatures|n_symmetry_ops",
        "GlobalSymmetryFeatures|crystal_system",
        "YangSolidSolution|Yang delta",
        "Miedema|Miedema_deltaH_inter",
        "AtomicPackingEfficiency|mean simul. packing efficiency",
        "Miedema|Miedema_deltaH_amor",
        "AtomicPackingEfficiency|mean abs simul. packing efficiency",
        "Miedema|Miedema_deltaH_ss_min",
    ]

    for col in allowed_bad_columns:
        if col in error_cols:
            error_cols.remove(col)

    assert not error_cols, f"Some columns contained errors: {error_cols}"


def test_nmi_target():

    # Test with linear data (should get 1.0 mutual information, or very close due to algorithm used
    # in mutual_info_regression)
    npoints = 31
    x = np.linspace(0.5, 3.5, npoints)
    y = 2 * x - 2
    z = 4 * x + 2

    df_feat = pd.DataFrame({"x": x, "y": y})
    df_target = pd.DataFrame({"z": z})

    # Here we fix the number of neighbors for the call to sklearn.feature_selection's mutual_info_regression to 2 so
    # that we get exactly 1 for the mutual information.
    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, n_neighbors=2)

    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0)

    # Same data shuffled
    # Shuffle the x, y and z
    indices = np.arange(npoints)
    np.random.seed(42)
    np.random.shuffle(indices)
    xs = x.take(indices)
    ys = y.take(indices)
    zs = z.take(indices)

    df_feat = pd.DataFrame({"x": xs, "y": ys})
    df_target = pd.DataFrame({"z": zs})

    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, n_neighbors=2)

    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0)

    # Test with one constant feature
    c = np.ones(npoints) * 1.4
    df_feat = pd.DataFrame({"x": x, "y": y, "c": c})
    df_target = pd.DataFrame({"z": z})

    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, n_neighbors=2)
    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0)

    df_nmi_target = nmi_target(
        df_feat=df_feat,
        df_target=df_target,
        drop_constant_features=False,
        n_neighbors=2,
    )
    assert df_nmi_target.shape == (3, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0)
    assert df_nmi_target.loc["c"]["z"] == pytest.approx(0.0)

    # Test with unrelated data (grid)
    x = np.linspace(start=2, stop=5, num=4)
    z = np.linspace(start=3, stop=7, num=5)
    x, z = np.meshgrid(x, z)
    x = x.flatten()
    z = z.flatten()
    df_feat = pd.DataFrame({"x": x})
    df_target = pd.DataFrame({"z": z})

    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target)
    assert df_nmi_target.shape == (1, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(0.0)

    # Test initial checks
    # Incompatible shapes
    x = np.linspace(start=2, stop=3, num=5)
    z = np.linspace(start=2, stop=3, num=8)
    df_feat = pd.DataFrame({"x": x})
    df_target = pd.DataFrame({"z": z})
    with pytest.raises(
        ValueError,
        match=r"The input features DataFrame and the target variable DataFrame "
        r"should contain the same number of data points.",
    ):
        nmi_target(df_feat=df_feat, df_target=df_target)
    # Target DataFrame does not have exactly one column
    x = np.linspace(start=2, stop=3, num=5)
    z = np.linspace(start=2, stop=3, num=5)
    df_feat = pd.DataFrame({"x": x})
    df_target = pd.DataFrame({"z2": z, "z": z})
    with pytest.raises(
        ValueError, match=r"The target DataFrame should have exactly one column."
    ):
        nmi_target(df_feat=df_feat, df_target=df_target)

    # Test with some more real data (for which NMI is not just 0.0 or 1.0)
    npoints = 200
    np.random.seed(42)
    x = np.random.rand(npoints)
    z = 4 * x + 1.0 * np.random.rand(npoints)

    df_feat = pd.DataFrame({"x": x})
    df_target = pd.DataFrame({"z": z})

    # Here we fix the random_state for the call to sklearn.feature_selection's mutual_info_regression so
    # that we always get the same value.
    df_nmi_target = nmi_target(df_feat=df_feat, df_target=df_target, random_state=42)
    assert df_nmi_target.shape == (1, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(0.3417665092162398)


def test_nmi_target_classif():
    # Test with linear discrete data (should get 1.0 mutual information, or very close due to algorithm used
    # in mutual_info_regression)
    npoints = 1500
    x = np.array([0] * 500 + [1] * 500 + [20] * 500, dtype="float")
    y = 2 * x
    z = np.array(x, dtype="int")
    print(z)

    df_feat = pd.DataFrame({"x": x, "y": y})
    df_target = pd.DataFrame({"z": z})
    df_nmi_target = nmi_target(
        df_feat=df_feat, df_target=df_target, task_type="classification", n_neighbors=2
    )

    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0, 0.02)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0, 0.02)

    # Same data shuffled
    # Shuffle the x, y and z
    indices = np.arange(npoints)
    np.random.seed(42)
    np.random.shuffle(indices)
    xs = x.take(indices)
    ys = y.take(indices)
    zs = z.take(indices)

    df_feat = pd.DataFrame({"x": xs, "y": ys})
    df_target = pd.DataFrame({"z": zs})

    df_nmi_target = nmi_target(
        df_feat=df_feat, df_target=df_target, task_type="classification", n_neighbors=2
    )

    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0, 0.02)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0, 0.02)

    # Test with one constant feature
    c = np.ones(npoints) * 1.4
    df_feat = pd.DataFrame({"x": x, "y": y, "c": c})
    df_target = pd.DataFrame({"z": z})

    df_nmi_target = nmi_target(
        df_feat=df_feat, df_target=df_target, task_type="classification", n_neighbors=2
    )
    assert df_nmi_target.shape == (2, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0, 0.02)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0, 0.02)

    df_nmi_target = nmi_target(
        df_feat=df_feat,
        df_target=df_target,
        task_type="classification",
        drop_constant_features=False,
        n_neighbors=2,
    )
    assert df_nmi_target.shape == (3, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(1.0, 0.02)
    assert df_nmi_target.loc["y"]["z"] == pytest.approx(1.0, 0.02)

    # Test with unrelated data (grid)
    x = np.linspace(start=2, stop=5, num=4)
    z = np.linspace(start=3, stop=7, num=5)
    x, z = np.meshgrid(x, z)
    x = x.flatten()
    z = z.flatten()
    z = np.array(z / 10, dtype="int")
    df_feat = pd.DataFrame({"x": x})
    df_target = pd.DataFrame({"z": z})

    df_nmi_target = nmi_target(
        df_feat=df_feat, df_target=df_target, task_type="classification"
    )
    assert df_nmi_target.shape == (1, 1)
    assert df_nmi_target.loc["x"]["z"] == pytest.approx(0.0, 0.02)


def test_get_cross_nmi():

    # Test with linear data (should get 1.0 mutual information, or very close due to algorithm used
    # in mutual_info_regression)
    npoints = 31
    x = np.linspace(0.5, 3.5, npoints)
    y = 2 * x - 2
    z = 4 * x + 2

    df_feat = pd.DataFrame({"x": x, "y": y, "z": z})

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

    df_feat = pd.DataFrame({"x": xs, "y": ys, "z": zs})

    df_cross_nmi = get_cross_nmi(df_feat=df_feat, n_neighbors=2)

    assert df_cross_nmi.shape == (3, 3)
    for idx in df_cross_nmi.index:
        for col in df_cross_nmi.columns:
            assert df_cross_nmi.loc[idx][col] == pytest.approx(1.0)

    # Test with one constant feature
    c = np.ones(npoints) * 1.4
    df_feat = pd.DataFrame({"x": x, "y": y, "z": z, "c": c})

    df_cross_nmi = get_cross_nmi(df_feat=df_feat, n_neighbors=2)
    assert df_cross_nmi.shape == (3, 3)
    expected = np.ones((3, 3))
    # expected[3, :] = expected[:, 3] = 0
    # expected[3, 3] = 0
    np.testing.assert_allclose(
        np.array(df_cross_nmi, dtype=np.float64),
        expected,
    )

    # Test with unrelated data (grid)
    x = np.linspace(start=2, stop=5, num=4)
    y = np.linspace(start=3, stop=7, num=5)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    df_feat = pd.DataFrame({"x": x, "y": y})

    df_cross_nmi = get_cross_nmi(df_feat=df_feat, n_neighbors=2)
    assert df_cross_nmi.shape == (2, 2)
    assert df_cross_nmi.loc["x"]["y"] == pytest.approx(0.0)
    assert df_cross_nmi.loc["y"]["x"] == pytest.approx(0.0)

    # Test with some more real data (for which NMI is not just 0.0 or 1.0)
    npoints = 200
    np.random.seed(42)
    x = np.random.rand(npoints)
    y = 4 * x + 1.0 * np.random.rand(npoints)

    df_feat = pd.DataFrame({"x": x, "y": y})

    # Here we fix the random_state for the call to sklearn.feature_selection's mutual_info_regression so
    # that we always get the same value.
    df_cross_nmi = get_cross_nmi(df_feat=df_feat, random_state=42)
    assert df_cross_nmi.shape == (2, 2)
    assert df_cross_nmi.loc["x"]["x"] == pytest.approx(1.0)
    assert df_cross_nmi.loc["y"]["y"] == pytest.approx(1.0)
    assert df_cross_nmi.loc["x"]["y"] == pytest.approx(0.3417665092162398)
    assert df_cross_nmi.loc["y"]["x"] == pytest.approx(0.3417665092162398)


def test_load_moddata_zip(subset_moddata):
    """This test checks that older MODData objects can still be loaded."""

    data = subset_moddata

    assert len(data.structures) == 100
    assert len(data.mpids) == 100
    assert len(data.df_structure) == 100
    assert len(data.df_featurized) == 100
    assert len(data.df_targets) == 100
    assert len(data.df_targets) == 100


@pytest.mark.parametrize("featurizer_mode", ["single", "multi"])
def test_small_moddata_featurization(small_moddata_2023, featurizer_mode):
    """This test creates a new MODData from the MP 2018.6 structures."""

    from modnet.featurizers.presets import Matminer2023Featurizer

    old = small_moddata_2023
    structures = old.structures
    targets = old.targets

    names = old.names
    featurizer = Matminer2023Featurizer()
    featurizer.featurizer_mode = featurizer_mode
    new = MODData(structures, targets, target_names=names, featurizer=featurizer)
    new.featurize(fast=False, n_jobs=1)
    check_column_values(new, old, tolerance=0.03)


@pytest.mark.parametrize("featurizer_mode", ["multi", "single"])
def test_small_moddata_composition_featurization(
    small_moddata_composition_2023, featurizer_mode
):
    """This test creates a new MODData from the MP 2018.6 structures."""
    from modnet.featurizers.presets import CompositionOnlyMatminer2023Featurizer

    reference = small_moddata_composition_2023
    compositions = reference.compositions

    featurizer = CompositionOnlyMatminer2023Featurizer()
    featurizer.featurizer_mode = featurizer_mode

    new = MODData(materials=compositions, featurizer=featurizer)
    new.featurize(fast=False, n_jobs=1)

    # assert relative error below 3 percent
    check_column_values(new, reference, tolerance=0.03)


def test_small_moddata_feature_selection_classif(small_moddata_2023):
    """This test creates classifier MODData and test the feature selection method"""

    x1 = np.array([0] * 500 + [1] * 500 + [2] * 500, dtype="float")
    x2 = np.random.choice(2, 1500)
    x3 = x1 * x2
    x4 = x1 + (x2 * 0.5)
    targets = np.array(x1, dtype="int").reshape(-1, 1)
    features = np.array([x1, x2, x3, x4]).T
    names = ["my_classes"]

    c_nmi = pd.DataFrame(
        [[1, 0, 0.5, 0.5], [0, 1, 0.5, 0.5], [0.5, 0.5, 1, 0.5], [0.5, 0.5, 0.5, 1]],
        columns=["f1", "f2", "f3", "f4"],
        index=["f1", "f2", "f3", "f4"],
    )

    classif_md = MODData(
        ["dummy"] * 1500, targets, target_names=names, num_classes={"my_classes": 3}
    )
    classif_md.df_featurized = pd.DataFrame(features, columns=["f1", "f2", "f3", "f4"])
    classif_md.feature_selection(n=3, cross_nmi=c_nmi)
    assert len(classif_md.get_optimal_descriptors()) == 3
    assert classif_md.get_optimal_descriptors() == ["f1", "f4", "f3"]


def test_merge_ranked():
    from modnet.preprocessing import merge_ranked

    # Test lists of the same length
    test_features = [["a", "b", "c"], ["d", "b", "e"]]

    expected = ["a", "d", "b", "c", "e"]
    assert merge_ranked(test_features) == expected

    # Test lists of different length
    test_features = [["a", "b", "c"], ["d", "b", "e", "g"]]

    expected = ["a", "d", "b", "c", "e", "g"]
    assert merge_ranked(test_features) == expected

    test_features = [["d", "b", "e", "g"], ["a", "b", "c"]]

    expected = ["d", "a", "b", "e", "c", "g"]
    assert merge_ranked(test_features) == expected

    # Test lists with other hashable types
    test_features = [["a", "b", "c"], ["d", "b", 2, "g"], ["c", 0, "e", "0"]]

    expected = ["a", "d", "c", "b", 0, 2, "e", "g", "0"]
    assert merge_ranked(test_features) == expected


@pytest.mark.slow
def test_load_precomputed_dataset():
    """Tries to load and unpack the dataset on figshare.

    Warning: Requires ~10 GB of memory.

    NB: This test will not redownload the dataset if it is already present.

    """

    from pathlib import Path

    from modnet.ext_data import load_ext_dataset

    path = load_ext_dataset("MP_2018.6", "MODData")
    assert path == Path(__file__).parent.parent.joinpath("data") / "MP_2018.6.zip"
    assert path.is_file()

    # Replace the path with some garbage file and make sure a RuntimeError is raised
    # when the dataset is loaded, as the hash will no longer match
    path.unlink()
    path.touch()

    try:
        with pytest.raises(
            RuntimeError, match="Precomputed MODData did not match expected MD5 from"
        ):
            path = load_ext_dataset("MP_2018.6", "MODData")
    finally:
        path.unlink(missing_ok=True)


def test_moddata_splits(subset_moddata):
    from sklearn.model_selection import KFold

    kf = KFold(5, shuffle=True, random_state=123)

    for split in kf.split(subset_moddata.df_featurized):
        train, test = subset_moddata.split(split)

        assert len(train.structure_ids) == 80
        assert len(train.df_featurized) == 80
        assert len(train.df_targets) == 80
        assert len(train.df_structure) == 80
        assert len(train.get_featurized_df()) == 80
        assert len(train.get_structure_df()) == 80
        assert len(train.get_target_df()) == 80

        assert len(test.structure_ids) == 20
        assert len(test.df_featurized) == 20
        assert len(test.df_targets) == 20
        assert len(test.df_structure) == 20
        assert len(test.get_featurized_df()) == 20
        assert len(test.get_structure_df()) == 20
        assert len(test.get_target_df()) == 20

        test_id_set = set(test.structure_ids)
        for _id in train.structure_ids:
            assert _id not in test_id_set

        break


def test_precomputed_cross_nmi(small_moddata_2020):

    new = MODData(
        materials=small_moddata_2020.structures,
        targets=small_moddata_2020.targets,
        target_names=small_moddata_2020.names,
        df_featurized=small_moddata_2020.df_featurized,
    )
    new.feature_selection(5, use_precomputed_cross_nmi=True)


def test_moddata_balancing():
    d = MODData(
        materials=[1, 2, 3, 4, 5, 6],
        targets=np.array([[0, 0, 0, 0, 1, 1]]).T,
        target_names=["C"],
        num_classes={"C": 2},
    )

    d.df_featurized = pd.DataFrame({"f": [1, 2, 3, 4, 5, 6]})
    d.rebalance()
    assert len(d.df_featurized) == 8
    assert len(d.df_targets) == 8
    assert len(d.df_structure) == 8
    assert d.df_targets["C"].values.sum() == 4
