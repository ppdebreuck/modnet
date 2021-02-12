#!/usr/bin/env python


def test_train_small_model_benchmark(subset_moddata, tf_session):
    """Tests the `fit_preset()` method."""
    from modnet.matbench.benchmark import matbench_benchmark

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    results = matbench_benchmark(
        data,
        [[["eform"]]],
        {"eform": 1},
        inner_feat_selection=False,
        fast=True,
        nested=2,
        n_jobs=1,
    )

    expected_keys = (
        "nested_losses",
        "nested_learning_curves",
        "best_learning_curves",
        "predictions",
        "targets",
        "errors",
        "scores",
        "best_presets",
    )

    assert all(key in results for key in expected_keys)
    assert all(len(results[key]) == 5 for key in expected_keys)
