#!/usr/bin/env python
import pytest


@pytest.mark.slow
def test_train_small_model_benchmark(small_moddata, tf_session):
    """Tests the `matbench_benchmark()` method with optional arguments."""
    from modnet.matbench.benchmark import matbench_benchmark

    data = small_moddata
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
        "model",
    )

    for key in expected_keys:
        assert key in results

    assert all(len(results[key]) == 5 for key in expected_keys)


@pytest.mark.slow
def test_train_small_ensemblemodel_benchmark(small_moddata, tf_session):
    """Tests the `matbench_benchmark()` method for ensemble models."""
    from modnet.matbench.benchmark import matbench_benchmark
    from modnet.models import EnsembleMODNetModel

    data = small_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    results = matbench_benchmark(
        data,
        [[["eform"]]],
        {"eform": 1},
        model_type=EnsembleMODNetModel,
        n_models=2,
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
        "stds",
        "targets",
        "errors",
        "scores",
        "best_presets",
        "model",
    )
    for key in expected_keys:
        assert key in results
    assert all(len(results[key]) == 5 for key in expected_keys)


@pytest.mark.slow
def test_train_small_model_benchmark_with_extra_args(small_moddata):
    """Tests the `matbench_benchmark()` method with some extra settings,
    parallelised over 2 jobs.

    """
    from modnet.matbench.benchmark import matbench_benchmark

    data = small_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    # Check that other settings don't break the model creation,
    # but that they do get used in fitting
    other_fit_settings = {
        "epochs": 10,
        "increase_bs": False,
        "callbacks": [],
    }

    results = matbench_benchmark(
        data,
        [[["eform"]]],
        {"eform": 1},
        fit_settings=other_fit_settings,
        inner_feat_selection=False,
        fast=True,
        nested=2,
        n_jobs=2,
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
        "model",
    )

    for key in expected_keys:
        assert key in results
    assert all(len(results[key]) == 5 for key in expected_keys)


@pytest.mark.slow
def test_ga_benchmark(small_moddata, tf_session):
    """Tests the `matbench_benchmark()` method with the GA strategy."""
    from modnet.matbench.benchmark import matbench_benchmark

    data = small_moddata
    # set regression problem
    data.num_classes = {"eform": 0}
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    results = matbench_benchmark(
        data,
        [[["eform"]]],
        {"eform": 1},
        inner_feat_selection=False,
        hp_optimization=True,
        hp_strategy="ga",
        ga_settings={
            "size_pop": 2,
            "num_generations": 2,
            "refit": False,
            "prob_mut": 0.4,
        },
        fast=True,
        n_jobs=1,
    )

    expected_keys = (
        "predictions",
        "targets",
        "errors",
        "scores",
        "model",
    )

    for key in expected_keys:
        assert key in results

    assert all(len(results[key]) == 5 for key in expected_keys)
