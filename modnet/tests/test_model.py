#!/usr/bin/env python
import pytest


def test_train_small_model_single_target(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = MODNetModel(
        [[["eform"]]],
        weights={"eform": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=5)
    model.predict(data)


def test_train_small_model_single_target_classif(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    def is_metal(egap):
        if egap == 0:
            return 1
        else:
            return 0

    data.df_targets["is_metal"] = data.df_targets["egap"].apply(is_metal)
    model = MODNetModel(
        [[["is_metal"]]],
        weights={"is_metal": 1},
        num_neurons=[[16], [8], [8], [4]],
        num_classes={"is_metal": 2},
        n_feat=10,
    )

    model.fit(data, epochs=5)


def test_train_small_model_multi_target(subset_moddata, tf_session):
    """Tests the multi-target training."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=5)
    model.predict(data)


def test_train_small_model_presets(subset_moddata, tf_session):
    """Tests the `fit_preset()` method."""
    from modnet.model_presets import gen_presets
    from modnet.models import MODNetModel

    modified_presets = gen_presets(100, 100)[:2]

    for ind, preset in enumerate(modified_presets):
        modified_presets[ind]["epochs"] = 5

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    # nested=0/False -> no inner loop, so only 1 model
    # nested=1/True -> inner loop, but default n_folds so 5
    for num_nested, nested_option in zip([5, 1, 5], [5, 0, 1]):
        results = model.fit_preset(data, presets=modified_presets, nested=nested_option, val_fraction=0.2)
        models = results[0]
        assert len(models) == len(modified_presets)
        assert len(models[0]) == num_nested


@pytest.mark.skip(msg="Until pickle bug is fixed")
def test_model_integration(subset_moddata, tf_session):
    """Tests training, saving, loading and predictions."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ][:10]

    model = MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=5)

    model.save("test")
    loaded_model = MODNetModel.load("test")

    assert model.predict(data) == loaded_model.predict(data)


###### Bayesian_MODNet Tests ######
def test_train_small_bayesian_single_target(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import Bayesian_MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = Bayesian_MODNetModel(
        [[["eform"]]],
        weights={"eform": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=5)
    model.predict(data)
    model.predict(data, return_unc=True)


def test_train_small_bayesian_single_target_classif(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import Bayesian_MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    def is_metal(egap):
        if egap == 0:
            return 1
        else:
            return 0

    data.df_targets["is_metal"] = data.df_targets["egap"].apply(is_metal)
    model = Bayesian_MODNetModel(
        [[["is_metal"]]],
        weights={"is_metal": 1},
        num_neurons=[[16], [8], [8], [4]],
        num_classes={"is_metal": 2},
        n_feat=10,
    )

    model.fit(data, epochs=5)
    model.predict(data)
    model.predict(data, return_unc=True)

def test_train_small_bayesian_multi_target(subset_moddata, tf_session):
    """Tests the multi-target training."""
    from modnet.models import Bayesian_MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = Bayesian_MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=5)
    model.predict(data)
    model.predict(data, return_unc=True)


###### Bootstrap_MODNet Tests ######

def test_train_small_bootstrap_single_target(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import Ensemble_MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = Ensemble_MODNetModel(
        [[["eform"]]],
        weights={"eform": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
        n_models=3,
        bootstrap=True,
    )

    model.fit(data, epochs=5)
    model.predict(data)
    model.predict(data, return_unc=True)


def test_train_small_bootstrap_single_target_classif(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import Ensemble_MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    def is_metal(egap):
        if egap == 0:
            return 1
        else:
            return 0

    data.df_targets["is_metal"] = data.df_targets["egap"].apply(is_metal)
    model = Ensemble_MODNetModel(
        [[["is_metal"]]],
        weights={"is_metal": 1},
        num_neurons=[[16], [8], [8], [4]],
        num_classes={"is_metal": 2},
        n_feat=10,
        n_models=3,
        bootstrap=True,
    )

    model.fit(data, epochs=5)
    model.predict(data, return_unc=True)
    model.predict(data, return_unc=True)

def test_train_small_bootstrap_multi_target(subset_moddata, tf_session):
    """Tests the multi-target training."""
    from modnet.models import Ensemble_MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = Ensemble_MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
        n_models=3,
        bootstrap=True,
    )

    model.fit(data, epochs=5)
    model.predict(data, return_unc=True)

def test_train_small_bootstrap_presets(subset_moddata, tf_session):
    """Tests the `fit_preset()` method."""
    from modnet.model_presets import gen_presets
    from modnet.models import Ensemble_MODNetModel

    modified_presets = gen_presets(100, 100)[:2]

    for ind, preset in enumerate(modified_presets):
        modified_presets[ind]["epochs"] = 5

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = Ensemble_MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
        n_models=2
    )

    # nested=0/False -> no inner loop, so only 1 model
    # nested=1/True -> inner loop, but default n_folds so 5
    for num_nested, nested_option in zip([5, 1, 5], [5, 0, 1]):
        results = model.fit_preset(data, presets=modified_presets, nested=nested_option, val_fraction=0.2)
        models = results[0]
        assert len(models) == len(modified_presets)
        assert len(models[0]) == num_nested
