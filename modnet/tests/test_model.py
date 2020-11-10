#!/usr/bin/env python
import pytest


def test_train_small_model_single_target(subset_moddata):
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
        num_neurons=([16], [8], [8], [4]),
        n_feat=10,
    )

    model.fit(data, epochs=5)
    model.predict(data)


def test_train_small_model_multi_target(subset_moddata):
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
        num_neurons=([16], [8], [8], [4]),
        n_feat=10,
    )

    model.fit(data, epochs=5)
    model.predict(data)


def test_train_small_model_presets(subset_moddata):
    """Tests the `fit_preset()` method."""
    from copy import deepcopy
    from modnet.model_presets import MODNET_PRESETS
    from modnet.models import MODNetModel

    modified_presets = deepcopy(MODNET_PRESETS)

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
        num_neurons=([16], [8], [8], [4]),
        n_feat=10,
    )

    model.fit_preset(data, presets=modified_presets, val_fraction=0.2)


@pytest.mark.skip(msg="Until pickle bug is fixed")
def test_model_integration(subset_moddata):
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
        num_neurons=([16], [8], [8], [4]),
        n_feat=10,
    )

    model.fit(data, epochs=5)

    model.save("test")
    loaded_model = MODNetModel.load("test")

    assert model.predict(data) == loaded_model.predict(data)
