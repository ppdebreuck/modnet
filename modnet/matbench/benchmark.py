from modnet.preprocessing import MODData
from modnet.models import MODNetModel
import numpy as np
import os
from traceback import print_exc
from typing import List

MATBENCH_SEED = 18012019


def matbench_kfold_splits(data: MODData):
    """Return the pre-defined k-fold splits to use when reporting matbench results.

    Arguments:
        data: The featurized MODData.

    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=MATBENCH_SEED)
    kf_splits = kf.split(data.df_featurized, y=data.df_targets)
    return kf_splits

def matbench_nested_cv(data: MODData, targets: List[str], use_precomputed_cross_nmi=True):

    for ind, (train, test) in enumerate(matbench_kfold_splits(data)):
        train_data, test_data = data.split((train, test))

        weights = {target: 1 for target in targets}
        train_data.feature_selection(
            n=-1, use_precomputed_cross_nmi=use_precomputed_cross_nmi
        )
        n_feat = len(train_data.optimal_features)

        model = MODNetModel(
            [[targets]],
            weights,
            n_feat=n_feat,
        )

        models, val_losses = model.fit_preset(train_data)


def matbench_benchmark(
    data: MODData,
    target,
    target_weights,
    fit_settings,
    classification=False,
    multi_target=False,
    save_folds=False,
    hp_optimization=True,
    inner_feat_selection=True,
    use_precomputed_cross_nmi=True,
    presets=None,
):
    """Run nested CV hyperparamter optimisation and benchmarking based on
    the hyperparameter presets in `modnet.model_presets`.

    """
    all_models = []
    all_predictions = []
    all_errors = []
    all_scores = []
    all_targets = []

    if "n_feat" not in fit_settings or "num_neurons" not in fit_settings:
        raise RuntimeError("Need to supply n_feat or num_neurons")

    for ind, (train, test) in enumerate(matbench_kfold_splits(data)):
        train_data, test_data = data.split((train, test))
        if inner_feat_selection:
            # the training data is featurized and saved
            path = "folds/train_moddata_f{}".format(ind + 1)
            if os.path.isfile(path):
                train_data = MODData.load(path)
            else:
                train_data.feature_selection(
                    n=-1, use_precomputed_cross_nmi=use_precomputed_cross_nmi
                )
            train_data.save(path)

        model = MODNetModel(
            target,
            target_weights,
            n_feat=fit_settings["n_feat"],
            num_neurons=fit_settings["num_neurons"],
            act=fit_settings["act"],
            num_classes=fit_settings.get("num_classes", None)
        )

        if hp_optimization:
            models, val_losses = model.fit_preset(train_data, presets=presets)
        else:
            if fit_settings["increase_bs"]:
                model.fit(
                    train_data,
                    lr=fit_settings["lr"],
                    epochs=fit_settings["epochs"],
                    batch_size=fit_settings["batch_size"],
                    loss="mse",
                )
                model.fit(
                    train_data,
                    lr=fit_settings["lr"] / 7,
                    epochs=fit_settings["epochs"] // 2,
                    batch_size=fit_settings["batch_size"] * 2,
                    loss=fit_settings["loss"],
                )
            else:
                # model.fit(train_data, callbacks=learning_callbacks(), **fit_settings)
                model.fit(train_data, **fit_settings)

        try:
            if classification:
                predictions = model.predict(test_data, return_prob=True)
            else:
                predictions = model.predict(test_data)
            targets = test_data.df_targets

            if classification:
                from sklearn.metrics import roc_auc_score
                from sklearn.preprocessing import OneHotEncoder

                y_true = OneHotEncoder().fit_transform(targets.values).toarray()
                score = roc_auc_score(y_true, predictions.values)
                pred_bool = model.predict(test_data, return_prob=False)
                errors = targets - pred_bool
                print(f"Model #{ind+1}: ROC_AUC = {score}")
            elif multi_target:
                errors = targets - predictions
                score = np.mean(np.abs(errors.values), axis=0)
                print(f"Model #{ind+1}: MAE = {score}")
            else:
                errors = targets - predictions
                score = np.mean(np.abs(errors.values))
                print(f"Model #{ind+1}: MAE = {score}")
        except Exception:
            print_exc()
            print("Something went wrong benchmarking this model.")
            predictions = None
            errors = None
            score = None

        if save_folds:
            opt_feat = train_data.optimal_features[: fit_settings["n_feat"]]
            df_train = train_data.df_featurized
            df_train = df_train[opt_feat]
            df_train.to_csv("folds/train_f{}.csv".format(ind + 1))
            df_test = test_data.df_featurized
            df_test = df_test[opt_feat]
            errors.columns = [x + "_error" for x in errors.columns]
            df_test = df_test.join(errors)
            df_test.to_csv("folds/test_f{}.csv".format(ind + 1))

        all_models.append(model)
        all_predictions.append(predictions)
        all_errors.append(errors)
        all_scores.append(score)
        all_targets.append(targets)

    results = {
        "models": all_models,
        "predictions": all_predictions,
        "targets": all_targets,
        "errors": all_errors,
        "scores": all_scores,
    }

    return results
