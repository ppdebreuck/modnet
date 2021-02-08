"""This submodule defines some heuristic presets to be used as the first step
of hyperparameter optimisation.

"""

from typing import List, Dict, Any
import itertools

from modnet.utils import LOG


def gen_presets(
    n_feat: int, n_samples: int, classification: bool = False
) -> List[Dict[str, Any]]:
    """Generates sensible preset architectures and learning parameters
    based on number of samples and features.

    Arguments:
        n_feat: The number of training features available to the model.
        n_samples: The number of training samples available to the model.

    Returns:
        List of dictionaries to individually pass as kwargs to `model.fit(...)`.

    """
    if n_samples < 1000:
        batch_sizes = [32, 64]
    else:
        batch_sizes = [64]
    learning_rates = [0.001, 0.005, 0.01]
    epochs = [1000]

    if classification:
        losses = ["categorical_crossentropy"]
    else:
        losses = ["mae"]

    activations = ["elu"]
    xscale = ["minmax", "standard"]

    n_feat_list = [64, 128, 256, 512]
    n_feat_list = [n for n in n_feat_list if n <= n_feat]
    n_feat_list = [n for n in n_feat_list if n > n_feat / 20]
    if len(n_feat_list) == 1:
        n_feat_list.append(n_feat)

    if len(n_feat_list) < 3:
        n_feat_list.append((n_feat_list[0] + n_feat_list[1]) // 2)
    n_feat_list = sorted(n_feat_list)

    archs = []
    for nf in n_feat_list:
        archs += [
            (nf, [[nf * 2], [nf // 2], [nf // 8], [nf // 8]]),
            (nf, [[nf], [nf // 2], [nf // 8], [nf // 8]]),
            (nf, [[nf // 2], [nf // 4], [nf // 8], [nf // 8]]),
        ]

    LOG.info(
        "Proceeding with grid search: archs: {}, batch sizes: {}, learning_rates: {}".format(
            archs, batch_sizes, learning_rates
        )
    )

    hyperparam_presets = []
    for a, bs, lr, e, l, act, scaler in itertools.product(
        archs, batch_sizes, learning_rates, epochs, losses, activations, xscale
    ):
        preset = {
            "batch_size": bs,
            "lr": lr,
            "n_feat": a[0],
            "num_neurons": a[1],
            "epochs": e,
            "loss": l,
            "act": act,
            "xscale": scaler,
        }
        hyperparam_presets.append(preset)

    return hyperparam_presets
