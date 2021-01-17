"""This submodule defines some heuristic presets to be used as the first step
of hyperparameter optimisation.

"""

from typing import List, Dict, Any
import itertools


def gen_presets(n_feat: int, n_samples: int) -> List[Dict[str, Any]]:
    """ "Generates sensible preset architectures and learning parameters
    based on number of samples and features:

      * Small limit, e.g. 100 features and 1000 samples yields architectures
        - [200, 100, 25, 25]
        - [50, 50, 25, 25].

      * Crossover, e.g. 100 features, ~2000 samples yields:

        If just above 2000 samples:

        - [200, [100, 100], 25, 25]
        - [50, [50, 50], 25, 25]

        or if just below 2000 samples:

        - [200, 100, 25, 25]
        - [50, 50, 25, 25].

      * Large limit, e.g. 500 features, 50000 samples yields architectures
        - [1000, [500, 500], 125, 125]
        - [250, [250, 250], 125, 125]

    Arguments:
        n_feat: The number of training features available to the model.
        n_samples: The number of training samples available to the model.

    Returns:
        List of dictionaries to pass as keywords to `model.fit(...)`.

    """

    batch_sizes = [32, 64]
    learning_rates = [0.01, 0.005]
    epochs = [500]
    losses = ["mae"]
    activations = ["elu"]

    n_feat_list = (min(0.2 * n_feat, 50), max(0.5 * n_feat, 500))

    archs = []
    for nf in n_feat_list:
        if n_samples < 2000:
            archs += [
                (nf, [[nf * 2], [nf], [nf // 4], [nf // 4]]),
                (nf, [[nf // 2], [nf // 2], [nf // 4], [nf // 4]]),
            ]
        else:
            archs += [
                (nf, [[nf * 2], [nf, nf], [nf // 4], [nf // 4]]),
                (nf, [[nf // 2], [nf // 2, nf // 2], [nf // 4], [nf // 4]]),
            ]

    hyperparam_presets = []
    for bs, lr, a, e, l, act in itertools.product(
        batch_sizes, learning_rates, archs, epochs, losses, activations
    ):
        preset = {
            "batch_size": bs,
            "lr": lr,
            "n_feat": a[0],
            "num_neurons": a[1],
            "epochs": e,
            "loss": l,
            "act": act,
        }
        hyperparam_presets.append(preset)

    return hyperparam_presets
