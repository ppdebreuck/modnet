batch_sizes = [64]

learning_rates = [0.01, 0.001]

archs = [
    (350, [[64], [8], [8], []]),
    (350, [[256], [128], [8], [8]]),
    (1000, [[256], [128], [8], []]),
    (350, [[128], [64], [32], [32]]),
]

epochs = [1000]

losses = ["mse"]

activations = ["elu"]

hyperparam_presets = []
for bs in batch_sizes:
    for lr in learning_rates:
        for a in archs:
            for e in epochs:
                for l in losses:
                    for act in activations:
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
