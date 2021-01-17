def gen_presets(n_feat,n_samples):

    batch_sizes = [32, 64]
    learning_rates = [0.01, 0.005]
    epochs = [500]
    losses = ["mae"]
    activations = ["elu"]

    n_feat_list = (min(0.2*n_feat, 50), max(0.5*n_feat, 500))

    archs = []
    for nf in n_feat_list:
        if n_samples < 2000:
            archs += [(nf,[[nf*2],[nf],[nf//4],[nf//4]]),
                       (nf, [[nf//2], [nf//2], [nf//4], [nf//4]])]
        else:
            archs += [(nf,[[nf*2],[nf,nf],[nf//4],[nf//4]]),
                       (nf, [[nf//2], [nf//2,nf//2], [nf//4], [nf//4]])]

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
    return hyperparam_presets