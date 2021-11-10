from __future__ import annotations
import random
from typing import List, Optional
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modnet.preprocessing import MODData
from modnet.models import MODNetModel, EnsembleMODNetModel
from modnet.utils import LOG
import multiprocessing
import tqdm


class Individual:

    """Class representing a set of hyperparameters for the genetic algorithm."""

    def __init__(self, max_feat: int, num_classes: dict) -> Individual:
        """
        Args:
            max_feat (int): Maximum number of features
            num_classes (dict): MODData num_classes parameter.Used for distinguishing between regression and classification.
        """

        self.max_feat = max_feat
        self.num_classes = num_classes

        self.xscale_list = ["minmax", "standard"]
        self.lr_list = [0.01, 0.005, 0.001]
        self.initial_batch_size_list = [8, 16, 32, 64, 128]
        self.fraction_list = [1, 0.75, 0.5, 0.25]

        self.genes = {
            "act": "elu",
            "loss": "mae",
            "n_neurons_first_layer": 32 * random.randint(1, 10),
            "fraction1": random.choice(self.fraction_list),
            "fraction2": random.choice(self.fraction_list),
            "fraction3": random.choice(self.fraction_list),
            "xscale": random.choice(self.xscale_list),
            "lr": random.choice(self.lr_list),
            "initial_batch_size": random.choice(self.initial_batch_size_list),
            "n_feat": 0,
        }

        if max_feat <= 100:
            b = int(max_feat / 2)
            self.genes["n_feat"] = random.randint(1, b) + b
        elif max_feat > 100 and max_feat < 2000:
            max = max_feat
            self.genes["n_feat"] = 10 * random.randint(1, int(max / 10))
        else:
            max = np.sqrt(max_feat)
            self.genes["n_feat"] = random.randint(1, max) ** 2

    def crossover(self, partner: Individual) -> Individual:
        """Does the crossover of two parents and returns a 'child' which has a mix of the parents hyperparams.

        Args:
            partner (Individual): Partner individual.
        Returns:
            Individual: Child.
        """

        genes_from_mother = random.sample(
            range(10), k=5
        )  # creates indices to take randomly 5 genes from one parent, and 5 genes from the other

        child_genes = {
            list(self.genes.keys())[i]: list(self.genes.values())[i]
            if i in genes_from_mother
            else list(partner.genes.values())[i]
            for i in range(10)
        }

        child = Individual(max_feat=self.max_feat, num_classes=self.num_classes)
        child.genes = child_genes
        return child

    def mutation(self, prob_mut: float) -> None:
        """Performs mutation in the hyper parameters in order to maintain diversity in the population.

        Args:
            prob_mut (float): Probability [0,1] of mutation.

        Returns: None (inplace operator).
        """

        if np.random.rand() < prob_mut:
            individual = Individual(
                max_feat=self.max_feat, num_classes=self.num_classes
            )
            # modification of the number of features in a [-10%, +10%] range
            self.genes["n_feat"] = np.absolute(
                int(
                    self.genes["n_feat"]
                    + random.randint(
                        -int(0.1 * self.max_feat),
                        int(0.1 * self.max_feat),
                    )
                )
            )
            if self.genes["n_feat"] <= 0:
                self.genes["n_feat"] = 1
            elif self.genes["n_feat"] > self.max_feat:
                self.genes["n_feat"] = self.max_feat
            # modification of the number of neurons in the first layer of [-64, -32, 0, 32, 64]
            self.genes["n_neurons_first_layer"] = np.absolute(
                self.genes["n_neurons_first_layer"] + 32 * random.randint(-2, 2)
            )
            if self.genes["n_neurons_first_layer"] == 0:
                self.genes["n_neurons_first_layer"] = 32
            # modification of the 1st, 2nd or 3rd fraction
            i = random.choices([1, 2, 3])
            if i == 1:
                self.genes["fraction1"] = individual.genes["fraction1"]
            elif i == 2:
                self.genes["fraction2"] = individual.genes["fraction2"]
            else:
                self.genes["fraction3"] = individual.genes["fraction3"]
            # multiplication of the initial batch size by a factor of [1/2, 1, 2]
            self.genes["initial_batch_size"] = int(
                self.genes["initial_batch_size"] * 2 ** random.randint(-1, 1)
            )
        else:
            pass
        return None

    def evaluate(self, train_data: MODData, val_data: MODData, fast: bool = False):
        """Internally evaluates the validation loss by setting self.val_loss

        Args:
            train_data (MODData): Training MODData
            val_data (MODData): Validation MODData
            fast (bool, optional): Limited epoch for testing or debugging only. Defaults to False.
        """

        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=100,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
        callbacks = [es]
        model = MODNetModel(
            targets=[[train_data.target_names]],
            weights={n: 1 for n in train_data.target_names},
            n_feat=self.genes["n_feat"],
            num_neurons=[
                [int(self.genes["n_neurons_first_layer"])],
                [int(self.genes["n_neurons_first_layer"] * self.genes["fraction1"])],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                    )
                ],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                        * self.genes["fraction3"]
                    )
                ],
            ],
            act=self.genes["act"],
            num_classes=self.num_classes,
        )

        model.fit(
            train_data,
            val_data=val_data,
            loss=self.genes["loss"],
            lr=self.genes["lr"],
            epochs=1000 if not fast else 1,
            batch_size=self.genes["initial_batch_size"],
            xscale=self.genes["xscale"],
            callbacks=callbacks,
            verbose=0,
        )

        self.val_loss = model.evaluate(val_data)
        self.model = model

    def refit_model(self, data: MODData, fast: bool = False):
        """Refit inner model on specified data.
        Args:
            data (MODData): Training data
            fast (bool, optional): Limited epoch for testing or debugging only. Defaults to False.
        """

        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=100,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
        callbacks = [es]
        model = MODNetModel(
            targets=[[data.target_names]],
            weights={n: 1 for n in data.target_names},
            n_feat=self.genes["n_feat"],
            num_neurons=[
                [int(self.genes["n_neurons_first_layer"])],
                [int(self.genes["n_neurons_first_layer"] * self.genes["fraction1"])],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                    )
                ],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                        * self.genes["fraction3"]
                    )
                ],
            ],
            act=self.genes["act"],
            num_classes=self.num_classes,
        )

        model.fit(
            data,
            val_fraction=0,
            loss=self.genes["loss"],
            lr=self.genes["lr"],
            epochs=1000 if not fast else 1,
            batch_size=self.genes["initial_batch_size"],
            xscale=self.genes["xscale"],
            callbacks=callbacks,
            verbose=0,
        )

        self.model = model
        return self.model


class FitGenetic:
    """Class optimizing the model parameters using a genetic algorithm."""

    def __init__(self, data: MODData, sample_threshold: int = 5000):
        """Genetic algorithm hyperparameter optimization for MODNet.

        Args:
            data (MODData): Training MODData
            sample_threshold (int, optional): If the dataset size exceeds this threshold, individuals are
                trained on sampled subsets of this size. Defaults to 5000.
        """
        self.data = data
        subset_ids = np.random.permutation(len(data.df_featurized))[:sample_threshold]
        self.train_data, _ = data.split((subset_ids, []))

        LOG.info("Targets:")
        for i, (k, v) in enumerate(data.num_classes.items()):
            if v >= 2:
                type = "classification"
            else:
                type = "regression"
            LOG.info(f"{i+1}){k}: {type}")

    def initialization_population(self, size_pop: int) -> None:
        """Initializes the initial population (Generation 0).

        Args:
            size_pop (int): Size of population.
        """

        self.pop = [
            Individual(
                max_feat=len(self.train_data.get_optimal_descriptors()),
                num_classes=self.train_data.num_classes,
            )
            for _ in range(size_pop)
        ]

    def function_fitness(
        self,
        pop: List[Individual],
        n_jobs: int,
        nested=5,
        val_fraction=0.1,
        fast=False,
    ) -> None:
        """Calculates the fitness of each model, which has the parameters contained in the pop argument.
        The function returns a list containing respectively the MAE calculated on the validation set, the model, and the parameters of that model.

        Args:
            pop (List[Individual]): List of individuals
            n_jobs (int): number of jobs to parallelize on.
            nested (int, optional): CV fold size. Defaults to 5. Use <=0 for hold-out validation.
            val_fraction (float, optional): Validation fraction if no CV is used. Defaults to 0.1.
            fast (bool, optional): Limited epochs for testing and debugging only. Defaults to False.

        Returns:
            val_losses, models, individuals
        """

        from modnet.matbench.benchmark import matbench_kfold_splits

        num_nested_folds = 5
        if nested:
            num_nested_folds = nested
        if num_nested_folds <= 1:
            num_nested_folds = 5

        # create tasks
        splits = matbench_kfold_splits(self.train_data, n_splits=num_nested_folds)
        if not nested:
            splits = [
                train_test_split(
                    range(len(self.train_data.df_featurized)), test_size=val_fraction
                )
            ]
            n_splits = 1
        else:
            n_splits = num_nested_folds
        train_val_datas = []
        for train, val in splits:
            train_val_datas.append(self.train_data.split((train, val)))

        tasks = []
        for i, individual in enumerate(pop):
            for j in range(n_splits):
                train_data, val_data = train_val_datas[j]
                tasks += [
                    {
                        "individual": individual,
                        "train_data": train_data,
                        "val_data": val_data,
                        "individual_id": i,
                        "fold_id": j,
                        "fast": fast,
                    }
                ]

        val_losses = 1e20 * np.ones((len(pop), n_splits))
        models = [[None for _ in range(n_splits)] for _ in range(len(pop))]
        individuals = [None for _ in range(len(pop))]

        if n_jobs is None:
            n_jobs = 4
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=n_jobs)
        LOG.info(
            "Multiprocessing on {} cores. Total of {} cores available.".format(
                n_jobs, multiprocessing.cpu_count()
            )
        )

        for res in tqdm.tqdm(
            pool.imap_unordered(_map_evaluate_individual, tasks, chunksize=1),
            total=len(tasks),
        ):
            individual, individual_id, fold_id = res
            individual.model._restore_model()
            val_losses[individual_id, fold_id] = individual.val_loss
            individuals[individual_id] = individual
            models[individual_id][fold_id] = individual.model

        models = [
            EnsembleMODNetModel(modnet_models=inner_models) for inner_models in models
        ]
        val_loss_per_individual = np.mean(val_losses, axis=1)
        res_str = "Loss per individual: "
        for ind, vl in enumerate(val_loss_per_individual):
            res_str += "ind {}: {:.3f} \t".format(ind, vl)
        LOG.info(res_str)

        pool.close()
        pool.join()

        return val_loss_per_individual, np.array(models), np.array(individuals)

    def run(
        self,
        size_pop: int = 20,
        num_generations: int = 10,
        prob_mut: Optional[int] = None,
        nested: Optional[int] = 5,
        n_jobs: Optional[int] = None,
        early_stopping: Optional[int] = 4,
        refit: Optional[int] = 5,
        fast=False,
    ) -> EnsembleMODNetModel:
        """Run the GA and return best model.

        Args:
            size_pop (int, optional): Size of the population per generation.. Defaults to 20.
            num_generations (int, optional): Size of the population per generation. Defaults to 10.
            prob_mut (Optional[int], optional): Probability of mutation. Defaults to None.
            nested (Optional[int], optional): CV fold size. Use <=0 for hold-out validation. Defaults to 5.
            n_jobs (Optional[int], optional): Number of jobs to parallelize on. Defaults to None.
            early_stopping (Optional[int], optional): Number of successive generations without improvement before stopping. Defaults to 4.
            refit (Optional[int], optional): Wether to refit (>0) the best hyperparameters on the whole dataset or use the best Individual instead (=0).
                The amount corresponds the the number of models used in the ensemble. Defaults to 0.
            fast (bool, optional): Use only for debugging and testing. A fast GA run with small number of epochs, generations, individuals and folds.
                Overrides the size_pop, num_generation and nested arguments.. Defaults to False.

        Returns:
            EnsembleMODNetModel: Fitted model with best hyperparameters
        """

        if fast:
            size_pop, num_generations, nested = 2, 2, 2

        LOG.info("Generation number 0")
        self.initialization_population(size_pop)  # initialization of the population
        val_loss, models, individuals = self.function_fitness(
            pop=self.pop,
            nested=nested,
            n_jobs=n_jobs,
            fast=fast,
        )
        ranking = val_loss.argsort()
        best_model_per_gen = [None for _ in range(num_generations)]
        self.best_model = models[ranking[0]]
        best_model_per_gen[0] = self.best_model

        for j in range(1, num_generations):
            LOG.info("Generation number {}".format(j))

            # select parents
            weights = [
                1 / lw ** 5 for lw in val_loss[ranking]
            ]  # **5 in order to give relatively more importance to the best individuals
            weights = [w / sum(weights) for w in weights]
            # selection: weighted choice of the parents -> parents with a low MAE have more chance to be selected
            parents_1 = random.choices(
                individuals[ranking], weights=weights, k=size_pop
            )
            parents_2 = random.choices(
                individuals[ranking], weights=weights, k=size_pop
            )

            # crossover
            children = [parents_1[i].crossover(parents_2[i]) for i in range(size_pop)]
            if prob_mut is None:
                prob_mut = 1 / size_pop
            for c in children:
                c.mutation(prob_mut)

            # calculates children's fitness to choose who will pass to the next generation
            (
                val_loss_children,
                models_children,
                individuals_children,
            ) = self.function_fitness(
                pop=children, nested=nested, n_jobs=n_jobs, fast=fast
            )
            val_loss = np.concatenate([val_loss, val_loss_children])
            models = np.concatenate([models, models_children])
            individuals = np.concatenate([individuals, individuals_children])

            ranking = val_loss.argsort()

            self.best_model = models[ranking[0]]
            self.best_individual = individuals[ranking[0]]
            best_model_per_gen[j] = self.best_model

            # early stopping if we have the same best_individual for early_stopping generations
            if (
                j >= early_stopping - 1
                and best_model_per_gen[j - (early_stopping - 1)]
                == best_model_per_gen[j]
            ):
                LOG.info(
                    "Early stopping: same best model for {} consecutive generations".format(
                        early_stopping
                    )
                )
                LOG.info("Early stopping at generation number {}".format(j))
                break

        if refit:
            LOG.info("Refit...")
            ensemble = []
            for i in range(refit):
                ensemble.append(self.best_individual.refit_model(self.data, fast=fast))
            self.best_model = EnsembleMODNetModel(modnet_models=ensemble)

        return self.best_model


def _map_evaluate_individual(kwargs):
    return _evaluate_individual(**kwargs)


def _evaluate_individual(
    individual: Individual,
    train_data: MODData,
    val_data: MODData,
    individual_id: int,
    fold_id: int,
    fast: bool = False,
):
    """Evaluate individual

    Args:
        individual (Individual): Individual to be evaluated
        train_data (MODData): Training MODData
        val_data (MODData): Validation MODData
        individual_id (int): Individual ID
        fold_id (int): Fold ID
        fast (bool, optional): Limited number of epochs for debugging and testing purposes only. Defaults to False.

    Returns:
        individual, individual_id, fold_id
        individual.val_loss contains the corresponding validation score.
    """
    individual.evaluate(train_data, val_data, fast=fast)
    individual.model._make_picklable()
    return individual, individual_id, fold_id
