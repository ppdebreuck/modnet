import copy
import os
import random
from typing import List, Tuple, Optional
from random import randint
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from modnet.preprocessing import MODData
from modnet.models import MODNetModel
from modnet.utils import LOG
from modnet.individual import Individual
import multiprocessing
import tqdm


class FitGenetic:
    """Class optimizing the model parameters using a genitic algorithm.
    """

    def __init__(
            self,
            data: MODData,
    ):

        """Initializes the MODData used in this class.
        Parameters:
            data: A 'MODData' that has been featurized and feature selected.
        """

        self.data = data

    def shuffle_MD(
            self,
            data: MODData,
            random_state: int
    ):

        """Shuffles the MODData data.
        Parameters:
            data: A 'MODData' that has been featurized and feature selected.
            random_state: It affects the ordering of the indices, which controls the randomness of each fold.
        """

        data = copy.deepcopy(data)
        ids = data.df_targets.sample(frac=1, random_state=random_state).index
        data.df_featurized = data.df_featurized.loc[ids]
        data.df_targets = data.df_targets.loc[ids]
        data.df_structure = data.df_structure.loc[ids]

        return data

    def MDKsplit(
            self,
            data: MODData,
            random_state: int,
            n_splits: int = 5
    ):

        """Provides train/test indices to split data in train/test sets. Splits MODData dataset into k consecutive folds.
        Parameters:
            data: A 'MODData' that has been featurized and feature selected.
            n_splits: Number of folds.
            random_state: It affects the ordering of the indices, which controls the randomness of each fold.
        """

        data = self.shuffle_MD(data, random_state=random_state)
        ids = np.array(data.structure_ids)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []
        for train_idx, val_idx in kf.split(ids):
            data_train = MODData(
                data.df_structure.iloc[train_idx]['structure'].values,
                data.df_targets.iloc[train_idx].values,
                target_names=data.df_targets.columns,
                structure_ids=ids[train_idx]
            )
            data_train.df_featurized = data.df_featurized.iloc[train_idx]
            data_train.optimal_features = data.optimal_features

            data_val = MODData(
                data.df_structure.iloc[val_idx]['structure'].values,
                data.df_targets.iloc[val_idx].values,
                target_names=data.df_targets.columns,
                structure_ids=ids[val_idx]
            )
            data_val.df_featurized = data.df_featurized.iloc[val_idx]
            data_val.optimal_features = data.optimal_features

            folds.append((data_train, data_val))

        return folds

    def initialization_population(
            self,
            size_pop: int
    ) -> None:

        """Inintializes the initial population (Generation 0).
        Paramter:
            size_pop: Size of the population.
        """

        self.pop = [{}] * size_pop

        for i in range(0, size_pop):
            individual = Individual(self.data)  # details about the possible values of each gene are in this class
            self.pop[i] = {
                'n_feat': individual.n_features,
                'n_neurons_first_layer': individual.n_neurons_first_layer,
                'fraction1': individual.fraction1,
                'fraction2': individual.fraction2,
                'fraction3': individual.fraction3,
                'act': individual.activation,
                'loss': individual.loss,
                'xscale': individual.xscale,
                'lr': individual.lr,
                'initial_batch_size': individual.initial_batch_size
            }
        return self.pop

    def crossover(
            self,
            mother: List,
            father: List
    ) -> None:

        """Does the crossover of two parents and returns a 'child' which have the combined genetic information of both parents.
        Parameters:
            mother: List containing the gentic information of the first parent.
            father: List containing the gentic information of the second parent.
        """

        genes_from_mother = random.sample(range(10),
                                          k=5)  # creates indices to take randomly 5 genes from one parent, and 5 genes from the other
        child = {
            list(mother.keys())[i]: list(mother.values())[i] if i in genes_from_mother else list(father.values())[i] for
            i in range(10)}
        return child

    def mutation(
            self,
            children: List,
            prob_mut: int
    ) -> None:

        """Performs mutation in the genetic information in order to maintain diversity in the population.
        Paramters:
            children: List containing the genetic information of the 'children'.
        """

        for c in range(0, len(children)):
            if np.random.rand() > prob_mut:
                individual = Individual(self.data)
                # modification of the number of features in a [-10%, +10%] range
                children[c]['n_feat'] = np.absolute(int(
                    children[c]['n_feat'] + randint(-int(0.1 * len(self.data.get_optimal_descriptors())),
                                                    int(0.1 * len(self.data.get_optimal_descriptors())))))
                # modification of the number of neurons in the first layer of [-64, -32, 0, 32, 64]
                children[c]['n_neurons_first_layer'] = np.absolute(
                    children[c]['n_neurons_first_layer'] + 32 * randint(-2, 2))
                if children[c]['n_neurons_first_layer'] == 0:
                    children[c]['n_neurons_first_layer'] = 32
                # modification of the 1st, 2nd or 3rd fraction
                i = random.choices([1, 2, 3])
                if i == 1:
                    children[c]['fraction1'] = individual.fraction1
                elif i == 2:
                    children[c]['fraction2'] = individual.fraction2
                else:
                    children[c]['fraction3'] = individual.fraction3
                # multiplication of the initial batch size by a factor of [1/2, 1, 2]
                children[c]['initial_batch_size'] = int(children[c]['initial_batch_size'] * 2 ** randint(-1, 1))
            else:
                pass
        return children

    def mae_of_individual(
            self,
            individual: List,
            fold: Tuple[MODData, MODData],
            individual_id: int,
            fold_id: int
    ):

        """Returns the MAE of a modnet model given some parameters stored in ind and given the training and validation sets sorted in fold.
        Paramters:
            individual: An individual of the population, which is a list wherein the parameters are stored.
            fold: Tuple giving the training and validation MODData.
        """

        es = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=300,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        callbacks = [es]
        md_train = fold[0]
        y_train = md_train.df_targets
        md_val = fold[1]
        y_val = md_val.df_targets
        modnet_model = MODNetModel(
            [[[y_train.columns[0]]]],
            {y_train.columns[0]: 1},
            n_feat=individual['n_feat'],
            num_neurons=[
                [int(individual['n_neurons_first_layer'])],
                [int(individual['n_neurons_first_layer'] * individual['fraction1'])],
                [int(individual['n_neurons_first_layer'] * individual['fraction1'] * individual['fraction2'])],
                [int(individual['n_neurons_first_layer'] * individual['fraction1'] * individual['fraction2'] * individual['fraction3'])]
            ],
            act=individual['act']
        )
        for i in range(4):
            modnet_model.fit(
                md_train,
                val_fraction=0,
                val_key=y_train.columns[0],
                loss=individual['loss'],
                lr=individual['lr'],
                epochs=250,
                batch_size=(2 ** i) * individual['initial_batch_size'],
                xscale=individual['xscale'],
                callbacks=callbacks,
                verbose=0
            )
        MAE = mae(modnet_model.predict(md_val), y_val)
        return MAE, individual, individual_id, fold_id

    def model_of_individual(
            self,
            individual: List,
            md: MODData,
            individual_id: int
    ):

        """Returns the MODNet model given some parameters stored in ind and given the dataset to train the model on.
        Paramters:
            ind: An individual of the population, which is a list wherein the parameters are stored.
            md: MODData where the model is trained on.
        """

        es = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=300,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        callbacks = [es]

        y = md.df_targets

        modnet_model = MODNetModel(
            [[[y.columns[0]]]],
            {y.columns[0]: 1},
            n_feat=individual['n_feat'],
            num_neurons=[
                [int(individual['n_neurons_first_layer'])],
                [int(individual['n_neurons_first_layer'] * individual['fraction1'])],
                [int(individual['n_neurons_first_layer'] * individual['fraction1'] * individual['fraction2'])],
                [int(individual['n_neurons_first_layer'] * individual['fraction1'] * individual['fraction2'] * individual['fraction3'])]
            ],
            act=individual['act']
        )
        for i in range(4):
            modnet_model.fit(
                md,
                val_fraction=0,
                val_key=y.columns[0],
                loss=individual['loss'],
                lr=individual['lr'],
                epochs=250,
                batch_size=(2 ** i) * individual['initial_batch_size'],
                xscale=individual['xscale'],
                callbacks=callbacks,
                verbose=0
            )
        
        modnet_model = modnet_model._make_picklable()
        
        return modnet_model, individual_id

    def function_fitness(
            self,
            pop: List,
            md: MODData,
            n_jobs=None
    ) -> None:

        """Calculates the fitness of each model, which has the parameters contained in the pop argument. The function returns a list containing respectively the MAE calculated on the validation set, the model, and the parameters of that model.
        Parameters:
            pop: List containing the genetic information (i.e., the parameters) of the model.
            md_train: Input MODData.
            n_jobs: Number of jobs for multiprocessing
        """

        tasks = []
        tasks_model = []
        fitness = []
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        folds = self.MDKsplit(md, n_splits=5, random_state=1)
        maes = 1e20 * np.ones((len(pop), len(folds)))
        models = [] * len(pop)
        individuals = [] * len(pop)

        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=n_jobs)
        LOG.info(
            f"Multiprocessing on {n_jobs} cores. Total of {multiprocessing.cpu_count()} cores available."
        )

        for i, individual in enumerate(pop):
            for j, fold in enumerate(folds):
                tasks += [
                    {
                        "individual": individual,
                        "fold": fold,
                        "individual_id": i,
                        "fold_id": j
                    }
                ]

        for i, individual in enumerate(pop):
            tasks_model += [
                {
                    "individual": individual,
                    "md": md,
                    "individual_id": i
                }
            ]

        for res in tqdm.tqdm(
                pool.imap_unordered(self._mae_of_individual, tasks, chunksize=1),
                total=len(tasks)
        ):
            mae, individual, individual_id, fold_id = res
            LOG.info(f"MAE evaluation of individual #{individual_id} finished, MAE: {mae}")
            maes[individual_id, fold_id] = mae
            if individual_id is None:
                individuals[individual_id] = individual

        mae_per_individual = np.mean(maes, axis=1)
        print('MAE = ', mae_per_individual)

        for res in tqdm.tqdm(
                pool.imap_unordered(self._model_of_individual, tasks_model, chunksize=1),
                total=len(tasks_model)
        ):
            modnet_model, individual_id = res
            LOG.info(f"Model of individual #{individual_id} fitted.")
            if modnet_model is not None:
                modnet_model = modnet_model._restore_model()
                models[individual_id] = modnet_model

        pool.close()
        pool.join()

        print('mae_per_individual =', mae_per_individual)
        print('models =', models)
        print('individual_id =', individual_id)

        for individual_id in range(len(pop)):
            fitness.append([mae_per_individual[individual_id], models[individual_id], individuals[individual_id]])
        print('fitness =', fitness)

        return fitness

    def gen_alg(
            self,
            md: MODData,
            size_pop: int,
            num_generations: int,
            prob_mut: int
    ) -> None:

        """Selects the best individual (the model with the best parameters) for the next generation. The selection is based on a minimisation of the MAE on the validation set.
        Parameters:
            md: A 'MODData' that has been featurized and feature selected.
            size_pop: Size of the population per generation.
            num_generations: Number of generations.
        """

        print('##########################################################################################')
        LOG.info('Generation number 0')
        pop = self.initialization_population(size_pop)  # initialization of the population
        fitness = self.function_fitness(pop, md)  # fitness evaluation of the population
        pop_fitness_sort = np.array(
            list(sorted(fitness, key=lambda x: x[0])))  # ranking of the fitness of each individual
        best_individuals = np.zeros(num_generations)

        for j in range(0, num_generations):
            print('------------------------------------------------------------------------------------------')
            LOG.info("Generation number {}".format(j + 1))
            length = len(pop_fitness_sort)

            # select parents
            liste = [1 / l ** 10 for l in
                     pop_fitness_sort[:, 0]]  # **10 in order to give relatively more importance to the best individuals
            weights = [l / sum(liste) for l in liste]
            weights = np.array(list(sorted(weights, reverse=True)))  # sorting the weights
            # selection: weighted choice of the parents -> parents with a low MAE have more chance to be selected
            parents_1 = random.choices(pop_fitness_sort[:, 2], weights=weights, k=length)
            parents_2 = random.choices(pop_fitness_sort[:, 2], weights=weights, k=length)

            # crossover
            children = [self.crossover(parents_1[i], parents_2[i]) for i in
                        range(0, np.min([len(parents_2), len(parents_1)]))]
            children = self.mutation(children, prob_mut)
            # calculates children's fitness to choose who will pass to the next generation
            fitness_children = self.function_fitness(children, md)
            pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_children))
            sort = np.array(list(sorted(pop_fitness_sort, key=lambda x: x[0])))

            # selects individuals of the next generation
            pop_fitness_sort = sort[0:size_pop, :]
            self.best_individual = sort[0][1]

            # early stopping if we have the same best_individual for 3 generations
            best_individuals[j] = sort[0][0]
            if j > 1 and best_individuals[j - 2] == best_individuals[j]:
                break

        return self.best_individual

    def get_model(
            self,
            size_pop: Optional[int] = 15,
            num_generations: Optional[int] = 6
    ) -> MODNetModel:

        """Generates the model with the optimized parameters.
        Parameter:
            size_pop: Size of the population per generation. Default = 15.
            num_epochs: Number of generations. Default = 6.
        """

        self.best_individual = self.gen_alg(self.data, size_pop, num_generations, prob_mut=0.6)

        return self.best_individual

    def _mae_of_individual(self, kwargs):
        return self.mae_of_individual(**kwargs)

    def _model_of_individual(self, kwargs):
        return self.model_of_individual(**kwargs)
