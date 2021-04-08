import os
import copy
import random
from typing import List, Optional
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
        ids = data.df_targets.sample(frac=1,random_state=random_state).index
        data.df_featurized = data.df_featurized.loc[ids]
        data.df_targets = data.df_targets.loc[ids]
        data.df_structure = data.df_structure.loc[ids]
    
        return data


    def MDKsplit(
        self,
        data: MODData,
        random_state: int,
        n_splits: int=5
        ):

        """Provides train/test indices to split data in train/test sets. Splits MODData dataset into k consecutive folds.

        Parameters:
            data: A 'MODData' that has been featurized and feature selected.
            n_splits: Number of folds.
            random_state: It affects the ordering of the indices, which controls the randomness of each fold.
        """

        data = self.shuffle_MD(data,random_state=random_state)
        ids = np.array(data.structure_ids)
        kf = KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
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

            folds.append((data_train,data_val))
        
        return folds    


    def train_val_split(
        self,
        data: MODData
        )->None:

        """Splits arrays or matrices into random train and validation subsets.
        
        Parameter:
            data: 'MODData' data which need to be splitted.
        """
        i = randint(0,4)
        self.md_train, self.md_val = self.MDKsplit(data, n_splits=5, random_state=i)[i]
        self.y_train = self.md_train.df_targets
        self.y_val = self.md_val.df_targets

        return self.md_train, self.md_val, self.y_train, self.y_val


    def initialization_population(
        self,
        size_pop: int
        )->None:

        """Inintializes the initial population (Generation 0).

        Paramter:
            size_pop: Size of the population.
        """

        self.pop =  [[]]*size_pop

        for i in range(0, size_pop):
            self.individual = Individual(self.data)
            self.pop[i] = [
                          self.individual.n_features,
                          self.individual.n_neurons_first_layer,
                          self.individual.fraction1,
                          self.individual.fraction2,
                          self.individual.fraction3,
                          self.individual.activation,
                          self.individual.loss,
                          self.individual.xscale,
                          self.individual.lr,
                          self.individual.initial_batch_size
                          ]
        return self.pop


    def crossover(
        self,
        mother: List,
        father: List
        )->None:

        """Does the c
rossover of two parents and returns a 'child' which have the combined genetic information of both parents.

        Parameters:
            mother: List containing the gentic information of the first parent.
            father: List containing the gentic information of the second parent.
        """

        genes_from_mother = random.sample(range(10), k=5)
        child = [mother[i] if i in genes_from_mother else father[i] for i in range(10)]   
        return child


    def mutation(
        self,
        children: List,
        prob_mut: int = 0.6
        )->None:

        """Performs mutation in the genetic information in order to maintain diversity in the population. 
        
        Paramters:
            children: List containing the genetic information of the 'children'.
        """

        for c in range(0, len(children)):
            if np.random.rand() > prob_mut:
                individual = Individual(self.data)
                children[c][0] = np.absolute(int(children[c][0] + randint(-int(0.1*len(self.data.get_optimal_descriptors())), int(0.1*len(self.data.get_optimal_descriptors())))))
                self.individual.n_features = children[c][0]
                children[c][1] = np.absolute(children[c][1] + 32*randint(-2,2))
                if children[c][1] == 0:
                    children[c][1] = 32
                self.individual.neurons_first_layer = children[c][1]
                i = random.choices([1, 2, 3])
                if i == 1:
                    children[c][2] = individual.fraction1
                    self.individual.fraction1 = children[c][2]
                elif i == 2:
                    children[c][3] = individual.fraction2
                    self.individual.fraction2 = children[c][3]
                else:
                    children[c][4] = individual.fraction3
                    self.individual.fraction3 = children[c][4]
                children[c][9] = int(children[c][9]*2**randint(-1,1))
                self.individual.initial_batch_size = children[c][9]
            else:
                pass
        return children


    def function_fitness(
        self,
        pop: List,
        md_train: MODData,
        y_train: pd.DataFrame,
        md_val: MODData,
        y_val: pd.DataFrame
        )->None:

        """Calculates the fitness of each model, which has the parameters contained in the pop argument. The function returns a list containing respectively the MAE calculated on the validation set, the model, and the parameters of that model.
        
        Parameters:
            pop: List containing the genetic information (i.e., the parameters) of the model.
            md_train: Input data of the training set.
            y_train: Target values of the training set.
            md_val: Input data of the validation set.
            y_val: Target values of the validation set.
        """

        self.fitness = []
        j = 0
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
        modnet_model = MODNetModel(
                                  [[[md_train.df_targets.columns[0]]]],
                                  {md_train.df_targets.columns[0]:1},
                                  n_feat = self.individual.n_features,
                                  num_neurons = [
                                                [self.individual.neurons_first_layer],
                                                [self.individual.neurons_first_layer * self.individual.fraction1],
                                                [self.individual.neurons_first_layer * self.individual.fraction1 * self.individual.fraction2],
                                                [self.individual.neurons_first_layer * self.individual.fraction1 * self.individual.fraction2 * self.individual.fraction3]
                                                ],
                                  act = self.individual.activation
                                  )
        try:
            for i in range(4):
                modnet_model.fit(
                                md_train,
                                val_fraction = 0,
                                val_key = md_train.df_targets.columns[0],
                                loss = self.individual.loss,
                                lr = self.individual.lr,
                                epochs = 250,
                                batch_size = (2**i) * self.individual.initial_batch_size,
                                xscale = self.individual.xscale,
                                callbacks = callbacks,
                                verbose = 0
                                )
            f = mae(modnet_model.predict(md_val), y_val)
            print('MAE = ', f)
            self.fitness.append([f, modnet_model, w])
        except:
            pass
        return self.fitness


    def gen_alg(
        self,
        md_train: MODData,
        y_train: pd.DataFrame,
        md_val: MODData,
        y_val: pd.DataFrame,
        size_pop: int,
        num_generations: int,
        prob_mut: int = 0.6
        )->None:

        """Selects the best individual (the model with the best parameters) for the next generation. The selection is based on a minimisation of the MAE on the validation set.

        Parameters:
            md_train: Input data of the training set.
            y_train: Target values of the training set.
            md_val: Input data of the validation set.
            y_val: Target values of the validation set.
            size_pop: Size of the population per generation.
            num_generations: Number of generations.
        """

        LOG.info('Generation number 0')
        pop = self.initialization_population(size_pop)
        fitness = self.function_fitness(pop, md_train, y_train, md_val, y_val)
        pop_fitness_sort = np.array(list(sorted(fitness, key=lambda x: x[0])))
        best_individuals = np.zeros(num_generations)

        for j in range(0, num_generations):
            LOG.info("Generation number {}".format(j+1))
            length = len(pop_fitness_sort)
            
            #select parents
            liste = [1/l**3 for l in pop_fitness_sort[:,0]] #**3 in order to give relatively more importance to the best individuals
            weights = [l/sum(liste) for l in liste]
            weights = np.array(list(sorted(weights))) #sorting the weights
            parents_1 = random.choices(pop_fitness_sort[:,2], weights=weights, k=length//2)
            parents_2 = random.choices(pop_fitness_sort[:,2], weights=weights, k=length//2)
            
            #crossover
            children = [self.crossover(parents_1[i], parents_2[i]) for i in range(0, np.min([len(parents_2), len(parents_1)]))]
            children = self.mutation(children, prob_mut)
            
            #calculates children's fitness to choose who will pass to the next generation
            fitness_children = self.function_fitness(children, md_train, y_train, md_val, y_val)
            pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_children))
            sort = np.array(list(sorted(pop_fitness_sort,key=lambda x: x[0])))        

            #selects individuals of the next generation
            pop_fitness_sort = sort[0:size_pop, :]
            self.best_individual = sort[0][1]
            
            #early stopping if we have the same best_individual for 3 generations
            best_individuals[j] = sort[0][0]
            if j > 2 and best_individuals[j-2] == best_individuals[j]:
                break

        return self.best_individual


    def get_model(
        self,
        size_pop: Optional[int] = 15,
        num_generations: Optional[int] = 6
        )->MODNetModel:

        """Generates the model with the optimized parameters.

        Parameter:
            size_pop: Size of the population per generation. Default = 15.
            num_epochs: Number of generations. Default = 6.
        """

        md_train, md_val, y_train, y_val = self.train_val_split(self.data)
        self.best_individual = self.gen_alg(md_train, y_train, md_val, y_val, size_pop, num_generations, prob_mut=0.6)

        return self.best_individual

