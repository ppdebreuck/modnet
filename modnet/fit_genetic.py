from random import randint
import os
import copy
import random
from typing import List, Optional
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from modnet.preprocessing import MODData
from modnet.models import MODNetModel
from modnet.utils import LOG


class FitGenetic:
    """Class optimizing the model parameters using a genitic algorithm.
    """

    def __init__(
        self,
        data: MODData,
        size_pop=15,
        num_epochs=5
        ):

        """Initializes parameters used in this class.

        Parameters:
            size_pop: Size of the population.
            num_epochs: Number of generations.
            data: A 'MODData' that has been featurized and feature selected.
        """

        self.size_pop = size_pop
        self.num_epochs = num_epochs
        self.data = data


    def shuffle_MD(
        self,
        data: MODData,
        random_state: int=10
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
        n_splits: int=10,
        random_state: int=10
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
            data_train = MODData(data.df_structure.iloc[train_idx]['structure'].values,data.df_targets.iloc[train_idx].values,target_names=data.df_targets.columns,structure_ids=ids[train_idx])
            data_train.df_featurized = data.df_featurized.iloc[train_idx]
            #data_train.optimal_features = data.optimal_features
        
            data_val = MODData(data.df_structure.iloc[val_idx]['structure'].values,data.df_targets.iloc[val_idx].values,target_names=data.df_targets.columns,structure_ids=ids[val_idx])
            data_val.df_featurized = data.df_featurized.iloc[val_idx]
            #data_val.optimal_features = data.optimal_features

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

        self.X_train, self.X_val = data.MDKsplit(data, n_splits=10, random_state=10)
        self.y_train = self.X_train.df_targets
        self.y_val = self.X_val.df_targets

        return self.X_train, self.X_val, self.y_train, self.y_val


    def initialization_population(
        self,
        size_pop: int
        )->None:

        """Inintializes the initial population (Generation 0).
       
        Paramter:
            size_pop: Size of the population.
        """

        self.pop =  [[]]*size_pop
        activation = ['elu']
        loss = ['mae']
        xscale = ['minmax', 'standard']
        selflr = [0.01, 0.005, 0.001]
        self.initial_batch_size = [8, 16, 32, 64, 128]
        self.fraction = [1, 0.75, 0.5, 0.25]

        n_features = 0 #initialization
        if len(self.X_train.get_optimal_descriptors()) <= 100:
            b = int(len(self.X_train.get_optimal_descriptors())/2)
            n_features = randint(1, b) + b
        elif len(self.X_train.get_optimal_descriptors()) > 100 and len(self.X_train.get_optimal_descriptors()) < 2000:
            max = len(self.X_train.get_optimal_descriptors())
            n_features = 10*randint(1,10*int(max/10))
        else:
            max = np.sqrt(len(self.X_train.get_optimal_descriptors()))
            n_features = randint(1,max)**2
        self.pop = [[n_features , 32*randint(1,10), random.choice(self.fraction), random.choice(self.fraction), random.choice(self.fraction), random.choice(activation), random.choice(loss), random.choice(xscale), random.choice(self.lr), random.choice(self.initial_batch_size)] for i in range(0, size_pop)]
        return self.pop


    def crossover(
        self,
        mother: List,
        father: List
        )->None:

        """Does the crossover of two parents and returns a 'child' which have the combined genetic information of both parents.

        Parameters:
            mother: List containing the gentic information of the first parent.
            father: List containing the gentic information of the second parent.
        """

        genes_from_mother = random.sample(range(10), k=5)
        child = [mother[i] if i in genes_from_mother else father[i] for i in range(10)]   
        return child


    def mutation(
        self,
        child: List,
        )->None:

        """Performs mutation in the genetic information in order to maintain diversity in the population. 
        
        Paramters:
            child: List containing the genetic information of the 'child'.
        """

        for c in range(0, len(child)):
                if child[c][0] < int(0.5*len(self.X_train.get_optimal_descriptors())):
                    child[c][0] = int(child[c][0] + randint(1, int(0.1*len(self.X_train.get_optimal_descriptors()))))
                    child[c][1] = child[c][1] + 32*randint(-2,2)
                    child[c][2] = random.choice(self.fraction)
                    child[c][3] = random.choice(self.fraction)
                    child[c][4] = random.choice(self.fraction)
                    child[c][8] = random.choice(self.lr)
                else:
                    child[c][0] = int(child[c][0] - randint(1, int(0.1*len(self.X_train.get_optimal_descriptors()))))
                    child[c][1] = child[c][1] + 32*randint(-2,2)
                    child[c][2] = random.choice(self.fraction)
                    child[c][3] = random.choice(self.fraction)
                    child[c][4] = random.choice(self.fraction)
                    child[c][8] = random.choice(self.lr)
         return child


    def function_fitness(
        self,
        pop: List,
        X_train: MODData,
        y_train: pd.DataFrame,
        X_val: MODData,
        y_val: pd.DataFrame
        )->None:

        """Calculates the fitness of each model, which has the parameters contained in the pop argument. The function returns a list containing respectively the MSE calculated on the validation set, the model, and the parameters of that model.
        
        Parameters:
            pop: List containing the genetic information (i.e., the parameters) of the model.
            X_train: Input data of the training set.
            y_train: Target values of the training set.
            X_val: Input data of the validation set.
            y_val: Target values of the validation set.
        """

        self.fitness = []
        j = 0
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
        for w in self.pop:
            modnet_model = MODNetModel([[['BV_Ea']]], {'BV_Ea':1}, n_feat=w[0], num_neurons=[[int(w[1])],[int(w[1]*w[2])],[int(w[1]*w[2]*w[3])],[int(w[1]*w[2]*w[3]*w[4])]], act=w[5])
            try:
                for i in range(4):
                    modnet_model.fit(X_train,val_fraction=0, val_key='BV_Ea',loss=w[6], lr=w[8], epochs = 250, batch_size = (2**i)*w[9], xscale=w[7], callbacks=callbacks, verbose=0)
                f = mse(modnet_model.predict(X_val),y_val)
                print('MSE = ', f)
                self.fitness.append([f, modnet_model, w])
            except:
                 pass
        return self.fitness


    def gen_alg(
        self,
        X_train: MODData,
        y_train: pd.DataFrame,
        X_val: MODData,
        y_val: pd.DataFrame,
        size_pop: int,
        num_epochs: int
        )->None:

        """Selects the best individual (the model with the best parameters) for the next generation. The selection is based on a minimisation of the MSE on the validation set.

        Parameters:
            X_train: Input data of the training set.
            y_train: Target values of the training set.
            X_val: Input data of the validation set.
            y_val: Target values of the validation set.
            size_pop: Size of the population per generation.
            num_epochs: Number of generations.
        """

        LOG.info('Generation number 0')
        pop = self.initialization_population(size_pop)
        fitness = self.function_fitness(pop,  X_train, y_train, X_val, y_val)
        pop_fitness_sort = np.array(list(sorted(fitness,key=lambda x: x[0])))
        scaled_pop_fitness = pop_fitness_sort[:,0]/sum(pop_fitness_sort[:,0])
        for j in range(0, num_epochs):
            print('Generation number ', j+1)
            length = len(pop_fitness_sort)
            #select parents
            parent_1 = random.choices(pop_fitness_sort[:,2], weights=scaled_pop_fitness, k=length//2)
            parent_2 = random.choices(pop_fitness_sort[:,2], weights=scaled_pop_fitness, k=length//2)
            #crossover
            child_1 = [self.crossover(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
            child_2 = self.mutation(child_1)

            #calculates children's fitness to choose who will pass to the next generation
            fitness_child_1 = self.function_fitness(child_1,X_train, y_train, X_val, y_val)
            fitness_child_2 = self.function_fitness(child_2, X_train, y_train, X_val, y_val)
            pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_child_1, fitness_child_2))
            sort = np.array(list(sorted(pop_fitness_sort,key=lambda x: x[0])))

            #selects individuals of the next generation
            pop_fitness_sort = sort[0:size_pop, :]
            self.best_individual = sort[0][1]

        return self.best_individual


    def get_model(
        self,
        data: MODData,
        size_pop: Optional[int] = 15,
        num_epochs: Optional[int] = 5
        )->MODNetModel:

        """Generates the model with the optimized parameters.

        Parameter:
            data: A 'MODData' that has been featurized and feature selected.
            size_pop: Size of the population per generation. Default = 15.
            num_epochs: Number of generations. Default = 5.
        """

        X_train, X_val, y_train, y_val = self.train_val_split(data)
        self.best_individual = self.gen_alg(X_train, y_train, X_val, y_val, size_pop, num_epochs)

        return self.best_individual

