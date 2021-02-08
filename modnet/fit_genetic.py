from random import randint
import random
from typing import List, Optional
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from modnet.preprocessing import MODData
from modnet.models import MODNetModel


class FitGenetic:
    """Class optimizing the model parameters using a genitic algorithm.
    """

    def __init__(
        self,
        data: MODData,
        size_pop=15,
        num_epochs=5,
        prob_mut=0.8
        ):

        """Initializes parameters used in this class.

        Parameters:
            size_pop: Size of the population.
            num_epochs: Number of generations.
            prob_mut: Probability the mutation occurs.
            data: A 'MODData' that has been featurized and feature selected.
        """

        self.size_pop = size_pop
        self.num_epochs = num_epochs
        self.prob_mut = prob_mut
        self.data = data


    def train_val_split(
        self,
        data: MODData
        )->None:

        """Splits arrays or matrices into random train and validation subsets.
        
        Parameter:
            data: 'MODData' data which need to be splitted.
        """

        split_border = int(0.9*data.df_targets.shape[0])
        final_border = int(data.df_targets.shape[0])
        self.X_train, self.X_val = data.split((range(split_border),range(split_border,final_border)))
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
        activation = ['relu', 'elu']
        loss = ['mae', 'mse']
        xscale = ['minmax', 'normal']
        lr = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
        initial_batch_size = [8, 16, 32, 64, 128]
        fraction = [1, 0.75, 0.5, 0.25]
        self.pop = [[10*randint(1,int(len(self.X_train.get_optimal_descriptors())/10)), 32*randint(1,10), random.choice(fraction), random.choice(fraction), random.choice(fraction), random.choice(activation), random.choice(loss), random.choice(xscale), random.choice(lr), random.choice(initial_batch_size)] for i in range(0, size_pop)]
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

        child = [mother[0], father[1], mother[2], father[3], mother[4], father[5], mother[6], father[7], mother[8], father[9]]   
        return child


    def mutation(
        self,
        child: List,
        prob_mut: float = 0.8
        )->None:

        """Performs mutation in the genetic information in order to maintain diversity in the population. 
        
        Paramters:
            child: List containing the genetic information of the 'child'.
            prob_mut: Probability the mutation occurs.
        """

        for c in range(0, len(child)):
            if np.random.rand() > prob_mut:
                if child[c][0] < int(0.9*len(self.X_train.get_optimal_descriptors())):
                    child[c][0] = int(child[c][0] + 10*randint(1, 6))
                else:
                    child[c][0] = int(child[c][0] - 10*randint(1, 6))
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
        rlr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=20,
            verbose=0,
            mode="auto",
            min_delta=0,
        )
        es = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=300,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        callbacks = [rlr, es]
        for w in self.pop:
            modnet_model = MODNetModel([[['BV_Ea']]], {'BV_Ea':1}, n_feat=w[0], num_neurons=[[int(w[1])],[int(w[1]*w[2])],[int(w[1]*w[2]*w[3])],[int(w[1]*w[2]*w[3]*w[4])]], act=w[5])
            try:
                modnet_model.fit(X_train,val_fraction=0, val_key='BV_Ea',loss=w[6], lr=w[8], epochs = 250, batch_size = w[9], xscale=w[7], callbacks=callbacks, verbose=0)
                modnet_model.fit(X_train,val_fraction=0, val_key='BV_Ea',loss=w[6], lr=w[8], epochs = 250, batch_size = 2*w[9], xscale=w[7], callbacks=callbacks, verbose=0)
                modnet_model.fit(X_train,val_fraction=0, val_key='BV_Ea',loss=w[6], lr=w[8], epochs = 250, batch_size = 4*w[9], xscale=w[7], callbacks=callbacks, verbose=0)
                modnet_model.fit(X_train,val_fraction=0, val_key='BV_Ea',loss=w[6], lr=w[8], epochs = 250, batch_size = 8*w[9], xscale=w[7], callbacks=callbacks, verbose=0)
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
        num_epochs: int,
        prob_mut: float = 0.8
        )->None:

        """Selects the best individual (the model with the best parameters) for the next generation. The selection is based on a minimisation of the MSE on the validation set.

        Parameters:
            X_train: Input data of the training set.
            y_train: Target values of the training set.
            X_val: Input data of the validation set.
            y_val: Target values of the validation set.
            size_pop: Size of the population per generation.
            num_epochs: Number of generations.
            prob_mut: Probability the mutation occurs.
        """

        print('Generation number 0')
        pop = self.initialization_population(size_pop)
        fitness = self.function_fitness(pop,  X_train, y_train, X_val, y_val)
        pop_fitness_sort = np.array(list(sorted(fitness,key=lambda x: x[0])))
        for j in range(0, num_epochs):
            print('Generation number ', j+1)
            length = len(pop_fitness_sort)
            #select parents
            parent_1 = pop_fitness_sort[:,2][:length//2]
            parent_2 = pop_fitness_sort[:,2][length//2:]
            #crossover
            child_1 = [self.crossover(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
            child_2 = [self.crossover(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
            child_2 = self.mutation(child_2, prob_mut)

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
        self.best_individual = self.gen_alg(X_train, y_train, X_val, y_val, size_pop, num_epochs, prob_mut=0.8)

        return self.best_individual

