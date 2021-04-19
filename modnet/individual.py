import random
from random import randint
from modnet.preprocessing import MODData
import numpy as np

class Individual:
    
    """Class containing each of the tuned hyperparameters for the genetic algorithm.
    """
    
    def __init__(self, data):
        self.xscale_list = ['minmax', 'standard']
        self.lr_list = [0.01, 0.005, 0.001]
        self.initial_batch_size_list = [8, 16, 32, 64, 128]
        self.fraction_list = [1, 0.75, 0.5, 0.25]

        self.activation = 'elu'
        self.loss = 'mae'
        self.n_neurons_first_layer = 32*randint(1,10)
        self.fraction1 = random.choice(self.fraction_list)
        self.fraction2 = random.choice(self.fraction_list)
        self.fraction3 = random.choice(self.fraction_list)
        self.xscale = random.choice(self.xscale_list)
        self.lr = random.choice(self.lr_list)
        self.initial_batch_size = random.choice(self.initial_batch_size_list)

        self.n_features = 0 #initialization
        if len(data.get_optimal_descriptors()) <= 100:
            b = int(len(data.get_optimal_descriptors())/2)
            self.n_features = randint(1, b) + b
        elif len(data.get_optimal_descriptors()) > 100 and len(data.get_optimal_descriptors()) < 2000:
            max = len(data.get_optimal_descriptors())
            self.n_features = 10*randint(1,int(max/10))
        else:
            max = np.sqrt(len(data.get_optimal_descriptors()))
            self.n_features = randint(1,max)**2
