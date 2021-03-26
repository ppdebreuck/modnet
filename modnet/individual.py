import random


class Individual:
    """Class containing each of the tuned hyperparameters for the genetic algorithm.
    """

    xscale_list = ['minmax', 'standard']
    lr_list = [0.01, 0.005, 0.001]
    initial_batch_size_list = [8, 16, 32, 64, 128]
    fraction_list = [1, 0.75, 0.5, 0.25]
   
    activation = 'elu'
    loss = 'mae'
    n_neurons_first_layer = 32*randint(1,10)
    fraction1 = random.choice(fraction_list)
    fraction2 = random.choice(fraction_list)
    fraction3 = random.choice(fraction_list)
    xscale = random.choice(xscale_list)
    lr = random.choice(lr_list)
    initial_batch_size = random.choice(initial_batch_size_list)
    

    def n_feat(
        self,
        data: MODData
        ):
        
        """Optimal number of features chosen on a sliding scale based on available features.
        """

        n_features = 0 #initialization
        if len(data.get_optimal_descriptors()) <= 100:
            b = int(len(data.get_optimal_descriptors())/2)
            n_features = randint(1, b) + b
        elif len(data.get_optimal_descriptors()) > 100 and len(data.get_optimal_descriptors()) < 2000:
            max = len(data.get_optimal_descriptors())
            n_features = 10*randint(1,int(max/10))
        else:
            max = np.sqrt(len(data.get_optimal_descriptors()))
            n_features = randint(1,max)**2
    
        return self.n_features
