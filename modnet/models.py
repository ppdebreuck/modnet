import pandas as pd
import numpy as np
import keras
from keras.layers import *
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import keras.backend as K
from keras.models import model_from_json
from modnet.preprocessing import MODData

class MODNetModel:
    
    def __init__(self,targets,weights,num_neurons=[[64],[32],[16],[16]], n_feat=300, loss='mse',act='relu'):
        
        self.n_feat = n_feat
        self.weights = weights

        num_layers = [len(x) for x in num_neurons]

        f_temp = [x for subl in targets for x in subl]
        self.targets_flatten = [x for subl in f_temp for x in subl]
        
        if len(self.targets_flatten) > 1:
            self.PP = True
        else:
            self.PP = False
    

        #Build first common block
        f_input = Input(shape=(n_feat,))
        previous_layer = f_input
        for i in range(num_layers[0]):
            previous_layer = Dense(num_neurons[0][i],activation=act)(previous_layer)
            if self.PP:
                previous_layer = BatchNormalization()(previous_layer)
        common_out = previous_layer

        # build intermediate representations
        intermediate_models_out = []
        for i in range(len(targets)):
            previous_layer = common_out
            for j in range(num_layers[1]):
                previous_layer = Dense(num_neurons[1][j],activation=act)(previous_layer)
                if self.PP:
                    previous_layer = BatchNormalization()(previous_layer)
            intermediate_models_out.append(previous_layer)

        #Build outputs
        final_out = []
        for group_idx,group in enumerate(targets):
            for prop_idx in range(len(group)):
                previous_layer = intermediate_models_out[group_idx]
                for k in range(num_layers[2]):
                    previous_layer = Dense(num_neurons[2][k],activation=act)(previous_layer)
                    if self.PP:
                        previous_layer = BatchNormalization()(previous_layer)
                clayer = previous_layer
                temps = []
                for pi in range(len(group[prop_idx])):
                    previous_layer = clayer
                    for li in range(num_layers[3]):
                        previous_layer = Dense(num_neurons[3][li])(previous_layer)
                    out = Dense(1,activation='linear',name=group[prop_idx][pi])(previous_layer)
                    final_out.append(out)

        self.model = keras.models.Model(inputs=f_input, outputs=final_out)

        
    def fit(self,data:MODData, val_fraction = 0.0, val_key = None, lr=0.001, epochs = 200, batch_size = 128, xscale='minmax',yscale=None):
        
        print('new')
        self.xscale = xscale
        self.target_names = data.names
        self.optimal_descriptors = data.get_optimal_descriptors()
        x = data.get_featurized_df()[self.optimal_descriptors[:self.n_feat]].values
        print(x.shape)
        y = data.get_target_df()[self.targets_flatten].values.transpose()
        print(y.shape)
        
        #Scale the input features:
        if self.xscale == 'minmax':
            self.xmin = x.min(axis=0)
            self.xmax = x.max(axis=0)
            x=(x-self.xmin)/(self.xmax-self.xmin) - 0.5
                
        elif self.xscale == 'standard':
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x)

        x = np.nan_to_num(x)
        
        if val_fraction > 0:
            if self.PP:
                print_callback = LambdaCallback(
                  on_epoch_end=lambda epoch,logs: print("epoch {}: loss: {:.3f}, val_loss:{:.3f} val_{}:{:.3f}".format(epoch,logs['loss'],logs['val_loss'],val_key,logs['val_{}_mae'.format(val_key)])))
            else:
                print_callback = LambdaCallback(
                  on_epoch_end=lambda epoch,logs: print("epoch {}: loss: {:.3f}, val_loss:{:.3f} val_{}:{:.3f}".format(epoch,logs['loss'],logs['val_loss'],val_key,logs['val_mae'])))
        else:
            print_callback = LambdaCallback(
              on_epoch_end=lambda epoch,logs: print("epoch {}: loss: {:.3f}".format(epoch,logs['loss']))) 

        
        fit_params = {
                      'x': x,
                      'y': list(y),
                      'epochs': epochs,
                      'batch_size': batch_size,
                      'verbose': 0,
                      'validation_split' : val_fraction,
                      'callbacks':[print_callback]
                  }
        print('compile',flush=True)
        self.model.compile(loss = 'mse',optimizer=keras.optimizers.Adam(lr=lr),metrics=['mae'],loss_weights=self.weights)
        print('fit',flush=True)     
        
        self.model.fit(**fit_params)
        
        
    def predict(self,data):
        
        df = pd.DataFrame(columns=self.optimal_descriptors[:self.n_feat])
        df = df.append(data.get_featurized_df()).replace([np.inf, -np.inf, np.nan], 0)
        x = df[self.optimal_descriptors[:self.n_feat]].values
        
        #Scale the input features:
        if self.xscale == 'minmax':
            x=(x-self.xmin)/(self.xmax-self.xmin) - 0.5
                
        elif self.xscale == 'standard':
            x = self.scaler.transform(x)
        
        x = np.nan_to_num(x)
        
        if self.PP:
            p = np.array(self.model.predict(x))[:,:,0].transpose()
        else:
            p = np.array(self.model.predict(x))[:,0].transpose()
        predictions = pd.DataFrame(p)
        predictions.columns = self.targets_flatten
        predictions.index = data.ids
        
        return predictions
    
    
    def save(self,filename):
        model = self.model
        self.model = None
        model_json = model.to_json()
        fp = open('{}.json'.format(filename), 'w')
        fp.write(model_json)
        fp.close()
        model.save_weights('{}.h5'.format(filename))
        fp = open('{}.pkl'.format(filename),'wb')
        pickle.dump(self,fp)
        fp.close()
        self.model = model
        print('Saved model')
    
    @staticmethod
    def load(filename):
        fp = open('{}.pkl'.format(filename),'rb')
        mod = pickle.load(fp)
        fp.close()
        fp = open('{}.json'.format(filename), 'r')
        model_json = fp.read()
        fp.close()
        mod.model = model_from_json(model_json)
        mod.model.load_weights('{}.h5'.format(filename))
        return mod