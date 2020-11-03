import logging
logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from modnet.preprocessing import MODData
from modnet.model_presets import MODNET_PRESETS

class MODNetModel:
    
    def __init__(self,targets,weights,num_neurons=[[64],[32],[16],[16]], n_feat=300,act='relu'):

        self.targets = targets
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

        self.model = tf.keras.models.Model(inputs=f_input, outputs=final_out)

    def fit(self,data:MODData, val_fraction = 0.0, val_key = None, lr=0.001, epochs = 200, batch_size = 128, xscale='minmax', loss='mse', callbacks=None, verbose=1):
        
        if self.n_feat > len(data.get_optimal_descriptors()):
            raise RuntimeError("The model requires more features than computed in data. Please reduce n_feat below or equal to {}".format(len(data.get_optimal_descriptors())))
        self.xscale = xscale
        self.target_names = data.names
        self.optimal_descriptors = data.get_optimal_descriptors()
        x = data.get_featurized_df()[self.optimal_descriptors[:self.n_feat]].values
        #print(x.shape)
        y = data.get_target_df()[self.targets_flatten].values.transpose()
        #print(y.shape)
        
        #Scale the input features:
        if self.xscale == 'minmax':
            self.xmin = x.min(axis=0)
            self.xmax = x.max(axis=0)
            x=(x-self.xmin)/(self.xmax-self.xmin) - 0.5
                
        elif self.xscale == 'standard':
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x)

        x = np.nan_to_num(x)
        
        if verbose and self.PP:
            if val_fraction>0:
                print_callback = LambdaCallback(
                  on_epoch_end=lambda epoch,logs: print("epoch {}: loss: {:.3f}, val_loss:{:.3f} val_{}:{:.3f}".format(epoch,logs['loss'],logs['val_loss'],val_key,logs['val_{}_mae'.format(val_key)])))
                verbose = 0
            else:
                print_callback = LambdaCallback(
                    on_epoch_end=lambda epoch,logs: print("epoch {}: loss: {:.3f}".format(epoch,logs['loss'])))
                verbose = 0
            if callbacks is None:
                callbacks = [print_callback]
            else:
                callbacks += [print_callback]

        fit_params = {
            'x': x,
            'y': list(y),
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': verbose,
            'validation_split': val_fraction,
            'callbacks': callbacks
        }

        #print('compile',flush=True)
        self.model.compile(loss = loss, optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['mae'], loss_weights=self.weights)
        #print('fit',flush=True)
        
        history = self.model.fit(**fit_params)

        return history


    def fit_preset(self, data:MODData,verbose=0):
        """
        Chooses an optimal hyper-parametered MODNet model from different presets .
        The data is first fitted on several well working MODNet presets with a validation set (20% of the furnished data).
        The best validating preset is then fitted again on the whole data, and the current model is updated accordingly.
        Args:
            data: MODData object contain training and validation samples.

        Returns: None, object is updated to fit the data.

        """
        rlr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=verbose, mode="auto", min_delta=0)
        es = EarlyStopping(monitor="loss", min_delta=0.001, patience=300, verbose=verbose, mode="auto", baseline=None,
                           restore_best_weights=True)
        callbacks = [rlr,es]
        val_losses = np.empty((len(MODNET_PRESETS),))
        for i,params in enumerate(MODNET_PRESETS):
            logging.info("Training preset #{}/{}".format(i+1,len(MODNET_PRESETS)))
            n_feat = min(len(data.get_optimal_descriptors()),params['n_feat'])
            self.model = MODNetModel(self.targets,self.weights,num_neurons=params['num_neurons'],n_feat=n_feat, act=params['act']).model
            self.n_feat = n_feat
            hist = self.fit(data, val_fraction=0.2, lr=params['lr'], epochs=params['epochs'], batch_size=params['batch_size'], loss=params['loss'], callbacks=callbacks, verbose=verbose)
            val_loss = np.array(hist.history['val_loss'])[-20:].mean()
            val_losses[i] = val_loss
            logging.info("Validation loss: {:.3f}".format(val_loss))
        best_preset = val_losses.argmin()
        logging.info("Preset #{} resulted in lowest validation loss.\nFitting all data...".format(best_preset+1))
        n_feat = min(len(data.get_optimal_descriptors()), MODNET_PRESETS[best_preset]['n_feat'])
        self.model = MODNetModel(self.targets, self.weights, num_neurons=MODNET_PRESETS[best_preset]['num_neurons'], n_feat=n_feat,
                                 act=MODNET_PRESETS[best_preset]['act']).model
        self.n_feat = n_feat
        self.fit(data, val_fraction=0.2, lr=MODNET_PRESETS[best_preset]['lr'], epochs=MODNET_PRESETS[best_preset]['epochs'],
                        batch_size=MODNET_PRESETS[best_preset]['batch_size'], loss=MODNET_PRESETS[best_preset]['loss'], callbacks=callbacks, verbose=verbose)


    def predict(self,data):
        
        x = data.get_featurized_df()[self.optimal_descriptors[:self.n_feat]].values
        
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
        predictions.index = data.structure_ids
        
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