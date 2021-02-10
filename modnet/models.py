import pickle
from typing import List, Tuple, Dict, Optional, Callable, Any, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow.keras as keras

from modnet.preprocessing import MODData
from modnet.matbench.benchmark import matbench_kfold_splits
from modnet.utils import LOG
from modnet import __version__


__all__ = ("MODNetModel",)


class MODNetModel:
    """Container class for the underlying Keras `Model`, that handles
    setting up the architecture, activations, training and learning curve.

    Attributes:
        n_feat: The number of features used in the model.
        weights: The relative loss weights for each target.
        optimal_descriptors: The list of column names used
            in training the model.
        model: The `keras.model.Model` of the network itself.
        target_names: The list of targets names that the model
            was trained for.

    """

    def __init__(
        self,
        targets: List,
        weights: Dict[str, float],
        num_neurons=([64], [32], [16], [16]),
        num_classes: Optional[Dict[str, int]] = None,
        n_feat: Optional[int] = 64,
        act="relu",
    ):
        """Initialise the model on the passed targets with the desired
        architecture, feature count and loss functions and activation functions.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            weights: The relative loss weights to apply for each target.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                 with n=0 for regression and n>=2 for classification with n the number of classes.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            n_feat: The number of features to use as model inputs.
            act: A string defining a Keras activation function to pass to use
                in the `keras.layers.Dense` layers.

        """

        self.__modnet_version__ = __version__

        if n_feat is None:
            n_feat = 64
        self.n_feat = n_feat
        self.weights = weights
        self.num_classes = num_classes

        self._scaler = None
        self.optimal_descriptors = None
        self.target_names = None
        self.targets = targets
        self.model = None

        f_temp = [x for subl in targets for x in subl]
        self.targets_flatten = [x for subl in f_temp for x in subl]
        self.num_classes = {name: 0 for name in self.targets_flatten}
        if num_classes is not None:
            self.num_classes.update(num_classes)
        self._multi_target = len(self.targets_flatten) > 1

        self.model = self.build_model(
            targets, n_feat, num_neurons, act=act, num_classes=self.num_classes
        )

    def build_model(
        self,
        targets: List,
        n_feat: int,
        num_neurons: Tuple[List[int], List[int], List[int], List[int]],
        num_classes: Optional[Dict[str, int]] = None,
        act: str = "relu",
    ):
        """Builds the Keras model and sets the `self.model` attribute.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            n_feat: The number of features to use as model inputs.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                with n=0 for regression and n>=2 for classification with n the number of classes.
            act: A string defining a Keras activation function to pass to use
                in the `keras.layers.Dense` layers.

        """

        num_layers = [len(x) for x in num_neurons]

        # Build first common block
        f_input = keras.layers.Input(shape=(n_feat,))
        previous_layer = f_input
        for i in range(num_layers[0]):
            previous_layer = keras.layers.Dense(num_neurons[0][i], activation=act)(
                previous_layer
            )
            if self._multi_target:
                previous_layer = keras.layers.BatchNormalization()(previous_layer)
        common_out = previous_layer

        # Build intermediate representations
        intermediate_models_out = []
        for _ in range(len(targets)):
            previous_layer = common_out
            for j in range(num_layers[1]):
                previous_layer = keras.layers.Dense(num_neurons[1][j], activation=act)(
                    previous_layer
                )
                if self._multi_target:
                    previous_layer = keras.layers.BatchNormalization()(previous_layer)
            intermediate_models_out.append(previous_layer)

        # Build outputs
        final_out = []
        for group_idx, group in enumerate(targets):
            for prop_idx in range(len(group)):
                previous_layer = intermediate_models_out[group_idx]
                for k in range(num_layers[2]):
                    previous_layer = keras.layers.Dense(
                        num_neurons[2][k], activation=act
                    )(previous_layer)
                    if self._multi_target:
                        previous_layer = keras.layers.BatchNormalization()(
                            previous_layer
                        )
                clayer = previous_layer
                for pi in range(len(group[prop_idx])):
                    previous_layer = clayer
                    for li in range(num_layers[3]):
                        previous_layer = keras.layers.Dense(num_neurons[3][li])(
                            previous_layer
                        )
                    n = num_classes[group[prop_idx][pi]]
                    if n >= 2:
                        out = keras.layers.Dense(
                            n, activation="softmax", name=group[prop_idx][pi]
                        )(previous_layer)
                    else:
                        out = keras.layers.Dense(
                            1, activation="linear", name=group[prop_idx][pi]
                        )(previous_layer)
                    final_out.append(out)

        return keras.models.Model(inputs=f_input, outputs=final_out)

    def fit(
        self,
        training_data: MODData,
        val_fraction: float = 0.0,
        val_key: Optional[str] = None,
        val_data: Optional[MODData] = None,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = "mse",
        **fit_params,
    ) -> None:
        """Train the model on the passed training `MODData` object.

        Paramters:
            training_data: A `MODData` that has been featurized and
                feature selected. The first `self.n_feat` entries in
                `training_data.get_optimal_descriptors()` will be used
                for training.
            val_fraction: The fraction of the training data to use as a
                validation set for tracking model performance during
                training.
            val_key: The target name to track on the validation set
                during training, if performing multi-target learning.
            lr: The learning rate.
            epochs: The maximum number of epochs to train for.
            batch_size: The batch size to use for training.
            xscale: The feature scaler to use, either `None`,
                `'minmax'` or `'standard'`.
            metrics: A list of Keras metrics to pass to `compile(...)`.
            loss: The built-in Keras loss to pass to `compile(...)`.
            fit_params: Any additional parameters to pass to `fit(...)`,
                these will be overwritten by the explicit keyword
                arguments above.

        """

        if self.n_feat > len(training_data.get_optimal_descriptors()):
            raise RuntimeError(
                "The model requires more features than computed in data. "
                f"Please reduce n_feat below or equal to {len(training_data.get_optimal_descriptors())}"
            )

        self.xscale = xscale
        self.target_names = list(self.weights.keys())
        self.optimal_descriptors = training_data.get_optimal_descriptors()

        x = training_data.get_featurized_df()[
            self.optimal_descriptors[: self.n_feat]
        ].values

        y = []
        for targ in self.targets_flatten:
            if self.num_classes[targ] >= 2:  # Classification
                y_inner = keras.utils.to_categorical(
                    training_data.df_targets[targ].values,
                    num_classes=self.num_classes[targ],
                )
                loss = "categorical_crossentropy"
            else:
                # This is a compatibility thing: we have some pre-featurized data that has differing
                # column names that we would like to use. Most of this is single-target, so using just the
                # one column name will always work. Otherwise this will raise a missing column error unless
                # columns are renamed.
                if len(self.targets_flatten) == 1:
                    targ = list(training_data.df_targets.columns)[0]
                y_inner = training_data.df_targets[targ].values.astype(
                    np.float, copy=False
                )
            y.append(y_inner)

        # Scale the input features:
        x = np.nan_to_num(x)
        if self.xscale == "minmax":
            self._scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

        elif self.xscale == "standard":
            self._scaler = StandardScaler()

        x = self._scaler.fit_transform(x)

        if val_data is not None:
            val_x = val_data.get_featurized_df()[
                self.optimal_descriptors[: self.n_feat]
            ].values
            val_x = np.nan_to_num(val_x)
            val_x = self._scaler.transform(val_x)
            if len(self.targets_flatten) == 1:
                val_y = list(
                    val_data.get_target_df().values.transpose()
                )
            else:
                val_y = list(
                    val_data.get_target_df()[self.targets_flatten].values.transpose()
                )
            validation_data = (val_x, val_y)
        else:
            validation_data = None

        # Optionally set up print callback
        if verbose:
            if val_fraction > 0 or validation_data:
                if self._multi_target and val_key is not None:
                    val_metric_key = f"val_{val_key}_mae"
                else:
                    val_metric_key = "val_mae"
                print_callback = keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}, "
                        f"val_loss:{logs['val_loss']:.3f} {val_metric_key}:{logs[val_metric_key]:.3f}"
                    )
                )

            else:
                print_callback = keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}"
                    )
                )

                if callbacks is None:
                    callbacks = [print_callback]
                else:
                    callbacks.append(print_callback)

        fit_params = {
            "x": x,
            "y": y,
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": verbose,
            "validation_split": val_fraction,
            "validation_data": validation_data,
            "callbacks": callbacks,
        }

        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(lr=lr),
            metrics=metrics,
            loss_weights=self.weights,
        )

        self.history = self.model.fit(**fit_params)

    def fit_preset(
        self,
        data: MODData,
        presets: List[Dict[str, Any]] = None,
        val_fraction: float = 0.15,
        verbose: int = 0,
        classification: bool = False,
        refit: bool = True,
        fast: bool = False,
        nested: [Union[bool, int]] = 5,
    ) -> None:
        """Chooses an optimal hyper-parametered MODNet model from different presets.

        This function implements the "inner loop" of a cross-validation workflow. It can
        either be run in full nested mode (i.e. train n_fold * n_preset models) or just
        with a simple random hold-out set.

        The data is first fitted on several well working MODNet presets
        with a validation set (10% of the furnished data by default).

        Sets the `self.model` attribute to the model with the lowest loss.

        Args:
            data: MODData object contain training and validation samples.
            presets: A list of dictionaries containing custom presets.
            verbose: The verbosity level to pass to Keras
            val_fraction: The fraction of the data to use for validation.
            classification: Whether or not we are performing classification.
            refit: Whether or not to refit the final model for each fold with
                the best-performing settings.
            fast: Used for debugging. If `True`, only fit the first 2 presets and
                reduce the number of epochs.
            nested: Whether or not to perform full nested CV. If `True`, set validation
                data based on the chosen folds, ignoring the `val_fraction` argument. If
                an integer, use this number of inner CV folds.

        """

        es = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=100,
            verbose=verbose,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        callbacks = [es]
        if presets is None:
            from modnet.model_presets import gen_presets
            presets = gen_presets(self.n_feat, len(data.df_targets), classification=classification)

        if fast and len(presets) >= 2:
            presets = presets[:2]
            for k, _ in enumerate(presets):
                presets[k]["epochs"] = 100

        val_losses = 1e20 * np.ones((len(presets),))

        if nested and isinstance(nested, bool):
            nested = 5

        best_model = None
        best_n_feat = None
        best_learning_curve = None
        models = []
        learning_curves = []
        best_presets = []
        best_scaler = None

        for i, params in enumerate(presets):
            LOG.info("Initialising preset #{}/{}: {}".format(i + 1, len(presets), params))
            n_feat = min(len(data.get_optimal_descriptors()), params["n_feat"])

            splits = matbench_kfold_splits(data, n_splits=nested, classification=classification)
            if not nested:
                splits = [next(splits)]

            nested_val_losses = []
            for ind, (train, test) in enumerate(splits):
                LOG.info("Initialising split #{} preset #{}/{}: {}".format(ind + 1, i + 1, len(presets), params))

                val_params = {}
                if nested:
                    train_data, test_data = data.split((train, test))
                    val_params["val_data"] = test_data
                else:
                    val_params["val_fraction"] = 0.15
                    train_data = data

                self.model = MODNetModel(
                    self.targets,
                    self.weights,
                    num_neurons=params["num_neurons"],
                    n_feat=n_feat,
                    act=params["act"],
                    num_classes=self.num_classes
                ).model
                self.n_feat = n_feat
                self.fit(
                    train_data,
                    lr=params["lr"],
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    loss=params["loss"],
                    callbacks=callbacks,
                    verbose=verbose,
                    **val_params
                )

                learning_curve = self.model.history.history["val_loss"]

                if any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks):
                    val_loss = np.min(learning_curve)
                    mean_val_loss = np.mean(learning_curve[-20:])
                    LOG.info(f"Taking loss {val_loss:3.3f} from EarlyStopping vs mean {mean_val_loss:3.3f}")
                else:
                    LOG.info(f"Loss from curve {val_loss:3.3f}")
                    val_loss = np.mean(learning_curve[-20:])

                nested_val_losses.append(val_loss)

            val_loss = np.mean(nested_val_losses)

            if val_loss < min(val_losses):
                best_model = self.model
                best_n_feat = n_feat
                best_scaler = self._scaler

            val_losses[i] = val_loss

            models.append(self.model)
            learning_curves.append(learning_curve)
            LOG.info("Validation loss for preset #{}, {}: {:.3f}".format(i+1, params["loss"], val_loss))

        best_preset_idx = val_losses.argmin()
        best_preset = presets[best_preset_idx]
        LOG.info(
            "Preset #{} resulted in lowest validation loss with params {}".format(
                best_preset_idx + 1, params
            )
        )

        best_presets.append(best_preset)

        if refit:
            LOG.info("Refitting with all data and parameters: {}".format(best_preset))
            n_feat = min(len(data.get_optimal_descriptors()), best_preset['n_feat'])
            self.model = MODNetModel(
                self.targets,
                self.weights,
                num_neurons=best_preset['num_neurons'],
                n_feat=n_feat,
                act=best_preset['act'],
                num_classes=self.num_classes).model
            self.n_feat = n_feat
            self.fit(
                data,
                val_fraction=0,
                lr=best_preset['lr'],
                epochs=best_preset['epochs'],
                batch_size=best_preset['batch_size'],
                loss=best_preset['loss'],
                callbacks=callbacks,
                verbose=verbose)
            best_learning_curve = self.model.history.history["loss"]
        else:
            self.n_feat = best_n_feat
            self.model = best_model
            self._scaler = best_scaler

        return models, val_losses, best_learning_curve, learning_curves, best_presets

    def predict(self, test_data: MODData, return_prob=False) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification tasks only: whether to return the probability of each
                class OR only return the most probable class.

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.


        """
        # prevents Nan predictions if some features are inf
        x = test_data.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            self.optimal_descriptors[:self.n_feat]
        ].values

        # Scale the input features:
        x = np.nan_to_num(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
            x = np.nan_to_num(x)

        p = np.array(self.model.predict(x))
        if len(p.shape) == 2:
            p = np.array([p])
        p_dic = {}
        for i, name in enumerate(self.targets_flatten):
            if self.num_classes[name] >= 2:
                if return_prob:
                    temp = p[i, :, :] / (p[i, :, :].sum(axis=1)).reshape((-1, 1))
                    for j in range(temp.shape[-1]):
                        p_dic['{}_prob_{}'.format(name, j)] = temp[:, j]
                else:
                    p_dic[name] = np.argmax(p[i, :, :], axis=1)
            else:
                p_dic[name] = p[i, :, 0]
        predictions = pd.DataFrame(p_dic)
        predictions.index = test_data.structure_ids

        return predictions

    def save(self, filename: str):
        """Save the `MODNetModel` across 3 files with the same base
        filename:

        * <filename>.json contains the Keras model JSON dump.
        * <filename>.pkl contains the `MODNetModel` object, excluding the
          Keras model.
        * <filename>.h5 contains the model weights.

        Parameters:
            filename: The base filename to save to.

        """

        LOG.info("Saving model...")
        model_json = self.model.to_json()
        with open(f"{filename}.json", "w") as f:
            f.write(model_json)
        self.model.save_weights(f"{filename}.h5")
        try:
            with open(f"{filename}.history.pkl", "wb") as f:
                pickle.dump(self.model.history.history, f)
        except Exception:
            import traceback
            traceback.print_exc()
            LOG.info("Failed to save model history.")

        model = self.model
        self.model = None
        self.history = None
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self, f)
        self.model = model
        LOG.info("Saved model to {}(.json/.h5/.pkl)".format(filename))

    @staticmethod
    def load(filename: str):
        """Load the `MODNetModel` from 3 files with the same base
        filename:

        * <filename>.json contains the Keras model JSON dump.
        * <filename>.pkl contains the `MODNetModel` object, excluding the
          Keras model.
        * <filename>.h5 contains the model weights.

        Returns:
            The loaded `MODNetModel` object.

        """

        LOG.info("Loading model from {}(.json/.h5/.pkl)".format(filename))

        with open(f"{filename}.pkl", "rb") as f:
            mod = pickle.load(f)

        if not isinstance(mod, MODNetModel):
            raise RuntimeError(
                "Pickled data in {filename}.pkl did not contain a `MODNetModel`."
            )

        with open(f"{filename}.json", "r") as f:
            model_json = f.read()

        mod.model = keras.models.model_from_json(model_json)
        mod.model.load_weights(f"{filename}.h5")

        if not hasattr(mod, "__modnet_version__"):
            mod.__modnet_version__ = "<=0.1.7"

        LOG.info(
            "Loaded `MODNetModel` created with modnet version {}.".format(
                mod.__modnet_version__
            )
        )

        return mod
