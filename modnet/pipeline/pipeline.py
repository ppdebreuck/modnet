from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from sklearn.base import BaseEstimator
from modnet.preprocessing import MODData
from modnet.featurizers.presets import (
    DEFAULT_FEATURIZER,
    DEFAULT_COMPOSITION_ONLY_FEATURIZER,
)
from modnet.hyper_opt import FitGenetic
from modnet.utils import LOG


class FitGeneticEstimator(BaseEstimator):
    def __init__(
        self,
        moddata_params: dict = None,
        featurize_params: dict = None,
        feature_selection_params: dict = None,
        fitgenetic_params: dict = None,
        run_params: dict = None,
    ):
        if moddata_params is None:
            moddata_params = {}

        to_pop_from_moddata_params = []
        if featurize_params is None:
            featurize_params = {}
            if moddata_params:
                for possible_key in ["fast", "db_file", "n_jobs"]:
                    if possible_key in moddata_params.keys():
                        featurize_params[possible_key] = moddata_params[possible_key]
                        to_pop_from_moddata_params.append(possible_key)

        if feature_selection_params is None:
            feature_selection_params = {}
            if moddata_params:
                for possible_key in [
                    "n",
                    "cross_nmi",
                    "use_precomputed_cross_nmi",
                    "n_samples",
                    "n_jobs",
                ]:
                    if possible_key in moddata_params.keys():
                        feature_selection_params[possible_key] = moddata_params[
                            possible_key
                        ]
                        to_pop_from_moddata_params.append(possible_key)

        if to_pop_from_moddata_params:
            for key in set(to_pop_from_moddata_params):
                moddata_params.pop(key)

        if fitgenetic_params is None:
            fitgenetic_params = {}

        if run_params is None:
            run_params = {}

        self.moddata_params = moddata_params
        self.featurize_params = featurize_params
        self.feature_selection_params = feature_selection_params
        self.fitgenetic_params = fitgenetic_params
        self.run_params = run_params

        self.data = None
        self.fg = None
        self.model = None

    def fit(
        self,
        X: Optional[List] = None,
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        data: Optional[MODData] = None,
        fg_targets: List = None,
    ):
        LOG.info("Starting the FitGenetic pipeline...")
        # Get the MODData object
        if data is None:
            # Compatibility checks
            if X is None:
                raise ValueError("At least one of X or data should be provided.")
            if y is None:
                raise ValueError("At least one of y or data should be provided.")

            # Define the featurizer
            if (
                isinstance(X[0], Composition)
                and "featurizer" not in self.moddata_params.keys()
            ):
                self.moddata_params["featurizer"] = DEFAULT_COMPOSITION_ONLY_FEATURIZER
            elif (
                isinstance(X[0], Structure)
                and "featurizer" not in self.moddata_params.keys()
            ):
                self.moddata_params["featurizer"] = DEFAULT_FEATURIZER

            # LOG.info(f"Chosen featurizer: {self.moddata_params['featurizer']}.")
            # Create the MODData object
            data = MODData(
                materials=X,
                targets=y,
                target_names=list(np.array(fg_targets).flatten())
                if fg_targets
                else None,
                **self.moddata_params,
            )

            LOG.info("MODData created!")

            # Featurization and feature selection
            data.featurize(**self.featurize_params)
            data.feature_selection(**self.feature_selection_params)

            # Making the MODData an attribute
            self.data = data
            LOG.info("The MODData has been obtained correctly!")

        else:  # data is provided
            LOG.info(
                "MODData object passed to fit, we will ignore X and y if provided as well."
            )
            if data.df_featurized is None:
                LOG.info("Featurization has not yet been performed, running now...")
                data.featurize(**self.featurize_params)
            if data.cross_nmi is None:
                LOG.info("Feature selection has not yet been performed, running now...")
                data.feature_selection(**self.feature_selection_params)

            # Making the MODData an attribute
            self.data = data
            LOG.info("The MODData has been obtained correctly!")

        # Define the FitGenetic object
        fg = FitGenetic(data=self.data, targets=fg_targets, **self.fitgenetic_params)
        self.fg = fg
        LOG.info("The FitGenetic has been defined.")

        # Get the model by running the FitGenetic
        model = fg.run(**self.run_params)
        self.model = model
        LOG.info("The hyperparameters have been obtained and the model obtained.")

        return model
