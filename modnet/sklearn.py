__author__ = "David Waroquiers, Matgenix SRL"
__credits__ = "David Waroquiers, Matgenix SRL"
"""Proposition of API for MODNet.


The idea is to do something like :
from sklearn.pipeline import Pipeline
modnet_featurizer = MODNetFeaturizer(...arguments here...)
rr_analysis = RR(...arguments here...)
modnet_model = MODNet(...arguments here...)
p = Pipeline([('featurizer', modnet_featurizer), ('rr', rr_analysis), ('modnet', modnet_model)])

* One note about scikit learn's steps when performing cross-validation:
  A given transformer (e.g. PCA, or RR) will not be executed at each step of the cross validation if you cache it, i.e.
  scikit-learn detects that the inputs are the same (if they are indeed the same) and does not do it several times
  (see https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation).
* Another note about the fit method of a Pipeline object:
  It is possible to pass fit parameters to each step of a Pipeline object
  (see https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit)
* Another note about the __init__ method of sklearn derived estimators:
  The method should ONLY set the instance variables. In particular, "every keyword argument accepted by __init__
  should correspond to an attribute on the instance" (as stated in sklearn's developer documentation)
  (see https://scikit-learn.org/stable/developers/develop.html#instantiation).
"""


from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from typing import Union
from typing import Dict


class MODNetFeaturizer(TransformerMixin, BaseEstimator):
    """"""

    def __init__(self, matminer_featurizers):
        """Constructor for MODNetFeaturizer"""
        self.matminer_featurizers = matminer_featurizers

    def fit(self, X, y=None):
        """Probably not needed except for some matminer featurizers (e.g. I think SOAP needs to be fitted before)."""
        return self

    def transform(self, X, y=None):
        """Transform the input data (i.e. containing the composition and/or structure to the features"""

    @classmethod
    def from_preset(cls, preset):
        """Initializes a MODNetFeaturizer class based on some preset.

        Args:
            preset: Name of the preset (e.g. "DeBreuck2020" could trigger the Structure+Composition featurizers)

        Notes:
            See in matminer how it is done, e.g. SiteStatsFingerprint.from_preset method.
        """


class RR(TransformerMixin, BaseEstimator):
    """Relevance-Redundancy (RR) analysis.

    Feature selection and ordering procedure based on the relevance-redundancy algorithm developed in ARTICLE.

    Blabla

    Notes:
        1. Uses preprocessing.get_features_relevance_redundancy under the hood.
        2. To check if it is ok to have y that is *NOT* ignored (i.e. not y=None) in the fit method. Usually a
           Transformer's fit method signature is fit(self, X, y=None, ...).
    """

    def __init__(self, n_feat: Union[None, int]=None, rr_parameters: Union[None, Dict]=None):
        """Constructor for RR transformer.

        Args:
            n_feat: Number of features to keep and reorder using the RR procedure (default: None, i.e. all features).
            rr_parameters: Allow to tune p and c parameters. (default: None, i.e. use the dynamical setting in ARTICLE).

        Notes:
            This method should not be changed
        """
        self.n_feat = n_feat
        self.rr_parameters = rr_parameters

    def fit(self, X, y, nmi_feats_target=None, cross_nmi_feats=None):
        """Fit the model with X and y.

        Allow to accept numpy arrays for X and y or pandas Dataframes ?

        Args:
            X: Training input data
            y: Training output data
            nmi_feats_target: If a precomputed nmi features=>target is available.
            cross_nmi_feats: If a precomputed cross-nmi features=>features is available.
        """

    def transform(self, X, y=None):
        """Transform the inputs X based on a fitted RR analysis.

        Notes:
            I don't remember if the signature should be transform(self, X, y=None) or transform(self, X).

        Args:
            X: Input data
            y: Output data (should be ignored)

        Returns:
            Filtered and ordered data according to the fitted RR analysis.
        """


class MODNet(RegressorMixin, BaseEstimator):
    """MODNet model.

    Blabla.

    Notes:
        No assumption on the features here, just a list of numbers.
        What makes a MODNet model special with respect to using directly keras ? I would say that it is always doing
        some joint learning, maybe something else ?
    """

    def __init__(self):
        """Constructor for MODNet model.

        Needs some thinking of what to put in __init__ and fit
        """

    def fit(self, X, y):
        """Fit a MODNet regression model."""

    def predict(self, X):
        """Predict output based on a fitted MODNet regression model"""
