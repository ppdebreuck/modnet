# coding: utf-8
# Distributed under the terms of the MIT License.

""" This module defines the :class:`MODData` class, featurizer functions
and functions to compute normalized mutual information (NMI) and relevance redundancy
(RR) between descriptors.

"""

import os
import logging

from pymatgen import Structure
from pymatgen.core.periodic_table import Element
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    AtomicOrbitals,
    AtomicPackingEfficiency,
    BandCenter,
    # CohesiveEnergy, - This descriptor was not used in the paper preset
    # ElectronAffinity, - This descriptor was not used in the paper preset
    ElectronegativityDiff,
    ElementFraction,
    ElementProperty,
    IonProperty,
    Miedema,
    OxidationStates,
    Stoichiometry,
    TMetalFraction,
    ValenceOrbital,
    YangSolidSolution,
)

from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.structure import (
    BagofBonds,
    BondFractions,
    ChemicalOrdering,
    CoulombMatrix,
    DensityFeatures,
    EwaldEnergy,
    GlobalSymmetryFeatures,
    MaximumPackingEfficiency,
    PartialRadialDistributionFunction,
    RadialDistributionFunction,
    SineCoulombMatrix,
    SiteStatsFingerprint,
    StructuralHeterogeneity,
    XRDPowderPattern,
)
from matminer.featurizers.site import (
    AGNIFingerprints,
    AverageBondAngle,
    AverageBondLength,
    BondOrientationalParameter,
    ChemEnvSiteFingerprint,
    CoordinationNumber,
    CrystalNNFingerprint,
    GaussianSymmFunc,
    GeneralizedRadialDistributionFunction,
    LocalPropertyDifference,
    OPSiteFingerprint,
    VoronoiFingerprint,
)
from pymatgen.analysis.local_env import VoronoiNN
from typing import Dict, List, Union, Optional, Callable, Hashable

DATABASE = pd.DataFrame([])

logging.getLogger().setLevel(logging.INFO)


def nmi_target(df_feat: pd.DataFrame, df_target: pd.DataFrame,
               drop_constant_features: bool = True, **kwargs) -> pd.DataFrame:
    """
    Computes the Normalized Mutual Information (NMI) between a list of
    input features and a target variable.

    Args:
        df_feat (pandas.DataFrame): Dataframe containing the input features for
            which the NMI with the target variable is to be computed.
        df_target (pandas.DataFrame): Dataframe containing the target variable.
            This DataFrame should contain only one column and have the same
            size as `df_feat`.
        drop_constant_features (bool): If True, the features that are constant
            across the entire data set will be dropped.
        **kwargs: Keyword arguments to be passed down to the
            :py:func:`mutual_info_regression` function from scikit-learn. This
            can be useful e.g. for testing purposes.

    Returns:
        pandas.DataFrame: Dataframe containing the NMI between each of
            the input features and the target variable.

    """
    # Initial checks
    if df_target.shape[1] != 1:
        raise ValueError('The target DataFrame should have exactly one column.')

    if len(df_feat) != len(df_target):
        raise ValueError(
            'The input features DataFrame and the target variable DataFrame '
            'should contain the same number of data points.'
        )

    # Drop features which have the same value for the entire data set
    if drop_constant_features:
        frange = df_feat.max(axis=0) - df_feat.min(axis=0)
        to_drop = frange[frange == 0].index
        df_feat = df_feat.drop(to_drop, axis=1)

    # Prepare the output DataFrame and compute the mutual information
    target_name = df_target.columns[0]
    out_df = pd.DataFrame([], columns=[target_name], index=df_feat.columns)
    out_df.loc[:, target_name] = (mutual_info_regression(df_feat, df_target[target_name], **kwargs))

    # Compute the "self" mutual information (i.e. information entropy) of the target variable and of the input features
    target_mi = mutual_info_regression(df_target[target_name].values.reshape(-1, 1),
                                       df_target[target_name], **kwargs)[0]
    diag = {}
    for x in df_feat.columns:
        diag[x] = (mutual_info_regression(df_feat[x].values.reshape(-1, 1), df_feat[x], **kwargs))[0]

    # Normalize the mutual information
    for x in out_df.index:
        out_df.loc[x, target_name] = out_df.loc[x, target_name] / ((target_mi + diag[x])/2)

    return out_df


def get_cross_nmi(df_feat: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Computes the Normalized Mutual Information (NMI) between input features.

    Args:
        df_feat (pandas.DataFrame): Dataframe containing the input features for
            which the NMI with the target variable is to be computed.
        **kwargs: Keyword arguments to be passed down to the
            :py:func:`mutual_info_regression` function from scikit-learn. This
            can be useful e.g. for testing purposes.

    Returns:
        pd.DataFrame: pandas.DataFrame containing the Normalized Mutual Information between features.
    """
    # Prepare the output DataFrame and compute the mutual information
    out_df = pd.DataFrame([], columns=df_feat.columns, index=df_feat.columns)
    for ifeat, feat_name in enumerate(out_df.columns, start=1):
        logging.info('Computing MI of feature #{:d}/{:d} ({}) with all other features'.format(ifeat,
                                                                                              len(out_df.columns),
                                                                                              feat_name))
        out_df.loc[:, feat_name] = (mutual_info_regression(df_feat, df_feat[feat_name], **kwargs))

    # Compute the "self" mutual information (i.e. information entropy) of the features
    logging.info('Computing "self" MI (i.e. information entropy) of features')
    diag = {}
    for x in df_feat.columns:
        diag[x] = (mutual_info_regression(df_feat[x].values.reshape(-1, 1), df_feat[x], **kwargs))[0]

    # Normalize the mutual information between features
    logging.info('Normalizing MI')
    for feat1 in out_df.index:
        for feat2 in out_df.columns:
            out_df.loc[feat1, feat2] = out_df.loc[feat1, feat2] / ((diag[feat1] + diag[feat2]) / 2)
        logging.debug('  => Computed NMI of feature "{}" with all other features :\n'
                      '{}'.format(feat1, '\n'.join(['      {} : {:.4f}'.format(feat2,
                                                                               out_df.loc[feat1, feat2])
                                                    for feat2 in out_df.loc[feat1, :].index])))
    return out_df


def get_rr_p_parameter_default(nn: int) -> float:
    """
    Returns p for the default expression outlined in arXiv:2004:14766.

    Args:
        nn (int): number of features currently in chosen subset.

    Returns:
        float: the value for p.

    """
    return max(0.1, 4.5 - 0.4 * nn ** 0.4)


def get_rr_c_parameter_default(nn: int) -> float:
    """
    Returns c for the default expression outlined in arXiv:2004:14766.

    Args:
        nn (int): number of features currently in chosen subset.

    Returns:
        float: the value for p.

    """
    return min(1e5, 1e-6 * nn ** 3)


def get_features_relevance_redundancy(
    target_nmi: pd.DataFrame,
    cross_nmi: pd.DataFrame,
    n_feat: Optional[int] = None,
    rr_parameters: Optional[Dict[str, Union[float, Callable[[int], float]]]] = None,
    return_pc: bool = False
) -> List:
    """
    Select features from the Relevance Redundancy (RR) score between the input
    features and the target output.

    The RR is defined following Equation 2 of De Breuck et al, arXiv:2004:14766,
    with default values,

    ..math:: p = \\max{0.1, 4.5 -  n^{0.4}},

    and

    ..math:: c = 10^{-6} n^3,

    where :math:`n` is the number of features in the "chosen" subset for that iteration.
    These values can be overriden with the `rr_parameters` dictionary argument.

    Args:
        target_nmi (pandas.DataFrame): dataframe  containing the Normalized
            Mutual Information (NMI) between a list of input features and a
            target variable, as computed from :py:func:`nmi_target`.
        cross_nmi (pandas.DataFrame): dataframe containing the NMI between the
            input features, as computed from :py:func:`get_cross_nmi`.
        n_feat (int): Number of features for which the RR score needs to be computed (default: all features).
        rr_parameters (dict): Allows tuning of p and c parameters. Currently
            allows fixing of p and c to constant values instead of using the
            dynamical evaluation. Expects to find keys `"p"` and `"c"`, containing
            either a callable that takes `n` as an argument and returns the
            desired `p` or `c`, or another dictionary containing the key `"value"`
            that stores a constant value of `p` or `c`.
        return_pc: Whether to return p and c values in the output dictionaries.

    Returns:
        list: List of dictionaries containing the results of the relevance-redundancy selection algorithm.

    """
    # Initial checks
    if set(cross_nmi.index) != set(cross_nmi.columns):
        raise ValueError('The cross_nmi DataFrame should have its indices and columns identical.')
    if not set(target_nmi.index).issubset(set(cross_nmi.index)):
        raise ValueError('The indices of the target DataFrame should be included in the cross_nmi DataFrame indices.')

    # Define the functions for the parameters
    if rr_parameters is None:
        get_p = get_rr_p_parameter_default
        get_c = get_rr_c_parameter_default
    else:
        if 'p' not in rr_parameters or 'c' not in rr_parameters:
            raise ValueError('When tuning p and c with rr_parameters in get_features_relevance_redundancy, '
                             'both parameters should be tuned')
        # Set up p
        if callable(rr_parameters["p"]):
            get_p = rr_parameters["p"]
        elif rr_parameters['p'].get('function') == 'constant':
            def get_p(_):
                return rr_parameters['p']['value']
        else:
            raise ValueError(
                'If not passing a callable, "p" dict must contain keys "function" and "value".'
            )
        # Set up c
        if callable(rr_parameters["c"]):
            get_c = rr_parameters["c"]
        elif rr_parameters['c'].get('function') == 'constant':
            def get_c(_):
                return rr_parameters['c']['value']
        else:
            raise ValueError(
                'If not passing a callable, "c" dict must contain keys "function" and "value".'
            )

    # Set up the output list
    out = []

    # The first feature is the one with the largest target NMI
    target_column = target_nmi.columns[0]
    first_feature = target_nmi.nlargest(1, columns=target_column).index[0]
    feature_set = [first_feature]
    feat_out = {'feature': first_feature, 'RR_score': None, 'NMI_target': target_nmi[target_column][first_feature]}
    if return_pc:
        feat_out['RR_p'] = None
        feat_out['RR_c'] = None
    out.append(feat_out)

    # Default is to get the RR score for all features
    if n_feat is None:
        n_feat = len(target_nmi.index)

    # Loop on the number of features
    for n in range(1, n_feat):
        logging.debug("In selection of feature {}/{} features...".format(n+1, n_feat))
        if (n+1) % 50 == 0:
            logging.info("Selected {}/{} features...".format(n, n_feat))
        p = get_p(n)
        c = get_c(n)

        # Compute the RR score
        score = cross_nmi.copy()
        score = score.loc[target_nmi.index, target_nmi.index]
        # Remove features already selected for the index
        score = score.drop(feature_set, axis=0)
        # Use features already selected to compute the maximum NMI between
        # the remaining features and those already selected
        score = score[feature_set]

        # Get the scores of the remaining features
        for i in score.index:
            row = score.loc[i, :]
            score.loc[i, :] = target_nmi.loc[i, target_column] / (row ** p + c)

        # Get the next feature (the one with the highest score)
        scores_remaining_features = score.min(axis=1)
        next_feature = scores_remaining_features.idxmax(axis=0)
        feature_set.append(next_feature)

        # Add the results for the next feature to the list
        feat_out = {'feature': next_feature, 'RR_score': scores_remaining_features[next_feature],
                    'NMI_target': target_nmi[target_column][next_feature]}
        if return_pc:
            feat_out['RR_p'] = p
            feat_out['RR_c'] = c

        out.append(feat_out)

    return out


def get_features_dyn(n_feat, cross_mi, target_mi):

    first_feature = target_mi.nlargest(1).index[0]
    feature_set = [first_feature]

    get_p = get_rr_p_parameter_default
    get_c = get_rr_c_parameter_default

    if n_feat == -1:
        n_feat = len(cross_mi.index)

    for n in range(n_feat-1):
        if (n+1) % 50 == 0:
            logging.info("Selected {}/{} features...".format(n+1, n_feat))

        p = get_p(n)
        c = get_c(n)

        score = cross_mi.copy()
        score = score.loc[target_mi.index, target_mi.index]
        score = score.drop(feature_set, axis=0)
        score = score[feature_set]

        for i in score.index:
            row = score.loc[i, :]
            score.loc[i, :] = target_mi[i] / (row**p+c)

        next_feature = score.min(axis=1).idxmax(axis=0)
        feature_set.append(next_feature)

    return feature_set


def merge_ranked(lists: List[List[Hashable]]) -> List[Hashable]:
    """ For multiple lists of ranked feature names/IDs (e.g. for different
    targets), work through the lists and merge them such that each
    feature is included once according to its highest rank across each
    list.

    Args:
        lists (List[List[Hashable]]): the list of lists to merge.

    Returns:
        List[Hashable]: list of merged and ranked feature names/IDs.

    """
    if not all(len(lists[0]) == len(sublist) for sublist in lists):
        # pad all lists to same length
        max_len = max(len(sublist) for sublist in lists)
        for ind, sublist in enumerate(lists):
            if len(sublist) < max_len:
                lists[ind].extend((max_len - len(sublist))*[None])

    total_set = set()
    ranked_list = []
    for subrank in zip(*lists):
        for feature in subrank:
            if feature not in total_set and feature is not None:
                ranked_list.append(feature)
                total_set.add(feature)

    return ranked_list


def clean_df(df):
    """ Cleans dataframe by dropping missing values, replacing NaN's and infinities
    and selecting only columns containing numerical data.

    Args:
        df (pd.DataFrame): the dataframe to clean.

    Returns:
        pd.DataFrame: the cleaned dataframe.

    """

    df = df.dropna(axis=1, how='all')
    df = df.replace([np.inf, -np.inf, np.nan], -1)
    df = df.select_dtypes(include='number')

    return df


def featurize_composition(df: pd.DataFrame) -> pd.DataFrame:
    """ Decorate input `pandas.DataFrame` of structures with composition
    features from matminer.

    Currently applies the set of all matminer composition features.

    Args:
        df (pandas.DataFrame): the input dataframe with `"structure"`
            column containing `pymatgen.Structure` objects.

    Returns:
        pandas.DataFrame: the decorated DataFrame.

    """
    logging.info("Applying composition featurizers...")
    df = df.copy()
    df['composition'] = df['structure'].apply(lambda s: s.composition)
    featurizer = MultipleFeaturizer([ElementProperty.from_preset("magpie"),
                                     AtomicOrbitals(),
                                     BandCenter(),
                                     # ElectronAffinity(), - This descriptor was not used in the paper preset
                                     Stoichiometry(),
                                     ValenceOrbital(),
                                     IonProperty(),
                                     ElementFraction(),
                                     TMetalFraction(),
                                     # CohesiveEnergy(), - This descriptor was not used in the paper preset
                                     Miedema(),
                                     YangSolidSolution(),
                                     AtomicPackingEfficiency(),
                                     ])

    df = featurizer.featurize_dataframe(df, "composition", multiindex=True, ignore_errors=True)
    df.columns = df.columns.map('|'.join).str.strip('|')

    ox_featurizer = MultipleFeaturizer([OxidationStates(),
                                        ElectronegativityDiff()
                                        ])

    df = CompositionToOxidComposition().featurize_dataframe(df, "Input Data|composition")

    df = ox_featurizer.featurize_dataframe(df, "composition_oxid", multiindex=True, ignore_errors=True)
    df = df.rename(columns={'Input Data': ''})
    df.columns = df.columns.map('|'.join).str.strip('|')

    _orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}

    df['AtomicOrbitals|HOMO_character'] = df['AtomicOrbitals|HOMO_character'].map(_orbitals)
    df['AtomicOrbitals|LUMO_character'] = df['AtomicOrbitals|LUMO_character'].map(_orbitals)

    df['AtomicOrbitals|HOMO_element'] = df['AtomicOrbitals|HOMO_element'].apply(
        lambda x: -1 if not isinstance(x, str) else Element(x).Z
    )
    df['AtomicOrbitals|LUMO_element'] = df['AtomicOrbitals|LUMO_element'].apply(
        lambda x: -1 if not isinstance(x, str) else Element(x).Z
    )

    df = df.replace([np.inf, -np.inf, np.nan], 0)

    return clean_df(df)


def featurize_structure(df: pd.DataFrame) -> pd.DataFrame:
    """ Decorate input `pandas.DataFrame` of structures with structural
    features from matminer.

    Currently applies the set of all matminer structure features.

    Args:
        df (pandas.DataFrame): the input dataframe with `"structure"`
            column containing `pymatgen.Structure` objects.

    Returns:
        pandas.DataFrame: the decorated DataFrame.

    """

    logging.info("Applying structure featurizers...")

    df = df.copy()

    structure_features = [
         DensityFeatures(),
         GlobalSymmetryFeatures(),
         RadialDistributionFunction(),
         CoulombMatrix(),
         PartialRadialDistributionFunction(),
         SineCoulombMatrix(),
         EwaldEnergy(),
         BondFractions(),
         StructuralHeterogeneity(),
         MaximumPackingEfficiency(),
         ChemicalOrdering(),
         XRDPowderPattern(),
         BagofBonds()
    ]

    featurizer = MultipleFeaturizer([feature.fit(df["structure"]) for feature in structure_features])

    df = featurizer.featurize_dataframe(df, "structure", multiindex=True, ignore_errors=True)
    df.columns = df.columns.map('|'.join).str.strip('|')

    dist = df["RadialDistributionFunction|radial distribution function"][0]['distances'][:50]
    for i, d in enumerate(dist):
        _rdf_key = "RadialDistributionFunction|radial distribution function|d_{:.2f}".format(d)
        df[_rdf_key] = df["RadialDistributionFunction|radial distribution function"].apply(lambda x: x['distribution'][i])

    df = df.drop("RadialDistributionFunction|radial distribution function", axis=1)

    _crystal_system = {
        "cubic": 1, "tetragonal": 2, "orthorombic": 3,
        "hexagonal": 4, "trigonal": 5, "monoclinic": 6, "triclinic": 7
    }

    df["GlobalSymmetryFeatures|crystal_system"] = df["GlobalSymmetryFeatures|crystal_system"].map(_crystal_system)
    df["GlobalSymmetryFeatures|is_centrosymmetric"] = df["GlobalSymmetryFeatures|is_centrosymmetric"].map(int)

    return clean_df(df)


def featurize_site(df: pd.DataFrame, site_stats=("mean", "std_dev")) -> pd.DataFrame:
    """ Decorate input `pandas.DataFrame` of structures with site
    features from matminer.

    Currently creates the set of all matminer structure features with
    the `matminer.featurizers.structure.SiteStatsFingerprint`.

    Args:
        df (pandas.DataFrame): the input dataframe with `"structure"`
            column containing `pymatgen.Structure` objects.
        site_stats (Tuple[str]): the matminer site stats to use in the
            `SiteStatsFingerprint` for all features.

    Returns:
        pandas.DataFrame: the decorated DataFrame.

    """

    logging.info("Applying site featurizers...")

    df = df.copy()
    df.columns = ["Input data|" + x for x in df.columns]

    site_fingerprints = (
        AGNIFingerprints(),
        GeneralizedRadialDistributionFunction.from_preset("gaussian"),
        OPSiteFingerprint(),
        CrystalNNFingerprint.from_preset("ops"),
        VoronoiFingerprint(),
        GaussianSymmFunc(),
        ChemEnvSiteFingerprint.from_preset("simple"),
        CoordinationNumber(),
        LocalPropertyDifference(),
        BondOrientationalParameter(),
        AverageBondLength(VoronoiNN()),
        AverageBondAngle(VoronoiNN())
    )

    for fingerprint in site_fingerprints:
        site_stats_fingerprint = SiteStatsFingerprint(
            fingerprint,
            stats=site_stats
        )

        df = site_stats_fingerprint.featurize_dataframe(
            df,
            "Input data|structure",
            multiindex=False,
            ignore_errors=True
        )

        fingerprint_name = fingerprint.__class__.__name__

        # rename some features for backwards compatibility with pretrained models
        if fingerprint_name == "GeneralizedRadialDistributionFunction":
            fingerprint_name = "GeneralizedRDF"
        elif fingerprint_name == "AGNIFingerprints":
            fingerprint_name = "AGNIFingerPrint"
        elif fingerprint_name == "BondOrientationalParameter":
            fingerprint_name = "BondOrientationParameter"
        elif fingerprint_name == "GaussianSymmFunc":
            fingerprint_name = "ChemEnvSiteFingerprint|GaussianSymmFunc"

        if "|" not in fingerprint_name:
            fingerprint_name += "|"

        df.columns = [f"{fingerprint_name}{x}" if "|" not in x else x for x in df.columns]

    df = df.loc[:, (df != 0).any(axis=0)]

    return clean_df(df)


class MODData:
    """ The MODData class takes takes a list of `pymatgen.Structure`
    objects and creates a `pandas.DataFrame` that contains many matminer
    features per structure. It then uses mutual information between
    features and targets, and between the features themselves, to
    perform feature selection using relevance-redundancy indices.

    Attributes:
        df_structure (pd.DataFrame): dataframe storing the `pymatgen.Structure`
            representations for each structured, indexed by ID.
        df_targets (pd.Dataframe): dataframe storing the prediction targets
            per structure, indexed by ID.
        df_featurized (pd.DataFrame): dataframe with columns storing all
            computed features per structure, indexed by ID.
        optimal_features (List[str]): if feature selection has been performed
            this attribute stores a list of the selected features.
        optimal_features_by_target (Dict[str, List[str]]): if feature selection has been performed
            this attribute stores a list of the selected features, broken down by
            target property.

    """

    def __init__(
        self,
        structures: Optional[List[Structure]] = None,
        targets: Optional[Union[List[float], np.ndarray, List[List[float]]]] = None,
        target_names: Optional[List[str]] = None,
        structure_ids: Optional[List[Hashable]] = None,
        df_featurized: Optional[pd.DataFrame] = None,
    ):
        """ Initialise the MODData object either from a list of structures
        or from an already featurized dataframe. Prediction targets per
        structure can be specified as lists or an array alongside their
        target names. A list of unique IDs can be provided to label the
        structures.

        Args:
            structures: list of structures to featurize and predict.
            targets: optional list or list of lists of prediction targets per structure.
            target_names: optional list of names of target properties to use in the dataframe.
            structure_ids: optional list of unique IDs to use instead of generated integers.
            df_featurized: optional featurized dataframe to use instead of
                featurizing a new one. Should be passed without structures.

        """

        self.df_featurized = df_featurized

        if structures is not None and self.df_featurized is not None:
            raise RuntimeError(
                "Only one of `structures` or `df_featurized` should be passed to `MODData`."
            )
        if structures is None and self.df_featurized is None:
            raise RuntimeError(
                "At least one of `structures` or `df_featurized` should be passed to `MODData`."
            )

        if structures is not None and targets is not None:
            if np.shape(targets)[0] != len(structures):
                raise ValueError(f"Targets must have same length as structures: {np.shape(targets)} vs {len(structures)}")

        if target_names:
            if np.shape(targets)[-1] != len(target_names):
                raise ValueError("Target names must be supplied for every target.")
        else:
            target_names = ['prop'+str(i) for i in range(len(targets))]

        if structure_ids:
            # for backwards compat, always store the *passed* list of
            # IDs, so they can be used when loading from a database file
            self.mpids = structure_ids
            # check ids are unique
            if len(set(structure_ids)) != len(structure_ids):
                raise ValueError("List of IDs (`structure_ids`) provided must be unique.")

            if len(structure_ids) != len(structures):
                raise ValueError("List of IDs (`structure_ids`) must have same length as list of structure.")

        else:
            self.mpids = None
            structure_ids = [f"id{i}" for i in range(len(structures))]

        if targets is None:
            # set up dataframe for targets with columns (id, property_1, ..., property_n)
            data = {name: target for name, target in zip(target_names, targets)}
            data["id"] = structure_ids

            self.df_targets = pd.DataFrame(data)
            self.df_targets.set_index('id', inplace=True)

        # set up dataframe for structures with columns (id, structure)
        self.df_structure = pd.DataFrame({'id': structure_ids, 'structure': structures})
        self.df_structure.set_index('id', inplace=True)

    def featurize(self, fast: bool = False, db_file: str = 'feature_database.pkl'):
        """ For the input structures, construct many matminer features
        and save a featurized dataframe. If `db_file` is specified, this
        method will try to load previous feature calculations for each
        structure ID instead of recomputing.

        Sets the `self.df_featurized` attribute.

        Args:
            fast (bool): whether or not to try to load from a backup.
            db_file (str): filename of a pickled dataframe containing
                with the same ID index as this `MODData` object.

        """

        logging.info('Computing features, this can take time...')

        df_done = None
        df_todo = None

        if self.df_featurized:
            raise RuntimeError("Not overwriting existing featurized dataframe.")

        if fast and self.mpids:
            logging.info('Fast featurization on, retrieving from database...')

            global DATABASE
            if DATABASE.empty:
                DATABASE = pd.read_pickle(db_file)

            mpids_done = [x for x in self.mpids if x in DATABASE.index]

            logging.info(f"Retrieved features for {len(mpids_done)} out of {len(self.mpids)} materials")
            df_done = DATABASE.loc[mpids_done]
            df_todo = self.df_structure.drop(mpids_done, axis=0)

        # if any structures were already loaded
        if df_done:
            # if any are left to compute, do them
            if len(df_todo) > 0:
                df_composition = featurize_composition(df_todo)
                df_structure = featurize_structure(df_todo)
                df_site = featurize_site(df_todo)
                df_final = df_done.append(df_composition.join(df_structure.join(df_site, lsuffix="l"), rsuffix="r"))

            # otherwise, all structures were successfully loaded
            else:
                df_final = df_done

        # otherwise, no structures were loaded, so we need to compute all
        else:
            df_composition = featurize_composition(self.df_structure)
            df_structure = featurize_structure(self.df_structure)
            df_site = featurize_site(self.df_structure)
            df_final = df_composition.join(df_structure.join(df_site, lsuffix="l"), rsuffix="r")

        if self.mpids:
            df_final = df_final.reindex(self.mpids)

        df_final = df_final.replace([np.inf, -np.inf, np.nan], 0)

        self.df_featurized = df_final
        logging.info('Data has successfully been featurized!')

    def feature_selection(self, n: int = 1500, full_cross_nmi: Optional[pd.DataFrame] = None):
        """ Compute the mutual information between features and targets,
        then apply relevance-redundancy rankings to choose the top `n`
        features.

        Sets the `self.optimal_features` attribute to a list of feature
        names.

        Args:
            n: number of desired features.
            full_cross_nmi: specify the cross NMI between features as a
                dataframe.

        """
        if getattr(self, "df_featurized") is None:
            raise RuntimeError("Mutual information feature selection requiresd featurized data, please call `.featurize()`")
        if getattr(self, "df_targets") is None:
            raise RuntimeError("Mutual information feature selection requires target properties")

        ranked_lists = []
        optimal_features_by_target = {}

        # Loading mutual information between features
        if full_cross_nmi is None:
            this_dir, this_filename = os.path.split(__file__)
            dp = os.path.join(this_dir, "data", "Features_cross")
            if os.path.isfile(dp):
                full_cross_nmi = pd.read_pickle(dp)
        else:
            full_cross_nmi = full_cross_nmi

        if full_cross_nmi is None:
            if full_cross_nmi is None:
                logging.info('Computing cross NMI between all features...')
                df = self.df_featurized.copy()
                cross_nmi = get_cross_nmi(df)

        for i, name in enumerate(self.names):
            logging.info("Starting target {}/{}: {} ...".format(i+1, len(self.targets), self.names[i]))

            # Computing mutual information with target
            logging.info("Computing mutual information between features and target...")
            df = self.df_featurized.copy()
            y_nmi = nmi_target(self.df_featurized, self.df_targets[[name]])[name]

            # remove columns from cross NMI if not present in feature NMI
            cross_nmi = full_cross_nmi.copy(deep=True)
            missing = [x for x in cross_nmi.index if x not in y_nmi.index]
            cross_nmi = cross_nmi.drop(missing, axis=0).drop(missing, axis=1)

            logging.info('Computing optimal features...')
            optimal_features_by_target[name] = get_features_dyn(min(n, len(y_nmi.index)), cross_nmi, y_nmi)
            ranked_lists.append(optimal_features_by_target[name])

            logging.info("Done with target {}/{}: {}.".format(i+1, len(self.targets), name))

        logging.info('Merging all features...')
        self.optimal_features = merge_ranked(ranked_lists)
        self.optimal_features_by_target = optimal_features_by_target
        logging.info('Done.')

    def shuffle(self):
        # caution, not fully implemented
        raise NotImplementedError("shuffle function not yet finished.")
        self.df_featurized = self.df_featurized.sample(frac=1)
        self.df_targets = self.df_targets.loc[self.df_featurized.index]

    @property
    def structures(self) -> List[Structure]:
        """Returns the list of `pymatgen.Structure` objects. """
        return self.df_structure["structure"]

    @property
    def targets(self) -> np.ndarray:
        """ Returns a list of lists of prediction targets. """
        return self.df_targets.values

    @property
    def names(self) -> List[str]:
        """ Returns the list of prediction target field names. """
        return list(self.df_targets)

    def save(self, filename):
        """ Pickle the contents of the `MODData` object
        so that it can be loaded in  with `MODData.load()`.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be compressed accordingly by `pandas.to_pickle(...)`.

        """
        pd.to_pickle(self, filename)
        logging.info(f'Data successfully saved as {filename}!')

    @staticmethod
    def load(filename):
        """ Load `MODData` object pickled by the `.save(...)` method.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be decompressed accordingly by `pandas.read_pickle(...)`.

        """
        pickled_data = pd.read_pickle(filename)
        if isinstance(pickled_data, MODData):
            return pickled_data

        raise ValueError(
            f"File {filename} did not contain compatible data to create a MODData object, "
            f"instead found {pickled_data.__class__.__name__}."
        )

    def get_structure_df(self):
        return self.df_structure

    def get_target_df(self):
        return self.df_targets

    def get_featurized_df(self):
        return self.df_featurized

    def get_optimal_descriptors(self):
        return self.optimal_features

    def get_optimal_df(self):
        return self.df_featurized[self.optimal_features].join(self.targets)
