from pymatgen import Structure
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
from matminer.utils.conversions import composition_to_oxidcomposition
from matminer.featurizers.composition import *
from matminer.featurizers.base import *
from matminer.featurizers.conversions import *
from pymatgen.core.periodic_table import *
from matminer.featurizers.structure import *
from matminer.featurizers.site import *
from pymatgen.analysis.local_env import VoronoiNN
from typing import Dict, List, Union
import pickle
import os
import logging
database = pd.DataFrame([])


def nmi_target(df_feat: pd.DataFrame, df_target: pd.DataFrame,
               drop_constant_features: bool=True, **kwargs) -> pd.DataFrame:
    """
    Computes the Normalized Mutual Information (NMI) between a list of input features and a target variable.

    Args:
        df_feat: panda's DataFrame containing the input features for which the NMI with the target variable is to be
            computed.
        df_target: panda's DataFrame containing the target variable. This DataFrame should contain only one column and
            have the same size as df_feat.
        drop_constant_features: If True, the features that are constant across the entire data set will be dropped.
        **kwargs: Keyword arguments to be passed down to the mutual_info_regression function from scikit-learn. This
            can be useful e.g. for testing purposes.

    Returns:
        panda's DataFrame: panda's DataFrame containing the NMI between each of the input features and the target
            variable.
    """
    # Initial checks
    if df_target.shape[1] != 1:
        raise ValueError('The target DataFrame should have exactly one column.')
    if len(df_feat) != len(df_target):
        raise ValueError('The input features DataFrame and the target variable DataFrame should contain the same '
                         'number of data points.')

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
    """Computes the Normalized Mutual Information (NMI) between input features.

    Args:
        df_feat: panda's DataFrame containing the input features for which the NMI with the target variable is to be
            computed.
        **kwargs: Keyword arguments to be passed down to the mutual_info_regression function from scikit-learn. This
            can be useful e.g. for testing purposes.

    Returns:
        pd.DataFrame: panda's DataFrame containing the Normalized Mutual Information between features.
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


def get_features_relevance_redundancy(target_nmi: pd.DataFrame, cross_nmi: pd.DataFrame,
                                      n_feat: Union[None, int]=None, rr_parameters: Union[None, Dict]=None,
                                      return_pc: bool=False) -> List:
    """
    Select features from the Relevance Redundancy (RR) score between the input features and the target output.

    Args:
        target_nmi: panda's DataFrame containing the Normalized Mutual Information (NMI) between a list of input
            features and a target variable, as computed from :py:func:`nmi_target`.
        cross_nmi: panda's DataFrame containing the Normalized Mutual Information (NMI) between the input features, as
            computed from :py:func:`get_cross_nmi`.
        n_feat: Number of features for which the RR score needs to be computed (default: all features).
        rr_parameters: Allow to tune p and c parameters. Currently allows to fix p and c to constant values instead
            of using the dynamical evaluation.
        return_pc: Whether to return the p and c values in the output dictionaries.

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
        def get_p(nn):
            p = 4.5 - (nn ** 0.4) * 0.4
            return 0.1 if p < 0.1 else p

        def get_c(nn):
            c = 0.000001 * nn ** 3
            return 100000 if c > 100000 else c
    else:
        if 'p' not in rr_parameters or 'c' not in rr_parameters:
            raise ValueError('When tuning p and c with rr_parameters in get_features_relevance_redundancy, '
                             'both parameters should be tuned')
        # Set up p
        if rr_parameters['p']['function'] == 'constant':
            def get_p(nn):
                return rr_parameters['p']['value']
        else:
            raise ValueError('Allowed function for p : constant')
        # Set up c
        if rr_parameters['c']['function'] == 'constant':
            def get_c(nn):
                return rr_parameters['c']['value']
        else:
            raise ValueError('Allowed function for c : constant')

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
        p = get_p(nn=n)
        c = get_c(nn=n)

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


def get_features_dyn(n_feat,cross_mi, target_mi):

    first_feature =target_mi.nlargest(1).index[0]
    feature_set = [first_feature]

    if n_feat == -1:
        n_feat = len(cross_mi.index)

    for n in range(n_feat-1):
        if (n+1)%50 ==0:
            print("Selected {}/{} features...".format(n+1,n_feat))
        p = 4.5-(n**0.4)*0.4
        c = 0.000001*n**3
        if c > 100000:
            c=100000
        if p < 0.1:
            p=0.1

        score = cross_mi.copy()
        score = score.loc[target_mi.index, target_mi.index]
        score = score.drop(feature_set,axis=0)
        score = score[feature_set]

        for i in score.index:
            row = score.loc[i,:]
            score.loc[i,:] = target_mi[i] /(row**p+c)

        next_feature = score.min(axis=1).idxmax(axis=0)
        feature_set.append(next_feature)

    return feature_set

def merge_ranked(lists):
    zipped_lists = zip(*lists)
    total_set = set()
    ranked_list = []
    for subrank in zipped_lists:
        for feature in subrank:
            if feature not in total_set:
                ranked_list.append(feature)
                total_set.add(feature)
    return ranked_list


def featurize_composition(df):

    df = df.copy()
    df['composition'] = df['structure'].apply(lambda s: s.composition)
    featurizer = MultipleFeaturizer([ElementProperty.from_preset("magpie"),
                                     AtomicOrbitals(),
                                     BandCenter(),
                                     ElectronAffinity(),
                                     Stoichiometry(),
                                     ValenceOrbital(),
                                     IonProperty(),
                                     ElementFraction(),
                                     TMetalFraction(),
                                     CohesiveEnergy(),
                                     Miedema(),
                                     YangSolidSolution(),
                                     AtomicPackingEfficiency(),
                                     ])


    df = featurizer.featurize_dataframe(df,"composition",multiindex=True,ignore_errors=True)
    df.columns = df.columns.map('|'.join).str.strip('|')


    ox_featurizer = MultipleFeaturizer([OxidationStates(),
                                        ElectronegativityDiff()
                                        ])

    df = CompositionToOxidComposition().featurize_dataframe(df,"Input Data|composition")

    df = ox_featurizer.featurize_dataframe(df,"composition_oxid",multiindex=True,ignore_errors=True)
    df=df.rename(columns = {'Input Data':''})
    df.columns = df.columns.map('|'.join).str.strip('|')

    df['AtomicOrbitals|HOMO_character'] = df['AtomicOrbitals|HOMO_character'].map({'s':1,'p':2,'d':3,'f':4})
    df['AtomicOrbitals|LUMO_character'] = df['AtomicOrbitals|LUMO_character'].map({'s':1,'p':2,'d':3,'f':4})
    df['AtomicOrbitals|HOMO_element'] = df['AtomicOrbitals|HOMO_element'].apply(lambda x: -1 if not isinstance(x, str) else Element(x).Z)
    df['AtomicOrbitals|LUMO_element'] = df['AtomicOrbitals|LUMO_element'].apply(lambda x: -1 if not isinstance(x, str) else Element(x).Z)

    df = df.dropna(axis=1,how='all')
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    df = df.select_dtypes(include='number')
    return df


def featurize_structure(df):

    df = df.copy()
    prdf = PartialRadialDistributionFunction()
    prdf.fit(df["structure"])
    cm = CoulombMatrix()
    cm.fit(df["structure"])
    scm = SineCoulombMatrix()
    scm.fit(df["structure"])
    bf = BondFractions()
    bf.fit(df["structure"])
    bob =  BagofBonds()
    bob.fit(df["structure"])

    featurizer = MultipleFeaturizer([DensityFeatures(),
                                     GlobalSymmetryFeatures(),
                                     RadialDistributionFunction(),
                                     cm,
                                     scm,
                                     EwaldEnergy(),
                                     bf,
                                     StructuralHeterogeneity(),
                                     MaximumPackingEfficiency(),
                                     ChemicalOrdering(),
                                     XRDPowderPattern(),
                                     bob
                                     ])


    df = featurizer.featurize_dataframe(df,"structure",multiindex=True,ignore_errors=True)
    df.columns = df.columns.map('|'.join).str.strip('|')

    dist = df["RadialDistributionFunction|radial distribution function"][1]['distances'][:50]
    for i,d in enumerate(dist):
        df["RadialDistributionFunction|radial distribution function|d_{:.2f}".format(d)] = df["RadialDistributionFunction|radial distribution function"].apply(lambda x: x['distribution'][i])
    df = df.drop("RadialDistributionFunction|radial distribution function",axis=1)

    df["GlobalSymmetryFeatures|crystal_system"] = df["GlobalSymmetryFeatures|crystal_system"].map({"cubic":1, "tetragonal":2, "orthorombic":3, "hexagonal":4, "trigonal":5, "monoclinic":6, "triclinic":7})
    df["GlobalSymmetryFeatures|is_centrosymmetric"] = df["GlobalSymmetryFeatures|is_centrosymmetric"].map({True:1, False:0})


    df = df.dropna(axis=1,how='all')
    df = df.replace([np.inf, -np.inf, np.nan], -1)
    df = df.select_dtypes(include='number')
    return df

def featurize_site(df):

    df = df.copy()
    grdf = SiteStatsFingerprint(GeneralizedRadialDistributionFunction.from_preset('gaussian'),stats=('mean', 'std_dev')).fit(df["structure"])

    df.columns = ["Input data|"+x for x in df.columns]

    df = SiteStatsFingerprint(AGNIFingerprints(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["AGNIFingerPrint|"+x if '|' not in x else x for x in df.columns]


    df = SiteStatsFingerprint(OPSiteFingerprint(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["OPSiteFingerprint|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(CrystalNNFingerprint.from_preset("ops"),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["CrystalNNFingerprint|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(VoronoiFingerprint(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["VoronoiFingerprint|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(GaussianSymmFunc(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["GaussianSymmFunc" + x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(ChemEnvSiteFingerprint.from_preset("simple"),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["ChemEnvSiteFingerprint|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(CoordinationNumber(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["CoordinationNumber|"+x if '|' not in x else x for x in df.columns]

    df = grdf.featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["GeneralizedRDF|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(LocalPropertyDifference(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["LocalPropertyDifference|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(BondOrientationalParameter(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["BondOrientationParameter|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(AverageBondLength(VoronoiNN()),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["AverageBondLength|"+x if '|' not in x else x for x in df.columns]

    df = SiteStatsFingerprint(AverageBondAngle(VoronoiNN()),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["AverageBondAngle|"+x if '|' not in x else x for x in df.columns]

    df = df.dropna(axis=1,how='all')
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.replace([np.inf, -np.inf, np.nan], -1)
    df = df.select_dtypes(include='number')
    return df


class MODData():
    def __init__(self, structures: Union[None, List[Structure]], targets: List=[], names: List=[], mpids:List=[],
                 df_featurized=None):
        """

        Args:
            structures:
            targets:
            names:
            mpids:
            df_featurized: Allow to pass an already featurized DataFrame.
        """
        self.structures = structures
        self.df_featurized = df_featurized

        if len(targets)==0:
            self.prediction = True
        else:
            self.prediction = False

        if np.array(targets).ndim == 2:
            self.targets = targets
            self.PP = True
        else:
            self.targets = [targets]
            self.PP = False

        if len(names)>0:
            self.names = names
        else:
            self.names = ['prop'+str(i) for i in range(len(self.targets))]

        self.mpids = mpids

        if len(mpids)>0:
            self.ids = mpids
        else:
            self.ids = ['id'+str(i) for i in range(len(self.structures))]

        if not self.prediction:
            data = {'id':self.ids,}
            for i,target in enumerate(self.targets):
                data[self.names[i]] = target
            self.df_targets = pd.DataFrame(data)
            self.df_targets.set_index('id',inplace=True)
        self.df_structure = pd.DataFrame({'id':self.ids, 'structure':self.structures})
        self.df_structure.set_index('id',inplace=True)

    def featurize(self,fast=0,db_file='feature_database.pkl'):
        print('Computing features, this can take time...')
        if fast and len(self.mpids)>0:
            print('Fast featurization on, retrieving from database...')
            this_dir, this_filename = os.path.split(__file__)
            global database
            if len(database) == 0:
                database = pd.read_pickle(db_file)
            mpids_done = [x for x in self.mpids if x in database.index]
            print('Retrieved features for {} out of {} materials'.format(len(mpids_done),len(self.mpids)))
            df_done = database.loc[mpids_done]
            df_todo = self.df_structure.drop(mpids_done,axis=0)

            if len(df_todo) > 0 and len(df_done) > 0:
                df_composition = featurize_composition(df_todo)
                df_structure = featurize_structure(df_todo)
                df_site = featurize_site(df_todo)

                df_final = df_done.append(df_composition.join(df_structure.join(df_site)))
                df_final = df_final.reindex(self.mpids)
            elif len(df_todo) == 0:
                df_final = df_done
            else:
                df_composition = featurize_composition(self.df_structure)
                df_structure = featurize_structure(self.df_structure)
                df_site = featurize_site(self.df_structure)

                df_final = df_composition.join(df_structure.join(df_site))

        else:
            df_composition = featurize_composition(self.df_structure)
            df_structure = featurize_structure(self.df_structure)
            df_site = featurize_site(self.df_structure)

            df_final = df_composition.join(df_structure.join(df_site))
        df_final = df_final.replace([np.inf, -np.inf, np.nan], 0)
        self.df_featurized = df_final
        print('Data has successfully been featurized!')

    def feature_selection(self,n=1500, full_cross_nmi=None):
        """

        Args:
            n:
            full_cross_nmi: Allow to use a specific Cross-Features NMI.
        """
        assert hasattr(self, 'df_featurized'), 'Please featurize the data first'
        assert not self.prediction, 'Please provide targets'

        ranked_lists = []

        # Loading mutual information between features
        if full_cross_nmi is None:
            this_dir, this_filename = os.path.split(__file__)
            dp = os.path.join(this_dir, "data", "Features_cross")
            full_cross_nmi = pd.read_pickle(dp)
        else:
            full_cross_nmi = full_cross_nmi

        for i,name in enumerate(self.names):
            print("Starting target {}/{}: {} ...".format(i+1,len(self.targets),self.names[i]))

            # Computing mutual information with target
            print("Computing mutual information ...")
            df = self.df_featurized.copy()
            y_nmi = nmi_target(self.df_featurized,self.df_targets[[name]])[name]

            print('Computing optimal features...')
            cross_nmi = full_cross_nmi.copy(deep=True)

            a = []
            for x in cross_nmi.index:
                if x not in y_nmi.index:
                    a.append(x)
            cross_nmi = cross_nmi.drop(a,axis=0).drop(a,axis=1)
            # opt_features = get_features_dyn(min(n,len(cross_nmi.index)),cross_nmi,y_nmi)
            opt_features = get_features_dyn(min(n,len(y_nmi.index)),cross_nmi,y_nmi)
            ranked_lists.append(opt_features)
            print("Done with target {}/{}: {}.".format(i+1,len(self.targets),self.names[i]))

        print('Merging all features...')
        self.optimal_features = merge_ranked(ranked_lists)
        print('Done.')

    def shuffle(self):
        # caution, not fully implemented
        self.df_featurized =self.df_featurized.sample(frac=1)
        self.df_targets = self.df_targets.loc[data.df_featurized.index]

    def save(self, filename):
        """ Pickle the contents of the `MODData` object
        so that it can be loaded in  with `MODData.load()`.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be compressed accordingly by `pandas.to_pickle(...)`.

        """
        pd.to_pickle(self, filename)
        print(f'Data successfully saved as {filename}!')

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
