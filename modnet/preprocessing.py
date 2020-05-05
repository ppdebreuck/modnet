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
from typing import Dict, List
import pickle
import os
database = None

def nmi_target(df_feat,df_target):

    target_name = df_target.columns[0]
    mi_df = pd.DataFrame([],columns=[target_name],index=df_feat.columns)

    mi_df.loc[:,target_name] = (mutual_info_regression(df_feat,df_target[target_name]))
    S_mi = mutual_info_regression(df_target[target_name].values.reshape(-1, 1),df_target[target_name])[0]

    diag={}
    to_drop=[]
    for x in df_feat.columns:
            diag[x]=(mutual_info_regression(df_feat[x].values.reshape(-1, 1),df_feat[x]))[0]
            if diag[x] == 0:
                to_drop.append(x) # features which have an entropy of zero are useless

    mi_df.drop(to_drop,inplace=True)
    for x in mi_df.index:
        mi_df.loc[x,target_name] = mi_df.loc[x,target_name] / ((S_mi + diag[x])/2)


    return mi_df

def get_features_dyn(n_feat,cross_mi,target_mi):
  
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
                                     XRDPowderPattern()
                                    ])


    df = featurizer.featurize_dataframe(df,"structure",multiindex=True,ignore_errors=True)
    df.columns = df.columns.map('|'.join).str.strip('|')
    
    dist = df["RadialDistributionFunction|radial distribution function"][1]['distances'][:50]
    for i,d in enumerate(dist):
      df["RadialDistributionFunction|radial distribution function|d_{:.2f}".format(d)] = df["RadialDistributionFunction|radial distribution function"].apply(lambda x: x['distribution'][i])
    df = df.drop("RadialDistributionFunction|radial distribution function",axis=1)

    df["GlobalSymmetryFeatures|crystal_system"] = df["GlobalSymmetryFeatures|crystal_system"].map({"cubic":1, "tetragonal":2, "orthorombic":3, "hexagonal":4, "trigonal=":5, "monoclinic":6, "triclinic":7})
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
    def __init__(self,structures:List[Structure],targets:List=[],names:List=[],mpids:List=[]):
        self.structures = structures
        self.df_featurized = None
        
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
        
    def feature_selection(self,n=1500):
        
        assert hasattr(self, 'df_featurized'), 'Please featurize the data first'
        assert not self.prediction, 'Please provide targets'

        ranked_lists = []
        for i,name in enumerate(self.names):
            print("Starting target {}/{}: {} ...".format(i+1,len(self.targets),self.names[i]))
            
            # Computing mutual information with target
            print("Computing mutual information ...")
            df = self.df_featurized.copy()            
            y_nmi = nmi_target(self.df_featurized,self.df_targets[[name]])[name]
            
            print('Computing optimal features...')
            #Loading mutual information between features
            this_dir, this_filename = os.path.split(__file__)
            dp = os.path.join(this_dir, "data", "Features_cross")
            cross_mi = pd.read_pickle(dp)
            a = []
            for x in cross_mi.index:
                if x not in y_nmi.index:
                    a.append(x)
            cross_mi = cross_mi.drop(a,axis=0).drop(a,axis=1)
            opt_features = get_features_dyn(min(n,len(cross_mi.index)),cross_mi,y_nmi)
            ranked_lists.append(opt_features)
            print("Done with target {}/{}: {}.".format(i+1,len(self.targets),self.names[i]))
            
        print('Merging all features...')
        self.optimal_features = merge_ranked(ranked_lists)
        print('Done.')
        
    def shuffle(self):
        # caution, not fully implemented
        self.df_featurized =self.df_featurized.sample(frac=1)
        self.df_targets = self.df_targets.loc[data.df_featurized.index]
        
    def save(self,filename):
        fp = open(filename,'wb')
        pickle.dump(self,fp)
        fp.close()
        print('Data successfully saved!')
    
    @staticmethod
    def load(filename):
        fp = open(filename,'rb')
        data = pickle.load(fp)
        fp.close()
        return data
    
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