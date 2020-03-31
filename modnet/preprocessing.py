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
                #print("WARNING: Entropy {} is zero".format(x),flush=True)

    mi_df.drop(to_drop,inplace=True)
    for x in mi_df.index:
        mi_df.loc[x,target_name] = mi_df.loc[x,target_name] / ((S_mi + diag[x])/2)


    return mi_df

def get_features_dyn(n_feat,cross_mi,target_mi):
  
    first_feature =target_mi.nlargest(1).index[0]
    feature_set = [first_feature]

    for n in range(n_feat-1):
        if (n+1)%50 ==0:
            print("Selected {}/{} features...".format(n+1,n_feat))
        p = 4.5-(n**0.4)*0.4
        c = 0.000001*n**3
        if c > 100000:
            c=100000
        if p < 0.1:
            p=0.1
            
        #0: p = 6-2*math.log10(n+1)
        #0: c = 0.0001*(2**(n/10))
        
        #1: p = 6-2*math.log10(n+1)
        #1: c = 0.0001*(4**(n/10))
        
        #print((n,p,c))
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
    #df = pd.DataFrame({'structure':structures, 'composition':[s.composition for s in structures]})
    
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
                                 CohesiveEnergy(), # equals the formation energy
                                 Miedema(), # Formation enthalpies
                                 YangSolidSolution(),
                                 AtomicPackingEfficiency(),
                                ])


    df = featurizer.featurize_dataframe(df,"composition",multiindex=True,ignore_errors=True)
    df.columns = df.columns.map('|'.join).str.strip('|')


    ox_featurizer = MultipleFeaturizer([OxidationStates(),
                                    ElectronegativityDiff()
                                ])


    #df["composition_oxid"] = composition_to_oxidcomposition(df["Input Data|composition"])
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
    # todo drop input data (structure,composition)
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
                                     #Dimensionality(),
                                     RadialDistributionFunction(),
                                     #prdf,
                                     #ElectronicRadialDistributionFunction(),
                                     cm,
                                     scm,
                                     #OrbitalFieldMatrix(),
                                     #MinimumRelativeDistances(),
                                     #SiteStatsFingerprint(),
                                     EwaldEnergy(),
                                     bf, # add bond type name, here no distinction                                 
                                     #bob,
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
    
    # todo drop input data (structure)
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
    
    #SiteStatsFingerprint(AngularFourierSeries.from_preset('gaussian'),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    #df.columns = ["AngularFourierSeries|"+x if '|' not in x else x for x in df.columns]
    
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
    
    # todo drop input data (structure)
    return df

class MODData():
    def __init__(self,structures:List[Structure],targets:List,names:List=[],mpids:List=[]):
        self.structures = structures
        self.df_featurized = None
        
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
        
        data = {'id':self.ids,}
        for i,target in enumerate(self.targets):
            data[self.names[i]] = target
        self.df_targets = pd.DataFrame(data)
        self.df_targets.set_index('id',inplace=True)
        self.df_structure = pd.DataFrame({'id':self.ids, 'structure':self.structures})
        self.df_structure.set_index('id',inplace=True)

    def featurize(self,fast=0):
        ####
        # TODO fast part taking already computed structures from file
        ####
        print('Computing features, this can take time...')
        if fast and len(self.mpids)>0:
            print('Fast featurization on, retrieving from database...')
            database = pd.read_pickle('../data/feature_database.pkl')
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
            #for name, target in zip(self.names,self.targets):
            #    df_final['Target|'+name] = target
            
        self.df_featurized = df_final
        print('Data has successfully been featurized!')
        
    def feature_selection(self,n=300):
        
        assert hasattr(self, 'df_featurized'), 'Please featurize the data first'

        ranked_lists = []
        for i,name in enumerate(self.names):
            print("Starting target {}/{}: {} ...".format(i+1,len(self.targets),self.names[i]))
            
            # Computing mutual information with target
            print("Computing mutual information ...")
            df = self.df_featurized.copy()
            
            #a = []
            #for x in df.columns:
            #    if 'Target' in x:
            #        a.append(x)
            #a.remove('Target|' + name)
            #df = df.drop(a,axis=1)
            
            y_nmi = nmi_target(self.df_featurized,self.df_targets[[name]])[name]
            #y_nmi = df1['Target|'+name]
            #y_nmi.drop('Target|'+name, inplace=True)
            
            print('Computing optimal features...')
            #Loading mutual information between features
            cross_mi = pd.read_pickle('../data/Features_cross')
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

    def get_optimal_features(self):
        return self.optimal_features
    
    def get_optimal_df(self):
        return self.df_featurized[self.optimal_features].join(self.targets)