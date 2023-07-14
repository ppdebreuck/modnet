""" This submodule contains the `Matminer2023Featurizer` class. """

import numpy as np
import modnet.featurizers
import contextlib


class Matminer2023Featurizer(modnet.featurizers.MODFeaturizer):
    """A "kitchen-sink" featurizer for features implemented in matminer
    at time of creation (matminer v0.8.0 from late 2022/early 2023).

    Follows the same philosophy and featurizer list as the `DeBreuck2020Featurizer`
    but with with many features changing their underlying matminer implementation,
    definition and behaviour since the creation of the former featurizer.

    """

    def __init__(self, fast_oxid: bool = False, continuous_only: bool = False):
        """Creates the featurizer and imports all featurizer functions.

        Parameters:
            fast_oxid: Whether to use the accelerated oxidation state parameters within
                pymatgen when constructing features that constrain oxidation states such
                that all sites with the same species in a structure will have the same
                oxidation state (recommended if featurizing any structure
                with large unit cells).

        """

        super().__init__()
        self.continuous_only = continuous_only
        self.fast_oxid = fast_oxid
        self.load_featurizers()

    def load_featurizers(self):
        with contextlib.redirect_stdout(None):
            from pymatgen.analysis.local_env import VoronoiNN
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
            from matminer.featurizers.structure import (
                # BagofBonds, - This descriptor was not used in the paper preset
                BondFractions,
                ChemicalOrdering,
                CoulombMatrix,
                DensityFeatures,
                EwaldEnergy,
                GlobalSymmetryFeatures,
                MaximumPackingEfficiency,
                # PartialRadialDistributionFunction,
                RadialDistributionFunction,
                SineCoulombMatrix,
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

            if self.continuous_only:
                magpie_featurizer = ElementProperty.from_preset("magpie")
                magpie_featurizer.stats = ["mean", "avg_dev"]

                self.composition_featurizers = (
                    BandCenter(),
                    ElementFraction(),
                    magpie_featurizer,
                    IonProperty(fast=self.fast_oxid),
                    Stoichiometry(p_list=[2, 3, 5, 7, 10]),
                    TMetalFraction(),
                    ValenceOrbital(props=["frac"]),
                )
            else:
                self.composition_featurizers = (
                    AtomicOrbitals(),
                    AtomicPackingEfficiency(),
                    BandCenter(),
                    ElementFraction(),
                    ElementProperty.from_preset("magpie"),
                    IonProperty(),
                    Miedema(),
                    Stoichiometry(),
                    TMetalFraction(),
                    ValenceOrbital(),
                    YangSolidSolution(),
                )

            self.oxid_composition_featurizers = (
                ElectronegativityDiff(),
                OxidationStates(),
            )

            self.structure_featurizers = (
                DensityFeatures(),
                GlobalSymmetryFeatures(),
                RadialDistributionFunction(),
                CoulombMatrix(),
                # PartialRadialDistributionFunction(),
                SineCoulombMatrix(),
                EwaldEnergy(),
                BondFractions(),
                StructuralHeterogeneity(),
                MaximumPackingEfficiency(),
                ChemicalOrdering(),
                XRDPowderPattern(),
                # BagofBonds(),
            )

            # Patch for matminer: see https://github.com/hackingmaterials/matminer/issues/864
            self.structure_featurizers[0].desired_features = None
            self.structure_featurizers[1].desired_features = None

            self.site_featurizers = (
                AGNIFingerprints(),
                AverageBondAngle(VoronoiNN()),
                AverageBondLength(VoronoiNN()),
                BondOrientationalParameter(),
                ChemEnvSiteFingerprint.from_preset("simple"),
                CoordinationNumber(),
                CrystalNNFingerprint.from_preset("ops"),
                GaussianSymmFunc(),
                GeneralizedRadialDistributionFunction.from_preset("gaussian"),
                LocalPropertyDifference(),
                OPSiteFingerprint(),
                VoronoiFingerprint(),
            )

    def featurize_composition(self, df):
        """Applies the preset composition featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """
        from pymatgen.core.periodic_table import Element

        df = super().featurize_composition(df)

        if not self.continuous_only:
            _orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}
            df["AtomicOrbitals|HOMO_character"] = df[
                "AtomicOrbitals|HOMO_character"
            ].map(_orbitals)
            df["AtomicOrbitals|LUMO_character"] = df[
                "AtomicOrbitals|LUMO_character"
            ].map(_orbitals)

            df["AtomicOrbitals|HOMO_element"] = df["AtomicOrbitals|HOMO_element"].apply(
                lambda x: -1 if not isinstance(x, str) else Element(x).Z
            )
            df["AtomicOrbitals|LUMO_element"] = df["AtomicOrbitals|LUMO_element"].apply(
                lambda x: -1 if not isinstance(x, str) else Element(x).Z
            )

        else:
            df.drop(columns=["IonProperty|max ionic char"], inplace=True)

        return modnet.featurizers.clean_df(df, drop_allnan=self.drop_allnan)

    def featurize_structure(self, df):
        """Applies the preset structural featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """

        if self.structure_featurizers:
            df = super().featurize_structure(df)

        _crystal_system = {
            "cubic": 1,
            "tetragonal": 2,
            "orthorombic": 3,
            "hexagonal": 4,
            "trigonal": 5,
            "monoclinic": 6,
            "triclinic": 7,
        }

        def _int_map(x):
            if x == np.nan:
                return 0
            elif x:
                return 1
            else:
                return 0

        df["GlobalSymmetryFeatures|crystal_system"] = df[
            "GlobalSymmetryFeatures|crystal_system"
        ].map(_crystal_system)
        df["GlobalSymmetryFeatures|is_centrosymmetric"] = df[
            "GlobalSymmetryFeatures|is_centrosymmetric"
        ].map(_int_map)

        return modnet.featurizers.clean_df(df, drop_allnan=self.drop_allnan)

    def featurize_site(self, df):
        """Applies the preset site featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """

        # rename some features for backwards compatibility with pretrained models
        aliases = {
            "GeneralizedRadialDistributionFunction": "GeneralizedRDF",
            "AGNIFingerprints": "AGNIFingerPrint",
            "BondOrientationalParameter": "BondOrientationParameter",
        }
        df = super().featurize_site(df, aliases=aliases)
        df = df.loc[:, (df != 0).any(axis=0)]

        return modnet.featurizers.clean_df(df, drop_allnan=self.drop_allnan)


class CompositionOnlyMatminer2023Featurizer(Matminer2023Featurizer):
    """This subclass simply disables structure and site-level features
    from the main `Matminer2023Featurizer` class.

    This should yield identical results to the original 2020 version.

    """

    def __init__(self, continuous_only: bool = False, fast_oxid: bool = False):
        super().__init__(fast_oxid=fast_oxid, continuous_only=continuous_only)
        self.oxid_composition_featurizers = ()
        self.structure_featurizers = ()
        self.site_featurizers = ()
