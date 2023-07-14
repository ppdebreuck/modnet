""" This submodule contains the DeBreuck2020Featurizer class. """

import numpy as np
import modnet.featurizers
import contextlib
import warnings


class DeBreuck2020Featurizer(modnet.featurizers.MODFeaturizer):
    """Featurizer presets used for the paper

        **Materials property prediction for limited datasets enabled
        by feature selection and joint learning with MODNet**,
        Pierre-Paul De Breuck, Geoffroy Hautier & Gian-Marco Rignanese
        npj Comp. Mat. 7(1) 1-8 (2021)
        10.1038/s41524-021-00552-2

    Uses most of the featurizers implemented by matminer at the time of
    writing with their default hyperparameters and presets.

    """

    package_version_requirements = {"matminer": "==0.6.2"}

    def __init__(self, fast_oxid: bool = False):
        """Creates the featurizer and imports all featurizer functions.

        Parameters:
            fast_oxid: Whether to use the accelerated oxidation state parameters within
                pymatgen when constructing features that constrain oxidation states such
                that all sites with the same species in a structure will have the same
                oxidation state (recommended if featurizing any structure
                with large unit cells).

        """
        import matminer

        if matminer.__version__ != self.package_version_requirements[
            "matminer"
        ].replace("==", ""):
            warnings.warn(
                f"The {self.__class__.__name__} preset was written for and tested only with matminer{self.package_version_requirements['matminer']}.\n"
                "Newer versions of matminer will not work, and older versions may not be compatible with newer MODNet versions due to other conflicts.\n"
                "To use this featurizer robustly, please install `modnet==0.1.13` with its pinned dependencies.\n\n"
                "This preset will now be initialised without importing matminer featurizers to enable use with existing previously featurized data, "
                "but attempts to perform further featurization will result in an error."
            )

        else:
            super().__init__()
            self.load_featurizers()
            self.fast_oxid = fast_oxid

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

        _orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}
        df["AtomicOrbitals|HOMO_character"] = df["AtomicOrbitals|HOMO_character"].map(
            _orbitals
        )
        df["AtomicOrbitals|LUMO_character"] = df["AtomicOrbitals|LUMO_character"].map(
            _orbitals
        )

        df["AtomicOrbitals|HOMO_element"] = df["AtomicOrbitals|HOMO_element"].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )
        df["AtomicOrbitals|LUMO_element"] = df["AtomicOrbitals|LUMO_element"].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )

        return modnet.featurizers.clean_df(df, drop_allnan=self.drop_allnan)

    def featurize_structure(self, df):
        """Applies the preset structural featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """

        df = super().featurize_structure(df)

        if "RadialDistributionFunction|radial distribution function" in df:
            dist = df["RadialDistributionFunction|radial distribution function"].iloc[
                0
            ]["distances"][:50]
            for i, d in enumerate(dist):
                _rdf_key = "RadialDistributionFunction|radial distribution function|d_{:.2f}".format(
                    d
                )
                df[_rdf_key] = df[
                    "RadialDistributionFunction|radial distribution function"
                ].apply(lambda x: x["distribution"][i])

            df = df.drop(
                "RadialDistributionFunction|radial distribution function", axis=1
            )

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


class CompositionOnlyFeaturizer(DeBreuck2020Featurizer):
    """This subclass simply disables structure and site-level features
    from the main `DeBreuck2020Featurizer` class.

        **Materials property prediction for limited datasets enabled
        by feature selection and joint learning with MODNet**
        Pierre-Paul De Breuck, Geoffroy Hautier & Gian-Marco Rignanese
        npj Comp. Mat. 7(1) 1-8 (2021)
        10.1038/s41524-021-00552-2

    Uses most of the featurizers implemented by matminer at the time of
    writing with their default hyperparameters and presets.

    """

    def __init__(self):
        super().__init__()
        self.oxid_composition_featurizers = ()
        self.structure_featurizers = ()
        self.site_featurizers = ()
