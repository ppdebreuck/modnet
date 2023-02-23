""" This submodule contains the `Matminer2023Featurizer` class. """

import numpy as np
import modnet.featurizers
import contextlib


class MatminerAll2023Featurizer(modnet.featurizers.MODFeaturizer):
    """A "kitchen-sink" featurizer for features implemented in matminer
    at time of creation (matminer v0.8.0 from late 2022/early 2023).

    Follows the same philosophy and featurizer list as the `DeBreuck2020Featurizer`
    but with many features changing their underlying matminer implementation,
    definition and behaviour since the creation of the former featurizer.

    """

    def __init__(self, fast_oxid: bool = False):
        """Creates the featurizer and imports all featurizer functions.

        Parameters:
            fast_oxid: Whether to use the accelerated oxidation state parameters within
                pymatgen when constructing features that constrain oxidation states such
                that all sites with the same species in a structure will have the same
                oxidation state (recommended if featurizing any structure
                with large unit cells).

        """

        super().__init__()
        self.fast_oxid = fast_oxid
        self.load_featurizers()

    def load_featurizers(self):
        with contextlib.redirect_stdout(None):
            from pymatgen.analysis.local_env import VoronoiNN
            from matminer.featurizers.composition import (
                AtomicOrbitals,
                AtomicPackingEfficiency,
                BandCenter,
                CationProperty,
                ElectronAffinity,
                ElectronegativityDiff,
                ElementFraction,
                ElementProperty,
                IonProperty,
                Miedema,
                OxidationStates,
                Stoichiometry,
                TMetalFraction,
                ValenceOrbital,
                WenAlloys,
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

            # Get additional ElementProperty featurizer, but
            # get only the features that are not yet present with another featurizer.
            # Also in the case of continuous features, use only the mean and avg_dev.
            from matminer.utils.data import PymatgenData, DemlData
            magpie_featurizer = ElementProperty.from_preset("magpie")
            magpie_featurizer.stats = ["mean", "avg_dev"]

            pymatgen_features = [
                "block",
                "mendeleev_no",
                "electrical_resistivity",
                "velocity_of_sound",
                "thermal_conductivity",
                "bulk_modulus",
                "coefficient_of_linear_thermal_expansion",
            ]
            pymatgen_featurizer = ElementProperty(
                data_source=PymatgenData(),
                stats=["mean", "avg_dev"],
                features=pymatgen_features,
            )

            deml_features = [
                "atom_radius",
                "molar_vol",
                "heat_fusion",
                "boiling_point",
                "heat_cap",
                "first_ioniz",
                "electric_pol",
                "GGAU_Etot",
                "mus_fere",
                "FERE correction",
            ]
            deml_featurizer = ElementProperty(
                data_source=DemlData(),
                stats=["mean", "avg_dev"],
                features=deml_features,
            )

            self.composition_continuous_featurizers = (
                BandCenter(),
                ElementFraction(),
                magpie_featurizer,
                pymatgen_featurizer,
                deml_featurizer,
                Stoichiometry(p_list=[2, 3, 5, 7, 10]),
                TMetalFraction(),
                ValenceOrbital(props=["frac"]),
                WenAlloys(),
            )

            # Get back the initial presets from Matminer, without the duplicate features from Magpie
            pymatgen_featurizer_full = ElementProperty(
                data_source=PymatgenData(),
                stats=["minimum", "maximum", "range", "mean", "std_dev"],
                features=pymatgen_features,
            )

            deml_featurizer_full = ElementProperty(
                data_source=DemlData(),
                stats=["minimum", "maximum", "range", "mean", "std_dev"],
                features=deml_features,
            )

            self.composition_featurizers = (
                AtomicOrbitals(),
                AtomicPackingEfficiency(),
                BandCenter(),
                ElementFraction(),
                ElementProperty.from_preset("magpie"),
                pymatgen_featurizer_full,
                deml_featurizer_full,
                Miedema(),
                Stoichiometry(),
                TMetalFraction(),
                ValenceOrbital(),
                WenAlloys(),
            )

            self.oxid_composition_continuous_featurizers = (
                IonProperty(fast=self.fast_oxid),
                OxidationStates(stats=["mean"]),
            )

            self.oxid_composition_featurizers = (
                CationProperty.from_preset("deml"),
                ElectronAffinity(),
                ElectronegativityDiff(),
                IonProperty(fast=self.fast_oxid),
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

        if self.composition_featurizers:
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

        if self.composition_continuous_featurizers:
            df.drop(
                columns=[
                    "WenAlloys|Yang omega",
                    "WenAlloys|Yang delta",
                    "WenAlloys|Radii gamma",
                    "WenAlloys|Lambda entropy",
                    "WenAlloys|APE mean",
                    "WenAlloys|Interant electrons",
                    "WenAlloys|Interant s electrons",
                    "WenAlloys|Interant p electrons",
                    "WenAlloys|Interant d electrons",
                    "WenAlloys|Interant f electrons",
                    "WenAlloys|Atomic weight mean",
                    "WenAlloys|Total weight",
                ],
                inplace=True
            )

        if self.oxid_composition_continuous_featurizers:
            df.drop(columns=["IonProperty|max ionic char"], inplace=True)

        return modnet.featurizers.clean_df(df)

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

        return modnet.featurizers.clean_df(df)

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

        return modnet.featurizers.clean_df(df)


class CompositionOnlyMatminerAll2023Featurizer(MatminerAll2023Featurizer):
    """This subclass simply disables structure and site-level features
    from the main `Matminer2023Featurizer` class.

    This should yield identical results to the original 2020 version.

    """

    def __init__(self, continuous_only: bool = False, oxidation_featurizers: bool = False, fast_oxid: bool = False):
        super().__init__(fast_oxid=fast_oxid)
        self.fast_oxid = fast_oxid
        self.structure_featurizers = ()
        self.site_featurizers = ()
        if continuous_only:
            self.composition_featurizers = ()
        else:
            self.composition_continuous_featurizers = ()

        if oxidation_featurizers:
            if continuous_only:
                self.oxid_composition_featurizers = ()
            else:
                self.oxid_composition_continuous_featurizers = ()
        else:
            self.oxid_composition_featurizers = ()
            self.oxid_composition_continuous_featurizers = ()
