"""This submodule contains the `Matminer2024FastFeaturizer` class. """

import numpy as np
import modnet.featurizers
import contextlib


class Matminer2024FastFeaturizer(modnet.featurizers.MODFeaturizer):
    """A set of efficient featurizers for features implemented in matminer
    at time of creation (matminer v0.9.2 from 2024).

    Removes featurizers that are known to be slow (i.e., orders of magnitude
    more intensive to compute than the rest of the featurizers).

    """

    def __init__(
        self,
        fast_oxid: bool = True,
        continuous_only: bool = True,
    ):
        """Creates the featurizer and imports all featurizer functions.

        Parameters:
            fast_oxid: Whether to use the accelerated oxidation state parameters within
                pymatgen when constructing features that constrain oxidation states such
                that all sites with the same species in a structure will have the same
                oxidation state (recommended if featurizing any structure
                with large unit cells).
            continuous_only: Whether to keep only the features that are continuous
                with respect to the composition (only for composition featurizers).
                Discontinuous features may lead to discontinuities in the model predictions.

        """

        super().__init__()
        self.drop_allnan = False
        self.fast_oxid = fast_oxid
        self.continuous_only = continuous_only
        self.load_featurizers()

    def load_featurizers(self):
        with contextlib.redirect_stdout(None):
            from matminer.featurizers.composition import (
                BandCenter,
                ElementFraction,
                ElementProperty,
                Stoichiometry,
                TMetalFraction,
                ValenceOrbital,
            )
            from matminer.featurizers.structure import (
                DensityFeatures,
                EwaldEnergy,
                GlobalSymmetryFeatures,
                StructuralComplexity,
            )
            from matminer.utils.data import (
                DemlData,
                PymatgenData,
            )

            pymatgen_features = [
                "block",
                "mendeleev_no",
                "electrical_resistivity",
                "velocity_of_sound",
                "thermal_conductivity",
                "bulk_modulus",
                "coefficient_of_linear_thermal_expansion",
            ]

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

            magpie_featurizer = ElementProperty.from_preset("magpie")
            magpie_featurizer.stats = ["mean", "avg_dev"]

            pymatgen_featurizer = ElementProperty(
                data_source=PymatgenData(),
                stats=["mean", "avg_dev"],
                features=pymatgen_features,
            )

            deml_featurizer = ElementProperty(
                data_source=DemlData(),
                stats=["mean", "avg_dev"],
                features=deml_features,
            )

            self.composition_featurizers = (
                BandCenter(),
                ElementFraction(),
                magpie_featurizer,
                pymatgen_featurizer,
                deml_featurizer,
                Stoichiometry(p_list=[2, 3, 5, 7, 10]),
                TMetalFraction(),
                ValenceOrbital(props=["frac"]),
            )

            self.oxid_composition_featurizers = []

            self.structure_featurizers = (
                DensityFeatures(),
                EwaldEnergy(),
                GlobalSymmetryFeatures(),
                StructuralComplexity(),
            )

            self.site_featurizers = []

    def featurize_composition(self, df):
        """Applies the preset composition featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """
        from pymatgen.core.periodic_table import Element

        df = super().featurize_composition(df)

        if self.composition_featurizers and not self.continuous_only:
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

        if self.continuous_only:
            # These are additional features that have shown discontinuities in my tests.
            # Hopefully, I got them all...
            df.drop(
                columns=[
                    "ElementProperty|DemlData mean electric_pol",
                    "ElementProperty|DemlData mean FERE correction",
                    "ElementProperty|DemlData mean GGAU_Etot",
                    "ElementProperty|DemlData mean heat_fusion",
                    "ElementProperty|DemlData mean mus_fere",
                ],
                inplace=True,
                errors="ignore",
            )

            if self.oxid_composition_featurizers:
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
