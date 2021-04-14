""" This submodule contains the DeBreuck2020Featurizer class. """

import numpy as np
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import VoronoiNN
import modnet.featurizers


class DeBreuck2020Featurizer(modnet.featurizers.MODFeaturizer):
    """Featurizer presets used for the paper 'Machine learning
    materials properties for small datasets' by Pierre-Paul De Breuck,
    Geoffroy Hautier & Gian-Marco Rignanese, arXiv:2004.14766 (2020).

    Uses most of the featurizers implemented by matminer at the time of
    writing with their default hyperparameters and presets.

    """

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

    composition_featurizers = (
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

    oxid_composition_featurizers = (
        ElectronegativityDiff(),
        OxidationStates(),
    )

    structure_featurizers = (
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
    site_featurizers = (
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

    def __init__(self, fast_oxid=False):
        super().__init__()
        self.fast_oxid = fast_oxid

    def featurize_composition(self, df):
        """Applies the preset composition featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """
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

        return modnet.featurizers.clean_df(df)

    def featurize_structure(self, df):
        """Applies the preset structural featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """

        df = super().featurize_structure(df)

        dist = df["RadialDistributionFunction|radial distribution function"].iloc[0][
            "distances"
        ][:50]
        for i, d in enumerate(dist):
            _rdf_key = "RadialDistributionFunction|radial distribution function|d_{:.2f}".format(
                d
            )
            df[_rdf_key] = df[
                "RadialDistributionFunction|radial distribution function"
            ].apply(lambda x: x["distribution"][i])

        df = df.drop("RadialDistributionFunction|radial distribution function", axis=1)

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


class CompositionOnlyFeaturizer(DeBreuck2020Featurizer):
    oxid_composition_featurizers = ()
    structure_featurizers = ()
    site_featurizers = ()
