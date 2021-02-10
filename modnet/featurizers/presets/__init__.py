__all__ = ("FEATURIZER_PRESETS",)

from .debreuck_2020 import DeBreuck2020Featurizer, CompositionOnlyFeaturizer

FEATURIZER_PRESETS = {
    "DeBreuck2020": DeBreuck2020Featurizer,
    "CompositionOnly": CompositionOnlyFeaturizer,
}
