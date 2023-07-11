__all__ = (
    "FEATURIZER_PRESETS",
    "DEFAULT_FEATURIZER",
    "DEFAULT_COMPOSITION_ONLY_FEATURIZER",
)

from typing import Dict, Type
from .debreuck_2020 import DeBreuck2020Featurizer, CompositionOnlyFeaturizer
from .matminer_2023 import Matminer2023Featurizer, CompositionOnlyMatminer2023Featurizer
from .matminer_all_2023 import (
    MatminerAll2023Featurizer,
    CompositionOnlyMatminerAll2023Featurizer,
)
from modnet.featurizers import MODFeaturizer

DEFAULT_FEATURIZER: str = "Matminer2023"
DEFAULT_COMPOSITION_ONLY_FEATURIZER: str = "CompositionOnlyMatminer2023"

FEATURIZER_PRESETS: Dict[str, Type[MODFeaturizer]] = {
    "DeBreuck2020": DeBreuck2020Featurizer,
    "CompositionOnly": CompositionOnlyFeaturizer,
    "Matminer2023": Matminer2023Featurizer,
    "MatminerAll2023": MatminerAll2023Featurizer,
    "CompositionOnlyMatminer2023": CompositionOnlyMatminer2023Featurizer,
    "CompositionOnlyMatminerAll2023": CompositionOnlyMatminerAll2023Featurizer,
}
