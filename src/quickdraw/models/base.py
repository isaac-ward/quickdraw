"""Back-compat shim. The data-space model now lives in `dsar.py` as `DataSpaceAR`, built on the
shared `SequenceWorldModel` ancestor. `BaseWorldModel`/`BaseModelConfig` are kept as aliases so
existing imports (training.setup, training.lit, eval, checkpoints) keep working unchanged.
"""

from __future__ import annotations

from .dsar import BaseModelConfig, DataSpaceAR
from .dsar import DataSpaceAR as BaseWorldModel

__all__ = ["BaseModelConfig", "BaseWorldModel", "DataSpaceAR"]
