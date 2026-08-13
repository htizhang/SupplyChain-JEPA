"""Typed container for a complete pretrained SupplyChain-JEPA release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from .data import EpisodeTensorizer
from .model import (
    SemanticConceptSlotPredictor,
    StatePoolPredictor,
    SupplyChainJEPACore,
)


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model modules and the exact preprocessing contract they require."""

    model: SupplyChainJEPACore
    state_pool_predictor: StatePoolPredictor | None
    semantic_concept_slot_predictor: SemanticConceptSlotPredictor | None
    grounding_head: nn.Linear
    tensorizer: EpisodeTensorizer
    config: dict[str, Any]
    tensorizer_config: dict[str, Any]
    schema: dict[str, Any]
    normalization: dict[str, Any]
    release_manifest: dict[str, Any]
    source: Path
    revision: str | None
