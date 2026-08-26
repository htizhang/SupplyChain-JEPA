"""Public contracts shared by every raw-to-canonical data adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

SourceKind = Literal["observational", "simulator"]


@dataclass(frozen=True)
class AdapterRequest:
    """Runtime inputs for one source conversion.

    ``raw_root`` points to separately acquired observational data. ``source_root``
    points to a pinned simulator checkout. Adapters only use the field relevant to
    their source kind; neither path is stored in model tensors.
    """

    raw_root: Path | None = None
    source_root: Path | None = None
    limit: int | None = None
    pilot: bool = False
    worlds: int = 8
    horizon: int = 32
    seed: int = 20260730
    include_state: bool = True
    include_dynamics: bool | None = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative or None")
        if self.worlds < 1:
            raise ValueError("worlds must be positive")
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")


class SourceAdapter(ABC):
    """Convert one external source into canonical SupplyChain-JEPA episodes."""

    source_id: ClassVar[str]
    source_kind: ClassVar[SourceKind]
    supports_state: ClassVar[bool] = True
    supports_dynamics: ClassVar[bool] = False

    @abstractmethod
    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        """Return deterministic canonical records for ``request``."""

    def resolved_dynamics_flag(self, request: AdapterRequest) -> bool:
        requested = (
            self.supports_dynamics if request.include_dynamics is None else request.include_dynamics
        )
        if requested and not self.supports_dynamics:
            raise ValueError(f"{self.source_id} does not support Dynamics JEPA")
        return bool(requested)


@dataclass(frozen=True)
class AdapterDescriptor:
    """Lazy registry entry exposed without importing optional dependencies."""

    source_id: str
    source_kind: SourceKind
    supports_state: bool
    supports_dynamics: bool
    class_path: str
    description: str
