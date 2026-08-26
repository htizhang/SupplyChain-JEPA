"""Shared deterministic builders for factual observational adapters."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .base import AdapterRequest, SourceAdapter
from .common import (
    ADAPTER_VERSION,
    SCHEMA_VERSION,
    make_record_id,
    observation_status,
    recursive_numbers_are_finite,
    stable_digest,
)

PILOT_DEFAULT_LIMIT = 8
MAX_STEPS = 64
OBSERVATIONAL_ADAPTER_VERSION = f"{ADAPTER_VERSION}-observational-v1"


@dataclass(frozen=True)
class SourceMetadata:
    """Published metadata for one separately acquired observational source."""

    name: str
    version: str
    source_repository: str
    license: str


class ObservationalAdapter(SourceAdapter):
    """Base contract for finalized records with no intervention semantics."""

    source_kind = "observational"
    supports_state = True
    supports_dynamics = False

    def prepare_request(self, request: AdapterRequest) -> tuple[Path | None, int | None]:
        """Validate objective flags and return the source-local root and limit."""
        self.resolved_dynamics_flag(request)
        limit = effective_limit(request.limit, request.pilot)
        if limit == 0:
            return None, 0
        if request.raw_root is None:
            raise ValueError(f"{self.source_id} requires request.raw_root")
        raw_root = request.raw_root.expanduser()
        if not raw_root.is_dir():
            raise FileNotFoundError(f"{self.source_id} raw_root is not a directory: {raw_root}")
        return raw_root, limit


def effective_limit(limit: int | None, pilot: bool) -> int | None:
    """Apply the bounded pilot default while preserving explicit limits."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return 0
    if limit is not None:
        return limit
    return PILOT_DEFAULT_LIMIT if pilot else None


def required_file(raw_root: Path, relative_path: str) -> Path:
    """Resolve a required source-local file without workspace-specific fallbacks."""
    path = raw_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required raw file not found: {path}")
    return path


def stable_split(source_id: str, world_id: str) -> str:
    bucket = int(stable_digest("v2-real-split", source_id, world_id, length=8), 16) % 100
    if bucket < 90:
        return "pretrain_train"
    if bucket < 95:
        return "pretrain_dev"
    return "pretrain_test"


def stable_seed(source_id: str, world_id: str) -> int:
    return int(stable_digest("v2-real-seed", source_id, world_id, length=8), 16)


def finite_value(value: Any) -> Any:
    """Convert pandas scalars/timestamps while replacing missing values with None."""
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        if not pd.notna(value):
            return None
        return float(value)
    if isinstance(value, int):
        return int(value)
    return str(value)


def features(**values: Any) -> dict[str, Any]:
    return {key: finite_value(value) for key, value in values.items()}


def clock_hours(value: Any, origin: pd.Timestamp) -> float:
    timestamp = pd.Timestamp(value)
    return max(0.0, float((timestamp - origin).total_seconds() / 3600.0))


def ranked_groups(
    groups: Iterable[tuple[str, pd.DataFrame]],
    *,
    source_id: str,
    limit: int | None,
) -> list[tuple[str, pd.DataFrame]]:
    """Select complete multi-step groups by stable content-independent ranking."""
    eligible = [(str(key), frame) for key, frame in groups if len(frame) >= 2]
    eligible.sort(key=lambda pair: stable_digest("v2-real-sample", source_id, pair[0]))
    return eligible if limit is None else eligible[:limit]


def make_record(
    *,
    source_id: str,
    metadata: SourceMetadata,
    world_id: str,
    time_unit: str,
    topology_nodes: list[dict[str, Any]],
    topology_edges: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    source_files: list[str],
    domain_parameters: dict[str, Any],
    state_eligible: bool,
) -> dict[str, Any]:
    """Build one canonical episode from finalized, non-causal observations."""
    if len(steps) < 2:
        raise ValueError(f"{source_id}:{world_id} has fewer than two steps")
    for index, step_value in enumerate(steps):
        step_value["t"] = index
        step_value["terminal"] = index == len(steps) - 1

    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": make_record_id(source_id, world_id, None),
        "simulator": {
            "family": source_id,
            "name": metadata.name,
            "version": metadata.version,
            "source_repository": metadata.source_repository,
            "source_commit": "not_applicable_observational_dataset",
            "license": metadata.license,
        },
        "split": stable_split(source_id, world_id),
        "episode": {
            "episode_id": f"{world_id}:single",
            "world_id": world_id,
            "config_id": f"{source_id}:observational-v1",
            "seed": stable_seed(source_id, world_id),
            "time_unit": time_unit,
            "horizon": len(steps),
            "exogenous_stream_id": f"{source_id}:recorded:{world_id}",
            "domain_parameters": {
                "data_kind": "finalized_recorded_observations",
                "hidden_truth_claimed": False,
                **domain_parameters,
            },
        },
        "branch": None,
        "objective_eligibility": {
            "state_jepa": bool(state_eligible),
            "dynamics_jepa": False,
            "decision_heads": False,
        },
        "topology": {
            "topology_id": f"{source_id}:{world_id}:topology",
            "nodes": topology_nodes,
            "edges": topology_edges,
        },
        "steps": steps,
        "provenance": {
            "adapter_version": OBSERVATIONAL_ADAPTER_VERSION,
            "generator": f"supplychain_jepa.adapters.{source_id}",
            "causal_semantics": (
                "observational finalized records only; no interventions or "
                "counterfactual action effects"
            ),
            "observation_semantics": (
                "features are factual fields recorded by the source dataset; "
                "latent_state mirrors the same recorded facts and is not hidden truth"
            ),
            "source_files": source_files,
            "runtime_dependencies": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "pandas": pd.__version__,
            },
        },
    }
    if not recursive_numbers_are_finite(record):
        raise ValueError(f"{source_id}:{world_id} contains non-finite numbers")
    return record


def make_step(
    *,
    clock: float,
    entity: dict[str, Any],
    event: dict[str, Any] | None = None,
    global_features: dict[str, Any] | None = None,
    outcomes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mirror factual records into state and observation without hidden-state claims."""
    events = [] if event is None else [event]
    state = {
        "entities": [entity],
        "edges": [],
        "global": global_features or {},
        "events": events,
    }
    observation = deepcopy(state)
    observation["status"] = [
        observation_status(
            "observation.entities",
            "observed",
            reason="factual field present in the finalized source record",
        )
    ]
    return {
        "t": 0,
        "clock": max(0.0, float(clock)),
        "latent_state": state,
        "observation": observation,
        "actions": [],
        "outcomes": outcomes or {},
        "exogenous_digest": stable_digest(
            "v2-real-observation", entity["entity_id"], entity["features"]
        ),
        "terminal": False,
    }
