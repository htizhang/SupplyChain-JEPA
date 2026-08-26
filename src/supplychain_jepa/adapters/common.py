"""Deterministic builders used by source-specific adapters."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "supplychain-jepa.synthetic-episode.v1"
ADAPTER_VERSION = "release-adapters-v1"
ALLOWED_OBSERVATION_STATUS = {
    "observed",
    "censored",
    "delayed",
    "missing",
    "outage",
    "not_applicable",
}
STATUS_PATH_PATTERN = re.compile(
    r"^(observation|latent_state)\.(entities|edges|global|events)"
    r"(?:\.[^.]+){0,2}$"
)


def json_safe(value: Any) -> Any:
    """Convert NumPy and path values to deterministic finite JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite floating-point value: {value!r}")
        return float(value)
    return value


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def stable_digest(*parts: Any, length: int | None = None) -> str:
    payload = "\x1f".join(canonical_json(part) for part in parts).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return digest if length is None else digest[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_record_id(
    source_family: str,
    world_id: str,
    branch_id: str | None,
    observation_id: str | None = None,
) -> str:
    parts: list[Any] = [SCHEMA_VERSION, source_family, world_id, branch_id]
    if observation_id is not None:
        parts.append(observation_id)
    suffix = stable_digest(*parts, length=16)
    branch_label = branch_id or "single"
    observation_label = "" if observation_id is None else f":{observation_id}"
    return f"{source_family}:{world_id}:{branch_label}{observation_label}:{suffix}"


def observation_status(
    path: str,
    status: str,
    *,
    age_steps: float = 0.0,
    reason: str,
) -> dict[str, Any]:
    if status not in ALLOWED_OBSERVATION_STATUS:
        raise ValueError(f"Unsupported observation status: {status}")
    if STATUS_PATH_PATTERN.fullmatch(path) is None:
        raise ValueError(f"Unsupported observation-status path: {path}")
    return {
        "path": path,
        "status": status,
        "age_steps": float(age_steps),
        "reason": reason,
    }


def make_entity(
    entity_id: str,
    entity_type: str,
    *,
    node_id: str | None,
    product_id: str | None,
    features: dict[str, Any],
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "node_id": node_id,
        "product_id": product_id,
        "features": json_safe(features),
    }


def make_edge_state(edge_id: str, features: dict[str, Any]) -> dict[str, Any]:
    return {"edge_id": edge_id, "features": json_safe(features)}


def make_event(
    event_id: str,
    event_type: str,
    entity_id: str,
    *,
    node_id: str | None,
    event_time: float,
    report_time: float | None,
    observed: bool,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "entity_id": entity_id,
        "node_id": node_id,
        "event_time": float(event_time),
        "report_time": None if report_time is None else float(report_time),
        "observed": bool(observed),
        "features": json_safe(features or {}),
    }


def make_action(
    action_type: str,
    actor_id: str,
    *,
    target_id: str | None,
    quantity: float,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "actor_id": actor_id,
        "target_id": target_id,
        "quantity": float(quantity),
        "features": json_safe(features or {}),
    }


def recursive_numbers_are_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(recursive_numbers_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(recursive_numbers_are_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record))
            handle.write("\n")
            count += 1
    return count
