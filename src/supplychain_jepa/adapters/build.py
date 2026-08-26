"""Corpus publication helpers for canonical source adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from supplychain_jepa.schemas import canonical_episode_schema

from .base import AdapterRequest
from .common import (
    canonical_json,
    file_sha256,
    recursive_numbers_are_finite,
    write_jsonl,
)
from .registry import DESCRIPTORS, load_adapter


def _validate_records(source_id: str, records: list[dict[str, Any]]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError(
            "Adapter validation requires jsonschema: uv sync --extra adapters"
        ) from exc

    descriptor = DESCRIPTORS[source_id]
    validator = Draft202012Validator(
        json.loads(canonical_episode_schema().read_text(encoding="utf-8"))
    )
    seen: set[str] = set()
    for index, record in enumerate(records):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.path) or "<record>"
            raise ValueError(
                f"{source_id} record {index} violates schema at {location}: {error.message}"
            )
        record_id = str(record["record_id"])
        if record_id in seen:
            raise ValueError(f"{source_id} emitted duplicate record_id {record_id!r}")
        seen.add(record_id)
        if record["simulator"]["family"] != source_id:
            raise ValueError(f"{source_id} emitted a mismatched source family")
        eligibility = record["objective_eligibility"]
        if eligibility["dynamics_jepa"] and not descriptor.supports_dynamics:
            raise ValueError(f"{source_id} emitted an unsupported Dynamics record")
        if not recursive_numbers_are_finite(record):
            raise ValueError(f"{source_id} emitted NaN or Infinity")


def convert_source(
    source_id: str,
    request: AdapterRequest,
    *,
    output_path: Path,
    validate: bool = True,
) -> dict[str, Any]:
    """Convert one source, publish deterministic JSONL, and return its manifest."""
    adapter = load_adapter(source_id)
    records = adapter.convert(request)
    if not records:
        raise ValueError(f"{source_id} adapter emitted no records")
    records = sorted(records, key=lambda record: str(record["record_id"]))
    if validate:
        _validate_records(source_id, records)
    count = write_jsonl(output_path, records)
    return {
        "source_id": source_id,
        "source_kind": adapter.source_kind,
        "supports_state": adapter.supports_state,
        "supports_dynamics": adapter.supports_dynamics,
        "request": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(request).items()
        },
        "path": output_path.name,
        "records": count,
        "steps": sum(len(record["steps"]) for record in records),
        "state_records": sum(
            bool(record["objective_eligibility"]["state_jepa"]) for record in records
        ),
        "dynamics_records": sum(
            bool(record["objective_eligibility"]["dynamics_jepa"]) for record in records
        ),
        "sha256": file_sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


def build_corpus(config: Mapping[str, Any], *, config_dir: Path) -> dict[str, Any]:
    """Build every source declared in an adapter configuration mapping."""
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = config_dir / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_configs = config.get("sources")
    if not isinstance(source_configs, Mapping) or not source_configs:
        raise ValueError("adapter config must contain a non-empty sources mapping")

    manifests: dict[str, Any] = {}
    for source_id, raw_values in source_configs.items():
        if source_id not in DESCRIPTORS:
            raise ValueError(f"adapter config contains unknown source {source_id!r}")
        values = dict(raw_values or {})
        for path_key in ("raw_root", "source_root"):
            if values.get(path_key) is not None:
                path = Path(values[path_key])
                if not path.is_absolute():
                    path = config_dir / path
                values[path_key] = path.resolve()
        request = AdapterRequest(**values)
        manifests[source_id] = convert_source(
            source_id,
            request,
            output_path=output_dir / f"{source_id}.jsonl",
            validate=bool(config.get("validate", True)),
        )

    manifest = {
        "schema_version": "supplychain-jepa.adapter-corpus-manifest.v1",
        "sources": manifests,
        "record_count": sum(item["records"] for item in manifests.values()),
        "step_count": sum(item["steps"] for item in manifests.values()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(canonical_json(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = ["build_corpus", "convert_source"]
