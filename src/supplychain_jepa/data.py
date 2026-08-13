"""Leakage-safe tensorization for canonical SupplyChain-JEPA episodes.

The central invariant is deliberately simple:

* context tensors are constructed only from topology, the canonical
  ``observation`` object, and recorded actions;
* target tensors are constructed only from ``latent_state``;
* whether a target was legally observable is represented by a separate mask;
* training objective masks are not created here (see :mod:`masking`).

This keeps latent targets out of encoder inputs while retaining enough metadata
to build structured state and future-prediction objectives.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

TENSORIZER_FORMAT = "supplychain-jepa-tensorizer"
SUPPORTED_TENSORIZER_FORMATS = frozenset({TENSORIZER_FORMAT})
PAD_ID = 0
UNK_ID = 1

STATUS_TO_ID = {
    "observed": 1,
    "censored": 2,
    "delayed": 3,
    "missing": 4,
    "outage": 5,
    "not_applicable": 6,
}
ID_TO_STATUS = {value: key for key, value in STATUS_TO_ID.items()}

_SEQUENCE_PREFIXES = ("context", "target")
_MATRIX_FIELDS = (
    "values",
    "categorical_ids",
    "feature_present",
    "legal_observation_mask",
    "status_ids",
    "age",
)
_VECTOR_FIELDS = (
    "time_index",
    "clock",
    "family_id",
    "token_kind_id",
    "type_id",
    "entity_index",
    "product_index",
    "node_index",
    "source_index",
    "target_index",
    "alignment_index",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _world_signature(record: Mapping[str, Any]) -> str:
    """Hash physical-world descriptors without trusting the display ID."""

    episode = record["episode"]
    payload = {
        "simulator_family": record["simulator"]["family"],
        # The adapter's topology ID is canonical; hashing raw node/edge labels
        # would make the control metadata depend on arbitrary identifier
        # spellings even though model tensors are rename-invariant.
        "topology_id": record["topology"]["topology_id"],
        "config_id": episode["config_id"],
        "domain_parameters": episode["domain_parameters"],
        "physical_seed": episode["seed"],
        "exogenous_stream_id": episode["exogenous_stream_id"],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_number(value: Any, *, booleans_are_numeric: bool = True) -> bool:
    return (
        isinstance(value, (int, float))
        and (booleans_are_numeric or not isinstance(value, bool))
        and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    )


def _categorical_text(value: Any) -> str:
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"num:{value!r}"
    return f"json:{_canonical_json(value)}"


def _feature_name(kind: str, name: str) -> str:
    return f"{kind}::{name}"


def _recursive_artifact_paths(value: Any) -> Iterator[str]:
    if isinstance(value, str) and (
        value.endswith(".jsonl")
        or (value.endswith(".json") and "manifest" in Path(value).name.lower())
    ):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _recursive_artifact_paths(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            yield from _recursive_artifact_paths(nested)


def resolve_jsonl_paths(source: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve canonical JSONL files from files, manifests, or run directories.

    Supported declaration chains include:

    * a dataset ``manifest.json`` -> JSONL parts;
    * global manifest -> child manifest -> JSONL;
    * a ``corpus-manifest.json`` -> declared shard directory ->
      ``shard-manifest.json`` -> ``content.path``.

    A corpus manifest is authoritative: undeclared sibling shards are never
    scanned. Absolute shard paths work in place; if a run root was relocated,
    an unavailable absolute path containing ``shards/...`` is safely rebased
    beneath the corpus manifest's current parent.
    """

    raw_sources: Sequence[str | Path]
    raw_sources = [source] if isinstance(source, (str, Path)) else source

    resolved: dict[str, Path] = {}
    visited: set[Path] = set()
    active: set[Path] = set()

    def locate_declared(raw_value: str | Path, *, parent: Path) -> Path:
        raw_path = Path(raw_value).expanduser()
        direct = raw_path if raw_path.is_absolute() else parent / raw_path
        direct = direct.resolve()
        if direct.exists():
            return direct

        if raw_path.is_absolute():
            parts = raw_path.parts
            if "shards" in parts:
                shard_offset = len(parts) - 1 - tuple(reversed(parts)).index("shards")
                relocated = (parent / Path(*parts[shard_offset:])).resolve()
                if relocated.exists():
                    return relocated
            relocated_leaf = (parent / raw_path.name).resolve()
            if relocated_leaf.exists():
                return relocated_leaf
        raise FileNotFoundError(
            f"Declared artifact {raw_value!s} does not exist relative to {parent}"
        )

    def resolve_shard_directory(shard_dir: Path) -> None:
        shard_dir = shard_dir.resolve()
        if not shard_dir.is_dir():
            raise NotADirectoryError(shard_dir)
        success_path = shard_dir / "_SUCCESS"
        manifest_path = shard_dir / "shard-manifest.json"
        if not success_path.is_file():
            raise FileNotFoundError(f"Declared shard lacks _SUCCESS: {shard_dir}")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Declared shard lacks shard-manifest.json: {shard_dir}"
            )
        resolve_path(manifest_path)

    def resolve_manifest(path: Path) -> None:
        path = path.resolve()
        if path in active:
            raise ValueError(f"Manifest reference cycle detected at {path}")
        if path in visited:
            return
        active.add(path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise ValueError(f"Manifest {path} is not a JSON object")
            if manifest.get("success") is False:
                raise ValueError(f"Manifest is not successfully published: {path}")

            found = False
            shards = manifest.get("shards")
            if shards is not None:
                if not isinstance(shards, Sequence) or isinstance(
                    shards, (str, bytes, bytearray)
                ):
                    raise ValueError(f"Manifest {path} has a non-list shards field")
                for shard in shards:
                    if not isinstance(shard, Mapping):
                        raise ValueError(
                            f"Manifest {path} has a non-object shard declaration"
                        )
                    if "path" not in shard:
                        continue
                    shard_path = locate_declared(str(shard["path"]), parent=path.parent)
                    if shard_path.is_dir():
                        resolve_shard_directory(shard_path)
                    else:
                        resolve_path(shard_path)
                    found = True

            for relative in _recursive_artifact_paths(manifest):
                candidate = locate_declared(relative, parent=path.parent)
                resolve_path(candidate)
                found = True
            if not found:
                raise ValueError(
                    f"Manifest {path} declares no JSONL parts, shard directories, "
                    "or child manifests"
                )
            visited.add(path)
        finally:
            active.remove(path)

    def resolve_path(path: Path) -> None:
        path = path.expanduser().resolve()
        if path.is_dir():
            corpus_manifest = path / "corpus-manifest.json"
            root_manifest = path / "manifest.json"
            shard_manifest = path / "shard-manifest.json"
            if corpus_manifest.is_file():
                resolve_manifest(corpus_manifest)
            elif root_manifest.is_file():
                resolve_manifest(root_manifest)
            elif shard_manifest.is_file():
                resolve_shard_directory(path)
            else:
                manifests = sorted(path.glob("**/shard-manifest.json"))
                if manifests:
                    for manifest_path in manifests:
                        resolve_shard_directory(manifest_path.parent)
                else:
                    for part in sorted(path.glob("**/*.jsonl")):
                        resolved[str(part.resolve())] = part.resolve()
            return
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            resolved[str(path)] = path
            return
        if path.suffix == ".json":
            resolve_manifest(path)
            return
        raise ValueError(f"Unsupported episode source: {path}")

    for raw_source in raw_sources:
        path = Path(raw_source).expanduser().resolve()
        resolve_path(path)
    return [resolved[key] for key in sorted(resolved)]


def iter_episode_records(
    source: str | Path | Sequence[str | Path],
) -> Iterator[dict[str, Any]]:
    """Stream canonical episode records in deterministic file/line order."""

    for path in resolve_jsonl_paths(source):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Episode at {path}:{line_number} is not an object"
                    )
                yield record


@dataclass(frozen=True)
class FittedTensorizerState:
    """Serializable train-only feature, vocabulary, and scaling state."""

    state_version: str
    train_splits: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    numeric_mean: tuple[float, ...]
    numeric_std: tuple[float, ...]
    categorical_vocabs: dict[str, dict[str, int]]
    target_categorical_vocabs: dict[str, dict[str, int]]
    metadata_vocabs: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if not (
            len(self.feature_kinds)
            == len(self.numeric_mean)
            == len(self.numeric_std)
            == size
        ):
            raise ValueError("Feature state vectors must have identical lengths")
        if any(kind not in {"numeric", "categorical"} for kind in self.feature_kinds):
            raise ValueError("Feature kinds must be numeric or categorical")

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def feature_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.feature_names)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FittedTensorizerState:
        return cls(
            state_version=str(payload["state_version"]),
            train_splits=tuple(payload["train_splits"]),
            feature_names=tuple(payload["feature_names"]),
            feature_kinds=tuple(payload["feature_kinds"]),
            numeric_mean=tuple(float(value) for value in payload["numeric_mean"]),
            numeric_std=tuple(float(value) for value in payload["numeric_std"]),
            categorical_vocabs={
                str(feature): {str(value): int(index) for value, index in vocab.items()}
                for feature, vocab in payload["categorical_vocabs"].items()
            },
            target_categorical_vocabs={
                str(feature): {str(value): int(index) for value, index in vocab.items()}
                for feature, vocab in payload.get(
                    "target_categorical_vocabs",
                    payload["categorical_vocabs"],
                ).items()
            },
            metadata_vocabs={
                str(name): {str(value): int(index) for value, index in vocab.items()}
                for name, vocab in payload["metadata_vocabs"].items()
            },
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> FittedTensorizerState:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class _Token:
    kind: str
    token_type: str
    primary_id: str
    entity_id: str | None
    product_id: str | None
    node_id: str | None
    source_id: str | None
    target_id: str | None
    time_index: int
    clock: float
    features: dict[str, Any]
    alignment_id: str
    statuses: dict[str, tuple[str, float]]


@dataclass
class _RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.count == 0:
            return 1.0
        return max(math.sqrt(self.m2 / self.count), 1e-6)


@dataclass(frozen=True)
class _CanonicalEpisodeIds:
    """Episode-local role/order indices; raw identifiers never leave tensorization."""

    object_indices: dict[str, int]
    product_indices: dict[str, int]
    alignment_indices: dict[str, int]

    @staticmethod
    def _lookup(mapping: Mapping[str, int], value: str | None) -> int:
        if value in (None, ""):
            return PAD_ID
        return mapping.get(str(value), UNK_ID)

    def object_index(self, value: str | None) -> int:
        return self._lookup(self.object_indices, value)

    def product_index(self, value: str | None) -> int:
        return self._lookup(self.product_indices, value)

    def alignment_index(self, value: str) -> int:
        return self._lookup(self.alignment_indices, value)


@dataclass
class EpisodeTensors:
    """One tensorized episode with separate context and latent target sequences."""

    record_id: str
    world_id: str  # non-model metadata; never encoded into tensor inputs
    world_signature: str  # independent descriptor hash; non-model metadata
    topology_id: str  # non-model shuffled-control diagnostic
    config_id: str  # non-model shuffled-control diagnostic
    family: str
    split: str
    time_unit: str
    objective_eligibility: dict[str, bool]
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    context_values: Tensor
    context_categorical_ids: Tensor
    context_feature_present: Tensor
    context_legal_observation_mask: Tensor
    context_status_ids: Tensor
    context_age: Tensor
    context_time_index: Tensor
    context_clock: Tensor
    context_family_id: Tensor
    context_token_kind_id: Tensor
    context_type_id: Tensor
    context_entity_index: Tensor
    context_product_index: Tensor
    context_node_index: Tensor
    context_source_index: Tensor
    context_target_index: Tensor
    context_alignment_index: Tensor
    target_values: Tensor
    target_categorical_ids: Tensor
    target_feature_present: Tensor
    target_legal_observation_mask: Tensor
    target_status_ids: Tensor
    target_age: Tensor
    target_time_index: Tensor
    target_clock: Tensor
    target_family_id: Tensor
    target_token_kind_id: Tensor
    target_type_id: Tensor
    target_entity_index: Tensor
    target_product_index: Tensor
    target_node_index: Tensor
    target_source_index: Tensor
    target_target_index: Tensor
    target_alignment_index: Tensor
    context_token_kinds: tuple[str, ...]
    target_token_kinds: tuple[str, ...]
    node_neighbors: dict[int, tuple[int, ...]]

    @property
    def context_length(self) -> int:
        return int(self.context_values.shape[0])

    @property
    def target_length(self) -> int:
        return int(self.target_values.shape[0])


def clone_episode_tensors(episode: EpisodeTensors) -> EpisodeTensors:
    """Return a value-identical copy safe for mutation by downstream callers.

    PyTorch tensors are mutable even when held by a frozen Python container.
    A transformed-episode cache therefore retains one private canonical object
    and returns cloned tensors on every access.  This makes cache use an
    operational optimization only: callers see exactly the same values and
    cannot mutate later examples through a shared storage alias.
    """

    values: dict[str, Any] = {}
    for field in fields(EpisodeTensors):
        value = getattr(episode, field.name)
        values[field.name] = (
            value.detach().clone(memory_format=torch.preserve_format)
            if isinstance(value, Tensor)
            else copy.deepcopy(value)
        )
    return EpisodeTensors(**values)


def episode_tensor_bytes(episode: EpisodeTensors) -> int:
    """Count bytes occupied by tensor payloads in one transformed episode."""

    return sum(
        int(value.numel()) * int(value.element_size())
        for field in fields(EpisodeTensors)
        if isinstance((value := getattr(episode, field.name)), Tensor)
    )


def _status_map(
    step: Mapping[str, Any],
) -> dict[tuple[str, str, str, str], tuple[str, float]]:
    result: dict[tuple[str, str, str, str], tuple[str, float]] = {}
    for entry in step["observation"].get("status", []):
        parts = str(entry["path"]).split(".")
        if len(parts) < 2:
            continue
        prefix, plural_kind = parts[:2]
        kind = {
            "entities": "entity",
            "edges": "edge",
            "events": "event",
            "global": "global",
        }.get(plural_kind)
        if kind is None:
            continue
        object_id = "*"
        feature = "*"
        if kind == "global" and len(parts) >= 3:
            feature = parts[2]
        elif len(parts) >= 3:
            object_id = parts[2]
            if len(parts) >= 4:
                feature = parts[3]
        result[(prefix, kind, object_id, feature)] = (
            str(entry["status"]),
            float(entry["age_steps"]),
        )
    return result


def _lookup_status(
    statuses: Mapping[tuple[str, str, str, str], tuple[str, float]],
    prefix: str,
    kind: str,
    object_id: str,
    feature: str,
    *,
    default: tuple[str, float],
) -> tuple[str, float]:
    for key in (
        (prefix, kind, object_id, feature),
        (prefix, kind, object_id, "*"),
        (prefix, kind, "*", feature),
        (prefix, kind, "*", "*"),
    ):
        if key in statuses:
            return statuses[key]
    return default


def _node_neighbors(record: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    neighbors: dict[str, set[str]] = {
        str(node["node_id"]): set() for node in record["topology"]["nodes"]
    }
    for edge in record["topology"]["edges"]:
        source = str(edge["source"])
        target = str(edge["target"])
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    return {node: tuple(sorted(values)) for node, values in sorted(neighbors.items())}


def _base_features(item: Mapping[str, Any], kind: str) -> dict[str, Any]:
    features = dict(item.get("features", {}))
    if kind == "event":
        features.update(
            {
                "__event_time": item.get("event_time"),
                "__report_time": item.get("report_time"),
                "__observed": item.get("observed"),
            }
        )
    elif kind == "action":
        features["quantity"] = item.get("quantity")
    return {_feature_name(kind, str(name)): value for name, value in features.items()}


def _dynamic_token(
    item: Mapping[str, Any],
    *,
    kind: str,
    time_index: int,
    clock: float,
    status_prefix: str,
    statuses: Mapping[tuple[str, str, str, str], tuple[str, float]],
    action_index: int | None = None,
    edge_endpoints: Mapping[str, tuple[str, str]] | None = None,
) -> _Token:
    if kind == "entity":
        primary_id = str(item["entity_id"])
        token_type = str(item["entity_type"])
        entity_id = primary_id
        product_id = item.get("product_id")
        node_id = item.get("node_id")
        source_id = None
        target_id = None
    elif kind == "edge":
        primary_id = str(item["edge_id"])
        token_type = "edge_state"
        entity_id = primary_id
        product_id = None
        node_id = None
        source_id, target_id = (edge_endpoints or {}).get(primary_id, (None, None))
    elif kind == "event":
        primary_id = str(item["event_id"])
        token_type = str(item["event_type"])
        entity_id = str(item["entity_id"])
        product_id = None
        node_id = item.get("node_id")
        source_id = None
        target_id = None
    elif kind == "global":
        primary_id = "global"
        token_type = "global_state"
        entity_id = "global"
        product_id = None
        node_id = None
        source_id = None
        target_id = None
    elif kind == "action":
        primary_id = (
            f"{action_index}:{item['action_type']}:"
            f"{item['actor_id']}:{item.get('target_id')}"
        )
        token_type = str(item["action_type"])
        entity_id = str(item["actor_id"])
        product_id = item.get("features", {}).get("item_id")
        node_id = str(item["actor_id"])
        source_id = str(item["actor_id"])
        target_id = item.get("target_id")
    else:
        raise ValueError(f"Unsupported dynamic token kind: {kind}")

    features = (
        _base_features(item, kind)
        if kind != "global"
        else {_feature_name(kind, str(name)): value for name, value in item.items()}
    )
    token_statuses = {
        feature_name: _lookup_status(
            statuses,
            status_prefix,
            kind,
            primary_id,
            feature_name.split("::", 1)[1],
            default=(
                ("observed", 0.0)
                if status_prefix == "observation"
                else ("missing", 0.0)
            ),
        )
        for feature_name in features
    }
    alignment_id = f"{kind}|{time_index}|{primary_id}"
    return _Token(
        kind=kind,
        token_type=token_type,
        primary_id=primary_id,
        entity_id=entity_id,
        product_id=None if product_id is None else str(product_id),
        node_id=None if node_id is None else str(node_id),
        source_id=None if source_id is None else str(source_id),
        target_id=None if target_id is None else str(target_id),
        time_index=time_index,
        clock=clock,
        features=features,
        alignment_id=alignment_id,
        statuses=token_statuses,
    )


def _topology_tokens(record: Mapping[str, Any]) -> list[_Token]:
    tokens: list[_Token] = []
    for node in record["topology"]["nodes"]:
        node_id = str(node["node_id"])
        tokens.append(
            _Token(
                kind="topology_node",
                token_type=str(node["node_type"]),
                primary_id=node_id,
                entity_id=node_id,
                product_id=None,
                node_id=node_id,
                source_id=None,
                target_id=None,
                time_index=-1,
                clock=-1.0,
                features={
                    _feature_name("topology_node", str(name)): value
                    for name, value in node["features"].items()
                },
                alignment_id=f"topology_node|-1|{node_id}",
                statuses={},
            )
        )
    for edge in record["topology"]["edges"]:
        edge_id = str(edge["edge_id"])
        tokens.append(
            _Token(
                kind="topology_edge",
                token_type=str(edge["edge_type"]),
                primary_id=edge_id,
                entity_id=edge_id,
                product_id=None,
                node_id=None,
                source_id=str(edge["source"]),
                target_id=str(edge["target"]),
                time_index=-1,
                clock=-1.0,
                features={
                    _feature_name("topology_edge", str(name)): value
                    for name, value in edge["features"].items()
                },
                alignment_id=f"topology_edge|-1|{edge_id}",
                statuses={},
            )
        )
    return tokens


def _episode_tokens(
    record: Mapping[str, Any],
) -> tuple[
    list[_Token],
    list[_Token],
    dict[int, tuple[int, ...]],
    _CanonicalEpisodeIds,
]:
    context = _topology_tokens(record)
    target: list[_Token] = []
    edge_endpoints = {
        str(edge["edge_id"]): (str(edge["source"]), str(edge["target"]))
        for edge in record["topology"]["edges"]
    }
    for step in record["steps"]:
        time_index = int(step["t"])
        clock = float(step["clock"])
        statuses = _status_map(step)
        observation = step["observation"]
        latent = step["latent_state"]

        for item in observation["entities"]:
            context.append(
                _dynamic_token(
                    item,
                    kind="entity",
                    time_index=time_index,
                    clock=clock,
                    status_prefix="observation",
                    statuses=statuses,
                )
            )
        for item in observation["edges"]:
            context.append(
                _dynamic_token(
                    item,
                    kind="edge",
                    time_index=time_index,
                    clock=clock,
                    status_prefix="observation",
                    statuses=statuses,
                    edge_endpoints=edge_endpoints,
                )
            )
        context.append(
            _dynamic_token(
                observation["global"],
                kind="global",
                time_index=time_index,
                clock=clock,
                status_prefix="observation",
                statuses=statuses,
            )
        )
        for item in observation["events"]:
            # Defense in depth: even if a future adapter accidentally places an
            # unreported event into observation.events, it is not a legal
            # context token until marked observed and reported.
            report_time = item.get("report_time")
            if not bool(item.get("observed")):
                continue
            if report_time is not None and float(report_time) > clock:
                continue
            context.append(
                _dynamic_token(
                    item,
                    kind="event",
                    time_index=time_index,
                    clock=clock,
                    status_prefix="observation",
                    statuses=statuses,
                )
            )
        for action_index, item in enumerate(step["actions"]):
            context.append(
                _dynamic_token(
                    item,
                    kind="action",
                    time_index=time_index,
                    clock=clock,
                    status_prefix="observation",
                    statuses=statuses,
                    action_index=action_index,
                )
            )

        for item in latent["entities"]:
            target.append(
                _dynamic_token(
                    item,
                    kind="entity",
                    time_index=time_index,
                    clock=clock,
                    status_prefix="latent_state",
                    statuses=statuses,
                )
            )
        for item in latent["edges"]:
            target.append(
                _dynamic_token(
                    item,
                    kind="edge",
                    time_index=time_index,
                    clock=clock,
                    status_prefix="latent_state",
                    statuses=statuses,
                    edge_endpoints=edge_endpoints,
                )
            )
        target.append(
            _dynamic_token(
                latent["global"],
                kind="global",
                time_index=time_index,
                clock=clock,
                status_prefix="latent_state",
                statuses=statuses,
            )
        )
        for item in latent["events"]:
            target.append(
                _dynamic_token(
                    item,
                    kind="event",
                    time_index=time_index,
                    clock=clock,
                    status_prefix="latent_state",
                    statuses=statuses,
                )
            )
    canonical_ids = _canonical_episode_ids(context, target)
    _canonicalize_identifier_features(context, canonical_ids)
    _canonicalize_identifier_features(target, canonical_ids)
    raw_neighbors = _node_neighbors(record)
    canonical_neighbors = {
        canonical_ids.object_index(node): tuple(
            canonical_ids.object_index(neighbor) for neighbor in neighbors
        )
        for node, neighbors in raw_neighbors.items()
    }
    return context, target, canonical_neighbors, canonical_ids


def _canonical_episode_ids(
    context: Sequence[_Token],
    target: Sequence[_Token],
) -> _CanonicalEpisodeIds:
    """Assign stable episode-local indices in canonical traversal order.

    Topology nodes/edges come first because ``context`` begins with those
    schema-defined arrays. Remaining objects, products, and alignments are
    assigned on first appearance in context, then target. Renaming arbitrary
    raw identifiers while preserving topology/record order therefore leaves
    every model-facing index unchanged.
    """

    object_indices: dict[str, int] = {}
    product_indices: dict[str, int] = {}
    alignment_indices: dict[str, int] = {}

    def add(mapping: dict[str, int], value: str | None) -> None:
        if value in (None, ""):
            return
        text = str(value)
        if text not in mapping:
            mapping[text] = max(mapping.values(), default=1) + 1

    def add_or_alias(mapping: dict[str, int], value: str) -> None:
        if value in mapping:
            return
        matches = {
            index
            for raw_value, index in mapping.items()
            if raw_value.rsplit(":", 1)[-1] == value
        }
        if len(matches) == 1:
            mapping[value] = next(iter(matches))
        else:
            add(mapping, value)

    for token in (*context, *target):
        for value in (
            token.entity_id,
            token.node_id,
            token.source_id,
            token.target_id,
        ):
            add(object_indices, value)
        add(product_indices, token.product_id)
        add(alignment_indices, token.alignment_id)
        for feature, value in token.features.items():
            if value is None or not isinstance(value, str):
                continue
            base_name = feature.split("::", 1)[-1].lower()
            if base_name in {"item_id", "product_id"}:
                add_or_alias(product_indices, str(value))
            elif base_name.endswith("_id"):
                add_or_alias(object_indices, str(value))

    def add_unambiguous_aliases(mapping: dict[str, int]) -> None:
        aliases: dict[str, set[int]] = {}
        for raw_value, index in list(mapping.items()):
            alias = raw_value.rsplit(":", 1)[-1]
            if alias != raw_value:
                aliases.setdefault(alias, set()).add(index)
        for alias, indices in aliases.items():
            if len(indices) == 1 and alias not in mapping:
                mapping[alias] = next(iter(indices))

    add_unambiguous_aliases(object_indices)
    add_unambiguous_aliases(product_indices)
    return _CanonicalEpisodeIds(
        object_indices=object_indices,
        product_indices=product_indices,
        alignment_indices=alignment_indices,
    )


def _canonicalize_identifier_features(
    tokens: Sequence[_Token],
    canonical_ids: _CanonicalEpisodeIds,
) -> None:
    """Replace ID-valued feature strings with episode-local numeric indices."""

    for token in tokens:
        for feature, value in list(token.features.items()):
            if value is None or not isinstance(value, str):
                continue
            base_name = feature.split("::", 1)[-1].lower()
            if base_name in {"item_id", "product_id"}:
                token.features[feature] = float(canonical_ids.product_index(str(value)))
            elif (
                base_name.endswith("_id") or str(value) in canonical_ids.object_indices
            ):
                token.features[feature] = float(canonical_ids.object_index(str(value)))
            elif str(value) in canonical_ids.product_indices:
                token.features[feature] = float(canonical_ids.product_index(str(value)))


def _fit_vocab(values: Iterable[str]) -> dict[str, int]:
    return {value: index + 2 for index, value in enumerate(sorted(set(values)))}


def _fit_context_first_vocab(
    context_values: Iterable[str],
    target_values: Iterable[str],
) -> dict[str, int]:
    """Keep context IDs stable, then append target-only meanings."""

    vocabulary = _fit_vocab(context_values)
    next_index = max(vocabulary.values(), default=1) + 1
    for value in sorted(set(target_values) - set(vocabulary)):
        vocabulary[value] = next_index
        next_index += 1
    return vocabulary


def fit_tensorizer_state(
    records: Iterable[Mapping[str, Any]],
    *,
    train_splits: Sequence[str] = ("pretrain_train",),
    state_version: str = TENSORIZER_FORMAT,
) -> FittedTensorizerState:
    """Fit all feature schemas, vocabularies, and scalers on train records only."""

    if state_version not in SUPPORTED_TENSORIZER_FORMATS:
        raise ValueError(f"Unsupported tensorizer state version {state_version!r}")
    booleans_are_numeric = False
    allowed_splits = tuple(str(split) for split in train_splits)
    observed_types: dict[str, set[str]] = {}
    context_numeric_stats: dict[str, _RunningStats] = {}
    target_numeric_stats: dict[str, _RunningStats] = {}
    context_categorical_values: dict[str, set[str]] = {}
    target_categorical_values: dict[str, set[str]] = {}
    metadata: dict[str, set[str]] = {
        "family": set(),
        "time_unit": set(),
        "context_token_kind": set(),
        "context_token_type": set(),
        "target_token_kind": set(),
        "target_token_type": set(),
    }
    selected = 0

    for record in records:
        if str(record["split"]) not in allowed_splits:
            continue
        selected += 1
        family = str(record["simulator"]["family"])
        metadata["family"].add(family)
        metadata["time_unit"].add(str(record["episode"]["time_unit"]))
        context, target, _, _ = _episode_tokens(record)
        for token in context:
            metadata["context_token_kind"].add(token.kind)
            metadata["context_token_type"].add(token.token_type)
            for feature, value in token.features.items():
                if value is None:
                    continue
                kind = (
                    "numeric"
                    if _is_number(value, booleans_are_numeric=booleans_are_numeric)
                    else "categorical"
                )
                observed_types.setdefault(feature, set()).add(kind)
                if kind == "numeric":
                    context_numeric_stats.setdefault(feature, _RunningStats()).update(
                        float(value)
                    )
                else:
                    context_categorical_values.setdefault(feature, set()).add(
                        _categorical_text(value)
                    )
        for token in target:
            metadata["target_token_kind"].add(token.kind)
            metadata["target_token_type"].add(token.token_type)
            for feature, value in token.features.items():
                if value is None:
                    continue
                kind = (
                    "numeric"
                    if _is_number(value, booleans_are_numeric=booleans_are_numeric)
                    else "categorical"
                )
                observed_types.setdefault(feature, set()).add(kind)
                if kind == "numeric":
                    target_numeric_stats.setdefault(feature, _RunningStats()).update(
                        float(value)
                    )
                else:
                    target_categorical_values.setdefault(feature, set()).add(
                        _categorical_text(value)
                    )

    if selected == 0:
        raise ValueError(f"No records matched train_splits={allowed_splits!r}")

    feature_names = tuple(sorted(observed_types))
    feature_kinds: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    cat_vocabs: dict[str, dict[str, int]] = {}
    target_cat_vocabs: dict[str, dict[str, int]] = {}
    for feature in feature_names:
        types = observed_types[feature]
        if len(types) != 1:
            raise ValueError(
                f"Feature {feature!r} has inconsistent value kinds in train data: "
                f"{sorted(types)}"
            )
        kind = next(iter(types))
        feature_kinds.append(kind)
        if kind == "numeric":
            # A target value may define a target-only feature schema, but it
            # must never change normalization of a feature seen by context.
            stats = context_numeric_stats.get(feature) or target_numeric_stats[feature]
            means.append(stats.mean)
            stds.append(stats.std)
        else:
            means.append(0.0)
            stds.append(1.0)
            shared_vocab = _fit_context_first_vocab(
                context_categorical_values.get(feature, set()),
                target_categorical_values.get(feature, set()),
            )
            cat_vocabs[feature] = shared_vocab
            target_cat_vocabs[feature] = dict(shared_vocab)

    metadata_vocabs = {
        "family": _fit_vocab(metadata["family"]),
        "time_unit": _fit_vocab(metadata["time_unit"]),
        "token_kind": _fit_context_first_vocab(
            metadata["context_token_kind"],
            metadata["target_token_kind"],
        ),
        "token_type": _fit_context_first_vocab(
            metadata["context_token_type"],
            metadata["target_token_type"],
        ),
    }
    return FittedTensorizerState(
        state_version=state_version,
        train_splits=allowed_splits,
        feature_names=feature_names,
        feature_kinds=tuple(feature_kinds),
        numeric_mean=tuple(means),
        numeric_std=tuple(stds),
        categorical_vocabs=cat_vocabs,
        target_categorical_vocabs=target_cat_vocabs,
        metadata_vocabs=metadata_vocabs,
    )


class EpisodeTensorizer:
    """Transform canonical episodes with one immutable fitted train state."""

    def __init__(self, state: FittedTensorizerState):
        if state.state_version not in SUPPORTED_TENSORIZER_FORMATS:
            raise ValueError(
                f"Unsupported tensorizer state {state.state_version!r}; "
                f"expected one of {sorted(SUPPORTED_TENSORIZER_FORMATS)!r}"
            )
        self.state = state
        self._feature_index = state.feature_index

    def _metadata_id(self, namespace: str, value: str) -> int:
        return self.state.metadata_vocabs[namespace].get(str(value), UNK_ID)

    def _encode_tokens(
        self,
        tokens: Sequence[_Token],
        *,
        family: str,
        canonical_ids: _CanonicalEpisodeIds,
        observed_lookup: Mapping[tuple[str, str], tuple[str, float]] | None = None,
        is_target: bool,
    ) -> dict[str, Any]:
        length = len(tokens)
        feature_count = self.state.feature_count
        values = torch.zeros((length, feature_count), dtype=torch.float32)
        categorical_ids = torch.zeros((length, feature_count), dtype=torch.long)
        present = torch.zeros((length, feature_count), dtype=torch.bool)
        legal = torch.zeros((length, feature_count), dtype=torch.bool)
        status_ids = torch.zeros((length, feature_count), dtype=torch.long)
        age = torch.zeros((length, feature_count), dtype=torch.float32)

        row_indices: list[int] = []
        column_indices: list[int] = []
        numeric_values: list[float] = []
        categorical_values: list[int] = []
        legal_values: list[bool] = []
        status_values: list[int] = []
        age_values: list[float] = []
        for token_index, token in enumerate(tokens):
            for feature, value in token.features.items():
                feature_index = self._feature_index.get(feature)
                if feature_index is None or value is None:
                    continue
                row_indices.append(token_index)
                column_indices.append(feature_index)
                feature_kind = self.state.feature_kinds[feature_index]
                if feature_kind == "numeric" and _is_number(
                    value,
                    booleans_are_numeric=False,
                ):
                    numeric_values.append(
                        (float(value) - self.state.numeric_mean[feature_index])
                        / self.state.numeric_std[feature_index]
                    )
                    categorical_values.append(0)
                else:
                    numeric_values.append(0.0)
                    categorical_values.append(
                        self.state.categorical_vocabs.get(feature, {}).get(
                            _categorical_text(value), UNK_ID
                        )
                    )

                if is_target:
                    observed = (
                        None
                        if observed_lookup is None
                        else observed_lookup.get((token.alignment_id, feature))
                    )
                    if observed is not None:
                        status, feature_age = observed
                        is_legal = status not in {
                            "missing",
                            "outage",
                            "not_applicable",
                        }
                    else:
                        status, feature_age = token.statuses.get(
                            feature, ("missing", 0.0)
                        )
                        is_legal = False
                else:
                    status, feature_age = token.statuses.get(feature, ("observed", 0.0))
                    is_legal = status not in {
                        "missing",
                        "outage",
                        "not_applicable",
                    }
                legal_values.append(is_legal)
                status_values.append(STATUS_TO_ID.get(status, STATUS_TO_ID["missing"]))
                age_values.append(float(feature_age))

        if row_indices:
            rows = torch.tensor(row_indices, dtype=torch.long)
            columns = torch.tensor(column_indices, dtype=torch.long)
            present[rows, columns] = True
            values[rows, columns] = torch.tensor(numeric_values, dtype=torch.float32)
            categorical_ids[rows, columns] = torch.tensor(
                categorical_values,
                dtype=torch.long,
            )
            legal[rows, columns] = torch.tensor(legal_values, dtype=torch.bool)
            status_ids[rows, columns] = torch.tensor(status_values, dtype=torch.long)
            age[rows, columns] = torch.tensor(age_values, dtype=torch.float32)

        family_id = self._metadata_id("family", family)
        return {
            "values": values,
            "categorical_ids": categorical_ids,
            "feature_present": present,
            "legal_observation_mask": legal & present,
            "status_ids": status_ids,
            "age": age,
            "time_index": torch.tensor(
                [token.time_index for token in tokens], dtype=torch.long
            ),
            "clock": torch.tensor(
                [token.clock for token in tokens], dtype=torch.float32
            ),
            "family_id": torch.full((length,), family_id, dtype=torch.long),
            "token_kind_id": torch.tensor(
                [self._metadata_id("token_kind", token.kind) for token in tokens],
                dtype=torch.long,
            ),
            "type_id": torch.tensor(
                [self._metadata_id("token_type", token.token_type) for token in tokens],
                dtype=torch.long,
            ),
            "entity_index": torch.tensor(
                [canonical_ids.object_index(token.entity_id) for token in tokens],
                dtype=torch.long,
            ),
            "product_index": torch.tensor(
                [canonical_ids.product_index(token.product_id) for token in tokens],
                dtype=torch.long,
            ),
            "node_index": torch.tensor(
                [canonical_ids.object_index(token.node_id) for token in tokens],
                dtype=torch.long,
            ),
            "source_index": torch.tensor(
                [canonical_ids.object_index(token.source_id) for token in tokens],
                dtype=torch.long,
            ),
            "target_index": torch.tensor(
                [canonical_ids.object_index(token.target_id) for token in tokens],
                dtype=torch.long,
            ),
            "alignment_index": torch.tensor(
                [canonical_ids.alignment_index(token.alignment_id) for token in tokens],
                dtype=torch.long,
            ),
            "token_kinds": tuple(token.kind for token in tokens),
        }

    def transform(self, record: Mapping[str, Any]) -> EpisodeTensors:
        context_tokens, target_tokens, neighbors, canonical_ids = _episode_tokens(
            record
        )
        family = str(record["simulator"]["family"])
        context = self._encode_tokens(
            context_tokens,
            family=family,
            canonical_ids=canonical_ids,
            observed_lookup=None,
            is_target=False,
        )
        observed_lookup: dict[tuple[str, str], tuple[str, float]] = {}
        for token in context_tokens:
            for feature, value in token.features.items():
                if value is None:
                    continue
                status, feature_age = token.statuses.get(feature, ("observed", 0.0))
                if status not in {"missing", "outage", "not_applicable"}:
                    observed_lookup[(token.alignment_id, feature)] = (
                        status,
                        feature_age,
                    )
        target = self._encode_tokens(
            target_tokens,
            family=family,
            canonical_ids=canonical_ids,
            observed_lookup=observed_lookup,
            is_target=True,
        )

        tensor_kwargs: dict[str, Any] = {}
        for prefix, encoded in (("context", context), ("target", target)):
            for field in (*_MATRIX_FIELDS, *_VECTOR_FIELDS):
                tensor_kwargs[f"{prefix}_{field}"] = encoded[field]
            tensor_kwargs[f"{prefix}_token_kinds"] = encoded["token_kinds"]

        return EpisodeTensors(
            record_id=str(record["record_id"]),
            world_id=str(record["episode"]["world_id"]),
            world_signature=_world_signature(record),
            topology_id=str(record["topology"]["topology_id"]),
            config_id=str(record["episode"]["config_id"]),
            family=family,
            split=str(record["split"]),
            time_unit=str(record["episode"]["time_unit"]),
            objective_eligibility={
                key: bool(value)
                for key, value in record["objective_eligibility"].items()
            },
            feature_names=self.state.feature_names,
            feature_kinds=self.state.feature_kinds,
            node_neighbors=neighbors,
            **tensor_kwargs,
        )


class JsonlEpisodeDataset(Dataset[EpisodeTensors]):
    """Deterministic map-style dataset backed by canonical JSONL offsets.

    ``cache_mode="disabled"`` preserves the historical lazy behavior.
    ``cache_mode="preload"`` transforms every selected episode exactly once
    and keeps the private canonical tensors in memory.  Cached access returns a
    deep tensor clone so training cannot mutate shared cache storage.
    """

    CACHE_MODES = frozenset(("disabled", "preload"))

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        tensorizer: EpisodeTensorizer,
        *,
        splits: Sequence[str] | None = None,
        cache_mode: str = "disabled",
    ):
        if cache_mode not in self.CACHE_MODES:
            raise ValueError(
                f"Unsupported transformed-episode cache mode {cache_mode!r}; "
                f"expected one of {sorted(self.CACHE_MODES)}"
            )
        self.paths = resolve_jsonl_paths(source)
        self.tensorizer = tensorizer
        self.splits = (
            None if splits is None else frozenset(str(value) for value in splits)
        )
        self._index: list[tuple[Path, int]] = []
        self._cache: tuple[EpisodeTensors, ...] | None = None
        self._cached_tensor_bytes = 0
        self._transform_calls = 0
        self._cache_hits = 0
        self._cache_misses = 0
        for path in self.paths:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    if self.splits is not None:
                        record = json.loads(line)
                        if str(record["split"]) not in self.splits:
                            continue
                    self._index.append((path, offset))
        if cache_mode == "preload":
            self.preload_transformed()

    def __len__(self) -> int:
        return len(self._index)

    def raw_record(self, index: int) -> dict[str, Any]:
        path, offset = self._index[index]
        with path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())

    def _transform_at(self, index: int) -> EpisodeTensors:
        self._transform_calls += 1
        return self.tensorizer.transform(self.raw_record(index))

    def preload_transformed(self) -> dict[str, Any]:
        """Populate the deterministic in-memory cache and return its attestation.

        Preloading is idempotent.  The complete tuple is published only after
        every selected record transforms successfully, so callers never observe
        a partially populated cache.
        """

        if self._cache is None:
            transformed = tuple(
                self._transform_at(index) for index in range(len(self._index))
            )
            if self.splits is not None:
                unexpected = sorted(
                    {
                        episode.split
                        for episode in transformed
                        if episode.split not in self.splits
                    }
                )
                if unexpected:
                    raise PermissionError(
                        f"Transformed cache contains undeclared splits: {unexpected}"
                    )
            self._cached_tensor_bytes = sum(
                episode_tensor_bytes(episode) for episode in transformed
            )
            self._cache = transformed
        return self.cache_stats()

    def cache_stats(self) -> dict[str, Any]:
        """Return deterministic cache inventory plus operational access counts."""

        return {
            "mode": "preload" if self._cache is not None else "disabled",
            "preloaded": self._cache is not None,
            "records": 0 if self._cache is None else len(self._cache),
            "tensor_bytes": int(self._cached_tensor_bytes),
            "source_parts": len(self.paths),
            "transform_calls": int(self._transform_calls),
            "cache_hits": int(self._cache_hits),
            "cache_misses": int(self._cache_misses),
            "returns_cloned_tensors": True,
        }

    def __getitem__(self, index: int) -> EpisodeTensors:
        if self._cache is None:
            self._cache_misses += 1
            return self._transform_at(index)
        self._cache_hits += 1
        return clone_episode_tensors(self._cache[index])


def _pad_tensor_sequence(
    tensors: Sequence[Tensor], padding_value: float | int | bool
) -> Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        list(tensors), batch_first=True, padding_value=padding_value
    )


def collate_episode_tensors(batch: Sequence[EpisodeTensors]) -> dict[str, Any]:
    """Pad variable-length episodes for ``batch_first=True`` Transformers.

    ``context_padding_mask`` and ``target_padding_mask`` follow PyTorch key
    padding semantics: ``True`` marks a padded position that must be ignored.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    feature_names = batch[0].feature_names
    if any(item.feature_names != feature_names for item in batch[1:]):
        raise ValueError("All episodes in a batch must use the same tensorizer state")

    result: dict[str, Any] = {
        "record_id": [item.record_id for item in batch],
        "family": [item.family for item in batch],
        "split": [item.split for item in batch],
        "time_unit": [item.time_unit for item in batch],
        "objective_eligibility": [
            copy.deepcopy(item.objective_eligibility) for item in batch
        ],
        "feature_names": feature_names,
        "feature_kinds": batch[0].feature_kinds,
        "context_token_kinds": [item.context_token_kinds for item in batch],
        "target_token_kinds": [item.target_token_kinds for item in batch],
        "node_neighbors": [copy.deepcopy(item.node_neighbors) for item in batch],
    }
    for prefix in _SEQUENCE_PREFIXES:
        lengths = torch.tensor(
            [getattr(item, f"{prefix}_length") for item in batch], dtype=torch.long
        )
        maximum = int(lengths.max().item())
        padding_mask = torch.ones((len(batch), maximum), dtype=torch.bool)
        for row, length in enumerate(lengths.tolist()):
            padding_mask[row, :length] = False
        result[f"{prefix}_lengths"] = lengths
        result[f"{prefix}_padding_mask"] = padding_mask

        for field in _MATRIX_FIELDS:
            tensors = [getattr(item, f"{prefix}_{field}") for item in batch]
            padding_value: float | int | bool = (
                False if tensors[0].dtype == torch.bool else 0
            )
            result[f"{prefix}_{field}"] = _pad_tensor_sequence(tensors, padding_value)
        for field in _VECTOR_FIELDS:
            tensors = [getattr(item, f"{prefix}_{field}") for item in batch]
            padding_value = -1 if field == "time_index" else 0
            result[f"{prefix}_{field}"] = _pad_tensor_sequence(tensors, padding_value)
    return result
