"""Bounded deterministic training windows for training.

This module is intentionally an adapter over the stable public objects in
``data.py`` and ``masking.py``.  It does not change tensorization.  The only
model-facing identities produced here are episode-local integer indices that
already exist in :class:`EpisodeTensors`.

Two invariants dominate the implementation:

* time is never silently truncated; causal history is preserved through global
  tokens and query/action-relevant trajectories, and an explicit error is
  raised when mandatory tokens alone exceed the budget;
* raw simulator identifiers, exogenous stream identifiers, and digests are
  never copied into model tensors.  A paired branch-group string is retained
  only as non-model batching metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .data import EpisodeTensorizer, EpisodeTensors
from .masking import MaskPlan

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
_SEQUENCE_FIELDS = _MATRIX_FIELDS + _VECTOR_FIELDS
ROLE_HIDDEN_CURRENT = 0
ROLE_ACTION_OMITTED_FUTURE = 1
ACTION_DIM = 8
DYNAMICS_FAMILIES = frozenset({"gym_invmgmt", "milestone_flow"})
CONTROL_VARIANTS = ("correct", "noop", "shuffled", "wrong_entity")
ACTION_CONDITIONING_MODES = frozenset({"legacy", "decision_only"})


def _derived_seed(seed: int, *parts: object) -> int:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


class WindowCapacityError(ValueError):
    """The declared token cap cannot contain the mandatory causal evidence."""


@dataclass(frozen=True)
class WindowSpec:
    max_tokens: int = 384
    target_window_steps: int = 4
    seed: int = 20260727

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 1 <= self.target_window_steps <= 4:
            raise ValueError("target_window_steps must be between one and four")


DEFAULT_WINDOW_SPEC = WindowSpec()


@dataclass(frozen=True)
class StateWindowGeometry:
    """Capacity-relevant geometry shared by production and audit paths."""

    query_type: str
    role_id: int
    horizon: int
    context_endpoint: int
    target_endpoint: int
    objective_indices: Tensor
    target_indices: Tensor
    eligible_context_mask: Tensor


@dataclass
class StateTrainingExample:
    record_id: str
    world_id: str  # batching/control metadata only; never a model tensor
    world_signature: str  # batching/control metadata only
    topology_id: str  # batching/control diagnostic only
    config_id: str  # batching/control diagnostic only
    family: str
    split: str
    query_type: str
    mask_kind: str
    role_id: int
    horizon: int
    context_endpoint: int
    target_endpoint: int
    selected_context_indices: Tensor
    selected_target_indices: Tensor
    context: dict[str, Tensor]
    target: dict[str, Tensor]
    context_token_kinds: tuple[str, ...]
    target_token_kinds: tuple[str, ...]
    objective_target_token_mask: Tensor
    objective_target_feature_mask: Tensor
    context_feature_keep_mask: Tensor


@dataclass
class DynamicsTrainingExample:
    record_id: str
    family: str
    split: str
    branch_group_id: str  # batching metadata only; never a model tensor
    branch_position: int
    control_variant: str
    decision_time: int
    horizon: int
    selected_context_indices: Tensor
    selected_target_indices: Tensor
    context: dict[str, Tensor]
    target: dict[str, Tensor]
    context_token_kinds: tuple[str, ...]
    target_token_kinds: tuple[str, ...]
    actions: Tensor
    action_actor_indices: Tensor
    action_target_indices: Tensor
    action_type_ids: Tensor
    action_times: Tensor
    action_quantities: Tensor
    known_exogenous: Tensor
    wrong_entity_topologically_distinct: bool | None


def _slice_episode(
    episode: EpisodeTensors, prefix: str, indices: Tensor
) -> dict[str, Tensor]:
    return {
        field: getattr(episode, f"{prefix}_{field}").index_select(0, indices)
        for field in _SEQUENCE_FIELDS
    }


def _query_identity_sets(episode: EpisodeTensors, target_indices: Tensor) -> set[int]:
    identities: set[int] = set()
    for field in ("entity_index", "node_index", "source_index", "target_index"):
        values = getattr(episode, f"target_{field}").index_select(0, target_indices)
        identities.update(int(value) for value in values.tolist() if int(value) > 1)
    return identities


def _expanded_neighbors(episode: EpisodeTensors, identities: set[int]) -> set[int]:
    result = set(identities)
    for identity in tuple(identities):
        result.update(int(value) for value in episode.node_neighbors.get(identity, ()))
    return result


def _token_touches(episode: EpisodeTensors, index: int, identities: set[int]) -> bool:
    return any(
        int(getattr(episode, f"context_{field}")[index]) in identities
        for field in ("entity_index", "node_index", "source_index", "target_index")
    )


def mandatory_context_indices(
    episode: EpisodeTensors,
    *,
    eligible: Tensor,
    target_indices: Tensor,
    extra_preserve_identities: set[int] | None = None,
) -> Tensor:
    """Return the exact mandatory subset used by production window selection.

    This helper is intentionally capacity-agnostic: callers may measure the
    mandatory floor without changing the production sampler or choosing a cap.
    """

    if eligible.shape != (episode.context_length,):
        raise ValueError("eligible mask is not aligned to context tokens")
    if eligible.dtype != torch.bool:
        raise ValueError("eligible mask must be boolean")
    if target_indices.ndim != 1:
        raise ValueError("target_indices must be one-dimensional")
    if target_indices.numel() and (
        bool((target_indices < 0).any())
        or bool((target_indices >= episode.target_length).any())
    ):
        raise ValueError("target_indices contain an out-of-range index")

    candidate_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    candidate_list = candidate_indices.tolist()
    core_identities = _query_identity_sets(episode, target_indices)
    core_identities.update(extra_preserve_identities or ())
    topology_identities = _expanded_neighbors(episode, core_identities)

    candidate_kinds = [episode.context_token_kinds[index] for index in candidate_list]
    global_action = torch.tensor(
        [kind in {"global", "action"} for kind in candidate_kinds],
        dtype=torch.bool,
        device=candidate_indices.device,
    )
    topology = torch.tensor(
        [kind.startswith("topology_") for kind in candidate_kinds],
        dtype=torch.bool,
        device=candidate_indices.device,
    )
    candidate_times = episode.context_time_index.index_select(0, candidate_indices)
    candidate_identities = tuple(
        getattr(episode, f"context_{field}").index_select(0, candidate_indices)
        for field in ("entity_index", "node_index", "source_index", "target_index")
    )

    mandatory = global_action.clone()
    if topology_identities and bool(topology.any()):
        topology_values = torch.tensor(
            sorted(topology_identities),
            dtype=candidate_identities[0].dtype,
            device=candidate_identities[0].device,
        )
        touches_topology = torch.zeros_like(mandatory)
        for identity_values in candidate_identities:
            touches_topology |= torch.isin(identity_values, topology_values)
        mandatory |= topology & touches_topology

    # Preserve the latest causal observation of every queried/acted identity.
    # Retaining every historical token for every future network-wide query can
    # itself exceed the declared cap; temporal coverage is carried separately
    # by mandatory global/action tokens and one-token-per-time fallback below.
    dynamic_observation = (candidate_times >= 0) & ~global_action
    for identity in sorted(core_identities):
        touches_identity = torch.zeros_like(mandatory)
        for identity_values in candidate_identities:
            touches_identity |= identity_values == identity
        touched = dynamic_observation & touches_identity
        if bool(touched.any()):
            latest = candidate_times[touched].max()
            mandatory |= touched & (candidate_times == latest)

    # A schema without dynamic global tokens still may not silently erase a
    # complete time point.  Keep one deterministic token for every causal time.
    for time_index in torch.unique(candidate_times[candidate_times >= 0]):
        time_positions = torch.nonzero(
            candidate_times == time_index,
            as_tuple=False,
        ).flatten()
        if not bool(mandatory.index_select(0, time_positions).any()):
            mandatory[time_positions[0]] = True

    return candidate_indices[mandatory].to(dtype=torch.long)


def _select_context_indices(
    episode: EpisodeTensors,
    *,
    eligible: Tensor,
    target_indices: Tensor,
    spec: WindowSpec,
    seed_parts: Sequence[object],
    seed_record_id: str | None = None,
    extra_preserve_identities: set[int] | None = None,
) -> Tensor:
    """Select causal context without silently dropping temporal coverage.

    Dynamic global tokens preserve every retained time point.  All causal
    actions, the latest causal queried/acted entities, and their topology
    neighborhoods are mandatory.  Only remaining entity/event/edge tokens are
    sampled.
    """

    candidates = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    if len(candidates) <= spec.max_tokens:
        return torch.tensor(candidates, dtype=torch.long)

    mandatory = set(
        mandatory_context_indices(
            episode,
            eligible=eligible,
            target_indices=target_indices,
            extra_preserve_identities=extra_preserve_identities,
        ).tolist()
    )

    if len(mandatory) > spec.max_tokens:
        raise WindowCapacityError(
            "mandatory causal/query/action tokens exceed max_tokens: "
            f"{len(mandatory)} > {spec.max_tokens}; increase the explicit cap"
        )

    remaining = [index for index in candidates if index not in mandatory]
    rng = random.Random(
        _derived_seed(
            spec.seed,
            episode.record_id if seed_record_id is None else seed_record_id,
            *seed_parts,
        )
    )
    rng.shuffle(remaining)
    chosen = mandatory | set(remaining[: spec.max_tokens - len(mandatory)])
    return torch.tensor(sorted(chosen), dtype=torch.long)


def _local_target_times(times: Sequence[int], endpoint: int, width: int) -> set[int]:
    ordered = sorted({int(value) for value in times if 0 <= int(value) <= endpoint})
    return set(ordered[-width:])


def state_window_geometry(
    episode: EpisodeTensors,
    plan: MaskPlan,
    *,
    spec: WindowSpec = DEFAULT_WINDOW_SPEC,
) -> StateWindowGeometry:
    """Compute exact State query endpoints, targets, and eligible context."""

    if plan.objective_target_token_mask.shape != (episode.target_length,):
        raise ValueError("objective target token mask is not aligned to episode")
    if plan.objective_target_feature_mask.shape != episode.target_feature_present.shape:
        raise ValueError("objective target feature mask is not aligned to episode")
    objective_indices = torch.nonzero(
        plan.objective_target_token_mask, as_tuple=False
    ).flatten()
    if objective_indices.numel() == 0:
        raise ValueError("State window requires at least one objective target")

    objective_times = episode.target_time_index.index_select(0, objective_indices)
    if bool((objective_times < 0).any()):
        raise ValueError("State objectives must refer to dynamic target tokens")

    if plan.kind == "future_horizon":
        query_type = "action_omitted_future"
        role_id = ROLE_ACTION_OMITTED_FUTURE
        if "cutoff" not in plan.metadata:
            raise ValueError("future_horizon plan requires an explicit cutoff")
        context_endpoint = int(plan.metadata["cutoff"])
        if bool((objective_times <= context_endpoint).any()):
            raise ValueError("future targets must occur strictly after the cutoff")
        target_endpoint = int(objective_times.max().item())
        horizon = int(plan.metadata.get("horizon", target_endpoint - context_endpoint))
        selected_times = _local_target_times(
            objective_times.tolist(), target_endpoint, spec.target_window_steps
        )
    else:
        query_type = "hidden_current"
        role_id = ROLE_HIDDEN_CURRENT
        target_endpoint = int(objective_times.max().item())
        context_endpoint = target_endpoint
        horizon = 0
        selected_times = _local_target_times(
            objective_times.tolist(), target_endpoint, spec.target_window_steps
        )

    target_time_keep = torch.tensor(
        [int(value) in selected_times for value in episode.target_time_index.tolist()],
        dtype=torch.bool,
    )
    target_indices = torch.nonzero(target_time_keep, as_tuple=False).flatten()
    if target_indices.numel() == 0:
        raise ValueError("Local target window is empty")
    if len(set(episode.target_time_index.index_select(0, target_indices).tolist())) > 4:
        raise AssertionError("local target window exceeded four steps")

    causal = (episode.context_time_index < 0) | (
        episode.context_time_index <= context_endpoint
    )
    eligible = causal & plan.context_token_keep_mask
    return StateWindowGeometry(
        query_type=query_type,
        role_id=role_id,
        horizon=horizon,
        context_endpoint=context_endpoint,
        target_endpoint=target_endpoint,
        objective_indices=objective_indices,
        target_indices=target_indices,
        eligible_context_mask=eligible,
    )


def build_state_example(
    episode: EpisodeTensors,
    plan: MaskPlan,
    *,
    spec: WindowSpec = DEFAULT_WINDOW_SPEC,
    epoch: int = 0,
) -> StateTrainingExample:
    """Build one endpoint-causal State JEPA example from a sampled mask plan."""

    geometry = state_window_geometry(episode, plan, spec=spec)
    context_indices = _select_context_indices(
        episode,
        eligible=geometry.eligible_context_mask,
        target_indices=geometry.objective_indices,
        spec=spec,
        seed_parts=(
            "state",
            epoch,
            plan.kind,
            geometry.context_endpoint,
            geometry.target_endpoint,
        ),
    )

    context = _slice_episode(episode, "context", context_indices)
    feature_keep = plan.context_feature_keep_mask.index_select(0, context_indices)
    context["feature_present"] &= feature_keep
    context["legal_observation_mask"] &= feature_keep
    context["values"] = context["values"].masked_fill(~feature_keep, 0.0)
    context["categorical_ids"] = context["categorical_ids"].masked_fill(
        ~feature_keep, 0
    )
    context["status_ids"] = context["status_ids"].masked_fill(~feature_keep, 0)
    context["age"] = context["age"].masked_fill(~feature_keep, 0.0)

    target = _slice_episode(episode, "target", geometry.target_indices)
    objective_token_mask = plan.objective_target_token_mask.index_select(
        0, geometry.target_indices
    )
    objective_feature_mask = plan.objective_target_feature_mask.index_select(
        0, geometry.target_indices
    )
    if not bool(objective_feature_mask.any()):
        raise ValueError("Local target window contains no objective features")

    return StateTrainingExample(
        record_id=episode.record_id,
        world_id=episode.world_id,
        world_signature=episode.world_signature,
        topology_id=episode.topology_id,
        config_id=episode.config_id,
        family=episode.family,
        split=episode.split,
        query_type=geometry.query_type,
        mask_kind=str(plan.kind),
        role_id=geometry.role_id,
        horizon=geometry.horizon,
        context_endpoint=geometry.context_endpoint,
        target_endpoint=geometry.target_endpoint,
        selected_context_indices=context_indices,
        selected_target_indices=geometry.target_indices,
        context=context,
        target=target,
        context_token_kinds=tuple(
            episode.context_token_kinds[i] for i in context_indices.tolist()
        ),
        target_token_kinds=tuple(
            episode.target_token_kinds[i] for i in geometry.target_indices.tolist()
        ),
        objective_target_token_mask=objective_token_mask,
        objective_target_feature_mask=objective_feature_mask,
        context_feature_keep_mask=feature_keep,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_pair(records: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    if len(records) < 2:
        raise ValueError("Dynamics requires at least two branches per group")
    first = records[0]
    family = str(first["simulator"]["family"])
    if family not in DYNAMICS_FAMILIES:
        raise ValueError(f"Dynamics rejects simulator family {family!r}")
    if not bool(first["objective_eligibility"].get("dynamics_jepa")):
        raise ValueError("Record is not eligible for Dynamics JEPA")
    branch = first.get("branch")
    if not isinstance(branch, Mapping):
        raise ValueError("Dynamics records require branch metadata")
    group_id = str(branch["branch_group_id"])
    branch_at = float(branch["branch_at"])
    reference_prefix = [
        step for step in first["steps"] if float(step["clock"]) < branch_at
    ]
    reference_topology = _canonical_json(first["topology"])
    exogenous_stream = first["episode"].get("exogenous_stream_id")

    for record in records[1:]:
        if str(record["simulator"]["family"]) != family:
            raise ValueError("A branch group cannot mix simulator families")
        if not bool(record["objective_eligibility"].get("dynamics_jepa")):
            raise ValueError("Every branch must be Dynamics eligible")
        other = record.get("branch")
        if (
            not isinstance(other, Mapping)
            or str(other.get("branch_group_id")) != group_id
        ):
            raise ValueError("Records do not share one branch group")
        if float(other.get("branch_at")) != branch_at:
            raise ValueError("Paired branches disagree on decision time")
        if _canonical_json(record["topology"]) != reference_topology:
            raise ValueError("Paired branches disagree on topology")
        prefix = [step for step in record["steps"] if float(step["clock"]) < branch_at]
        if _canonical_json(prefix) != _canonical_json(reference_prefix):
            raise ValueError(
                "Paired branches do not have an identical pre-action prefix"
            )
        if record["episode"].get("exogenous_stream_id") != exogenous_stream:
            raise ValueError("Paired branches do not share an exogenous stream")
    return group_id, branch_at


def _decision_step(record: Mapping[str, Any], branch_at: float) -> Mapping[str, Any]:
    matches = [step for step in record["steps"] if float(step["clock"]) == branch_at]
    if len(matches) != 1:
        raise ValueError("Each branch must have exactly one decision step")
    return matches[0]


def _action_positions(episode: EpisodeTensors, decision_time: int) -> list[int]:
    return [
        index
        for index, (kind, time_index) in enumerate(
            zip(episode.context_token_kinds, episode.context_time_index.tolist(), strict=False)
        )
        if kind == "action" and int(time_index) == decision_time
    ]


def _extract_actions(
    record: Mapping[str, Any],
    episode: EpisodeTensors,
    *,
    action_step_times: Sequence[int],
) -> dict[str, Tensor]:
    """Extract the supplied action sequence over the half-open interval [t,t+h).

    A simulator step with no action contributes no token. In particular, the
    previous action is never replayed to stand in for a genuine no-op step.
    """

    allowed_times = tuple(int(value) for value in action_step_times)
    if not allowed_times:
        raise ValueError("Dynamics action interval cannot be empty")
    vectors: list[list[float]] = []
    actor_indices: list[int] = []
    target_indices: list[int] = []
    type_ids: list[int] = []
    quantities: list[float] = []
    times: list[int] = []
    steps_by_time = {int(step["t"]): step for step in record["steps"]}
    for action_time in allowed_times:
        if action_time not in steps_by_time:
            raise ValueError(
                f"Action interval references absent simulator time {action_time}"
            )
        actions = list(steps_by_time[action_time]["actions"])
        positions = _action_positions(episode, action_time)
        if len(actions) != len(positions):
            raise ValueError(
                "Action sequence is not aligned with tensorized action tokens"
            )
        for action, position in zip(actions, positions, strict=False):
            # Ignore adapter-private labels; retain only the portable arrival lag.
            quantity = float(action.get("quantity", 0.0))
            actor = int(episode.context_entity_index[position])
            target = int(episode.context_target_index[position])
            product = int(episode.context_product_index[position])
            type_id = int(episode.context_type_id[position])
            observed_time = int(episode.context_time_index[position])
            if observed_time != action_time:
                raise ValueError("Tensorized action carries the wrong step time")
            features = action.get("features")
            arrival_day = (
                features.get("arrival_day")
                if isinstance(features, Mapping)
                else None
            )
            arrival_lag = (
                math.log1p(max(float(arrival_day) - action_time, 0.0))
                if arrival_day is not None
                else 0.0
            )
            vectors.append(
                [
                    quantity,
                    float(type_id),
                    float(observed_time),
                    float(actor),
                    float(target),
                    float(product),
                    1.0,
                    arrival_lag,
                ]
            )
            actor_indices.append(actor)
            target_indices.append(target)
            type_ids.append(type_id)
            quantities.append(quantity)
            times.append(observed_time)
    if not vectors:
        raise ValueError("Dynamics branch has no action in the supplied interval")
    return {
        "vectors": torch.tensor(vectors, dtype=torch.float32),
        "actor_indices": torch.tensor(actor_indices, dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "type_ids": torch.tensor(type_ids, dtype=torch.long),
        "quantities": torch.tensor(quantities, dtype=torch.float32),
        "times": torch.tensor(times, dtype=torch.long),
    }


def _topology_maps(
    record: Mapping[str, Any], episode: EpisodeTensors
) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[str, str]]]:
    node_positions = [
        index
        for index, kind in enumerate(episode.context_token_kinds)
        if kind == "topology_node"
    ]
    edge_positions = [
        index
        for index, kind in enumerate(episode.context_token_kinds)
        if kind == "topology_edge"
    ]
    nodes = list(record["topology"]["nodes"])
    edges = list(record["topology"]["edges"])
    if len(nodes) != len(node_positions) or len(edges) != len(edge_positions):
        raise ValueError("Topology tensors do not align with raw topology order")
    node_map = {
        str(node["node_id"]): int(episode.context_node_index[position])
        for node, position in zip(nodes, node_positions, strict=False)
    }
    edge_map = {
        str(edge["edge_id"]): int(episode.context_entity_index[position])
        for edge, position in zip(edges, edge_positions, strict=False)
    }
    endpoints = {
        str(edge["edge_id"]): (str(edge["source"]), str(edge["target"]))
        for edge in edges
    }
    return node_map, edge_map, endpoints


def _graph_distances(record: Mapping[str, Any], start: str) -> dict[str, int]:
    adjacency: dict[str, set[str]] = {
        str(node["node_id"]): set() for node in record["topology"]["nodes"]
    }
    for edge in record["topology"]["edges"]:
        source, target = str(edge["source"]), str(edge["target"])
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    if start not in adjacency:
        return {}
    distances = {start: 0}
    frontier = [start]
    while frontier:
        current = frontier.pop(0)
        for neighbor in sorted(adjacency[current]):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                frontier.append(neighbor)
    return distances


def _wrong_entity_actions(
    record: Mapping[str, Any],
    episode: EpisodeTensors,
    correct: dict[str, Tensor],
    *,
    seed: int,
) -> tuple[dict[str, Tensor], bool]:
    step = _decision_step(record, float(record["branch"]["branch_at"]))
    raw_actions = list(step["actions"])
    node_map, edge_map, endpoints = _topology_maps(record, episode)
    vectors = correct["vectors"].clone()
    actors = correct["actor_indices"].clone()
    targets = correct["target_indices"].clone()
    all_distinct = True

    for row, action in enumerate(raw_actions):
        raw_target = str(action.get("target_id"))
        rng = random.Random(_derived_seed(seed, episode.record_id, "wrong", row))
        if raw_target in endpoints:
            original_endpoints = set(endpoints[raw_target])
            preferred = [
                edge_id
                for edge_id, pair in sorted(endpoints.items())
                if edge_id != raw_target and original_endpoints.isdisjoint(pair)
            ]
            fallback = [
                edge_id for edge_id in sorted(endpoints) if edge_id != raw_target
            ]
            candidates = preferred or fallback
            if candidates:
                chosen_edge = candidates[rng.randrange(len(candidates))]
                source, target_node = endpoints[chosen_edge]
                chosen_actor = target_node if target_node in node_map else source
                actors[row] = node_map[chosen_actor]
                targets[row] = edge_map[chosen_edge]
                vectors[row, 3] = float(actors[row])
                vectors[row, 4] = float(targets[row])
                all_distinct &= chosen_edge in preferred
                continue
        if raw_target in node_map:
            distances = _graph_distances(record, raw_target)
            preferred_nodes = sorted(
                node for node, distance in distances.items() if distance >= 2
            )
            fallback_nodes = sorted(node for node in node_map if node != raw_target)
            candidates = preferred_nodes or fallback_nodes
            if candidates:
                chosen = candidates[rng.randrange(len(candidates))]
                targets[row] = node_map[chosen]
                vectors[row, 4] = float(targets[row])
                all_distinct &= chosen in preferred_nodes
                continue
        # If actor/target are not topology objects, choose a distinct feasible
        # topology node for the actor while preserving the action semantics.
        actor_candidates = sorted(
            value for value in node_map.values() if value != int(actors[row])
        )
        if actor_candidates:
            actors[row] = actor_candidates[rng.randrange(len(actor_candidates))]
            vectors[row, 3] = float(actors[row])
            all_distinct = False
        else:
            raise ValueError("No feasible wrong-entity control exists")

    altered = bool(
        (actors != correct["actor_indices"]).any()
        or (targets != correct["target_indices"]).any()
    )
    if not altered:
        raise ValueError("Wrong-entity control did not alter an entity")
    result = {key: value.clone() for key, value in correct.items()}
    result["vectors"] = vectors
    result["actor_indices"] = actors
    result["target_indices"] = targets
    return result, all_distinct


def _action_variant(
    correct: dict[str, Tensor],
    variant: str,
    *,
    shuffled: dict[str, Tensor] | None = None,
    wrong: tuple[dict[str, Tensor], bool] | None = None,
) -> tuple[dict[str, Tensor], bool | None]:
    if variant == "correct":
        return {key: value.clone() for key, value in correct.items()}, None
    if variant == "noop":
        result = {key: value.clone() for key, value in correct.items()}
        result["vectors"][:, 0] = 0.0
        result["quantities"].zero_()
        return result, None
    if variant == "shuffled":
        if shuffled is None:
            raise ValueError("shuffled action source is required")
        return {key: value.clone() for key, value in shuffled.items()}, None
    if variant == "wrong_entity":
        if wrong is None:
            raise ValueError("wrong-entity action source is required")
        return ({key: value.clone() for key, value in wrong[0].items()}, wrong[1])
    raise ValueError(f"Unsupported control variant {variant!r}")


def build_dynamics_examples(
    records: Sequence[Mapping[str, Any]],
    tensorizer: EpisodeTensorizer,
    *,
    horizons: Sequence[int] = (1, 2, 4),
    spec: WindowSpec = DEFAULT_WINDOW_SPEC,
    include_controls: bool = True,
    action_conditioning: str = "legacy",
) -> list[DynamicsTrainingExample]:
    """Build paired, endpoint-causal Dynamics examples from raw branch records.

    The branch-group ID is copied only to ``branch_group_id`` for grouped
    sampling/bootstrap.  It and ``exogenous_stream_id`` are absent from every
    model tensor.
    """

    group_id, branch_at = _validate_pair(records)
    if action_conditioning not in ACTION_CONDITIONING_MODES:
        raise ValueError(
            "Dynamics action conditioning must be one of "
            f"{sorted(ACTION_CONDITIONING_MODES)}, got {action_conditioning!r}"
        )
    requested_horizons = tuple(int(value) for value in horizons)
    if not requested_horizons or any(value < 1 for value in requested_horizons):
        raise ValueError("Dynamics horizons must be positive")

    episodes = [tensorizer.transform(record) for record in records]
    variants = CONTROL_VARIANTS if include_controls else ("correct",)
    output: list[DynamicsTrainingExample] = []

    for branch_position, (record, episode) in enumerate(zip(records, episodes, strict=False)):
        decision_step = _decision_step(record, branch_at)
        decision_time = int(decision_step["t"])
        step_times = [int(step["t"]) for step in record["steps"]]
        if decision_time not in step_times:
            raise ValueError("Decision time is absent from episode steps")
        decision_position = step_times.index(decision_time)

        for horizon in requested_horizons:
            endpoint_position = decision_position + horizon
            if endpoint_position >= len(step_times):
                continue
            action_step_times = (
                (decision_time,)
                if action_conditioning == "decision_only"
                else step_times[decision_position:endpoint_position]
            )
            correct = _extract_actions(
                record,
                episode,
                action_step_times=action_step_times,
            )
            shuffled_position = (branch_position + 1) % len(records)
            shuffled_record = records[shuffled_position]
            shuffled_episode = episodes[shuffled_position]
            shuffled_step_times = [int(step["t"]) for step in shuffled_record["steps"]]
            shuffled_decision_time = int(
                _decision_step(shuffled_record, branch_at)["t"]
            )
            shuffled_decision_position = shuffled_step_times.index(
                shuffled_decision_time
            )
            shuffled_action_step_times = (
                (shuffled_decision_time,)
                if action_conditioning == "decision_only"
                else shuffled_step_times[
                    shuffled_decision_position : shuffled_decision_position + horizon
                ]
            )
            shuffled_source = _extract_actions(
                shuffled_record,
                shuffled_episode,
                action_step_times=shuffled_action_step_times,
            )
            wrong = _wrong_entity_actions(record, episode, correct, seed=spec.seed)

            action_identities = {
                int(value)
                for value in correct["actor_indices"].tolist()
                if int(value) > 1
            }
            action_identities.update(
                int(value)
                for value in correct["target_indices"].tolist()
                if int(value) > 1
            )
            future_times = step_times[decision_position + 1 : endpoint_position + 1]
            local_times = set(future_times[-spec.target_window_steps :])
            target_keep = torch.tensor(
                [
                    int(value) in local_times
                    for value in episode.target_time_index.tolist()
                ],
                dtype=torch.bool,
            )
            target_indices = torch.nonzero(target_keep, as_tuple=False).flatten()
            if target_indices.numel() == 0:
                raise ValueError("Dynamics local future target is empty")
            if (
                len(
                    set(
                        episode.target_time_index.index_select(
                            0, target_indices
                        ).tolist()
                    )
                )
                > 4
            ):
                raise AssertionError("Dynamics target window exceeded four steps")

            # Current decision actions are represented only in the action input,
            # never duplicated in state context.  Earlier actions remain causal.
            # The canonical schema does not promise that a snapshot stamped at
            # branch_at is pre-action.  Use only the strictly shared prefix; this
            # prevents same-step post-action state from leaking into context.
            context_eligible = (episode.context_time_index < 0) | (
                episode.context_time_index < decision_time
            )
            current_action = torch.tensor(
                [
                    kind == "action" and int(time_index) == decision_time
                    for kind, time_index in zip(
                        episode.context_token_kinds,
                        episode.context_time_index.tolist(), strict=False,
                    )
                ],
                dtype=torch.bool,
            )
            context_eligible &= ~current_action
            context_indices = _select_context_indices(
                episode,
                eligible=context_eligible,
                target_indices=target_indices,
                spec=spec,
                # Paired branches share one pre-action information state.  Use
                # the branch-group identity rather than branch/record identity
                # so context subsampling cannot leak which action was taken.
                seed_parts=("dynamics", horizon),
                seed_record_id=group_id,
                extra_preserve_identities=action_identities,
            )
            context = _slice_episode(episode, "context", context_indices)
            target = _slice_episode(episode, "target", target_indices)

            for variant in variants:
                selected_action, distinct = _action_variant(
                    correct,
                    variant,
                    shuffled=shuffled_source,
                    wrong=wrong,
                )
                if action_conditioning == "decision_only" and not torch.all(
                    selected_action["times"] == decision_time
                ):
                    raise AssertionError(
                        "decision_only action tensor contains a post-decision action"
                    )
                output.append(
                    DynamicsTrainingExample(
                        record_id=episode.record_id,
                        family=episode.family,
                        split=episode.split,
                        branch_group_id=group_id,
                        branch_position=branch_position,
                        control_variant=variant,
                        decision_time=decision_time,
                        horizon=horizon,
                        selected_context_indices=context_indices.clone(),
                        selected_target_indices=target_indices.clone(),
                        context={key: value.clone() for key, value in context.items()},
                        target={key: value.clone() for key, value in target.items()},
                        context_token_kinds=tuple(
                            episode.context_token_kinds[i]
                            for i in context_indices.tolist()
                        ),
                        target_token_kinds=tuple(
                            episode.target_token_kinds[i]
                            for i in target_indices.tolist()
                        ),
                        actions=selected_action["vectors"],
                        action_actor_indices=selected_action["actor_indices"],
                        action_target_indices=selected_action["target_indices"],
                        action_type_ids=selected_action["type_ids"],
                        action_times=selected_action["times"],
                        action_quantities=selected_action["quantities"],
                        known_exogenous=torch.empty((1, 0), dtype=torch.float32),
                        wrong_entity_topologically_distinct=distinct,
                    )
                )
    if not output:
        raise ValueError("No requested Dynamics horizon fits this branch group")
    return output


def _pad_sequences(
    values: Sequence[Tensor], padding_value: float | int | bool = 0
) -> Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        list(values), batch_first=True, padding_value=padding_value
    )


def _collate_views(
    examples: Sequence[StateTrainingExample | DynamicsTrainingExample],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix in ("context", "target"):
        lengths = torch.tensor(
            [
                example.context["values"].shape[0]
                if prefix == "context"
                else example.target["values"].shape[0]
                for example in examples
            ],
            dtype=torch.long,
        )
        maximum = int(lengths.max().item())
        padding_mask = torch.ones((len(examples), maximum), dtype=torch.bool)
        for row, length in enumerate(lengths.tolist()):
            padding_mask[row, :length] = False
        result[f"{prefix}_lengths"] = lengths
        result[f"{prefix}_padding_mask"] = padding_mask
        for field in _SEQUENCE_FIELDS:
            tensors = [getattr(example, prefix)[field] for example in examples]
            if field == "time_index":
                padding: float | int | bool = -1
            elif tensors[0].dtype == torch.bool:
                padding = False
            else:
                padding = 0
            result[f"{prefix}_{field}"] = _pad_sequences(tensors, padding)
    return result


def collate_state_examples(batch: Sequence[StateTrainingExample]) -> dict[str, Any]:
    """Pad State windows into the key layout consumed by the encoder/predictor."""

    if not batch:
        raise ValueError("Cannot collate an empty State batch")
    result = _collate_views(batch)
    result.update(
        {
            "record_id": [example.record_id for example in batch],
            "world_id": [example.world_id for example in batch],
            "world_signature": [example.world_signature for example in batch],
            "topology_id": [example.topology_id for example in batch],
            "config_id": [example.config_id for example in batch],
            "family": [example.family for example in batch],
            "split": [example.split for example in batch],
            "query_type": [example.query_type for example in batch],
            "mask_kind": [example.mask_kind for example in batch],
            "selected_context_indices": _pad_sequences(
                [example.selected_context_indices for example in batch], -1
            ),
            "selected_target_indices": _pad_sequences(
                [example.selected_target_indices for example in batch], -1
            ),
            "objective_target_token_mask": _pad_sequences(
                [example.objective_target_token_mask for example in batch], False
            ),
            "objective_target_feature_mask": _pad_sequences(
                [example.objective_target_feature_mask for example in batch], False
            ),
            "context_feature_keep_mask": _pad_sequences(
                [example.context_feature_keep_mask for example in batch], False
            ),
            "role_ids": _pad_sequences(
                [
                    torch.full(
                        (example.target["values"].shape[0],),
                        example.role_id,
                        dtype=torch.long,
                    )
                    for example in batch
                ],
                0,
            ),
            "horizons": _pad_sequences(
                [
                    torch.full(
                        (example.target["values"].shape[0],),
                        float(example.horizon),
                        dtype=torch.float32,
                    )
                    for example in batch
                ],
                0.0,
            ),
        }
    )
    return result


def collate_dynamics_examples(
    batch: Sequence[DynamicsTrainingExample],
) -> dict[str, Any]:
    """Pad Dynamics windows and actions into model-ready tensors.

    ``batching_metadata`` is deliberately separated from model tensors.  It is
    suitable for paired-world sampling and bootstrap grouping only.
    """

    if not batch:
        raise ValueError("Cannot collate an empty Dynamics batch")
    result = _collate_views(batch)
    action_lengths = torch.tensor([example.actions.shape[0] for example in batch])
    max_actions = int(action_lengths.max().item())
    action_padding_mask = torch.ones((len(batch), max_actions), dtype=torch.bool)
    for row, length in enumerate(action_lengths.tolist()):
        action_padding_mask[row, :length] = False
    result.update(
        {
            "record_id": [example.record_id for example in batch],
            "family": [example.family for example in batch],
            "split": [example.split for example in batch],
            "control_variant": [example.control_variant for example in batch],
            "batching_metadata": [
                {
                    "branch_group_id": example.branch_group_id,
                    "branch_position": example.branch_position,
                }
                for example in batch
            ],
            "selected_context_indices": _pad_sequences(
                [example.selected_context_indices for example in batch], -1
            ),
            "selected_target_indices": _pad_sequences(
                [example.selected_target_indices for example in batch], -1
            ),
            "actions": _pad_sequences([example.actions for example in batch], 0.0),
            "action_padding_mask": action_padding_mask,
            "action_actor_indices": _pad_sequences(
                [example.action_actor_indices for example in batch], 0
            ),
            "action_target_indices": _pad_sequences(
                [example.action_target_indices for example in batch], 0
            ),
            "action_type_ids": _pad_sequences(
                [example.action_type_ids for example in batch], 0
            ),
            "action_times": _pad_sequences(
                [example.action_times for example in batch], 0
            ),
            "action_quantities": _pad_sequences(
                [example.action_quantities for example in batch], 0.0
            ),
            "known_exogenous": torch.cat(
                [example.known_exogenous for example in batch], dim=0
            ),
            "decision_times": torch.tensor(
                [example.decision_time for example in batch], dtype=torch.long
            ),
            "horizons": torch.tensor(
                [[float(example.horizon)] for example in batch], dtype=torch.float32
            ),
        }
    )
    if result["actions"].shape[-1] != ACTION_DIM:
        raise AssertionError("Dynamics action vectors do not match model action_dim")
    return result
