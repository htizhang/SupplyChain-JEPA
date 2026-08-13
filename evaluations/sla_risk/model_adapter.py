"""Leakage-safe Cargo2000 adapter for the released State encoder."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from supplychain_jepa import ModelBundle
from supplychain_jepa.data import STATUS_TO_ID, FittedTensorizerState

from .protocol import EVENT_SPECS, INCOMING_PREFIXES, LEG_PREFIXES

PLAN_TOKEN_COUNT = 32
ENTITY_TOKEN_COUNT = 4
EVENT_TOKEN_COUNT = 32
PLAN_START = 0
ENTITY_START = PLAN_TOKEN_COUNT
EVENT_START = ENTITY_START + ENTITY_TOKEN_COUNT
GLOBAL_POSITION = EVENT_START + EVENT_TOKEN_COUNT
TOKEN_COUNT = GLOBAL_POSITION + 1
CHECKPOINT_TO_CLOCK = {"plan": 0.0, "in_rcs": 1.0, "in_seg1": 2.0, "pre_join": 3.0}


@dataclass(frozen=True)
class DurationScaler:
    source_median: float
    source_upper: float

    @classmethod
    def fit(cls, raw: pd.DataFrame, source_process_ids: set[int]) -> DurationScaler:
        source = raw[raw["nr"].isin(source_process_ids)]
        planned_columns = [spec.planned for spec in EVENT_SPECS]
        values = source[planned_columns].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if not len(values):
            raise ValueError("No positive source planned durations")
        return cls(float(np.median(values)), float(np.quantile(values, 0.995)))

    def transform(self, raw: float, *, target_median: float, target_upper: float) -> float:
        clipped = float(np.clip(raw, 0.0, self.source_upper))
        mapped = target_median * math.log1p(clipped / max(self.source_median, 1e-6))
        mapped /= math.log(2.0)
        return float(np.clip(mapped, 0.0, target_upper))


def _metadata_id(state: FittedTensorizerState, namespace: str, value: str) -> int:
    return int(state.metadata_vocabs[namespace].get(value, 1))


def _categorical_id(state: FittedTensorizerState, feature: str, value: bool | str) -> int:
    key = f"bool:{str(value).lower()}" if isinstance(value, bool) else f"str:{value}"
    return int(state.categorical_vocabs.get(feature, {}).get(key, 1))


def _resolve_feature(index: Mapping[str, int], *names: str, purpose: str) -> str:
    for name in names:
        if name in index:
            return name
    raise ValueError(f"Released tensorizer has no feature for {purpose}: {names}")


def _event_type_name(state: FittedTensorizerState, event_type: str) -> str:
    candidates = {
        "rcs": ("RCS",),
        "dep": ("DEP",),
        "rcf": ("RCF", "ARR_HUB"),
        "dlv": ("DLV", "delivery_completed"),
    }[event_type]
    vocabulary = state.metadata_vocabs["token_type"]
    return next((candidate for candidate in candidates if candidate in vocabulary), candidates[0])


def _event_rank(spec: Any) -> int:
    if spec.event_type == "rcs":
        return 0
    if spec.event_type == "dep":
        return 1 + 2 * (spec.segment - 1)
    if spec.event_type == "rcf":
        return 2 + 2 * (spec.segment - 1)
    if spec.event_type == "dlv":
        return 7
    raise ValueError(spec.event_type)


def _leg_index(leg: str) -> int:
    return LEG_PREFIXES.index(leg)


def _set_numeric(
    values: np.ndarray,
    present: np.ndarray,
    status: np.ndarray,
    state: FittedTensorizerState,
    index: Mapping[str, int],
    row: int,
    position: int,
    feature: str,
    raw: float,
) -> None:
    if feature not in index or not np.isfinite(raw):
        return
    column = index[feature]
    values[row, position, column] = (raw - state.numeric_mean[column]) / max(
        state.numeric_std[column], 1e-6
    )
    present[row, position, column] = True
    status[row, position, column] = STATUS_TO_ID["observed"]


def _set_categorical(
    categorical: np.ndarray,
    present: np.ndarray,
    status: np.ndarray,
    state: FittedTensorizerState,
    index: Mapping[str, int],
    row: int,
    position: int,
    feature: str,
    raw: bool | str,
) -> None:
    if feature not in index:
        return
    column = index[feature]
    categorical[row, position, column] = _categorical_id(state, feature, raw)
    present[row, position, column] = True
    status[row, position, column] = STATUS_TO_ID["observed"]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def cargo_state_batch(
    examples: pd.DataFrame,
    state: FittedTensorizerState,
    duration_scaler: DurationScaler,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Map corrupted Cargo prefixes to the released canonical tensor interface."""

    batch_size = len(examples)
    feature_count = state.feature_count
    index = state.feature_index
    values = np.zeros((batch_size, TOKEN_COUNT, feature_count), np.float32)
    categorical = np.zeros((batch_size, TOKEN_COUNT, feature_count), np.int64)
    present = np.zeros((batch_size, TOKEN_COUNT, feature_count), bool)
    status = np.zeros((batch_size, TOKEN_COUNT, feature_count), np.int64)
    age = np.zeros_like(values)
    integer_fields = {
        name: np.zeros((batch_size, TOKEN_COUNT), np.int64)
        for name in (
            "time_index",
            "token_kind_id",
            "type_id",
            "entity_index",
            "product_index",
            "node_index",
            "source_index",
            "target_index",
        )
    }
    integer_fields["product_index"].fill(2)
    clock = np.zeros((batch_size, TOKEN_COUNT), np.float32)
    family = np.full(
        (batch_size, TOKEN_COUNT),
        _metadata_id(state, "family", "milestone_flow"),
        np.int64,
    )
    alignment = np.broadcast_to(
        np.arange(2, TOKEN_COUNT + 2, dtype=np.int64), (batch_size, TOKEN_COUNT)
    ).copy()
    padding = np.ones((batch_size, TOKEN_COUNT), bool)
    masks = {name: np.zeros_like(padding) for name in ("plan", "event", "state")}

    lead_feature = _resolve_feature(
        index,
        "topology_edge::lead_time",
        "topology_edge::delivery_lead_time",
        "topology_edge::downstream_shipment_lead_time",
        purpose="planned lead time",
    )
    travel_feature = _resolve_feature(
        index,
        "topology_edge::travel_time_days",
        "topology_edge::lead_time",
        purpose="planned travel time",
    )
    event_time_feature = _resolve_feature(
        index,
        "event::__event_time",
        "event::arrived_time",
        "entity::event_timestamp",
        purpose="observed event time",
    )
    event_due_feature = _resolve_feature(
        index, "event::due_time", "event::arrived_time", "entity::due_time", purpose="due time"
    )
    event_delay_feature = _resolve_feature(
        index,
        "event::failure_delay",
        "event::consolidation_wait",
        "entity::sequence_deviation",
        "entity::queue_wait",
        purpose="event delay",
    )
    entity_due_feature = _resolve_feature(
        index, "entity::due_time", "event::due_time", purpose="entity due time"
    )

    def upper(feature: str) -> float:
        column = index[feature]
        return max(state.numeric_mean[column] + 6.0 * state.numeric_std[column], 1.0)

    for row_index, (_, example) in enumerate(examples.iterrows()):
        checkpoint_clock = CHECKPOINT_TO_CLOCK[str(example["checkpoint"])]
        cumulative_plan = dict.fromkeys(LEG_PREFIXES, 0.0)
        cumulative_effective = dict.fromkeys(LEG_PREFIXES, 0.0)
        latest: dict[str, tuple[str, float]] = {}
        structural_legs: set[str] = set()
        observed_count = late_count = delivered_count = 0
        for spec_index, spec in enumerate(EVENT_SPECS):
            planned = example[spec.planned]
            if not np.isfinite(planned):
                continue
            structural_legs.add(spec.leg)
            cumulative_plan[spec.leg] += float(planned)
            leg_index = _leg_index(spec.leg)
            source_node = 2 + 16 * leg_index + _event_rank(spec)
            target_node = source_node + 1
            position = PLAN_START + spec_index
            padding[row_index, position] = False
            masks["plan"][row_index, position] = True
            integer_fields["token_kind_id"][row_index, position] = _metadata_id(
                state, "token_kind", "topology_edge"
            )
            integer_fields["type_id"][row_index, position] = _metadata_id(
                state, "token_type", "last_mile" if spec.event_type == "dlv" else "transport_leg"
            )
            integer_fields["entity_index"][row_index, position] = 2 + leg_index
            integer_fields["node_index"][row_index, position] = target_node
            integer_fields["source_index"][row_index, position] = source_node
            integer_fields["target_index"][row_index, position] = target_node
            integer_fields["time_index"][row_index, position] = _event_rank(spec)
            clock[row_index, position] = checkpoint_clock
            _set_numeric(
                values,
                present,
                status,
                state,
                index,
                row_index,
                position,
                lead_feature,
                duration_scaler.transform(
                    float(planned),
                    target_median=state.numeric_mean[index[lead_feature]],
                    target_upper=upper(lead_feature),
                ),
            )
            _set_numeric(
                values,
                present,
                status,
                state,
                index,
                row_index,
                position,
                travel_feature,
                duration_scaler.transform(
                    float(planned),
                    target_median=state.numeric_mean[index[travel_feature]],
                    target_upper=upper(travel_feature),
                ),
            )
            _set_categorical(
                categorical, present, status, state, index, row_index, position, "topology_edge::mode", "air"
            )
            _set_categorical(
                categorical, present, status, state, index, row_index, position, "topology_edge::route", "main"
            )
            if not bool(example[f"{spec.effective}__observed"] > 0.5):
                continue
            effective = float(example[spec.effective])
            cumulative_effective[spec.leg] += effective
            observed_count += 1
            late = effective > float(planned)
            late_count += int(late)
            delivered_count += int(spec.leg in INCOMING_PREFIXES and spec.event_type == "dlv")
            event_time = duration_scaler.transform(
                cumulative_effective[spec.leg],
                target_median=state.numeric_mean[index[event_time_feature]],
                target_upper=upper(event_time_feature),
            )
            due = duration_scaler.transform(
                cumulative_plan[spec.leg],
                target_median=state.numeric_mean[index[event_due_feature]],
                target_upper=upper(event_due_feature),
            )
            delay = duration_scaler.transform(
                max(effective - float(planned), 0.0),
                target_median=max(state.numeric_mean[index[event_delay_feature]], 0.5),
                target_upper=8.0,
            )
            position = EVENT_START + spec_index
            padding[row_index, position] = False
            masks["event"][row_index, position] = True
            masks["state"][row_index, position] = True
            integer_fields["token_kind_id"][row_index, position] = _metadata_id(
                state, "token_kind", "event"
            )
            integer_fields["type_id"][row_index, position] = _metadata_id(
                state, "token_type", _event_type_name(state, spec.event_type)
            )
            integer_fields["entity_index"][row_index, position] = 2 + leg_index
            integer_fields["node_index"][row_index, position] = target_node
            integer_fields["source_index"][row_index, position] = source_node
            integer_fields["target_index"][row_index, position] = target_node
            integer_fields["time_index"][row_index, position] = _event_rank(spec)
            clock[row_index, position] = checkpoint_clock
            for feature, raw in (
                (event_time_feature, event_time),
                ("event::__report_time", event_time),
                ("event::arrival_day", due),
                (event_due_feature, due),
                (event_delay_feature, delay),
            ):
                _set_numeric(values, present, status, state, index, row_index, position, feature, raw)
            for feature, raw in (("event::__observed", True), ("event::late", late), ("event::route", "main")):
                _set_categorical(
                    categorical, present, status, state, index, row_index, position, feature, raw
                )
            latest[spec.leg] = (_event_type_name(state, spec.event_type), event_time)

        for leg in sorted(structural_legs, key=_leg_index):
            leg_index = _leg_index(leg)
            position = ENTITY_START + leg_index
            padding[row_index, position] = False
            masks["state"][row_index, position] = True
            integer_fields["token_kind_id"][row_index, position] = _metadata_id(
                state, "token_kind", "entity"
            )
            integer_fields["type_id"][row_index, position] = _metadata_id(
                state, "token_type", "shipment_report"
            )
            integer_fields["entity_index"][row_index, position] = 2 + leg_index
            integer_fields["node_index"][row_index, position] = 2 + 16 * leg_index
            integer_fields["time_index"][row_index, position] = int(checkpoint_clock)
            clock[row_index, position] = checkpoint_clock
            last_event, last_time = latest.get(leg, ("NONE", 0.0))
            due = duration_scaler.transform(
                cumulative_plan[leg],
                target_median=state.numeric_mean[index[entity_due_feature]],
                target_upper=upper(entity_due_feature),
            )
            for feature, raw in (
                (entity_due_feature, due),
                ("entity::last_event_time", last_time),
                ("entity::latest_physical_event_time", last_time),
            ):
                _set_numeric(values, present, status, state, index, row_index, position, feature, raw)
            for feature, raw in (
                ("entity::last_event", last_event),
                ("entity::latest_reported_milestone", last_event),
                ("entity::report_available", last_event != "NONE"),
                ("entity::released", last_event != "NONE"),
                ("entity::delivered", last_event in {"DLV", "delivery_completed"}),
                ("entity::delivery_reported", last_event in {"DLV", "delivery_completed"}),
            ):
                _set_categorical(
                    categorical, present, status, state, index, row_index, position, feature, raw
                )

        position = GLOBAL_POSITION
        padding[row_index, position] = False
        masks["state"][row_index, position] = True
        integer_fields["token_kind_id"][row_index, position] = _metadata_id(
            state, "token_kind", "global"
        )
        integer_fields["type_id"][row_index, position] = _metadata_id(
            state, "token_type", "global_state"
        )
        integer_fields["entity_index"][row_index, position] = 2
        integer_fields["node_index"][row_index, position] = 2
        integer_fields["time_index"][row_index, position] = int(checkpoint_clock)
        clock[row_index, position] = checkpoint_clock
        for feature, raw in {
            "global::clock": checkpoint_clock,
            "global::period": checkpoint_clock,
            "global::reported_events": float(observed_count),
            "global::late_shipments": float(min(late_count, 8)),
            "global::delivered_shipments": float(delivered_count),
            "global::pipeline_in_transit": float(max(len(structural_legs) - delivered_count, 0)),
            "global::released_shipments": float(sum(event != "NONE" for event, _ in latest.values())),
        }.items():
            _set_numeric(values, present, status, state, index, row_index, position, feature, raw)

    def tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    batch = {
        "context_values": tensor(values, torch.float32),
        "context_categorical_ids": tensor(categorical, torch.long),
        "context_feature_present": tensor(present, torch.bool),
        "context_legal_observation_mask": tensor(present, torch.bool),
        "context_status_ids": tensor(status, torch.long),
        "context_age": tensor(age, torch.float32),
        "context_clock": tensor(clock, torch.float32),
        "context_family_id": tensor(family, torch.long),
        "context_alignment_index": tensor(alignment, torch.long),
        "context_padding_mask": tensor(padding, torch.bool),
    }
    for name, array in integer_fields.items():
        batch[f"context_{name}"] = tensor(array, torch.long)
    tensor_masks = {name: tensor(mask, torch.bool) for name, mask in masks.items()}
    tensor_masks["valid"] = ~batch["context_padding_mask"]
    return batch, tensor_masks


def encode_current_state(
    examples: pd.DataFrame,
    bundle: ModelBundle,
    duration_scaler: DurationScaler,
    *,
    encoder: torch.nn.Module | None = None,
    batch_size: int = 128,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Return the four pooled current-state blocks used by the fixed readout."""

    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path)
        expected = (len(examples), 4 * bundle.model.config.d_model)
        if cached.shape == expected:
            return cached
    encoder = bundle.model.online_encoder if encoder is None else encoder
    device = next(encoder.parameters()).device
    output = np.empty((len(examples), 4 * bundle.model.config.d_model), np.float32)
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            stop = min(start + batch_size, len(examples))
            batch, masks = cargo_state_batch(
                examples.iloc[start:stop], bundle.tensorizer.state, duration_scaler, device=device
            )
            tokens = encoder.encode_with_special_tokens(batch, "context").token_latents
            pooled = torch.cat(
                tuple(_masked_mean(tokens, masks[name]) for name in ("valid", "plan", "event", "state")),
                dim=1,
            )
            output[start:stop] = pooled.cpu().numpy()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npy")
        np.save(temporary, output)
        temporary.replace(cache_path)
    return output


def cache_key(label: str, examples: pd.DataFrame) -> str:
    digest = hashlib.sha256(label.encode())
    digest.update(
        pd.util.hash_pandas_object(
            examples[["process_id", "checkpoint"]], index=False
        ).to_numpy(dtype=np.uint64).tobytes()
    )
    return digest.hexdigest()[:20]
