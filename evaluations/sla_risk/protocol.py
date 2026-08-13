"""Cargo2000 prefixes and the critical-leg delayed-reporting protocol."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CHECKPOINTS = ("plan", "in_rcs", "in_seg1", "pre_join")
LEG_PREFIXES = ("i1", "i2", "i3", "o")
INCOMING_PREFIXES = ("i1", "i2", "i3")
HOP_COLUMNS = ("i1_hops", "i2_hops", "i3_hops", "o_hops")
PRIMARY_CHECKPOINT = "in_seg1"
DELTA_FROM_CHECKPOINT = "in_rcs"


@dataclass(frozen=True)
class EventSpec:
    leg: str
    event_type: str
    segment: int
    planned: str
    effective: str
    place: str | None


def _event_specs() -> tuple[EventSpec, ...]:
    specs: list[EventSpec] = []
    for leg in LEG_PREFIXES:
        specs.append(EventSpec(leg, "rcs", 0, f"{leg}_rcs_p", f"{leg}_rcs_e", None))
        for segment in range(1, 4):
            specs.extend(
                (
                    EventSpec(
                        leg,
                        "dep",
                        segment,
                        f"{leg}_dep_{segment}_p",
                        f"{leg}_dep_{segment}_e",
                        f"{leg}_dep_{segment}_place",
                    ),
                    EventSpec(
                        leg,
                        "rcf",
                        segment,
                        f"{leg}_rcf_{segment}_p",
                        f"{leg}_rcf_{segment}_e",
                        f"{leg}_rcf_{segment}_place",
                    ),
                )
            )
        specs.append(EventSpec(leg, "dlv", 0, f"{leg}_dlv_p", f"{leg}_dlv_e", None))
    return tuple(specs)


EVENT_SPECS = _event_specs()
PLANNED_COLUMNS = tuple(spec.planned for spec in EVENT_SPECS)
EFFECTIVE_COLUMNS = tuple(spec.effective for spec in EVENT_SPECS)
PLACE_COLUMNS = tuple(spec.place for spec in EVENT_SPECS if spec.place is not None)


def load_cargo_csv(path: str | Any) -> pd.DataFrame:
    frame = pd.read_csv(path, na_values="?").dropna(subset=["nr"]).copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["nr"] = frame["nr"].astype(int)
    frame["legs"] = frame["legs"].astype(int)
    return frame.reset_index(drop=True)


def _leg_duration(frame: pd.DataFrame, leg: str, suffix: str) -> pd.Series:
    columns = [f"{leg}_rcs_{suffix}"]
    for segment in range(1, 4):
        columns.extend((f"{leg}_dep_{segment}_{suffix}", f"{leg}_rcf_{segment}_{suffix}"))
    columns.append(f"{leg}_dlv_{suffix}")
    return frame[columns].sum(axis=1, min_count=1)


def _process_duration(frame: pd.DataFrame, suffix: str) -> pd.Series:
    incoming = pd.concat(
        [_leg_duration(frame, leg, suffix) for leg in INCOMING_PREFIXES], axis=1
    ).max(axis=1)
    return incoming + _leg_duration(frame, "o", suffix)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["process_plan"] = _process_duration(result, "p")
    result["process_effective"] = _process_duration(result, "e")
    result["label"] = (result["process_effective"] > result["process_plan"]).astype(int)
    return result


def is_revealed(spec: EventSpec, checkpoint: str) -> bool:
    if checkpoint == "plan" or spec.leg == "o":
        return False
    if checkpoint == "in_rcs":
        return spec.event_type == "rcs"
    if checkpoint == "in_seg1":
        return spec.event_type == "rcs" or (
            spec.event_type in {"dep", "rcf"} and spec.segment == 1
        )
    if checkpoint == "pre_join":
        return True
    raise ValueError(f"Unknown checkpoint: {checkpoint}")


def _visible_aggregates(row: pd.Series) -> dict[str, float]:
    planned_legs: dict[str, float] = {}
    projected_legs: dict[str, float] = {}
    visible_deltas: list[float] = []
    observed_events = 0
    positive_deviations = 0
    for leg in LEG_PREFIXES:
        planned_total = 0.0
        projected_total = 0.0
        for spec in EVENT_SPECS:
            if spec.leg != leg or not np.isfinite(row[spec.planned]):
                continue
            planned = float(row[spec.planned])
            planned_total += planned
            if bool(row[f"{spec.effective}__observed"] > 0.5):
                effective = float(row[spec.effective])
                projected_total += effective
                visible_deltas.append(effective - planned)
                observed_events += 1
                positive_deviations += int(effective > planned)
            else:
                projected_total += planned
        if planned_total:
            planned_legs[leg] = planned_total
            projected_legs[leg] = projected_total

    planned_process = max(planned_legs[leg] for leg in INCOMING_PREFIXES if leg in planned_legs)
    planned_process += planned_legs["o"]
    projected_process = max(
        projected_legs[leg] for leg in INCOMING_PREFIXES if leg in projected_legs
    )
    projected_process += projected_legs["o"]
    revealed_planned = [
        sum(
            float(row[spec.planned])
            for spec in EVENT_SPECS
            if spec.leg == leg
            and np.isfinite(row[spec.planned])
            and bool(row[f"{spec.effective}__observed"] > 0.5)
        )
        for leg in INCOMING_PREFIXES
        if leg in planned_legs
    ]
    progress = max(revealed_planned, default=0.0)
    maximum_delta = max(visible_deltas, default=0.0)
    overrun = projected_process - planned_process
    return {
        "projected_process": projected_process,
        "planned_process_visible": planned_process,
        "projected_overrun": overrun,
        "projected_overrun_ratio": overrun / max(planned_process, 1.0),
        "max_observed_delta": maximum_delta,
        "max_observed_delta_ratio": maximum_delta / max(planned_process, 1.0),
        "observed_violation_fraction": (
            positive_deviations / observed_events if observed_events else 0.0
        ),
        "remaining_planned_fraction": float(
            np.clip(1.0 - progress / max(planned_process, 1.0), 0.0, 1.0)
        ),
    }


def build_prefix_examples(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in frame.iterrows():
        for checkpoint in CHECKPOINTS:
            item: dict[str, Any] = {
                "process_id": int(source["nr"]),
                "checkpoint": checkpoint,
                "label": int(source["label"]),
                "legs": float(source["legs"]),
            }
            item.update({column: float(source[column]) for column in HOP_COLUMNS})
            for column in PLACE_COLUMNS:
                value = source[column]
                item[column] = "__structural_missing__" if pd.isna(value) else f"airport_{int(value)}"
            for spec in EVENT_SPECS:
                planned = source[spec.planned]
                structural = float(pd.notna(planned))
                item[spec.planned] = float(planned) if structural else np.nan
                item[f"{spec.effective}__structural"] = structural
                visible = bool(structural and is_revealed(spec, checkpoint))
                item[f"{spec.effective}__observed"] = float(visible)
                effective = float(source[spec.effective]) if visible else np.nan
                item[spec.effective] = effective
                item[f"{spec.effective}__delta"] = (
                    effective - float(planned) if visible else np.nan
                )
            item.update(_visible_aggregates(pd.Series(item)))
            rows.append(item)
    examples = pd.DataFrame(rows)
    examples["checkpoint"] = pd.Categorical(examples["checkpoint"], CHECKPOINTS, ordered=True)
    return examples


def critical_incoming_leg(row: pd.Series) -> str:
    """Select the longest incoming leg using planned information only."""

    candidates = []
    for leg in INCOMING_PREFIXES:
        planned = [
            float(row[spec.planned])
            for spec in EVENT_SPECS
            if spec.leg == leg and np.isfinite(row[spec.planned])
        ]
        if planned:
            candidates.append((sum(planned), leg))
    if not candidates:
        raise ValueError("Shipment has no incoming leg")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def apply_critical_prefix_delay(examples: pd.DataFrame) -> pd.DataFrame:
    """Hide one coherent incoming-leg report block and recompute summaries."""

    corrupted = examples.copy(deep=True)
    for _, positions in corrupted.groupby("process_id", sort=False).groups.items():
        positions = list(positions)
        reference = next(
            position
            for position in positions
            if str(corrupted.at[position, "checkpoint"]) == PRIMARY_CHECKPOINT
        )
        critical_leg = critical_incoming_leg(corrupted.loc[reference])
        for position in positions:
            checkpoint = str(corrupted.at[position, "checkpoint"])
            if checkpoint not in {DELTA_FROM_CHECKPOINT, PRIMARY_CHECKPOINT}:
                continue
            for spec in EVENT_SPECS:
                should_hide = spec.leg == critical_leg and (
                    spec.event_type == "rcs"
                    or (
                        checkpoint == PRIMARY_CHECKPOINT
                        and spec.event_type in {"dep", "rcf"}
                        and spec.segment == 1
                    )
                )
                if should_hide:
                    corrupted.at[position, f"{spec.effective}__observed"] = 0.0
                    corrupted.at[position, spec.effective] = np.nan
                    corrupted.at[position, f"{spec.effective}__delta"] = np.nan
            for name, value in _visible_aggregates(corrupted.loc[position]).items():
                corrupted.at[position, name] = value
    assert_no_hidden_value_leakage(corrupted)
    return corrupted


def assert_no_hidden_value_leakage(examples: pd.DataFrame) -> None:
    for spec in EVENT_SPECS:
        observed = examples[f"{spec.effective}__observed"].astype(bool)
        if examples.loc[~observed, spec.effective].notna().any():
            raise AssertionError(f"Hidden value leaked through {spec.effective}")
        if examples.loc[~observed, f"{spec.effective}__delta"].notna().any():
            raise AssertionError(f"Hidden delta leaked through {spec.effective}")


def feature_columns(examples: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["checkpoint", *PLACE_COLUMNS]
    excluded = {"process_id", "label", "remaining_planned_fraction", *categorical}
    numeric = [
        column
        for column in examples.columns
        if column not in excluded and not examples[column].isna().all()
    ]
    return numeric, categorical


def subset(examples: pd.DataFrame, process_ids: Iterable[int]) -> pd.DataFrame:
    return examples[examples["process_id"].isin(set(process_ids))].reset_index(drop=True)


def deterministic_rank_fraction(values: Sequence[float], process_ids: Sequence[int]) -> np.ndarray:
    score = np.asarray(values, dtype=np.float64)
    ids = np.asarray(process_ids, dtype=np.int64)
    if score.shape != ids.shape or not np.all(np.isfinite(score)):
        raise ValueError("Scores and process IDs must be finite arrays with equal shape")
    order = np.lexsort((ids, score))
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    return np.ones(len(score)) if len(score) <= 1 else ranks / float(len(score) - 1)


def rank_matched_probabilities(
    baseline: Sequence[float], priority: Sequence[float], process_ids: Sequence[int]
) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.float64)
    priority = np.asarray(priority, dtype=np.float64)
    ids = np.asarray(process_ids, dtype=np.int64)
    if not (baseline.shape == priority.shape == ids.shape):
        raise ValueError("Rank-matching inputs have inconsistent shapes")
    order = np.lexsort((ids, priority))
    output = np.empty_like(baseline)
    output[order] = np.sort(baseline)
    return output
