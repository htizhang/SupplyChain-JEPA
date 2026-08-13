from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from evaluations.sla_risk.evaluation import (
    RANDOM_SEEDS,
    _resolve_data_file,
    _verify_external_file,
    aggregate_random_controls,
    blend_ranking,
    fit_prototype,
    stage_delta,
)
from evaluations.sla_risk.model_adapter import TOKEN_COUNT, DurationScaler, cargo_state_batch
from evaluations.sla_risk.protocol import (
    CHECKPOINTS,
    EVENT_SPECS,
    HOP_COLUMNS,
    PRIMARY_CHECKPOINT,
    add_targets,
    apply_critical_prefix_delay,
    build_prefix_examples,
    critical_incoming_leg,
)
from supplychain_jepa.data import TENSORIZER_FORMAT, FittedTensorizerState


def synthetic_raw(processes: int = 6) -> pd.DataFrame:
    rows = []
    for process_id in range(processes):
        row: dict[str, float] = {
            "nr": process_id,
            "legs": 4,
            **{column: 1.0 for column in HOP_COLUMNS},
        }
        for spec in EVENT_SPECS:
            planned = {"i1": 1.0, "i2": 3.0, "i3": 2.0, "o": 1.0}[spec.leg]
            row[spec.planned] = planned
            row[spec.effective] = planned + (1.0 if process_id % 2 and spec.leg == "i2" else 0.0)
            if spec.place is not None:
                row[spec.place] = 100 + process_id
        rows.append(row)
    return pd.DataFrame(rows)


def test_critical_prefix_hides_coherent_block_without_leakage() -> None:
    examples = build_prefix_examples(add_targets(synthetic_raw(2)))
    before = examples[
        (examples.process_id == 0) & (examples.checkpoint.astype(str) == PRIMARY_CHECKPOINT)
    ].iloc[0]
    assert critical_incoming_leg(before) == "i2"

    delayed = apply_critical_prefix_delay(examples)
    in_rcs = delayed[(delayed.process_id == 0) & (delayed.checkpoint.astype(str) == "in_rcs")].iloc[0]
    in_seg1 = delayed[
        (delayed.process_id == 0) & (delayed.checkpoint.astype(str) == "in_seg1")
    ].iloc[0]

    assert in_rcs["i2_rcs_e__observed"] == 0
    assert pd.isna(in_rcs["i2_rcs_e"])
    for column in ("i2_rcs_e", "i2_dep_1_e", "i2_rcf_1_e"):
        assert in_seg1[f"{column}__observed"] == 0
        assert pd.isna(in_seg1[column])
        assert pd.isna(in_seg1[f"{column}__delta"])
    assert in_seg1["i1_dep_1_e__observed"] == 1
    assert in_seg1["observed_violation_fraction"] == 0


def test_critical_leg_selection_uses_only_planned_values() -> None:
    examples = build_prefix_examples(add_targets(synthetic_raw(1)))
    row = examples[examples.checkpoint.astype(str) == "in_seg1"].iloc[0].copy()
    assert critical_incoming_leg(row) == "i2"
    for spec in EVENT_SPECS:
        if spec.leg == "i1":
            row[spec.effective] = 1_000_000.0
    assert critical_incoming_leg(row) == "i2"


def test_external_file_resolution_verifies_sha(tmp_path: Path) -> None:
    payload = b"external-data-fixture"
    path = tmp_path / "file.csv"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert _resolve_data_file(tmp_path, ("file.csv",), expected) == path
    assert _verify_external_file(path, expected) == path
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _resolve_data_file(tmp_path, ("file.csv",), "0" * 64)


def test_stage_delta_prototype_and_rank_blend_are_deterministic() -> None:
    examples = pd.DataFrame(
        [
            {"process_id": pid, "checkpoint": checkpoint, "label": pid % 2}
            for pid in range(6)
            for checkpoint in CHECKPOINTS
        ]
    )
    latent = np.zeros((len(examples), 4), np.float32)
    for index, row in examples.iterrows():
        if row.checkpoint == "in_seg1":
            latent[index] = np.array([row.label * 3.0, row.process_id, 1.0, -row.label])
    delta = stage_delta(examples, latent)
    prototype = fit_prototype(delta, frozenset(range(6)))
    evidence = prototype.score(delta.values)
    baseline = np.linspace(0.1, 0.9, 6)
    first = blend_ranking(baseline, evidence, delta.process_ids)
    second = blend_ranking(baseline, evidence, delta.process_ids)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.sort(baseline))
    assert evidence[delta.labels == 1].mean() > evidence[delta.labels == 0].mean()


def test_random_control_aggregation_requires_canonical_ten_seeds() -> None:
    rows = [{"seed": seed, "auprc": 0.35 + index / 1000} for index, seed in enumerate(RANDOM_SEEDS)]
    summary = aggregate_random_controls(rows)
    assert summary["seeds"] == list(RANDOM_SEEDS)
    assert summary["auprc_mean"] == pytest.approx(np.mean([row["auprc"] for row in rows]))
    assert summary["auprc_sample_sd"] > 0
    with pytest.raises(ValueError, match="exactly 10"):
        aggregate_random_controls(rows[:-1])


def test_cargo_adapter_uses_released_tensor_contract_without_a_model() -> None:
    features = (
        "topology_edge::lead_time",
        "topology_edge::travel_time_days",
        "event::__event_time",
        "event::due_time",
        "event::failure_delay",
        "entity::due_time",
    )
    state = FittedTensorizerState(
        state_version=TENSORIZER_FORMAT,
        train_splits=(),
        feature_names=features,
        feature_kinds=("numeric",) * len(features),
        numeric_mean=(1.0,) * len(features),
        numeric_std=(1.0,) * len(features),
        categorical_vocabs={},
        target_categorical_vocabs={},
        metadata_vocabs={
            "family": {"milestone_flow": 2},
            "token_kind": {"topology_edge": 2, "event": 3, "entity": 4, "global": 5},
            "token_type": {
                "transport_leg": 2,
                "last_mile": 3,
                "RCS": 4,
                "DEP": 5,
                "RCF": 6,
                "DLV": 7,
                "shipment_report": 8,
                "global_state": 9,
            },
        },
    )
    examples = apply_critical_prefix_delay(build_prefix_examples(add_targets(synthetic_raw(1))))
    row = examples[examples.checkpoint.astype(str) == PRIMARY_CHECKPOINT]
    batch, masks = cargo_state_batch(
        row,
        state,
        DurationScaler(source_median=2.0, source_upper=10.0),
        device=torch.device("cpu"),
    )
    assert batch["context_values"].shape == (1, TOKEN_COUNT, len(features))
    assert batch["context_legal_observation_mask"].equal(batch["context_feature_present"])
    assert masks["valid"].sum() > masks["event"].sum() > 0
    # i2 is the critical leg; its RCS event token (the ninth event) is absent.
    assert bool(batch["context_padding_mask"][0, 32 + 4 + 8])


def test_reference_boundary_is_explicitly_negative() -> None:
    reference = json.loads(
        json.dumps(
            {
                "baseline": 0.499222,
                "pretrained": 0.477540,
                "matched_random": 0.479024,
            }
        )
    )
    assert reference["pretrained"] < reference["baseline"]
    assert reference["matched_random"] < reference["baseline"]
