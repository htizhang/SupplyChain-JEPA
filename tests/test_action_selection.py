from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluations.action_selection import (
    EvaluationConfig,
    ReconstructionConfig,
    TinyBackbone,
    build_world_splits,
    generate_reconstruction_worlds,
    run_evaluation,
)
from evaluations.action_selection.protocol import summarize_protocol_results


def _small_config() -> EvaluationConfig:
    return EvaluationConfig(
        reconstruction=ReconstructionConfig(
            worlds=30,
            lookback=16,
            horizon=4,
            future_replications=3,
            seed=7713,
        ),
        split_salts=(101, 202, 303, 404, 505),
        learning_rates=(3e-4, 1e-4),
        epochs=2,
        batch_size=64,
        extra_trees_estimators=16,
        device="cpu",
    )


def test_reconstruction_is_deterministic_and_quantity_matched() -> None:
    config = ReconstructionConfig(
        worlds=4,
        lookback=16,
        horizon=4,
        future_replications=3,
        seed=991,
    )
    first = generate_reconstruction_worlds(config)
    second = generate_reconstruction_worlds(config)

    assert first == second
    assert len(first) == 4
    for world in first:
        assert len(world.public_history) == 16
        assert world.regular_plan.total_quantity == pytest.approx(
            world.split_plan.total_quantity
        )
        assert len(world.regular_plan.orders) == 1
        assert len(world.split_plan.orders) == 2
        assert world.continuation_policy == "no_further_orders"
        assert len(world.paired_future_costs) == 3
        for pair in world.paired_future_costs:
            assert pair.noise_digest
            assert np.isfinite(pair.regular_cost)
            assert np.isfinite(pair.split_cost)


def test_world_splits_are_disjoint_and_repeatable() -> None:
    world_ids = [f"world-{index:03d}" for index in range(25)]
    first = build_world_splits(world_ids, salt=17)
    second = build_world_splits(world_ids, salt=17)

    assert first == second
    assert set(first.fit).isdisjoint(first.select)
    assert set(first.fit).isdisjoint(first.report)
    assert set(first.select).isdisjoint(first.report)
    assert set(first.fit) | set(first.select) | set(first.report) == set(world_ids)


def test_small_end_to_end_writes_per_split_json(tmp_path: Path) -> None:
    config = _small_config()
    backend = TinyBackbone(
        history_features=12,
        state_dim=16,
        seed=config.reconstruction.seed,
    )

    summary = run_evaluation(config, backend=backend, output_dir=tmp_path)

    assert summary["provenance"]["kind"] == "clean_reconstruction"
    assert summary["contract"]["lookback"] == 16
    assert summary["contract"]["horizon"] == 4
    assert summary["contract"]["continuation_policy"] == "no_further_orders"
    assert summary["contract"]["quantity_matched"] is True
    assert summary["training"]["optimizer"] == "AdamW"
    assert summary["training"]["learning_rates"] == [3e-4, 1e-4]
    assert len(summary["splits"]) == 5
    assert np.isfinite(summary["metrics"]["pretrained_regret_mean"])
    assert np.isfinite(summary["metrics"]["matched_random_regret_mean"])
    assert np.isfinite(summary["metrics"]["extra_trees_regret_mean"])

    for index in range(5):
        path = tmp_path / f"split_{index:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["provenance"]["kind"] == "clean_reconstruction"
        assert payload["split_index"] == index
        assert set(payload["worlds"]) == {"fit", "select", "report"}
        assert payload["worlds"]["report"]
        assert set(payload["regret"]) == {
            "pretrained",
            "matched_random",
            "extra_trees",
        }

    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk == summary


def test_tiny_state_encoders_remain_frozen_during_adaptation(tmp_path: Path) -> None:
    config = _small_config()
    backend = TinyBackbone(history_features=12, state_dim=12, seed=123)
    before = {
        name: value.detach().clone()
        for name, value in backend.pretrained_state_encoder.state_dict().items()
    }

    run_evaluation(config, backend=backend, output_dir=tmp_path)

    assert all(
        not parameter.requires_grad
        for parameter in backend.pretrained_state_encoder.parameters()
    )
    for name, value in backend.pretrained_state_encoder.state_dict().items():
        assert torch.equal(value, before[name])


def test_canonical_protocol_summary_aggregates_five_splits(tmp_path: Path) -> None:
    paths = []
    pretrained = [5.0, 5.5, 6.0, 5.2, 5.8]
    matched_random = [6.0, 6.2, 5.9, 6.1, 6.4]
    for index, (candidate, comparator) in enumerate(zip(pretrained, matched_random, strict=True)):
        path = tmp_path / f"split-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "attribution": {
                        "split": "base" if index == 0 else f"s{index}",
                        "pretrained_report_regret": candidate,
                        "random_mean_report_regret": comparator,
                    }
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    summary = summarize_protocol_results(
        paths,
        model_revision="a" * 40,
        public_history_baseline=9.0,
        corpus_sha256="b" * 64,
    )

    assert summary["status"] == "complete"
    assert summary["metrics"]["pretrained_regret_mean"] == pytest.approx(5.5)
    assert summary["metrics"]["pretrained_split_wins_vs_matched_random"] == 4
    assert len(summary["data"]["input_sha256"]) == 5
