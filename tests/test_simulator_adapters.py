from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from supplychain_jepa.adapters import AdapterRequest, load_adapter
from supplychain_jepa.data import EpisodeTensorizer, fit_tensorizer_state
from supplychain_jepa.schemas import canonical_episode_schema


def _validator() -> Draft202012Validator:
    schema = json.loads(canonical_episode_schema().read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _assert_dynamics_pair(records: list[dict[str, object]]) -> None:
    paired = [record for record in records if record["objective_eligibility"]["dynamics_jepa"]]
    assert len(paired) == 2
    assert {record["branch"]["branch_id"] for record in paired} == {"low", "high"}
    assert len({record["branch"]["branch_group_id"] for record in paired}) == 1
    assert all(record["branch"]["common_random_numbers"] for record in paired)

    branch_at = paired[0]["branch"]["branch_at"]
    assert {record["branch"]["branch_at"] for record in paired} == {branch_at}

    def prefix(record: dict[str, object]) -> list[dict[str, object]]:
        return [step for step in record["steps"] if step["t"] < branch_at]

    def decision(record: dict[str, object]) -> dict[str, object]:
        return next(step for step in record["steps"] if step["t"] == branch_at)

    assert prefix(paired[0]) == prefix(paired[1])
    assert decision(paired[0])["actions"] != decision(paired[1])["actions"]


def test_milestone_flow_validates_is_deterministic_and_tensorizes() -> None:
    request = AdapterRequest(worlds=1, horizon=8, seed=19)
    first = load_adapter("milestone_flow").convert(request)
    second = load_adapter("milestone_flow").convert(request)
    assert first == second
    assert len(first) == 1

    record = first[0]
    _validator().validate(record)
    assert record["simulator"]["family"] == "milestone_flow"
    assert record["branch"] is None
    assert record["objective_eligibility"] == {
        "state_jepa": True,
        "dynamics_jepa": False,
        "decision_heads": False,
    }
    assert any(
        item["status"] != "observed"
        for step in record["steps"]
        for item in step["observation"]["status"]
    )

    state = fit_tensorizer_state([record], train_splits=(record["split"],))
    episode = EpisodeTensorizer(state).transform(record)
    assert episode.context_length > 0
    assert episode.target_length > 0


@pytest.mark.external_data
@pytest.mark.slow
def test_all_external_simulators_validate_pair_and_tensorize() -> None:
    root_value = os.environ.get("SCJEPA_SIMULATOR_CHECKOUTS")
    if not root_value:
        pytest.skip("set SCJEPA_SIMULATOR_CHECKOUTS to the pinned checkout directory")
    root = Path(root_value)
    source_directories = {
        "stockpyl": "stockpyl",
        "supplynetpy": "supplynetpy",
        "mabim": "mabim",
        "isomorph": "isomorph",
    }

    validator = _validator()
    records: list[dict[str, object]] = []
    for source_id, directory in source_directories.items():
        converted = load_adapter(source_id).convert(
            AdapterRequest(
                source_root=root / directory,
                worlds=1,
                horizon=8,
                seed=19,
            )
        )
        expected_count = 3 if source_id in {"stockpyl", "mabim"} else 1
        assert len(converted) == expected_count
        for record in converted:
            validator.validate(record)
            assert record["simulator"]["family"] == source_id
        if source_id in {"stockpyl", "mabim"}:
            _assert_dynamics_pair(converted)
        else:
            assert all(record["branch"] is None for record in converted)
            assert all(not record["objective_eligibility"]["dynamics_jepa"] for record in converted)
        records.extend(converted)

    state = fit_tensorizer_state(
        records,
        train_splits=("pretrain_train", "pretrain_dev", "pretrain_test"),
    )
    episodes = [EpisodeTensorizer(state).transform(record) for record in records]
    assert {episode.family for episode in episodes} == set(source_directories)
    assert all(episode.context_length > 0 for episode in episodes)
    assert all(episode.target_length > 0 for episode in episodes)
