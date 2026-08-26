from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.slow
def test_canonical_state_to_frozen_dynamics_training_chain(tmp_path: Path) -> None:
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_training_smoke.py"),
            "--work-dir",
            str(tmp_path),
            "--updates",
            "1",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    state_path = tmp_path / "state" / "state_checkpoint.pt"
    dynamics_path = tmp_path / "dynamics" / "dynamics_checkpoint.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    dynamics = torch.load(dynamics_path, map_location="cpu", weights_only=False)

    assert state["stage"] == "state"
    assert state["updates"] == 1
    assert state["metrics"]["train_record_count"] == 2
    assert state["metrics"]["dev_record_count"] == 2
    assert state["sources"]["state"] == ["tiny_inventory"]

    assert dynamics["stage"] == "dynamics"
    assert dynamics["updates"] == 1
    assert dynamics["parent_checkpoint"]["stage"] == "state"
    assert dynamics["parent_checkpoint"]["sha256"] == _sha256(state_path)
    assert dynamics["metrics"]["train_branch_group_count"] == 1
    assert dynamics["metrics"]["dev_branch_group_count"] == 1
    assert dynamics["metrics"]["train_control_comparison_count"] > 0
    assert dynamics["metrics"]["train_branch_comparison_count"] > 0
    assert dynamics["metrics"]["action_sensitivity"] > 0.0
    assert state["tensorizer_state"] == dynamics["tensorizer_state"]

    for prefix in ("online_encoder.", "target_encoder.", "state_predictor."):
        keys = [key for key in state["model_state"] if key.startswith(prefix)]
        assert keys
        assert all(
            torch.equal(state["model_state"][key], dynamics["model_state"][key])
            for key in keys
        )
    dynamics_keys = [
        key for key in state["model_state"] if key.startswith("dynamics_predictor.")
    ]
    assert dynamics_keys
    assert any(
        not torch.equal(state["model_state"][key], dynamics["model_state"][key])
        for key in dynamics_keys
    )

    finite_metrics = (
        state["metrics"]["train_loss"],
        state["metrics"]["dev_loss"],
        state["metrics"]["gradient_norm"],
        dynamics["metrics"]["train_loss"],
        dynamics["metrics"]["dev_loss"],
        dynamics["metrics"]["gradient_norm"],
    )
    assert all(math.isfinite(float(value)) for value in finite_metrics)

    summary = json.loads((tmp_path / "smoke-summary.json").read_text(encoding="utf-8"))
    assert summary["corpus_records"] == 4
    assert summary["state_features"] > 0
