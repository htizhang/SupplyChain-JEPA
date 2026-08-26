"""Run a bounded CPU State-to-Dynamics training chain on generated records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "examples" / "training" / "generate_tiny_corpus.py"
STATE_CONFIG = ROOT / "examples" / "training" / "state-smoke.json"
DYNAMICS_CONFIG = ROOT / "examples" / "training" / "dynamics-smoke.json"
MODEL_CONFIG = ROOT / "configs" / "model.json"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("--updates must be positive")

    work_dir = args.work_dir.resolve()
    corpus = work_dir / "tiny-corpus.jsonl"
    state_dir = work_dir / "state"
    dynamics_dir = work_dir / "dynamics"
    _run([sys.executable, str(GENERATOR), "--output", str(corpus)])
    _run(
        [
            sys.executable,
            "-m",
            "supplychain_jepa.train",
            "state",
            "--data",
            str(corpus),
            "--model-config",
            str(MODEL_CONFIG),
            "--config",
            str(STATE_CONFIG),
            "--output-dir",
            str(state_dir),
            "--max-updates",
            str(args.updates),
            "--batch-size",
            "2",
            "--eval-samples",
            "2",
            "--device",
            "cpu",
            "--disable-amp",
            "--small-model",
        ]
    )
    state_checkpoint = state_dir / "state_checkpoint.pt"
    _run(
        [
            sys.executable,
            "-m",
            "supplychain_jepa.train",
            "dynamics",
            "--data",
            str(corpus),
            "--model-config",
            str(MODEL_CONFIG),
            "--config",
            str(DYNAMICS_CONFIG),
            "--checkpoint",
            str(state_checkpoint),
            "--output-dir",
            str(dynamics_dir),
            "--max-updates",
            str(args.updates),
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--disable-amp",
            "--small-model",
        ]
    )

    dynamics_checkpoint = dynamics_dir / "dynamics_checkpoint.pt"
    state_payload = torch.load(state_checkpoint, map_location="cpu", weights_only=False)
    dynamics_payload = torch.load(
        dynamics_checkpoint, map_location="cpu", weights_only=False
    )
    if state_payload["stage"] != "state" or dynamics_payload["stage"] != "dynamics":
        raise RuntimeError("training chain produced an unexpected stage")
    summary = {
        "corpus_records": sum(1 for line in corpus.read_text().splitlines() if line),
        "state_checkpoint": str(state_checkpoint),
        "state_features": len(state_payload["tensorizer_state"]["feature_names"]),
        "dynamics_checkpoint": str(dynamics_checkpoint),
        "dynamics_action_sensitivity": dynamics_payload["metrics"]["action_sensitivity"],
    }
    (work_dir / "smoke-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
