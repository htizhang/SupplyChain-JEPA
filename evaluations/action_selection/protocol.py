"""Summarize the canonical paired-plan protocol from compact run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_protocol_results(
    paths: list[Path],
    *,
    model_revision: str,
    public_history_baseline: float,
    corpus_sha256: str | None = None,
) -> dict[str, Any]:
    """Aggregate five fit/select/report runs without retaining raw world records."""

    if len(paths) != 5:
        raise ValueError("exactly five world-split result files are required")

    per_split: list[dict[str, Any]] = []
    seen_splits: set[str] = set()
    input_hashes: dict[str, str] = {}
    for index, path in enumerate(sorted(paths)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        attribution = payload["attribution"]
        split = str(attribution.get("split") or payload.get("recipe", {}).get("split_salt") or "base")
        if split in seen_splits:
            raise ValueError(f"duplicate world split {split!r}")
        seen_splits.add(split)

        pretrained = float(attribution["pretrained_report_regret"])
        matched_random = float(attribution["random_mean_report_regret"])
        per_split.append(
            {
                "split": split,
                "pretrained_regret": pretrained,
                "matched_random_regret": matched_random,
                "relative_reduction_vs_matched_random": 1.0 - pretrained / matched_random,
                "pretrained_wins": pretrained < matched_random,
            }
        )
        input_hashes[f"split_{index:02d}"] = _sha256(path)

    pretrained_values = [row["pretrained_regret"] for row in per_split]
    random_values = [row["matched_random_regret"] for row in per_split]
    pretrained_mean = mean(pretrained_values)
    random_mean = mean(random_values)
    split_wins = sum(bool(row["pretrained_wins"]) for row in per_split)

    return {
        "schema_version": 1,
        "status": "complete",
        "task": "short-horizon paired replenishment-plan choice",
        "model": {
            "revision": model_revision,
            "state_encoder_frozen": True,
            "dynamics_predictor_adapted": True,
        },
        "protocol": {
            "lookback_periods": 16,
            "decision_horizon": 4,
            "candidate_plans": ["regular", "split"],
            "quantity_matched": True,
            "continuation_policy": "no_further_orders",
            "world_splits": 5,
        },
        "data": {
            "generated": True,
            "corpus_sha256": corpus_sha256,
            "raw_results_redistributed": False,
            "input_sha256": input_hashes,
        },
        "metrics": {
            "public_history_baseline_regret": public_history_baseline,
            "pretrained_regret_mean": pretrained_mean,
            "pretrained_regret_sample_sd": stdev(pretrained_values),
            "matched_random_regret_mean": random_mean,
            "matched_random_regret_sample_sd": stdev(random_values),
            "relative_reduction_vs_public_history_baseline": (
                1.0 - pretrained_mean / public_history_baseline
            ),
            "relative_reduction_vs_matched_random": 1.0 - pretrained_mean / random_mean,
            "pretrained_split_wins_vs_matched_random": split_wins,
            "total_splits": len(per_split),
        },
        "per_split": sorted(per_split, key=lambda row: row["split"]),
        "claim_boundary": (
            "The public-history comparison evaluates the complete pretrained decision system. "
            "The matched-random comparison is an attribution diagnostic and may be numerically "
            "sensitive to downstream adaptation."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=5, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--public-history-baseline", required=True, type=float)
    parser.add_argument("--corpus-sha256")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = summarize_protocol_results(
        args.inputs,
        model_revision=args.model_revision,
        public_history_baseline=args.public_history_baseline,
        corpus_sha256=args.corpus_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
