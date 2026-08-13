"""Command-line entry point for the bounded action-selection reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import ModelBundleBackbone, TinyBackbone
from .contracts import EvaluationConfig, ReconstructionConfig
from .evaluate import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--tiny", action="store_true", help="use the synthetic smoke backend")
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--future-replications", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--extra-trees-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--require-reference-gates",
        action="store_true",
        help="return a nonzero status when the reconstruction misses retained reference gates",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reconstruction = ReconstructionConfig(
        worlds=args.worlds,
        future_replications=args.future_replications,
        seed=args.seed,
    )
    config = EvaluationConfig(
        reconstruction=reconstruction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        extra_trees_estimators=args.extra_trees_estimators,
        device=args.device,
        threads=args.threads,
    )
    if args.tiny:
        backend = TinyBackbone(history_features=12, state_dim=32, seed=args.seed)
    else:
        backend = ModelBundleBackbone.from_pretrained(
            args.model_dir,
            local_files_only=False,
            seed=args.seed,
        )
    summary = run_evaluation(config, backend=backend, output_dir=args.output_dir)
    print(json.dumps(summary["metrics"], sort_keys=True))
    if args.require_reference_gates and not summary["all_reproduction_gates_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
