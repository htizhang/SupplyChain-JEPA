"""Command-line entry point for demand recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .pipeline import EvaluationConfig, run_evaluation
from .representation import DEFAULT_MODEL_ID, DEFAULT_REVISION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen SupplyChain-JEPA state representations on "
            "controlled-stockout demand recovery."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing canonical train.parquet and eval.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--label-budget-percent", type=int, default=5)
    parser.add_argument("--adaptation-pool", type=int, default=60_000)
    parser.add_argument("--validation-pool", type=int, default=20_000)
    parser.add_argument(
        "--test-cap",
        type=int,
        help="Development-only cap; omit for the canonical final evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    config = EvaluationConfig(
        label_budget_percent=args.label_budget_percent,
        adaptation_pool=args.adaptation_pool,
        validation_pool=args.validation_pool,
        test_cap=args.test_cap,
        pca_components=args.pca_components,
        encode_batch_size=args.encode_batch_size,
        threads=args.threads,
        device=args.device,
        model_id=args.model_id,
        revision=args.revision,
    )
    result = run_evaluation(
        args.data_root,
        args.output,
        cache_dir=args.cache_dir,
        artifact_dir=args.artifact_dir,
        config=config,
    )
    print(json.dumps(result["test"]["pooled"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
