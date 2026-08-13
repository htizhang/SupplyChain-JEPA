"""Load and verify the immutable pretrained model release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supplychain_jepa import DEFAULT_MODEL_ID, DEFAULT_REVISION, load_pretrained


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional downloaded release directory")
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = load_pretrained(
        args.source,
        repo_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        device=args.device,
        verify=True,
    )
    result = {
        "model_id": DEFAULT_MODEL_ID,
        "revision": DEFAULT_REVISION,
        "core_parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
        "feature_count": bundle.model.config.feature_count,
        "device": args.device,
        "verified": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
