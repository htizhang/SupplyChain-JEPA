"""Command-line interface for raw-to-canonical corpus conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .base import AdapterRequest
from .build import build_corpus, convert_source
from .common import canonical_json
from .registry import list_adapters


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m supplychain_jepa.adapters")
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="list all released source adapters")
    listing.add_argument("--json", action="store_true")

    convert = commands.add_parser("convert", help="convert one source to canonical JSONL")
    convert.add_argument("--source", required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--raw-root", type=Path)
    convert.add_argument("--source-root", type=Path)
    convert.add_argument("--limit", type=int)
    convert.add_argument("--pilot", action="store_true")
    convert.add_argument("--worlds", type=int, default=8)
    convert.add_argument("--horizon", type=int, default=32)
    convert.add_argument("--seed", type=int, default=20260730)
    convert.add_argument("--no-state", action="store_true")
    dynamics = convert.add_mutually_exclusive_group()
    dynamics.add_argument("--dynamics", action="store_true")
    dynamics.add_argument("--no-dynamics", action="store_true")
    convert.add_argument("--no-validate", action="store_true")

    build = commands.add_parser("build", help="build all sources declared in a JSON config")
    build.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "list":
        descriptors = list_adapters()
        if args.json:
            print(canonical_json([descriptor.__dict__ for descriptor in descriptors], indent=2))
        else:
            for descriptor in descriptors:
                dynamics = "State+Dynamics" if descriptor.supports_dynamics else "State"
                print(f"{descriptor.source_id:24} {descriptor.source_kind:13} {dynamics}")
        return

    if args.command == "build":
        config_path = args.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        print(canonical_json(build_corpus(config, config_dir=config_path.parent), indent=2))
        return

    include_dynamics = None
    if args.dynamics:
        include_dynamics = True
    elif args.no_dynamics:
        include_dynamics = False
    request = AdapterRequest(
        raw_root=args.raw_root.resolve() if args.raw_root else None,
        source_root=args.source_root.resolve() if args.source_root else None,
        limit=args.limit,
        pilot=args.pilot,
        worlds=args.worlds,
        horizon=args.horizon,
        seed=args.seed,
        include_state=not args.no_state,
        include_dynamics=include_dynamics,
    )
    manifest = convert_source(
        args.source,
        request,
        output_path=args.output.resolve(),
        validate=not args.no_validate,
    )
    print(canonical_json(manifest, indent=2))


if __name__ == "__main__":
    main()
