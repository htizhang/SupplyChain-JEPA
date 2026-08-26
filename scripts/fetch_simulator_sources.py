#!/usr/bin/env python3
"""Fetch the exact external simulator revisions required by data adapters."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PINS = ROOT / "configs" / "source_pins.json"


def _run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def fetch_source(source_id: str, pin: dict[str, Any], destination: Path) -> None:
    repository = str(pin["repository"])
    revision = str(pin["revision"])
    if repository == "project-authored":
        return
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(f"Existing destination is not a Git checkout: {destination}")
        actual = _run("git", "rev-parse", "HEAD", cwd=destination)
        if actual != revision:
            raise RuntimeError(
                f"{source_id} checkout is {actual}; expected {revision}. "
                "Use an empty destination rather than modifying an existing checkout."
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run("git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination))
    _run("git", "checkout", "--detach", revision, cwd=destination)
    actual = _run("git", "rev-parse", "HEAD", cwd=destination)
    if actual != revision:
        raise RuntimeError(f"Failed to pin {source_id}: expected {revision}, found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--sources", nargs="*", default=None)
    args = parser.parse_args()

    manifest = json.loads(args.pins.read_text(encoding="utf-8"))
    pins = manifest["simulators"]
    selected = list(pins) if args.sources is None else args.sources
    unknown = sorted(set(selected) - set(pins))
    if unknown:
        raise ValueError(f"Unknown simulator sources: {unknown}")
    for source_id in selected:
        destination = args.root.resolve() / source_id
        fetch_source(source_id, pins[source_id], destination)
        if pins[source_id]["repository"] != "project-authored":
            print(f"{source_id}: {pins[source_id]['revision']} -> {destination}")


if __name__ == "__main__":
    main()
