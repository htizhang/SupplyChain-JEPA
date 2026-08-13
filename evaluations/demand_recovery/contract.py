"""Pinned data and headline-result contracts for demand recovery."""

from __future__ import annotations

import hashlib
from pathlib import Path

DATASET_FILES = {
    "train.parquet": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    "eval.parquet": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
}
REFERENCE_RESULT = {
    "baseline_wape": 0.41701661785209065,
    "jepa_system_wape": 0.3978750022662813,
    "relative_wape_improvement": 0.04590132566994868,
    "winning_test_seeds": 3,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(data_root: Path) -> dict[str, str]:
    """Fail closed unless both canonical external parquet files are present."""

    root = data_root.expanduser().resolve()
    actual: dict[str, str] = {}
    for filename, expected in DATASET_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {filename} under --data-root {root}; dataset files are not bundled"
            )
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected}, got {observed}"
            )
        actual[filename] = observed
    return actual
