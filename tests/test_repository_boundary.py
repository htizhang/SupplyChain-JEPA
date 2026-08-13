from __future__ import annotations

import json
from pathlib import Path

from scripts.check_repository_boundary import check_repository

ROOT = Path(__file__).resolve().parents[1]


def test_repository_boundary() -> None:
    assert check_repository() == []


def test_machine_readable_manifests_are_valid_json() -> None:
    paths = [*ROOT.glob("data/manifests/*.json"), *ROOT.glob("results/*.json")]
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get("schema_version") == 1
