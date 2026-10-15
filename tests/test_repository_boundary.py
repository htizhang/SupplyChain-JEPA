from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_repository_boundary as boundary
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


@pytest.mark.parametrize("size", ["8xlarge", "xlarge", "metal"])
@pytest.mark.parametrize("prefix", ["", "ml."])
def test_repository_boundary_rejects_machine_types(tmp_path, monkeypatch, size, prefix):
    path = tmp_path / "example.py"
    family = "g" + "4" + "dn"
    path.write_text(f"# Runs on {prefix}{family}.{size}.\n", encoding="utf-8")
    monkeypatch.setattr(boundary, "ROOT", tmp_path)
    monkeypatch.setattr(boundary, "repository_files", lambda: [path])
    assert any("machine type" in failure for failure in boundary.check_repository())


def test_repository_boundary_accepts_neutral_hardware_description(tmp_path, monkeypatch):
    path = tmp_path / "example.py"
    path.write_text("# Keep the host machine within its memory limit.\n", encoding="utf-8")
    monkeypatch.setattr(boundary, "ROOT", tmp_path)
    monkeypatch.setattr(boundary, "repository_files", lambda: [path])
    assert boundary.check_repository() == []
