from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

from supplychain_jepa.adapters import AdapterRequest, list_adapters, load_adapter
from supplychain_jepa.adapters.build import build_corpus

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCES = {
    "m5",
    "bpi2019",
    "lade",
    "olist",
    "planned_actual_routes",
    "stockpyl",
    "supplynetpy",
    "mabim",
    "isomorph",
    "milestone_flow",
}


def test_registry_covers_every_released_pretraining_source() -> None:
    descriptors = list_adapters()
    assert {descriptor.source_id for descriptor in descriptors} == EXPECTED_SOURCES
    assert {descriptor.source_id for descriptor in descriptors if descriptor.supports_dynamics} == {
        "stockpyl",
        "mabim",
    }


def test_adapter_request_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="worlds must be positive"):
        AdapterRequest(worlds=0)
    with pytest.raises(ValueError, match="horizon must be at least 2"):
        AdapterRequest(horizon=1)
    with pytest.raises(ValueError, match="limit must be non-negative"):
        AdapterRequest(limit=-1)


@pytest.mark.parametrize(
    "source_id",
    sorted(EXPECTED_SOURCES - {"stockpyl", "mabim"}),
)
def test_state_only_adapters_reject_dynamics(source_id: str) -> None:
    with pytest.raises(ValueError, match="does not support Dynamics JEPA"):
        load_adapter(source_id).convert(AdapterRequest(limit=0, include_dynamics=True))


def test_adapter_cli_lists_all_sources_as_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "supplychain_jepa.adapters", "list", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert {item["source_id"] for item in payload} == EXPECTED_SOURCES


def test_build_config_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        build_corpus(
            {"output_dir": str(tmp_path / "out"), "sources": {"unknown": {}}},
            config_dir=tmp_path,
        )


def test_simulator_source_pins_match_adapter_constants() -> None:
    manifest = json.loads((ROOT / "configs" / "source_pins.json").read_text(encoding="utf-8"))
    pins = manifest["simulators"]
    assert set(pins) == {
        "stockpyl",
        "supplynetpy",
        "mabim",
        "isomorph",
        "milestone_flow",
    }
    for source_id, pin in pins.items():
        module = import_module(f"supplychain_jepa.adapters.{source_id}")
        assert pin["revision"] == module.SOURCE_COMMIT
