"""Lazy registry for every pretraining source used by the released lineage."""

from __future__ import annotations

from importlib import import_module

from .base import AdapterDescriptor, SourceAdapter

_DESCRIPTORS = (
    AdapterDescriptor(
        "m5", "observational", True, False, ".m5:M5Adapter", "M5 item-store daily sales"
    ),
    AdapterDescriptor(
        "bpi2019",
        "observational",
        True,
        False,
        ".bpi2019:BPI2019Adapter",
        "BPI 2019 procurement events",
    ),
    AdapterDescriptor(
        "lade", "observational", True, False, ".lade:LaDeAdapter", "LaDe last-mile operations"
    ),
    AdapterDescriptor(
        "olist", "observational", True, False, ".olist:OlistAdapter", "Olist order lifecycles"
    ),
    AdapterDescriptor(
        "planned_actual_routes",
        "observational",
        True,
        False,
        ".planned_actual_routes:PlannedActualRoutesAdapter",
        "Planned versus actual route execution",
    ),
    AdapterDescriptor(
        "stockpyl",
        "simulator",
        True,
        True,
        ".stockpyl:StockpylAdapter",
        "Stockpyl inventory-network simulation",
    ),
    AdapterDescriptor(
        "supplynetpy",
        "simulator",
        True,
        False,
        ".supplynetpy:SupplyNetPyAdapter",
        "SupplyNetPy discrete-event networks",
    ),
    AdapterDescriptor(
        "mabim",
        "simulator",
        True,
        True,
        ".mabim:MABIMAdapter",
        "MABIM multi-agent inventory simulation",
    ),
    AdapterDescriptor(
        "isomorph",
        "simulator",
        True,
        False,
        ".isomorph:ISOMORPHAdapter",
        "ISOMORPH supply-network simulation",
    ),
    AdapterDescriptor(
        "milestone_flow",
        "simulator",
        True,
        False,
        ".milestone_flow:MilestoneFlowAdapter",
        "Project-authored milestone-flow simulation",
    ),
)
DESCRIPTORS = {descriptor.source_id: descriptor for descriptor in _DESCRIPTORS}


def list_adapters() -> tuple[AdapterDescriptor, ...]:
    return _DESCRIPTORS


def load_adapter(source_id: str) -> SourceAdapter:
    try:
        descriptor = DESCRIPTORS[source_id]
    except KeyError as exc:
        allowed = ", ".join(DESCRIPTORS)
        raise ValueError(f"Unknown source {source_id!r}; choose one of: {allowed}") from exc
    module_name, class_name = descriptor.class_path.split(":", maxsplit=1)
    module = import_module(module_name, package=__package__)
    adapter_class = getattr(module, class_name)
    adapter = adapter_class()
    if not isinstance(adapter, SourceAdapter):
        raise TypeError(f"{descriptor.class_path} is not a SourceAdapter")
    return adapter
