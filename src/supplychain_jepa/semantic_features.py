"""Canonical supply-chain concepts shared across heterogeneous source schemas.

The pretraining corpus contains hundreds of source-specific column names.  Treating every
name as an unrelated physical quantity lets the encoder memorize a dataset
schema even when several columns describe the same operational state.  This
module supplies a deliberately small, deterministic bridge from raw feature
names to reusable supply-chain concepts.  It contains no downstream dataset
names and can therefore be fitted before downstream adaptation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

FEATURE_CONCEPTS = (
    "generic",
    "identity",
    "action_order",
    "demand_sales",
    "inventory",
    "backlog_shortage",
    "fulfillment_service",
    "shipment_transit",
    "lead_time_delay",
    "capacity_queue",
    "cost_price",
    "event_progress",
    "calendar_time",
    "product_attributes",
    "reporting_observability",
    "risk_regime",
    "topology_location",
)

_CONCEPT_INDEX = {name: index for index, name in enumerate(FEATURE_CONCEPTS)}


def _has(text: str, *patterns: str) -> bool:
    return any(pattern in text for pattern in patterns)


def feature_concept(feature_name: str) -> str:
    """Map one canonical feature name to a stable operational concept.

    Rules are ordered from economically specific quantities to metadata.  For
    example, ``inventory_reported`` remains inventory rather than being reduced
    to a generic reporting flag.  Namespace and suffix are both considered,
    but no simulator or downstream dataset identity is used.
    """

    normalized = str(feature_name).strip().lower()
    namespace, _, field = normalized.partition("::")
    field = re.sub(r"[^a-z0-9]+", "_", field).strip("_")

    if _has(field, "cost", "price", "freight_value", "net_worth", "selling"):
        return "cost_price"
    if _has(field, "demand", "sales", "sale", "units_sold"):
        return "demand_sales"
    if _has(
        field,
        "inventory",
        "in_stock",
        "on_hand",
        "base_stock",
        "order_up_to",
        "reorder_point",
    ):
        return "inventory"
    if _has(
        field,
        "backlog",
        "backorder",
        "shortage",
        "stockout",
        "unfilled",
        "excess",
        "overflow",
    ):
        return "backlog_shortage"
    if _has(
        field,
        "lead_time",
        "travel_time",
        "duration",
        "delay",
        "wait",
        "vlt",
        "deviation",
        "vs_estimate",
    ):
        return "lead_time_delay"
    if _has(
        field,
        "shipment",
        "in_transit",
        "pipeline",
        "freight",
        "container",
        "dispatch",
    ):
        return "shipment_transit"
    if _has(
        field,
        "fulfilled",
        "served",
        "fill_rate",
        "delivered",
        "delivery_indicator",
        "accepted",
    ):
        return "fulfillment_service"
    if _has(
        field,
        "capacity",
        "queue",
        "busy",
        "congest",
        "reservable",
        "volume",
    ):
        return "capacity_queue"
    if _has(
        field,
        "calendar",
        "day",
        "week",
        "period",
        "clock",
        "date",
        "weekday",
    ):
        return "calendar_time"
    if _has(
        field,
        "replenish",
        "order",
        "policy",
        "procurement",
        "purchase",
        "release",
    ) or namespace == "action":
        return "action_order"
    if _has(
        field,
        "event",
        "milestone",
        "sequence",
        "actual_index",
        "planned_index",
        "arriv",
        "due_time",
        "earliest_time",
        "latest_time",
        "timestamp",
        "distance",
    ):
        return "event_progress"
    if _has(
        field,
        "shock",
        "risk",
        "late",
        "reliability",
        "regime",
        "failure",
        "status",
        "active",
    ):
        return "risk_regime"
    if _has(
        field,
        "product",
        "item_category",
        "item_type",
        "sku_count",
        "height",
        "width",
        "weight",
        "length_cm",
        "category_id",
        "department_id",
    ):
        return "product_attributes"
    if _has(
        field,
        "route",
        "mode",
        "city",
        "state_id",
        "country",
        "region",
        "latitude",
        "longitude",
        "gps",
        "destination",
        "origin",
        "source_node",
        "echelon",
        "depot",
        "external_supply",
    ) or namespace.startswith("topology"):
        return "topology_location"
    if _has(
        field,
        "report",
        "observed",
        "censored",
        "published",
        "visible",
        "snapshot",
    ):
        return "reporting_observability"
    if _has(
        field,
        "_id",
        "address",
        "company",
        "courier",
        "driver",
        "vendor",
        "document",
        "resource",
    ):
        return "identity"
    return "generic"


def feature_concept_ids(feature_names: Sequence[str]) -> tuple[int, ...]:
    """Return concept IDs aligned exactly with a fitted feature vocabulary."""

    return tuple(_CONCEPT_INDEX[feature_concept(name)] for name in feature_names)


__all__ = ["FEATURE_CONCEPTS", "feature_concept", "feature_concept_ids"]
