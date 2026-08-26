"""Adapter for factual Olist seller order-item lifecycle records."""

from __future__ import annotations

import zipfile
from typing import Any

import pandas as pd

from ._observational import (
    MAX_STEPS,
    ObservationalAdapter,
    SourceMetadata,
    clock_hours,
    features,
    make_record,
    make_step,
    ranked_groups,
    required_file,
)
from .base import AdapterRequest
from .common import make_entity, make_event

_METADATA = SourceMetadata(
    name="Olist Brazilian E-Commerce",
    version="downloaded public release",
    source_repository="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
    license="see upstream dataset terms",
)


class OlistAdapter(ObservationalAdapter):
    """Convert finalized seller order-item lifecycle sequences."""

    source_id = "olist"

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        raw_root, limit = self.prepare_request(request)
        if limit == 0:
            return []
        assert raw_root is not None
        archive = required_file(raw_root, "olist.zip")
        with zipfile.ZipFile(archive) as bundle:
            orders = pd.read_csv(bundle.open("olist_orders_dataset.csv"))
            items = pd.read_csv(bundle.open("olist_order_items_dataset.csv"))
            products = pd.read_csv(bundle.open("olist_products_dataset.csv"))
            sellers = pd.read_csv(bundle.open("olist_sellers_dataset.csv"))
        frame = items.merge(orders, on="order_id", how="inner", validate="many_to_one")
        frame = frame.merge(products, on="product_id", how="left", validate="many_to_one")
        frame = frame.merge(sellers, on="seller_id", how="left", validate="many_to_one")
        timestamp_columns = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        for column in timestamp_columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame = frame.dropna(subset=["seller_id", "order_purchase_timestamp"])
        groups = ranked_groups(
            (
                (str(seller_id), group)
                for seller_id, group in frame.groupby("seller_id", sort=False)
            ),
            source_id=self.source_id,
            limit=limit,
        )

        records: list[dict[str, Any]] = []
        for world_id, group in groups:
            group = group.sort_values(
                ["order_purchase_timestamp", "order_id", "order_item_id"],
                kind="stable",
            ).head(MAX_STEPS)
            origin = group["order_purchase_timestamp"].iloc[0]
            first = group.iloc[0]
            node_id = f"seller:{world_id}"
            steps: list[dict[str, Any]] = []
            for _, row in group.iterrows():
                delivered = row["order_delivered_customer_date"]
                purchase = row["order_purchase_timestamp"]
                delivery_hours = (
                    None
                    if pd.isna(delivered)
                    else max(0.0, float((delivered - purchase).total_seconds() / 3600))
                )
                lateness_hours = (
                    None
                    if pd.isna(delivered) or pd.isna(row["order_estimated_delivery_date"])
                    else float(
                        (delivered - row["order_estimated_delivery_date"]).total_seconds() / 3600
                    )
                )
                entity_id = f"{row['order_id']}:{int(row['order_item_id'])}"
                entity = make_entity(
                    entity_id,
                    "order_item_lifecycle_record",
                    node_id=node_id,
                    product_id=str(row["product_id"]),
                    features=features(
                        order_status=row["order_status"],
                        purchase_timestamp=purchase,
                        approved_timestamp=row["order_approved_at"],
                        carrier_handoff_timestamp=row["order_delivered_carrier_date"],
                        delivered_timestamp=delivered,
                        estimated_delivery_timestamp=row["order_estimated_delivery_date"],
                        recorded_delivery_hours=delivery_hours,
                        delivery_vs_estimate_hours=lateness_hours,
                        price=row["price"],
                        freight_value=row["freight_value"],
                        product_category=row["product_category_name"],
                        product_weight_g=row["product_weight_g"],
                        product_length_cm=row["product_length_cm"],
                        product_height_cm=row["product_height_cm"],
                        product_width_cm=row["product_width_cm"],
                        seller_city=row["seller_city"],
                        seller_state=row["seller_state"],
                    ),
                )
                clock = clock_hours(purchase, origin)
                event = make_event(
                    f"olist:{entity_id}:purchase",
                    "order_item_purchased",
                    entity_id,
                    node_id=node_id,
                    event_time=clock,
                    report_time=clock,
                    observed=True,
                    features=features(order_status=row["order_status"]),
                )
                steps.append(
                    make_step(
                        clock=clock,
                        entity=entity,
                        event=event,
                        outcomes=features(
                            recorded_delivery_hours=delivery_hours,
                            delivery_vs_estimate_hours=lateness_hours,
                        ),
                    )
                )
            records.append(
                make_record(
                    source_id=self.source_id,
                    metadata=_METADATA,
                    world_id=world_id,
                    time_unit="hour",
                    topology_nodes=[
                        {
                            "node_id": node_id,
                            "node_type": "seller",
                            "features": features(
                                seller_city=first["seller_city"],
                                seller_state=first["seller_state"],
                            ),
                        }
                    ],
                    topology_edges=[],
                    steps=steps,
                    source_files=[
                        "olist.zip::olist_orders_dataset.csv",
                        "olist.zip::olist_order_items_dataset.csv",
                        "olist.zip::olist_products_dataset.csv",
                        "olist.zip::olist_sellers_dataset.csv",
                    ],
                    domain_parameters=features(
                        sequence_semantics=("finalized seller order-item lifecycle records")
                    ),
                    state_eligible=request.include_state,
                )
            )
        return records
