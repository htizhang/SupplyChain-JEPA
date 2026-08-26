"""Adapter for factual BPI Challenge 2019 procurement event histories."""

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
    name="BPI Challenge 2019",
    version="2019",
    source_repository="https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853",
    license="see upstream 4TU dataset terms",
)


class BPI2019Adapter(ObservationalAdapter):
    """Convert recorded purchase-order-item event traces."""

    source_id = "bpi2019"

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        raw_root, limit = self.prepare_request(request)
        if limit == 0:
            return []
        assert raw_root is not None
        archive = required_file(raw_root, "BPIChallenge2019CSV.zip")
        with zipfile.ZipFile(archive) as bundle:
            frame = pd.read_csv(
                bundle.open("BPI_Challenge_2019.csv"),
                nrows=60_000 if request.pilot else None,
                low_memory=False,
                encoding="cp1252",
            )
        frame["event time:timestamp"] = pd.to_datetime(
            frame["event time:timestamp"],
            format="%d-%m-%Y %H:%M:%S.%f",
            errors="coerce",
        )
        frame = frame.dropna(subset=["case concept:name", "event time:timestamp"])
        groups = ranked_groups(
            (
                (str(case_id), group)
                for case_id, group in frame.groupby("case concept:name", sort=False)
            ),
            source_id=self.source_id,
            limit=limit,
        )

        records: list[dict[str, Any]] = []
        for world_id, group in groups:
            group = group.sort_values(["event time:timestamp", "eventID "], kind="stable").head(
                MAX_STEPS
            )
            origin = group["event time:timestamp"].iloc[0]
            first = group.iloc[0]
            node_id = str(first["case Company"])
            steps: list[dict[str, Any]] = []
            for _, row in group.iterrows():
                timestamp = row["event time:timestamp"]
                entity = make_entity(
                    world_id,
                    "purchase_order_item_case",
                    node_id=node_id,
                    product_id=str(row["case Item"]),
                    features=features(
                        purchasing_document=row["case Purchasing Document"],
                        company=row["case Company"],
                        vendor=row["case Vendor"],
                        document_type=row["case Document Type"],
                        item_type=row["case Item Type"],
                        item_category=row["case Item Category"],
                        goods_receipt_required=row["case Goods Receipt"],
                        event_name=row["event concept:name"],
                        event_resource=row["event org:resource"],
                        cumulative_net_worth_eur=row["event Cumulative net worth (EUR)"],
                        event_timestamp=timestamp,
                    ),
                )
                clock = clock_hours(timestamp, origin)
                event = make_event(
                    str(row["eventID "]),
                    str(row["event concept:name"]),
                    world_id,
                    node_id=node_id,
                    event_time=clock,
                    report_time=clock,
                    observed=True,
                    features=features(resource=row["event org:resource"]),
                )
                steps.append(
                    make_step(
                        clock=clock,
                        entity=entity,
                        event=event,
                        outcomes=features(
                            recorded_cumulative_net_worth_eur=row[
                                "event Cumulative net worth (EUR)"
                            ]
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
                            "node_type": "purchasing_company",
                            "features": features(company=node_id),
                        }
                    ],
                    topology_edges=[],
                    steps=steps,
                    source_files=["BPIChallenge2019CSV.zip::BPI_Challenge_2019.csv"],
                    domain_parameters=features(
                        trace_semantics="recorded purchase-order-item event history"
                    ),
                    state_eligible=request.include_state,
                )
            )
        return records
