"""Adapter for factual LaDe completed-delivery records."""

from __future__ import annotations

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
    name="LaDe delivery records",
    version="downloaded research release",
    source_repository="https://github.com/wenhaomin/LaDe",
    license="research use; see upstream dataset terms",
)


class LaDeAdapter(ObservationalAdapter):
    """Convert courier-day sequences ordered by recorded delivery completion."""

    source_id = "lade"

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        raw_root, limit = self.prepare_request(request)
        if limit == 0:
            return []
        assert raw_root is not None
        path = required_file(raw_root, "delivery_jl.csv")
        frame = pd.read_csv(
            path,
            nrows=80_000 if request.pilot else None,
            low_memory=False,
        )
        frame["accept_ts"] = pd.to_datetime(
            frame["accept_time"], format="%m-%d %H:%M:%S", errors="coerce"
        )
        frame["delivery_ts"] = pd.to_datetime(
            frame["delivery_time"], format="%m-%d %H:%M:%S", errors="coerce"
        )
        frame = frame.dropna(subset=["courier_id", "ds", "accept_ts", "delivery_ts"])
        frame["world_id"] = (
            frame["city"].astype(str)
            + ":"
            + frame["courier_id"].astype(str)
            + ":"
            + frame["ds"].astype(str)
        )
        groups = ranked_groups(
            ((str(world_id), group) for world_id, group in frame.groupby("world_id", sort=False)),
            source_id=self.source_id,
            limit=limit,
        )

        records: list[dict[str, Any]] = []
        for world_id, group in groups:
            group = group.sort_values(["delivery_ts", "order_id"], kind="stable").head(MAX_STEPS)
            origin = group["delivery_ts"].iloc[0]
            first = group.iloc[0]
            node_id = f"region:{first['region_id']}"
            courier_id = str(first["courier_id"])
            steps: list[dict[str, Any]] = []
            for _, row in group.iterrows():
                duration_hours = max(
                    0.0,
                    float((row["delivery_ts"] - row["accept_ts"]).total_seconds() / 3600),
                )
                entity = make_entity(
                    f"order:{row['order_id']}",
                    "completed_delivery_record",
                    node_id=node_id,
                    product_id=None,
                    features=features(
                        courier_id=courier_id,
                        city=row["city"],
                        region_id=row["region_id"],
                        aoi_id=row["aoi_id"],
                        aoi_type=row["aoi_type"],
                        destination_lng=row["lng"],
                        destination_lat=row["lat"],
                        accept_timestamp=row["accept_ts"],
                        delivery_timestamp=row["delivery_ts"],
                        recorded_delivery_duration_hours=duration_hours,
                        delivery_gps_lng=row["delivery_gps_lng"],
                        delivery_gps_lat=row["delivery_gps_lat"],
                    ),
                )
                clock = clock_hours(row["delivery_ts"], origin)
                event = make_event(
                    f"lade:{row['order_id']}:delivered",
                    "delivery_completed",
                    f"order:{row['order_id']}",
                    node_id=node_id,
                    event_time=clock,
                    report_time=clock,
                    observed=True,
                    features=features(courier_id=courier_id),
                )
                steps.append(
                    make_step(
                        clock=clock,
                        entity=entity,
                        event=event,
                        outcomes=features(recorded_delivery_duration_hours=duration_hours),
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
                            "node_type": "delivery_region",
                            "features": features(region_id=first["region_id"], city=first["city"]),
                        }
                    ],
                    topology_edges=[],
                    steps=steps,
                    source_files=["delivery_jl.csv"],
                    domain_parameters=features(
                        sequence_semantics="finalized courier-day delivery records",
                        courier_id=courier_id,
                    ),
                    state_eligible=request.include_state,
                )
            )
        return records
