"""Adapter for factual planned-versus-actual last-mile route records."""

from __future__ import annotations

import zipfile
from typing import Any

import pandas as pd

from ._observational import (
    MAX_STEPS,
    ObservationalAdapter,
    SourceMetadata,
    features,
    make_record,
    make_step,
    ranked_groups,
    required_file,
)
from .base import AdapterRequest
from .common import make_entity, make_event

_METADATA = SourceMetadata(
    name="Last-mile planned versus actual routes",
    version="v1",
    source_repository="https://data.mendeley.com/",
    license="see upstream Mendeley dataset terms",
)


class PlannedActualRoutesAdapter(ObservationalAdapter):
    """Convert routes in actual stop order with factual plan-versus-execution fields."""

    source_id = "planned_actual_routes"

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        raw_root, limit = self.prepare_request(request)
        if limit == 0:
            return []
        assert raw_root is not None
        archive = required_file(raw_root, "last-mile-route-deviations-v1.zip")
        with zipfile.ZipFile(archive) as bundle:
            try:
                member = next(name for name in bundle.namelist() if name.endswith(".xlsx"))
            except StopIteration as exc:
                raise FileNotFoundError(f"no .xlsx workbook found in {archive}") from exc
            with bundle.open(member) as handle:
                frame = pd.read_excel(
                    handle,
                    engine="openpyxl",
                    nrows=50_000 if request.pilot else None,
                )
        frame = frame.dropna(subset=["Route ID", "IndexA", "Stop ID"])
        groups = ranked_groups(
            ((str(route_id), group) for route_id, group in frame.groupby("Route ID", sort=False)),
            source_id=self.source_id,
            limit=limit,
        )

        records: list[dict[str, Any]] = []
        for world_id, group in groups:
            group = group.sort_values(["IndexA", "Stop ID"], kind="stable").head(MAX_STEPS)
            first = group.iloc[0]
            depot_id = f"route:{world_id}:depot"
            steps: list[dict[str, Any]] = []
            cumulative_actual_distance = 0.0
            cumulative_planned_distance = 0.0
            for step_index, (_, row) in enumerate(group.iterrows()):
                actual_distance = float(row["DistanceA"])
                planned_distance = float(row["DistanceP"])
                cumulative_actual_distance += actual_distance
                cumulative_planned_distance += planned_distance
                arrived = float(row["Arrived Time"])
                earliest = float(row["Earliest Time"])
                latest = float(row["Latest Time"])
                entity_id = f"route:{world_id}:stop:{row['Stop ID']}"
                entity = make_entity(
                    entity_id,
                    "route_stop_execution_record",
                    node_id=depot_id,
                    product_id=None,
                    features=features(
                        driver_id=row["Driver ID"],
                        address_id=row["Address ID"],
                        week_id=row["Week ID"],
                        country=row["Country"],
                        day_of_week=row["Day of Week"],
                        planned_index=row["IndexP"],
                        actual_index=row["IndexA"],
                        sequence_deviation=float(row["IndexA"]) - float(row["IndexP"]),
                        arrived_time=arrived,
                        earliest_time=earliest,
                        latest_time=latest,
                        arrived_before_window=arrived < earliest,
                        arrived_after_window=arrived > latest,
                        planned_leg_distance=planned_distance,
                        actual_leg_distance=actual_distance,
                        cumulative_planned_distance=cumulative_planned_distance,
                        cumulative_actual_distance=cumulative_actual_distance,
                        depot_indicator=row["Depot"],
                        delivery_indicator=row["Delivery"],
                    ),
                )
                event = make_event(
                    f"route:{world_id}:arrival:{row['Stop ID']}",
                    "recorded_stop_arrival",
                    entity_id,
                    node_id=depot_id,
                    event_time=float(step_index),
                    report_time=float(step_index),
                    observed=True,
                    features=features(arrived_time=arrived),
                )
                steps.append(
                    make_step(
                        clock=float(step_index),
                        entity=entity,
                        event=event,
                        outcomes=features(
                            recorded_actual_leg_distance=actual_distance,
                            recorded_sequence_deviation=float(row["IndexA"]) - float(row["IndexP"]),
                        ),
                    )
                )
            records.append(
                make_record(
                    source_id=self.source_id,
                    metadata=_METADATA,
                    world_id=world_id,
                    time_unit="route_stop",
                    topology_nodes=[
                        {
                            "node_id": depot_id,
                            "node_type": "route_depot",
                            "features": features(
                                driver_id=first["Driver ID"],
                                country=first["Country"],
                                week_id=first["Week ID"],
                            ),
                        }
                    ],
                    topology_edges=[],
                    steps=steps,
                    source_files=[
                        "last-mile-route-deviations-v1.zip::"
                        "Last-mile delivery route deviations dataset planned vs. "
                        "actual routes/routes_performance.xlsx"
                    ],
                    domain_parameters=features(
                        sequence_semantics=("recorded planned-versus-actual route execution")
                    ),
                    state_eligible=request.include_state,
                )
            )
        return records
