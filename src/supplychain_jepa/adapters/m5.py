"""Adapter for factual M5 item-store daily sales histories."""

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
    required_file,
)
from .base import AdapterRequest
from .common import make_entity, stable_digest

_METADATA = SourceMetadata(
    name="M5 Forecasting Accuracy",
    version="competition-release",
    source_repository="https://www.kaggle.com/competitions/m5-forecasting-accuracy",
    license="see upstream competition data terms",
)


class M5Adapter(ObservationalAdapter):
    """Convert recorded unit sales without re-labeling them as hidden demand."""

    source_id = "m5"

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        raw_root, limit = self.prepare_request(request)
        if limit == 0:
            return []
        assert raw_root is not None
        archive = required_file(raw_root, "m5-forecasting-accuracy.zip")
        candidate_rows = 128 if request.pilot else None
        with zipfile.ZipFile(archive) as bundle:
            calendar = pd.read_csv(bundle.open("calendar.csv"))
            sales = pd.read_csv(
                bundle.open("sales_train_validation.csv"),
                nrows=candidate_rows,
                low_memory=False,
            )

        day_columns = [column for column in sales.columns if column.startswith("d_")]
        calendar = calendar.set_index("d")
        candidates = list(sales.to_dict("records"))
        candidates.sort(
            key=lambda row: stable_digest("v2-real-sample", self.source_id, str(row["id"]))
        )
        if limit is not None:
            candidates = candidates[:limit]
        episode_days = 32 if request.pilot else MAX_STEPS

        records: list[dict[str, Any]] = []
        for row in candidates:
            world_id = str(row["id"])
            max_start = max(0, len(day_columns) - episode_days)
            start = (
                int(stable_digest("m5-window", world_id, length=8), 16) % (max_start + 1)
                if max_start
                else 0
            )
            selected_days = day_columns[start : start + episode_days]
            steps: list[dict[str, Any]] = []
            for day_index, day in enumerate(selected_days):
                calendar_row = calendar.loc[day]
                state_code = str(row["state_id"])
                entity = make_entity(
                    f"{world_id}:{day}",
                    "item_store_daily_sales",
                    node_id=str(row["store_id"]),
                    product_id=str(row["item_id"]),
                    features=features(
                        units_sold=row[day],
                        store_id=row["store_id"],
                        state_id=state_code,
                        department_id=row["dept_id"],
                        category_id=row["cat_id"],
                        date=calendar_row["date"],
                        weekday=calendar_row["weekday"],
                        week_id=calendar_row["wm_yr_wk"],
                        event_name_1=calendar_row["event_name_1"],
                        event_type_1=calendar_row["event_type_1"],
                        snap_eligible=calendar_row.get(f"snap_{state_code}"),
                    ),
                )
                steps.append(
                    make_step(
                        clock=float(day_index),
                        entity=entity,
                        global_features=features(calendar_day=day),
                        outcomes=features(recorded_units_sold=row[day]),
                    )
                )
            records.append(
                make_record(
                    source_id=self.source_id,
                    metadata=_METADATA,
                    world_id=world_id,
                    time_unit="day",
                    topology_nodes=[
                        {
                            "node_id": str(row["store_id"]),
                            "node_type": "retail_store",
                            "features": features(state_id=row["state_id"]),
                        }
                    ],
                    topology_edges=[],
                    steps=steps,
                    source_files=[
                        "m5-forecasting-accuracy.zip::calendar.csv",
                        "m5-forecasting-accuracy.zip::sales_train_validation.csv",
                    ],
                    domain_parameters=features(
                        series_semantics="recorded item-store unit sales",
                        window_start=selected_days[0],
                        window_end=selected_days[-1],
                    ),
                    state_eligible=request.include_state,
                )
            )
        return records
