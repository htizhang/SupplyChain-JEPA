from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator

from supplychain_jepa.adapters import AdapterRequest, load_adapter
from supplychain_jepa.data import EpisodeTensorizer, fit_tensorizer_state
from supplychain_jepa.schemas import canonical_episode_schema


def _write_zip(path: Path, members: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)


def _m5(root: Path) -> None:
    calendar = pd.DataFrame(
        {
            "d": ["d_1", "d_2"],
            "date": ["2011-01-29", "2011-01-30"],
            "weekday": ["Saturday", "Sunday"],
            "wm_yr_wk": [11101, 11101],
            "event_name_1": [None, "Event"],
            "event_type_1": [None, "Cultural"],
            "snap_CA": [0, 1],
        }
    )
    sales = pd.DataFrame(
        {
            "id": ["item_1_store_1_validation"],
            "item_id": ["item_1"],
            "dept_id": ["dept_1"],
            "cat_id": ["cat_1"],
            "store_id": ["store_1"],
            "state_id": ["CA"],
            "d_1": [2],
            "d_2": [3],
        }
    )
    _write_zip(
        root / "m5-forecasting-accuracy.zip",
        {
            "calendar.csv": calendar.to_csv(index=False),
            "sales_train_validation.csv": sales.to_csv(index=False),
        },
    )


def _bpi2019(root: Path) -> None:
    frame = pd.DataFrame(
        {
            "case concept:name": ["case-1", "case-1"],
            "event time:timestamp": [
                "01-01-2019 00:00:00.000",
                "01-01-2019 01:00:00.000",
            ],
            "eventID ": ["event-1", "event-2"],
            "case Company": ["company-1", "company-1"],
            "case Item": ["item-1", "item-1"],
            "case Purchasing Document": ["po-1", "po-1"],
            "case Vendor": ["vendor-1", "vendor-1"],
            "case Document Type": ["standard", "standard"],
            "case Item Type": ["material", "material"],
            "case Item Category": ["normal", "normal"],
            "case Goods Receipt": [True, True],
            "event concept:name": ["Create Purchase Order", "Record Goods Receipt"],
            "event org:resource": ["buyer", "warehouse"],
            "event Cumulative net worth (EUR)": [10.0, 10.0],
        }
    )
    _write_zip(
        root / "BPIChallenge2019CSV.zip",
        {"BPI_Challenge_2019.csv": frame.to_csv(index=False)},
    )


def _lade(root: Path) -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["order-1", "order-2"],
            "courier_id": ["courier-1", "courier-1"],
            "ds": ["2023-01-01", "2023-01-01"],
            "city": ["Jilin", "Jilin"],
            "region_id": [1, 1],
            "aoi_id": [11, 12],
            "aoi_type": ["residential", "commercial"],
            "lng": [126.5, 126.6],
            "lat": [43.8, 43.9],
            "accept_time": ["01-01 08:00:00", "01-01 09:00:00"],
            "delivery_time": ["01-01 10:00:00", "01-01 11:00:00"],
            "delivery_gps_lng": [126.5, 126.6],
            "delivery_gps_lat": [43.8, 43.9],
        }
    )
    frame.to_csv(root / "delivery_jl.csv", index=False)


def _olist(root: Path) -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["order-1", "order-2"],
            "order_status": ["delivered", "delivered"],
            "order_purchase_timestamp": ["2018-01-01", "2018-01-02"],
            "order_approved_at": ["2018-01-01", "2018-01-02"],
            "order_delivered_carrier_date": ["2018-01-02", "2018-01-03"],
            "order_delivered_customer_date": ["2018-01-04", "2018-01-05"],
            "order_estimated_delivery_date": ["2018-01-06", "2018-01-06"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["order-1", "order-2"],
            "order_item_id": [1, 1],
            "product_id": ["product-1", "product-2"],
            "seller_id": ["seller-1", "seller-1"],
            "price": [10.0, 12.0],
            "freight_value": [2.0, 2.5],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["product-1", "product-2"],
            "product_category_name": ["food", "food"],
            "product_weight_g": [100.0, 120.0],
            "product_length_cm": [10.0, 11.0],
            "product_height_cm": [5.0, 5.0],
            "product_width_cm": [8.0, 8.0],
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["seller-1"],
            "seller_city": ["Sao Paulo"],
            "seller_state": ["SP"],
        }
    )
    _write_zip(
        root / "olist.zip",
        {
            "olist_orders_dataset.csv": orders.to_csv(index=False),
            "olist_order_items_dataset.csv": items.to_csv(index=False),
            "olist_products_dataset.csv": products.to_csv(index=False),
            "olist_sellers_dataset.csv": sellers.to_csv(index=False),
        },
    )


def _planned_actual_routes(root: Path) -> None:
    frame = pd.DataFrame(
        {
            "Route ID": ["route-1", "route-1"],
            "IndexA": [1, 2],
            "IndexP": [1, 2],
            "Stop ID": ["stop-1", "stop-2"],
            "DistanceA": [1.0, 1.5],
            "DistanceP": [1.1, 1.4],
            "Arrived Time": [9.0, 10.0],
            "Earliest Time": [8.0, 9.0],
            "Latest Time": [10.0, 11.0],
            "Driver ID": ["driver-1", "driver-1"],
            "Address ID": ["address-1", "address-2"],
            "Week ID": [1, 1],
            "Country": ["NL", "NL"],
            "Day of Week": ["Monday", "Monday"],
            "Depot": [0, 0],
            "Delivery": [1, 1],
        }
    )
    workbook = io.BytesIO()
    frame.to_excel(workbook, index=False, engine="openpyxl")
    _write_zip(root / "last-mile-route-deviations-v1.zip", {"routes.xlsx": workbook.getvalue()})


BUILDERS = {
    "m5": _m5,
    "bpi2019": _bpi2019,
    "lade": _lade,
    "olist": _olist,
    "planned_actual_routes": _planned_actual_routes,
}


def test_all_observational_adapters_validate_and_tensorize(tmp_path: Path) -> None:
    schema = json.loads(canonical_episode_schema().read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    records = []
    for source_id, build_fixture in BUILDERS.items():
        source_root = tmp_path / source_id
        source_root.mkdir()
        build_fixture(source_root)
        converted = load_adapter(source_id).convert(
            AdapterRequest(raw_root=source_root, limit=1, pilot=True)
        )
        assert converted
        for record in converted:
            validator.validate(record)
            assert record["simulator"]["family"] == source_id
            assert record["branch"] is None
            assert record["objective_eligibility"] == {
                "state_jepa": True,
                "dynamics_jepa": False,
                "decision_heads": False,
            }
        records.extend(converted)

    state = fit_tensorizer_state(
        records,
        train_splits=("pretrain_train", "pretrain_dev", "pretrain_test"),
    )
    tensorizer = EpisodeTensorizer(state)
    tensors = [tensorizer.transform(record) for record in records]
    assert len(tensors) == len(BUILDERS)
    assert {episode.family for episode in tensors} == set(BUILDERS)
    assert all(episode.context_length > 0 for episode in tensors)
    assert all(episode.target_length > 0 for episode in tensors)
