"""State-only adapter for a request-scoped, pinned ISOMORPH checkout.

ISOMORPH supplies long-horizon state and demand variation but no external
step-wise action interface.  The smoke records therefore label its internal
fixed-policy dispatches as non-interventional and explicitly disable
Dynamics-JEPA eligibility.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import AdapterRequest, SourceAdapter
from .common import (
    SCHEMA_VERSION,
    json_safe,
    make_action,
    make_edge_state,
    make_entity,
    make_event,
    make_record_id,
    observation_status,
    stable_digest,
)

SOURCE_COMMIT = "1316f44b24678c8f2d32d10534118330eb02d194"
SOURCE_VERSION = "commit-1316f44b2467"
SOURCE_REPOSITORY = "https://github.com/tuhinsahai/ISOMORPH"
SOURCE_LICENSE = "MIT"
SOURCE_FILE = Path("simulator/Supplychaingeo_item50.py")
ADAPTER_VERSION = "v2-isomorph-state-v1"
GENERATOR_NAME = "supplychain_jepa.adapters.isomorph.ISOMORPHAdapter"
HORIZON = 12
SELECTED_ITEMS = ("I01", "I02", "I03")


def _checkout_commit(root: Path) -> str:
    """Read and verify an immutable checkout revision from Git or a sidecar."""
    if (root / ".git").exists():
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot verify pinned ISOMORPH checkout: {root}") from exc
    pin_file = root / ".scjepa-source-commit"
    if pin_file.is_file():
        return pin_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"Cannot verify pinned ISOMORPH checkout: {root}")


def _source_path(source_root: Path) -> Path:
    path = source_root.resolve() / SOURCE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Pinned ISOMORPH source not found: {path}")
    return path


def _load_source_module(source_root: Path):
    """Load the pinned source file without writing into the upstream checkout."""
    root = source_root.resolve()
    revision = _checkout_commit(root)
    if revision != SOURCE_COMMIT:
        raise RuntimeError(
            f"ISOMORPH checkout revision mismatch: expected {SOURCE_COMMIT}, found {revision}"
        )
    path = _source_path(root)
    module_name = f"_sc_jepa_isomorph_{stable_digest(str(root), length=12)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _scenario_parameters(name: str) -> dict[str, Any]:
    common = {
        "base_lambda_lo": 32.0,
        "base_lambda_hi": 48.0,
        "ss_scale": 0.35,
        "burst_rate_scale": 0.0,
        "shock_count_scale": 0.0,
    }
    if name == "baseline":
        return {
            **common,
            "containers_scale": 1.0,
            "leadtime_scale": 1.0,
            "seasonal_scale": 0.5,
        }
    return {
        **common,
        "containers_scale": 0.35,
        "leadtime_scale": 1.75,
        "seasonal_scale": 0.8,
        "demand_shock_start": 4.0,
        "demand_shock_end": 8.0,
        "demand_shock_multiplier": 3.0,
    }


def _restrict_items(sim: Any, net: Any, items: dict[str, Any]) -> None:
    selected = {item_id: items[item_id] for item_id in SELECTED_ITEMS}
    sim.items = selected
    sim.item_order = sorted(selected)
    sim.demand_ema = {item_id: sim.demand_ema[item_id] for item_id in selected}
    sim.item_intransit = {item_id: sim.item_intransit[item_id] for item_id in selected}
    sim.svc_demand = {item_id: 0 for item_id in selected}
    sim.svc_served = {item_id: 0 for item_id in selected}
    sim.svc_backlog = {item_id: 0 for item_id in selected}
    for node in net.nodes.values():
        for field_name in (
            "inventory",
            "s_levels",
            "S_levels",
            "lead_time_mean",
            "outstanding_orders",
            "backlog",
        ):
            values = getattr(node, field_name)
            setattr(
                node,
                field_name,
                {item_id: values.get(item_id) for item_id in SELECTED_ITEMS if item_id in values},
            )


def _install_demand_scenario(
    sim: Any,
    scenario_name: str,
    *,
    shock_multiplier: float | None = None,
) -> None:
    base_demand_fn = sim.demand_fn
    multiplier_value = 3.0 if shock_multiplier is None else float(shock_multiplier)

    def demand_fn(day: int) -> dict[str, int]:
        values = base_demand_fn(day)
        multiplier = 1.0
        if scenario_name == "shock" and 4 <= day <= 8:
            multiplier = multiplier_value
        return {
            item_id: int(round(values.get(item_id, 0) * multiplier)) for item_id in SELECTED_ITEMS
        }

    sim.demand_fn = demand_fn


def _topology(net: Any) -> dict[str, Any]:
    nodes = []
    for node_id, node in sorted(net.nodes.items()):
        nodes.append(
            {
                "node_id": f"node:{node_id}",
                "node_type": (
                    "destination"
                    if node.is_destination
                    else "source"
                    if node.is_source
                    else "warehouse"
                ),
                "features": {
                    "source_node_id": node_id,
                    "latitude": float(node.lat),
                    "longitude": float(node.lon),
                    "is_destination": bool(node.is_destination),
                    "is_source": bool(node.is_source),
                },
            }
        )
    edges = []
    for (source, target), edge in sorted(net.edges.items()):
        edges.append(
            {
                "edge_id": f"edge:{source}->{target}",
                "source": f"node:{source}",
                "target": f"node:{target}",
                "edge_type": "transport",
                "features": {
                    "travel_time_days": float(edge.travel_time_days),
                    "container_volume": float(edge.container_volume),
                    "containers_per_day": float(edge.num_containers_per_day),
                    "daily_total_capacity": float(edge.daily_total_capacity),
                },
            }
        )
    return {
        "topology_id": f"isomorph:usa-dag:{stable_digest(nodes, edges, length=12)}",
        "nodes": nodes,
        "edges": edges,
    }


def _daily_rows(sim: Any, day: int) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in sim.daily_records:
        if int(row["day"]) == day and row["item"] in SELECTED_ITEMS:
            rows[row["item"]] = row
    return rows


def _daily_shipments(sim: Any, day: int) -> list[dict[str, Any]]:
    return [
        row for row in sim.shipments_log if int(row["day"]) == day and row["item"] in SELECTED_ITEMS
    ]


def _snapshot_entities(sim: Any, day: int) -> list[dict[str, Any]]:
    entities = []
    for node_id, node in sorted(sim.network.nodes.items()):
        for item_id in SELECTED_ITEMS:
            order = node.outstanding_orders.get(item_id)
            arrival_day = None
            open_quantity = 0.0
            if order is not None:
                arrival_day, open_quantity = order
            entities.append(
                make_entity(
                    f"node-item:{node_id}:{item_id}",
                    "node_product_state",
                    node_id=f"node:{node_id}",
                    product_id=f"product:{item_id}",
                    features={
                        "inventory_on_hand": float(node.inventory.get(item_id, 0)),
                        "inventory": float(node.inventory.get(item_id, 0)),
                        "backlog": float(node.backlog.get(item_id, 0)),
                        "pipeline_in_transit": (
                            float(sim.item_intransit.get(item_id, 0))
                            if node_id == sim.destination_id
                            else 0.0
                        ),
                        "destination_in_transit": (
                            float(sim.item_intransit.get(item_id, 0))
                            if node_id == sim.destination_id
                            else 0.0
                        ),
                        "open_order_quantity": float(open_quantity),
                        "open_order_arrival_day": (
                            float(arrival_day) if arrival_day is not None else -1.0
                        ),
                        "snapshot_day": float(day),
                    },
                )
            )
    return entities


def _edge_states(sim: Any, shipments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states = []
    for (source, target), edge in sorted(sim.network.edges.items()):
        dispatched = 0.0
        for shipment in shipments:
            path = list(
                zip(
                    shipment["path_nodes"][:-1],
                    shipment["path_nodes"][1:],
                    strict=True,
                )
            )
            if (source, target) in path:
                dispatched += float(shipment["units"])
        states.append(
            make_edge_state(
                f"edge:{source}->{target}",
                {
                    "shipment_units_dispatched": dispatched,
                    "remaining_container_volume": float(sum(edge.daily_containers)),
                    "nominal_daily_capacity": float(edge.daily_total_capacity),
                },
            )
        )
    return states


def _latent_state(
    sim: Any,
    day: int,
    daily: dict[str, dict[str, Any]],
    shipments: list[dict[str, Any]],
) -> dict[str, Any]:
    demand_total = float(sum(row["demand"] for row in daily.values()))
    served_total = float(sum(row["served_from_stock"] for row in daily.values()))
    backlog_added = float(sum(row["new_backlog_today"] for row in daily.values()))
    events = []
    for item_id, row in sorted(daily.items()):
        events.append(
            make_event(
                f"demand:{day}:{item_id}",
                "destination_demand",
                f"node-item:{sim.destination_id}:{item_id}",
                node_id=f"node:{sim.destination_id}",
                event_time=day,
                report_time=day + 1,
                observed=False,
                features={
                    "demand": float(row["demand"]),
                    "served_from_stock": float(row["served_from_stock"]),
                    "new_backlog": float(row["new_backlog_today"]),
                },
            )
        )
    for index, shipment in enumerate(shipments):
        events.append(
            make_event(
                f"shipment:{day}:{index}:{shipment['item']}",
                "shipment_dispatch",
                f"node-item:{shipment['from']}:{shipment['item']}",
                node_id=f"node:{shipment['from']}",
                event_time=day,
                report_time=day + 1,
                observed=False,
                features={
                    "quantity": float(shipment["units"]),
                    "arrival_day": float(shipment["arrival_day"]),
                    "destination": str(shipment["to"]),
                },
            )
        )
    return {
        "entities": _snapshot_entities(sim, day),
        "edges": _edge_states(sim, shipments),
        "global": {
            "day": float(day),
            "demand_true": demand_total,
            "demand_total": demand_total,
            "sales_observed": served_total,
            "served_total": served_total,
            "new_backlog_total": backlog_added,
            "inventory_total": float(
                sum(
                    node.inventory.get(item_id, 0)
                    for node in sim.network.nodes.values()
                    for item_id in SELECTED_ITEMS
                )
            ),
            "backlog_total": float(
                sum(
                    node.backlog.get(item_id, 0)
                    for node in sim.network.nodes.values()
                    for item_id in SELECTED_ITEMS
                )
            ),
            "destination_in_transit_total": float(
                sum(sim.item_intransit.get(item_id, 0) for item_id in SELECTED_ITEMS)
            ),
            "shipment_units_dispatched_total": float(
                sum(shipment["units"] for shipment in shipments)
            ),
        },
        "events": events,
    }


def _observation(
    *,
    sim: Any,
    day: int,
    daily: dict[str, dict[str, Any]],
    latent: dict[str, Any],
    prior_entities: list[dict[str, Any]] | None,
    event_history: list[dict[str, Any]],
    scenario_name: str,
) -> dict[str, Any]:
    outage_nodes = {"Atlanta", "Philadelphia"} if 6 <= day <= 7 else set()
    source_entities = prior_entities if prior_entities is not None else latent["entities"]
    entities = []
    statuses = []
    for entity in source_entities:
        node_id = entity["node_id"].split(":", 1)[1]
        if node_id in outage_nodes:
            entities.append(
                make_entity(
                    entity["entity_id"],
                    "node_product_report",
                    node_id=entity["node_id"],
                    product_id=entity["product_id"],
                    features={"report_available": False},
                )
            )
            statuses.append(
                observation_status(
                    f"observation.entities.{entity['entity_id']}.report_available",
                    "outage",
                    reason="structured warehouse reporting outage",
                )
            )
        else:
            reported = dict(entity["features"])
            reported["report_available"] = True
            entities.append(
                make_entity(
                    entity["entity_id"],
                    "node_product_report",
                    node_id=entity["node_id"],
                    product_id=entity["product_id"],
                    features=reported,
                )
            )
            statuses.append(
                observation_status(
                    f"observation.entities.{entity['entity_id']}.snapshot_day",
                    "delayed" if prior_entities is not None else "observed",
                    age_steps=1 if prior_entities is not None else 0,
                    reason=(
                        "inventory and backlog reports arrive one day late"
                        if prior_entities is not None
                        else "initial same-day report"
                    ),
                )
            )

    edge_states = []
    for edge_state in latent["edges"]:
        source_target = edge_state["edge_id"].split(":", 1)[1]
        source, target = source_target.split("->", 1)
        if source in outage_nodes or target in outage_nodes:
            edge_states.append(make_edge_state(edge_state["edge_id"], {"report_available": False}))
            statuses.append(
                observation_status(
                    f"observation.edges.{edge_state['edge_id']}.report_available",
                    "missing",
                    reason="transport report unavailable during adjacent-node outage",
                )
            )
        else:
            features = dict(edge_state["features"])
            features["report_available"] = True
            edge_states.append(make_edge_state(edge_state["edge_id"], features))

    demand_reported = 0.0
    sales_observed = 0.0
    demand_observation_features: dict[str, Any] = {}
    for item_id, row in sorted(daily.items()):
        demand = float(row["demand"])
        sales = float(row["served_from_stock"])
        stockout = bool(sales + 1e-9 < demand)
        demand_reported += sales if stockout else demand
        sales_observed += sales
        demand_observation_features[f"demand_reported_{item_id}"] = sales if stockout else demand
        demand_observation_features[f"sales_observed_{item_id}"] = sales
        statuses.append(
            observation_status(
                f"observation.global.demand_reported_{item_id}",
                "censored" if stockout else "observed",
                reason=(
                    "stockout-censored sales are a lower bound on latent demand"
                    if stockout
                    else "demand observed because inventory was sufficient"
                ),
            )
        )

    reported_events = []
    for event in event_history:
        if event["report_time"] is not None and event["report_time"] <= day:
            reported = dict(event)
            reported["observed"] = True
            reported_events.append(reported)
    for event in latent["events"]:
        report_time = event["report_time"]
        if report_time is None or report_time > day:
            statuses.append(
                observation_status(
                    f"latent_state.events.{event['event_id']}",
                    "missing" if report_time is None else "delayed",
                    reason=(
                        "physical event is permanently absent from the report stream"
                        if report_time is None
                        else "physical event has not yet reached the reporting stream"
                    ),
                )
            )
    return {
        "entities": entities,
        "edges": edge_states,
        "global": {
            "day": float(day),
            "demand_reported_total": demand_reported,
            "sales_observed_total": sales_observed,
            "scenario_shock_flag": scenario_name == "shock" and 4 <= day <= 8,
            **demand_observation_features,
        },
        "events": reported_events,
        "status": statuses,
    }


def _actions(sim: Any, day: int, shipments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not shipments:
        return [
            make_action(
                "fixed_policy_no_dispatch",
                f"node:{sim.destination_id}",
                target_id=None,
                quantity=0.0,
                features={
                    "interventional": False,
                    "policy": "internal_sS_plus_reactive_dispatch",
                    "day": float(day),
                },
            )
        ]
    return [
        make_action(
            "fixed_policy_dispatch",
            f"node:{shipment['from']}",
            target_id=f"node:{shipment['to']}",
            quantity=float(shipment["units"]),
            features={
                "interventional": False,
                "policy": "internal_sS_plus_reactive_dispatch",
                "item_id": str(shipment["item"]),
                "arrival_day": float(shipment["arrival_day"]),
            },
        )
        for shipment in shipments
    ]


def _build_record(
    scenario_name: str,
    seed: int,
    *,
    source_root: Path,
    split: str = "pretrain_train",
    config_id: str | None = None,
    horizon: int = HORIZON,
    scenario_parameters: dict[str, Any] | None = None,
    pipeline_multiplier: float = 2.0,
    destination_initial_inventory: int | None = None,
    dispatch_cap: int | None = None,
    shock_multiplier: float | None = None,
    domain_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = _load_source_module(source_root)
    scenario = (
        _scenario_parameters(scenario_name)
        if scenario_parameters is None
        else dict(scenario_parameters)
    )
    random_state = random.getstate()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sim, net, items, _ = source.build_example_simulation_from_adjacency(
                seed=seed,
                horizon_days=horizon,
                pipeline_multiplier=pipeline_multiplier,
                streaming_out_dir=None,
                packing="greedy",
                scenario=scenario,
            )
        _restrict_items(sim, net, items)
        _install_demand_scenario(
            sim,
            scenario_name,
            shock_multiplier=shock_multiplier,
        )
        destination_initial_inventory = (
            (45 if scenario_name == "baseline" else 25)
            if destination_initial_inventory is None
            else int(destination_initial_inventory)
        )
        dispatch_cap = (
            (25 if scenario_name == "baseline" else 12)
            if dispatch_cap is None
            else int(dispatch_cap)
        )
        for item_id in SELECTED_ITEMS:
            net.nodes[sim.destination_id].inventory[item_id] = destination_initial_inventory
        sim.per_item_daily_cap_units = dispatch_cap

        steps = []
        prior_entities = None
        event_history: list[dict[str, Any]] = []
        for day in range(horizon):
            sim.step(day)
            daily = _daily_rows(sim, day)
            shipments = _daily_shipments(sim, day)
            latent = _latent_state(sim, day, daily, shipments)
            event_history.extend(latent["events"])
            demand_vector = [float(daily[item_id]["demand"]) for item_id in SELECTED_ITEMS]
            demand_total = float(sum(demand_vector))
            served_total = float(
                sum(daily[item_id]["served_from_stock"] for item_id in SELECTED_ITEMS)
            )
            steps.append(
                {
                    "t": day,
                    "clock": float(day),
                    "latent_state": latent,
                    "observation": _observation(
                        sim=sim,
                        day=day,
                        daily=daily,
                        latent=latent,
                        prior_entities=prior_entities,
                        event_history=event_history,
                        scenario_name=scenario_name,
                    ),
                    "actions": _actions(sim, day, shipments),
                    "outcomes": {
                        "demand_total": demand_total,
                        "served_total": served_total,
                        "new_backlog_total": float(
                            sum(daily[item_id]["new_backlog_today"] for item_id in SELECTED_ITEMS)
                        ),
                        "fill_rate": (served_total / demand_total if demand_total > 0 else 1.0),
                        "shipment_units_dispatched_total": float(
                            sum(row["units"] for row in shipments)
                        ),
                    },
                    "exogenous_digest": stable_digest(
                        "isomorph",
                        scenario_name,
                        seed,
                        day,
                        demand_vector,
                    ),
                    "terminal": day == horizon - 1,
                }
            )
            prior_entities = latent["entities"]
    finally:
        random.setstate(random_state)

    world_id = f"isomorph-{scenario_name}-seed-{seed}"
    episode_domain_parameters = (
        {
            "scenario": scenario_name,
            "selected_item_count": float(len(SELECTED_ITEMS)),
            "selected_items": ",".join(SELECTED_ITEMS),
            "pipeline_multiplier": 2.0,
            "fixed_policy": "internal_sS_plus_reactive_dispatch",
            "actions_interventional": False,
            "containers_scale": float(scenario["containers_scale"]),
            "leadtime_scale": float(scenario["leadtime_scale"]),
            "destination_initial_inventory_per_item": float(destination_initial_inventory),
            "per_item_daily_dispatch_cap": float(dispatch_cap),
        }
        if domain_parameters is None
        else dict(domain_parameters)
    )
    return json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "record_id": make_record_id("isomorph", world_id, None),
            "simulator": {
                "family": "isomorph",
                "name": "ISOMORPH",
                "version": SOURCE_VERSION,
                "source_repository": SOURCE_REPOSITORY,
                "source_commit": SOURCE_COMMIT,
                "license": "MIT",
            },
            "split": split,
            "episode": {
                "episode_id": world_id,
                "world_id": world_id,
                "config_id": config_id or f"item50-{scenario_name}-subset3-v1",
                "seed": seed,
                "time_unit": "day",
                "horizon": horizon,
                "exogenous_stream_id": stable_digest(
                    "isomorph-demand-stream",
                    scenario_name,
                    seed,
                    [step["exogenous_digest"] for step in steps],
                    length=20,
                ),
                "domain_parameters": episode_domain_parameters,
            },
            "branch": None,
            "objective_eligibility": {
                "state_jepa": True,
                "dynamics_jepa": False,
                "decision_heads": True,
            },
            "topology": _topology(net),
            "steps": steps,
            "provenance": {
                "adapter_version": ADAPTER_VERSION,
                "generator": "scripts.adapters.isomorph.generate_smoke_records",
                "causal_semantics": (
                    "Dispatches and replenishments are generated by the "
                    "simulator's fixed internal policy; they are observational, "
                    "not externally randomized interventions."
                ),
                "observation_semantics": (
                    "Latent inventory, backlog, open orders, destination "
                    "in-transit quantities, shipments, and demand are complete; "
                    "reports are one day delayed, stockout-censored, and include "
                    "a two-day structured warehouse outage."
                ),
                "source_files": [str(SOURCE_FILE)],
            },
        }
    )


_FORBIDDEN_LINEAGE_MARKERS = (
    "freshretailnet",
    "cargo2000",
    "gym-invmgmt",
    "gym_invmgmt",
    "or-gym",
    "or_gym",
    "or-gymnax",
    "downstream-application",
)


def _split_for(world_id: str) -> str:
    bucket = int(stable_digest("isomorph-v2-split", world_id, length=8), 16) % 20
    if bucket == 18:
        return "pretrain_dev"
    if bucket == 19:
        return "pretrain_test"
    return "pretrain_train"


def _world_seed(seed: int, world_index: int) -> int:
    return int(
        stable_digest("isomorph-v2-world-seed", seed, world_index, length=8),
        16,
    )


def _generated_world_parameters(
    seed: int,
    world_index: int,
    scenario_name: str,
) -> dict[str, Any]:
    rng = random.Random(
        int(
            stable_digest(
                "isomorph-v2-generated-inputs",
                seed,
                world_index,
                length=16,
            ),
            16,
        )
    )
    scenario = _scenario_parameters(scenario_name)
    demand_scale = rng.uniform(0.72, 1.38)
    scenario.update(
        {
            "base_lambda_lo": scenario["base_lambda_lo"] * demand_scale,
            "base_lambda_hi": scenario["base_lambda_hi"] * demand_scale,
            "containers_scale": rng.uniform(0.45, 1.25),
            "leadtime_scale": rng.uniform(0.75, 1.65),
            "ss_scale": rng.uniform(0.22, 0.58),
        }
    )
    return {
        "scenario": scenario,
        "demand_scale": demand_scale,
        "pipeline_multiplier": rng.uniform(1.25, 2.75),
        "initial_inventory": rng.randint(18, 64),
        "dispatch_cap": rng.randint(10, 32),
        "shock_multiplier": rng.uniform(1.8, 3.8) if scenario_name == "shock" else 1.0,
    }


def _release_record(
    *,
    source_root: Path,
    seed: int,
    world_index: int,
    horizon: int,
) -> dict[str, Any]:
    scenario_name = "baseline" if world_index % 2 == 0 else "shock"
    generated = _generated_world_parameters(seed, world_index, scenario_name)
    world_seed = _world_seed(seed, world_index)
    world_id = f"isomorph-v2-seed-{seed}-world-{world_index:06d}"
    split = _split_for(world_id)
    record = _build_record(
        scenario_name,
        world_seed,
        source_root=source_root,
        split=split,
        config_id=stable_digest(
            "isomorph-v2-config",
            scenario_name,
            generated["scenario"],
            generated["pipeline_multiplier"],
            generated["initial_inventory"],
            generated["dispatch_cap"],
            generated["shock_multiplier"],
            length=16,
        ),
        horizon=horizon,
        scenario_parameters=generated["scenario"],
        pipeline_multiplier=generated["pipeline_multiplier"],
        destination_initial_inventory=generated["initial_inventory"],
        dispatch_cap=generated["dispatch_cap"],
        shock_multiplier=generated["shock_multiplier"],
    )

    record["record_id"] = make_record_id("isomorph", world_id, None)
    record["split"] = split
    record["episode"].update(
        {
            "episode_id": f"{world_id}:state",
            "world_id": world_id,
            "seed": world_seed,
            "horizon": horizon,
            "exogenous_stream_id": stable_digest(
                "isomorph-v2-generated-stream",
                seed,
                world_index,
                [step["exogenous_digest"] for step in record["steps"]],
                length=20,
            ),
        }
    )
    record["episode"]["domain_parameters"].update(
        {
            "generated_inputs_only": True,
            "downstream_lineage_used": False,
            "world_index": float(world_index),
            "root_seed": float(seed),
            "demand_scale": float(generated["demand_scale"]),
            "pipeline_multiplier": float(generated["pipeline_multiplier"]),
        }
    )
    record["branch"] = None
    record["objective_eligibility"] = {
        "state_jepa": True,
        "dynamics_jepa": False,
        "decision_heads": False,
    }
    record["simulator"].update(
        {
            "family": "isomorph",
            "name": "ISOMORPH",
            "version": SOURCE_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "license": SOURCE_LICENSE,
        }
    )
    record["provenance"] = {
        "adapter_version": ADAPTER_VERSION,
        "generator": GENERATOR_NAME,
        "causal_semantics": (
            "Generated observational rollout under the simulator's fixed internal "
            "policy; emitted actions are not interventions."
        ),
        "observation_semantics": (
            "Latent inventory, backlog, open orders, in-transit quantities, shipments, "
            "and demand are complete; reports are delayed, stockout-censored, and "
            "subject to a structured warehouse outage."
        ),
        "source_files": [SOURCE_FILE.as_posix()],
        "runtime_dependencies": {
            "isomorph_revision": SOURCE_COMMIT,
            "source_import_mode": "request_source_root",
        },
    }
    serialized = repr(record).lower()
    found = [marker for marker in _FORBIDDEN_LINEAGE_MARKERS if marker in serialized]
    if found:
        raise RuntimeError(f"Forbidden downstream lineage marker emitted: {found}")
    return record


def generate_isomorph_records(
    *,
    source_root: Path,
    worlds: int = 8,
    horizon: int = 32,
    seed: int = 20260730,
    include_state: bool = True,
    include_dynamics: bool = False,
) -> list[dict[str, Any]]:
    """Generate deterministic, State-only ISOMORPH episodes."""
    if worlds < 0:
        raise ValueError("worlds must be non-negative")
    if horizon < 4:
        raise ValueError("horizon must be at least 4")
    if include_dynamics:
        raise ValueError("ISOMORPH adapter is State-only")
    if not include_state or worlds == 0:
        return []

    root = source_root.resolve()
    # Fail once before doing any expensive simulator work.
    _source_path(root)
    revision = _checkout_commit(root)
    if revision != SOURCE_COMMIT:
        raise RuntimeError(
            f"ISOMORPH checkout revision mismatch: expected {SOURCE_COMMIT}, found {revision}"
        )
    return [
        _release_record(
            source_root=root,
            seed=seed,
            world_index=world_index,
            horizon=horizon,
        )
        for world_index in range(worlds)
    ]


class ISOMORPHAdapter(SourceAdapter):
    """Convert generated ISOMORPH worlds from a pinned external checkout."""

    source_id = "isomorph"
    source_kind = "simulator"
    supports_state = True
    supports_dynamics = False

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        include_dynamics = self.resolved_dynamics_flag(request)
        if not request.include_state:
            return []
        if request.source_root is None:
            raise ValueError("isomorph requires AdapterRequest.source_root")
        worlds = request.worlds
        if request.limit is not None:
            worlds = min(worlds, request.limit)
        return generate_isomorph_records(
            source_root=request.source_root,
            worlds=worlds,
            horizon=request.horizon,
            seed=request.seed,
            include_state=request.include_state,
            include_dynamics=include_dynamics,
        )


__all__ = ["ADAPTER_VERSION", "ISOMORPHAdapter", "SOURCE_COMMIT"]
