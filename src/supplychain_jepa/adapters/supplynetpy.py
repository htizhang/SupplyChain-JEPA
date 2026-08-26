"""State-only adapter for a request-scoped, pinned SupplyNetPy checkout.

The adapter generates every network, demand stream, policy, and observation
schedule locally.  It never reads SupplyNetPy's packaged examples or datasets.
Replenishment actions are fixed-policy observations, not interventions.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

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

SOURCE_COMMIT = "00c23633d5ae92d9437032f27789f298f73f86cf"
SOURCE_VERSION = "0.1.12"
SOURCE_REPOSITORY = "https://github.com/SupplyChainSimulation/SupplyNetPy"
SOURCE_LICENSE = "MIT"
ADAPTER_VERSION = "v2-supplynetpy-state-only-v1"
GENERATOR_NAME = "supplychain_jepa.adapters.supplynetpy.SupplyNetPyAdapter"
_CORE_RELATIVE = Path("src/SupplyNetPy/Components/core.py")


def _checkout_revision(root: Path) -> tuple[str, str]:
    """Return the exact checkout revision, preferring Git over the sidecar."""
    if (root / ".git").exists():
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot verify pinned SupplyNetPy checkout: {root}") from exc
        return revision, "git"

    sidecar = root / ".scjepa-source-commit"
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8").strip(), "sidecar"
    raise RuntimeError(f"Cannot verify pinned SupplyNetPy checkout: {root}")


def _package(name: str, path: Path) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(path)]  # type: ignore[attr-defined]
    return package


def _load_supplynetpy(source_root: Path) -> tuple[Any, str, str]:
    """Import only the core API from the exact request-scoped checkout.

    Loading ``SupplyNetPy.Components`` executes visualization imports that this
    adapter does not use.  Synthetic package shells let Python load ``core.py``
    and its logger dependency without importing those unrelated optional APIs.
    """
    root = source_root.resolve()
    expected = root / _CORE_RELATIVE
    if not expected.is_file():
        raise FileNotFoundError(f"Pinned SupplyNetPy checkout not found: {root}")

    revision, verification_mode = _checkout_revision(root)
    if revision != SOURCE_COMMIT:
        raise RuntimeError(
            f"SupplyNetPy checkout revision mismatch: expected {SOURCE_COMMIT}, found {revision}"
        )

    existing = sys.modules.get("SupplyNetPy.Components.core")
    if existing is not None:
        module_file = Path(getattr(existing, "__file__", "")).resolve()
        if module_file == expected:
            return existing, _CORE_RELATIVE.as_posix(), verification_mode
        for name in tuple(sys.modules):
            if name == "SupplyNetPy" or name.startswith("SupplyNetPy."):
                del sys.modules[name]

    package_root = root / "src" / "SupplyNetPy"
    components_root = package_root / "Components"
    sys.modules["SupplyNetPy"] = _package("SupplyNetPy", package_root)
    sys.modules["SupplyNetPy.Components"] = _package("SupplyNetPy.Components", components_root)

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        core = importlib.import_module("SupplyNetPy.Components.core")
    except ImportError as exc:
        for name in tuple(sys.modules):
            if name == "SupplyNetPy" or name.startswith("SupplyNetPy."):
                del sys.modules[name]
        if exc.name == "simpy":
            raise RuntimeError(
                "SupplyNetPy conversion requires its runtime dependency 'simpy'"
            ) from exc
        raise
    finally:
        sys.dont_write_bytecode = previous

    module_path = Path(core.__file__).resolve()
    if module_path != expected:
        raise ImportError(f"SupplyNetPy resolved outside the pinned checkout: {module_path}")
    return core, _CORE_RELATIVE.as_posix(), verification_mode


class _ValidationSafeSequence:
    """Replay values while making SupplyNetPy's constructor probe non-consuming."""

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._index = 0
        self._constructor_probe = True

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        if self._constructor_probe:
            self._constructor_probe = False
        else:
            self._index += 1
        return float(value)


def _split_for(world_id: str) -> str:
    bucket = int(stable_digest("supplynetpy-v2-split", world_id, length=8), 16) % 20
    if bucket == 18:
        return "pretrain_dev"
    if bucket == 19:
        return "pretrain_test"
    return "pretrain_train"


def _world_spec(seed: int, world_index: int, horizon: int) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, world_index]))
    base_demand = float(rng.uniform(7.0, 18.0))
    periods = np.arange(horizon, dtype=float)
    seasonal = 1.0 + float(rng.uniform(0.08, 0.30)) * np.sin(
        2.0 * np.pi * periods / 7.0 + float(rng.uniform(0.0, 2.0 * np.pi))
    )
    demand = rng.poisson(np.maximum(base_demand * seasonal, 0.5)).astype(float)
    if horizon >= 6:
        shock_at = int(rng.integers(horizon // 3, max(horizon // 3 + 1, horizon - 2)))
        shock_width = min(int(rng.integers(1, 4)), horizon - shock_at)
        demand[shock_at : shock_at + shock_width] += rng.integers(
            max(2, int(base_demand * 0.5)),
            max(3, int(base_demand * 1.4)),
            size=shock_width,
        )
    demand = np.maximum(demand, 1.0)

    retailer_capacity = float(round(base_demand * rng.uniform(7.0, 11.0), 4))
    warehouse_capacity = float(round(retailer_capacity * rng.uniform(2.0, 3.2), 4))
    retailer_order_up_to = float(round(retailer_capacity * rng.uniform(0.78, 0.94), 4))
    warehouse_order_up_to = float(round(warehouse_capacity * rng.uniform(0.78, 0.94), 4))
    retailer_reorder = float(round(retailer_order_up_to * rng.uniform(0.35, 0.55), 4))
    warehouse_reorder = float(round(warehouse_order_up_to * rng.uniform(0.35, 0.55), 4))
    return {
        "demand": demand.tolist(),
        "base_demand": base_demand,
        "warehouse_capacity": warehouse_capacity,
        "warehouse_initial": float(round(warehouse_order_up_to * 0.72, 4)),
        "warehouse_s": warehouse_reorder,
        "warehouse_S": warehouse_order_up_to,
        "retailer_capacity": retailer_capacity,
        "retailer_initial": float(round(retailer_order_up_to * 0.72, 4)),
        "retailer_s": retailer_reorder,
        "retailer_S": retailer_order_up_to,
        "supplier_to_warehouse_lead": float(rng.integers(1, 4)),
        "warehouse_to_retailer_lead": float(rng.integers(1, 3)),
        "supplier_to_warehouse_cost": float(round(rng.uniform(12.0, 35.0), 4)),
        "warehouse_to_retailer_cost": float(round(rng.uniform(5.0, 18.0), 4)),
        "warehouse_holding_cost": float(round(rng.uniform(0.03, 0.14), 6)),
        "retailer_holding_cost": float(round(rng.uniform(0.08, 0.28), 6)),
        "warehouse_buy_price": float(round(rng.uniform(4.0, 8.0), 4)),
        "retailer_buy_price": float(round(rng.uniform(9.0, 14.0), 4)),
        "retailer_sell_price": float(round(rng.uniform(18.0, 28.0), 4)),
        "report_outage_node": "warehouse" if world_index % 2 == 0 else "retailer",
        "report_outage_offset": int(rng.integers(0, 7)),
    }


def _build_network(
    components: Any,
    spec: dict[str, Any],
    world_seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    try:
        import simpy
    except ImportError as exc:  # pragma: no cover - raised earlier in normal use
        raise RuntimeError("SupplyNetPy conversion requires 'simpy'") from exc

    components.set_seed(world_seed)
    env = simpy.Environment()
    supplier = components.Supplier(
        env,
        ID="supplier",
        name="Generated infinite supplier",
        node_type="infinite_supplier",
        logging=False,
    )
    warehouse = components.InventoryNode(
        env,
        ID="warehouse",
        name="Generated warehouse",
        node_type="warehouse",
        capacity=spec["warehouse_capacity"],
        initial_level=spec["warehouse_initial"],
        inventory_holding_cost=spec["warehouse_holding_cost"],
        replenishment_policy=components.SSReplenishment,
        policy_param={"s": spec["warehouse_s"], "S": spec["warehouse_S"]},
        product_buy_price=spec["warehouse_buy_price"],
        product_sell_price=spec["retailer_buy_price"],
        logging=False,
    )
    retailer = components.InventoryNode(
        env,
        ID="retailer",
        name="Generated retailer",
        node_type="retailer",
        capacity=spec["retailer_capacity"],
        initial_level=spec["retailer_initial"],
        inventory_holding_cost=spec["retailer_holding_cost"],
        replenishment_policy=components.SSReplenishment,
        policy_param={"s": spec["retailer_s"], "S": spec["retailer_S"]},
        product_buy_price=spec["retailer_buy_price"],
        product_sell_price=spec["retailer_sell_price"],
        logging=False,
    )
    supplier_link = components.Link(
        env,
        ID="supplier_to_warehouse",
        source=supplier,
        sink=warehouse,
        cost=spec["supplier_to_warehouse_cost"],
        lead_time=lambda: spec["supplier_to_warehouse_lead"],
    )
    retailer_link = components.Link(
        env,
        ID="warehouse_to_retailer",
        source=warehouse,
        sink=retailer,
        cost=spec["warehouse_to_retailer_cost"],
        lead_time=lambda: spec["warehouse_to_retailer_lead"],
    )
    demand = components.Demand(
        env,
        ID="customer_demand",
        name="Generated customer demand",
        order_arrival_model=lambda: 1.0,
        order_quantity_model=_ValidationSafeSequence(spec["demand"]),
        demand_node=retailer,
        tolerance=0.0,
        consume_available=True,
        logging=False,
    )
    nodes = {
        "supplier": supplier,
        "warehouse": warehouse,
        "retailer": retailer,
        "customer_demand": demand,
    }
    links = {
        "supplier_to_warehouse": supplier_link,
        "warehouse_to_retailer": retailer_link,
    }
    return env, nodes, links


def _topology(spec: dict[str, Any], world_id: str) -> dict[str, Any]:
    return {
        "topology_id": stable_digest("supplynetpy-v2-topology", world_id, length=16),
        "nodes": [
            {
                "node_id": "supplier",
                "node_type": "infinite_supplier",
                "features": {"generated": True, "external_supply": True},
            },
            {
                "node_id": "warehouse",
                "node_type": "warehouse",
                "features": {
                    "capacity": spec["warehouse_capacity"],
                    "holding_cost": spec["warehouse_holding_cost"],
                    "reorder_point": spec["warehouse_s"],
                    "order_up_to": spec["warehouse_S"],
                    "policy": "sS",
                },
            },
            {
                "node_id": "retailer",
                "node_type": "retailer",
                "features": {
                    "capacity": spec["retailer_capacity"],
                    "holding_cost": spec["retailer_holding_cost"],
                    "reorder_point": spec["retailer_s"],
                    "order_up_to": spec["retailer_S"],
                    "policy": "sS",
                },
            },
            {
                "node_id": "customer_demand",
                "node_type": "demand",
                "features": {"generated": True, "arrival_interval": 1.0},
            },
        ],
        "edges": [
            {
                "edge_id": "supplier_to_warehouse",
                "source": "supplier",
                "target": "warehouse",
                "edge_type": "replenishment",
                "features": {
                    "lead_time": spec["supplier_to_warehouse_lead"],
                    "transportation_cost": spec["supplier_to_warehouse_cost"],
                },
            },
            {
                "edge_id": "warehouse_to_retailer",
                "source": "warehouse",
                "target": "retailer",
                "edge_type": "replenishment",
                "features": {
                    "lead_time": spec["warehouse_to_retailer_lead"],
                    "transportation_cost": spec["warehouse_to_retailer_cost"],
                },
            },
            {
                "edge_id": "retailer_to_customer",
                "source": "retailer",
                "target": "customer_demand",
                "edge_type": "demand_fulfillment",
                "features": {"delivery_lead_time": 0.0},
            },
        ],
    }


def _stats_snapshot(nodes: dict[str, Any]) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for node_id, node in nodes.items():
        stats = node.stats
        snapshot[node_id] = {
            "demand_placed_count": float(stats.demand_placed[0]),
            "demand_placed_qty": float(stats.demand_placed[1]),
            "demand_received_qty": float(stats.demand_received[1]),
            "demand_fulfilled_qty": float(stats.demand_fulfilled[1]),
            "fulfillment_received_qty": float(stats.fulfillment_received[1]),
            "shortage_qty": float(stats.shortage[1]),
            "backorder_qty": float(stats.backorder[1]),
            "inventory_carry_cost": float(stats.inventory_carry_cost),
            "inventory_spend_cost": float(stats.inventory_spend_cost),
            "transportation_cost": float(stats.transportation_cost),
            "revenue": float(stats.revenue),
            "profit": float(stats.profit),
        }
    return snapshot


def _delta(
    current: dict[str, dict[str, float]],
    previous: dict[str, dict[str, float]],
    node_id: str,
    field: str,
) -> float:
    return float(current[node_id][field] - previous[node_id][field])


def _state(
    nodes: dict[str, Any],
    links: dict[str, Any],
    stats: dict[str, dict[str, float]],
    previous: dict[str, dict[str, float]],
    *,
    t: int,
) -> dict[str, Any]:
    entities = [
        make_entity(
            "node:supplier",
            "supplier_state",
            node_id="supplier",
            product_id="product:generated",
            features={
                "external_supply": True,
                "node_active": nodes["supplier"].node_status == "active",
            },
        )
    ]
    for node_id in ("warehouse", "retailer"):
        node = nodes[node_id]
        entities.append(
            make_entity(
                f"node:{node_id}",
                "inventory_node_state",
                node_id=node_id,
                product_id="product:generated",
                features={
                    "inventory_level": float(node.inventory.level),
                    "inventory_position": float(node.inventory.on_hand),
                    "capacity": float(node.inventory.capacity),
                    "backorder_quantity": stats[node_id]["backorder_qty"],
                    "pending_orders": float(node.pending_orders),
                    "demand_received_cumulative": stats[node_id]["demand_received_qty"],
                    "demand_fulfilled_cumulative": stats[node_id]["demand_fulfilled_qty"],
                    "shortage_cumulative": stats[node_id]["shortage_qty"],
                    "node_active": node.node_status == "active",
                },
            )
        )
    entities.append(
        make_entity(
            "node:customer_demand",
            "demand_process_state",
            node_id="customer_demand",
            product_id="product:generated",
            features={
                "orders_cumulative": stats["customer_demand"]["demand_placed_count"],
                "demand_cumulative": stats["customer_demand"]["demand_placed_qty"],
                "fulfilled_cumulative": stats["customer_demand"]["fulfillment_received_qty"],
            },
        )
    )

    edge_states = [
        make_edge_state(
            edge_id,
            {
                "active": link.status == "active",
                "lead_time": float(link.lead_time()),
                "transportation_cost": float(link.cost),
                "sink_pending_orders": float(link.sink.pending_orders),
            },
        )
        for edge_id, link in links.items()
    ]
    edge_states.append(
        make_edge_state(
            "retailer_to_customer",
            {
                "active": True,
                "demand_this_period": _delta(
                    stats, previous, "customer_demand", "demand_placed_qty"
                ),
                "fulfilled_this_period": _delta(
                    stats, previous, "customer_demand", "fulfillment_received_qty"
                ),
            },
        )
    )

    events = []
    demand = _delta(stats, previous, "customer_demand", "demand_placed_qty")
    if demand > 0:
        events.append(
            make_event(
                f"demand:{t}",
                "customer_order",
                "node:customer_demand",
                node_id="retailer",
                event_time=float(t),
                report_time=float(t),
                observed=True,
                features={"quantity": demand},
            )
        )
    for node_id, upstream in (("warehouse", "supplier"), ("retailer", "warehouse")):
        ordered = _delta(stats, previous, node_id, "demand_placed_qty")
        if ordered > 0:
            events.append(
                make_event(
                    f"replenishment:{node_id}:{t}",
                    "replenishment_order",
                    f"node:{node_id}",
                    node_id=node_id,
                    event_time=float(t),
                    report_time=float(t),
                    observed=True,
                    features={"quantity": ordered, "upstream": upstream},
                )
            )
        received = _delta(stats, previous, node_id, "fulfillment_received_qty")
        if received > 0:
            events.append(
                make_event(
                    f"receipt:{node_id}:{t}",
                    "shipment_received",
                    f"node:{node_id}",
                    node_id=node_id,
                    event_time=float(t),
                    report_time=float(t),
                    observed=True,
                    features={"quantity": received, "upstream": upstream},
                )
            )

    inventory_total = sum(
        float(nodes[node_id].inventory.level) for node_id in ("warehouse", "retailer")
    )
    return {
        "entities": entities,
        "edges": edge_states,
        "global": {
            "inventory_total": inventory_total,
            "inventory_position_total": sum(
                float(nodes[node_id].inventory.on_hand) for node_id in ("warehouse", "retailer")
            ),
            "backorder_total": sum(
                stats[node_id]["backorder_qty"] for node_id in ("warehouse", "retailer")
            ),
            "demand_this_period": demand,
            "sales_this_period": _delta(
                stats, previous, "customer_demand", "fulfillment_received_qty"
            ),
            "shortage_this_period": _delta(stats, previous, "retailer", "shortage_qty"),
            "replenishment_this_period": sum(
                _delta(stats, previous, node_id, "demand_placed_qty")
                for node_id in ("warehouse", "retailer")
            ),
        },
        "events": events,
    }


def _observation(
    latent: dict[str, Any],
    spec: dict[str, Any],
    *,
    t: int,
) -> dict[str, Any]:
    outage_node = spec["report_outage_node"]
    outage = (t + spec["report_outage_offset"]) % 7 == 3
    entities = []
    statuses = []
    for entity in latent["entities"]:
        node_id = entity["node_id"]
        if outage and node_id == outage_node:
            statuses.append(
                observation_status(
                    f"observation.entities.node:{node_id}",
                    "outage",
                    age_steps=1.0,
                    reason="generated one-period operational reporting outage",
                )
            )
            continue
        entities.append(deepcopy(entity))
        statuses.append(
            observation_status(
                f"observation.entities.{entity['entity_id']}",
                "observed",
                reason="simulator state reported at the period boundary",
            )
        )

    visible_inventory = sum(
        float(entity["features"].get("inventory_level", 0.0)) for entity in entities
    )
    events = [
        deepcopy(event)
        for event in latent["events"]
        if not outage or event["node_id"] != outage_node
    ]
    return {
        "entities": entities,
        "edges": deepcopy(latent["edges"]),
        "global": {
            "visible_inventory_total": visible_inventory,
            "demand_reported_this_period": latent["global"]["demand_this_period"],
            "sales_reported_this_period": latent["global"]["sales_this_period"],
            "reporting_outage_active": outage,
        },
        "events": events,
        "status": statuses,
    }


def _actions(
    stats: dict[str, dict[str, float]],
    previous: dict[str, dict[str, float]],
    *,
    t: int,
) -> list[dict[str, Any]]:
    actions = []
    for node_id, upstream in (("warehouse", "supplier"), ("retailer", "warehouse")):
        quantity = _delta(stats, previous, node_id, "demand_placed_qty")
        if quantity <= 0:
            continue
        actions.append(
            make_action(
                "replenish",
                node_id,
                target_id=upstream,
                quantity=quantity,
                features={
                    "policy": "sS",
                    "policy_semantics": "observational",
                    "period": float(t),
                },
            )
        )
    return actions


def _run_world(
    components: Any,
    spec: dict[str, Any],
    *,
    horizon: int,
    world_seed: int,
) -> list[dict[str, Any]]:
    env, nodes, links = _build_network(components, spec, world_seed)
    previous = _stats_snapshot(nodes)
    steps = []
    for t in range(horizon):
        env.run(until=float(t + 1))
        for node_id in ("warehouse", "retailer"):
            nodes[node_id].stats.update_stats()
        current = _stats_snapshot(nodes)
        latent = _state(nodes, links, current, previous, t=t)
        demand = float(spec["demand"][t])
        steps.append(
            {
                "t": t,
                "clock": float(t),
                "latent_state": latent,
                "observation": _observation(latent, spec, t=t),
                "actions": _actions(current, previous, t=t),
                "outcomes": {
                    "demand": demand,
                    "sales": latent["global"]["sales_this_period"],
                    "shortage": latent["global"]["shortage_this_period"],
                    "inventory": latent["global"]["inventory_total"],
                    "replenishment": latent["global"]["replenishment_this_period"],
                },
                "exogenous_digest": stable_digest(
                    "supplynetpy-v2-generated-demand", world_seed, t, demand
                ),
                "terminal": t == horizon - 1,
            }
        )
        previous = current
    return steps


def _record(
    *,
    world_id: str,
    world_seed: int,
    spec: dict[str, Any],
    steps: list[dict[str, Any]],
    module_relative_path: str,
    verification_mode: str,
) -> dict[str, Any]:
    input_digest = stable_digest("supplynetpy-v2-generated-inputs", spec)
    return json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "record_id": make_record_id("supplynetpy", world_id, None),
            "simulator": {
                "family": "supplynetpy",
                "name": "SupplyNetPy",
                "version": SOURCE_VERSION,
                "source_repository": SOURCE_REPOSITORY,
                "source_commit": SOURCE_COMMIT,
                "license": SOURCE_LICENSE,
            },
            "split": _split_for(world_id),
            "episode": {
                "episode_id": f"{world_id}:state",
                "world_id": world_id,
                "config_id": stable_digest(
                    "supplynetpy-v2-config",
                    {key: value for key, value in spec.items() if key != "demand"},
                    length=16,
                ),
                "seed": int(world_seed),
                "time_unit": "day",
                "horizon": len(steps),
                "exogenous_stream_id": stable_digest(
                    "supplynetpy-v2-demand-stream",
                    world_seed,
                    spec["demand"],
                    length=20,
                ),
                "domain_parameters": {
                    "echelon_count": 2.0,
                    "sku_count": 1.0,
                    "base_demand": spec["base_demand"],
                    "generated_inputs_only": True,
                    "packaged_examples_used": False,
                    "packaged_datasets_used": False,
                    "downstream_lineage_used": False,
                    "generated_input_digest": input_digest,
                },
            },
            "branch": None,
            "objective_eligibility": {
                "state_jepa": True,
                "dynamics_jepa": False,
                "decision_heads": False,
            },
            "topology": _topology(spec, world_id),
            "steps": steps,
            "provenance": {
                "adapter_version": ADAPTER_VERSION,
                "generator": GENERATOR_NAME,
                "causal_semantics": (
                    "Observational trajectory under generated sS policies; recorded "
                    "actions are not interventions or action-effect labels."
                ),
                "observation_semantics": (
                    "Simulator inventory, logistics, demand, and policy events are "
                    "reported at daily boundaries; a deterministic generated reporting "
                    "protocol withholds one node for selected periods."
                ),
                "source_files": [
                    module_relative_path,
                    f"generated://supplynetpy/{world_id}/{input_digest}",
                ],
                "runtime_dependencies": {
                    "supplynetpy_revision": SOURCE_COMMIT,
                    "source_import_mode": "request_source_root",
                    "revision_verification": verification_mode,
                    "generated_inputs_only": True,
                    "simpy": "required by SupplyNetPy",
                },
            },
        }
    )


def generate_supplynetpy_records(
    *,
    source_root: Path,
    worlds: int = 8,
    horizon: int = 32,
    seed: int = 20260730,
    include_state: bool = True,
    include_dynamics: bool = False,
) -> list[dict[str, Any]]:
    """Generate one canonical State-only SupplyNetPy record per world."""
    if worlds < 0:
        raise ValueError("worlds must be non-negative")
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if include_dynamics:
        raise ValueError("SupplyNetPy adapter is State-only")
    if not include_state or worlds == 0:
        return []

    components, module_relative_path, verification_mode = _load_supplynetpy(source_root)
    records = []
    for world_index in range(worlds):
        world_seed = int(np.random.SeedSequence([seed, world_index]).generate_state(1)[0])
        world_id = f"supplynetpy-v2-seed-{seed}-world-{world_index:06d}"
        spec = _world_spec(seed, world_index, horizon)
        steps = _run_world(components, spec, horizon=horizon, world_seed=world_seed)
        records.append(
            _record(
                world_id=world_id,
                world_seed=world_seed,
                spec=spec,
                steps=steps,
                module_relative_path=module_relative_path,
                verification_mode=verification_mode,
            )
        )
    return records


class SupplyNetPyAdapter(SourceAdapter):
    """Convert generated SupplyNetPy worlds from a pinned external checkout."""

    source_id = "supplynetpy"
    source_kind = "simulator"
    supports_state = True
    supports_dynamics = False

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        include_dynamics = self.resolved_dynamics_flag(request)
        if not request.include_state:
            return []
        if request.source_root is None:
            raise ValueError("supplynetpy requires AdapterRequest.source_root")
        worlds = request.worlds
        if request.limit is not None:
            worlds = min(worlds, request.limit)
        return generate_supplynetpy_records(
            source_root=request.source_root,
            worlds=worlds,
            horizon=request.horizon,
            seed=request.seed,
            include_state=request.include_state,
            include_dynamics=include_dynamics,
        )


__all__ = ["ADAPTER_VERSION", "SOURCE_COMMIT", "SupplyNetPyAdapter"]
