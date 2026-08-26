"""Generated-input Stockpyl adapter for canonical pretraining episodes."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
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
    stable_digest,
)

SOURCE_COMMIT = "097d99e2c0cd469465933f1aec3608796177010d"
SOURCE_TAG = "v1.0.2"
SOURCE_REPOSITORY = "https://github.com/LarrySnyder/stockpyl"
SOURCE_LICENSE = "MIT"
ADAPTER_VERSION = "v2-stockpyl-v1"
GENERATOR_NAME = "supplychain_jepa.adapters.stockpyl.generate_stockpyl_records"


def _checkout_commit(root: Path) -> str:
    """Read a checkout revision from Git or its immutable runtime sidecar."""
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
            raise RuntimeError(f"Cannot verify pinned Stockpyl checkout: {root}") from exc
    pin_file = root / ".scjepa-source-commit"
    if pin_file.is_file():
        return pin_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"Cannot verify pinned Stockpyl checkout: {root}")


def _load_stockpyl(source_root: Path) -> tuple[Any, Any, Any]:
    """Load Stockpyl from, and only from, the requested pinned checkout."""
    checkout_root = source_root.resolve()
    checkout_src = checkout_root / "src"
    expected = checkout_src / "stockpyl" / "sim.py"
    if not expected.is_file():
        raise FileNotFoundError(f"Pinned Stockpyl checkout not found: {checkout_root}")

    actual_commit = _checkout_commit(checkout_root)
    if actual_commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"Stockpyl checkout revision mismatch: expected {SOURCE_COMMIT}, found {actual_commit}"
        )

    checkout_text = str(checkout_src)
    if checkout_text in sys.path:
        sys.path.remove(checkout_text)
    sys.path.insert(0, checkout_text)

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from stockpyl.sim import close, initialize, step
        from stockpyl.supply_chain_network import serial_system
    finally:
        sys.dont_write_bytecode = previous

    module_path = Path(sys.modules[initialize.__module__].__file__).resolve()
    if checkout_root not in module_path.parents:
        raise ImportError(f"Stockpyl resolved outside the pinned checkout: {module_path}")
    return serial_system, (initialize, step, close), module_path


def _sum_nested(value: Any) -> float:
    if isinstance(value, dict):
        return float(sum(_sum_nested(item) for item in value.values()))
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(sum(_sum_nested(item) for item in value))
    if value is None:
        return 0.0
    return float(value)


def _split_for(world_id: str) -> str:
    bucket = int(stable_digest("stockpyl-v2-split", world_id, length=8), 16) % 20
    if bucket == 18:
        return "pretrain_dev"
    if bucket == 19:
        return "pretrain_test"
    return "pretrain_train"


def _world_spec(seed: int, world_index: int, horizon: int) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, world_index]))
    num_nodes = int(rng.integers(3, 5))
    mean_demand = float(rng.uniform(8.0, 24.0))
    season_amplitude = float(rng.uniform(0.05, 0.30))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    periods = np.arange(horizon, dtype=float)
    rates = mean_demand * (1.0 + season_amplitude * np.sin((2.0 * np.pi * periods / 7.0) + phase))
    demand = rng.poisson(np.maximum(rates, 0.25)).astype(float)
    if horizon >= 6:
        shock_start = int(rng.integers(horizon // 3, max(horizon // 3 + 1, 2 * horizon // 3)))
        shock_width = min(int(rng.integers(2, 5)), horizon - shock_start)
        demand[shock_start : shock_start + shock_width] += rng.integers(
            max(2, int(mean_demand * 0.4)),
            max(3, int(mean_demand * 1.2)),
            size=shock_width,
        )

    lead_times = rng.integers(1, 4, size=num_nodes).astype(int).tolist()
    holding_costs = rng.uniform(0.2, 1.5, size=num_nodes).round(6).tolist()
    base_stocks = [
        float(round(mean_demand * (1.25 + 0.55 * (num_nodes - index)), 4))
        for index in range(num_nodes)
    ]
    branch_at = max(1, min(horizon - 2, horizon // 3))
    prefix = float(round(mean_demand, 4))
    low = float(round(0.45 * mean_demand, 4))
    high = float(round(1.65 * mean_demand, 4))
    return {
        "num_nodes": num_nodes,
        "mean_demand": mean_demand,
        "season_amplitude": season_amplitude,
        "demand": demand.tolist(),
        "lead_times": lead_times,
        "holding_costs": holding_costs,
        "stockout_cost": float(round(rng.uniform(6.0, 18.0), 6)),
        "base_stocks": base_stocks,
        "branch_at": branch_at,
        "prefix_action": prefix,
        "low_action": low,
        "high_action": high,
    }


def _build_network(serial_system: Any, spec: dict[str, Any]) -> Any:
    return serial_system(
        num_nodes=spec["num_nodes"],
        node_order_in_system=list(range(spec["num_nodes"])),
        local_holding_cost=spec["holding_costs"],
        stockout_cost=spec["stockout_cost"],
        shipment_lead_time=spec["lead_times"],
        demand_type="D",
        demand_list=spec["demand"],
        policy_type="BS",
        base_stock_level=spec["base_stocks"],
        initial_inventory_level=spec["base_stocks"],
    )


def _topology(network: Any, world_id: str) -> dict[str, Any]:
    nodes = []
    for node in sorted(network.nodes, key=lambda item: item.index):
        nodes.append(
            {
                "node_id": f"node:{node.index}",
                "node_type": (
                    "supplier"
                    if not node.predecessors()
                    else "retailer"
                    if not node.successors()
                    else "inventory_echelon"
                ),
                "features": {
                    "local_holding_cost": float(node.local_holding_cost or 0.0),
                    "stockout_cost": float(node.stockout_cost or 0.0),
                    "shipment_lead_time": float(node.shipment_lead_time or 0.0),
                    "base_stock_level": float(node.inventory_policy.base_stock_level or 0.0),
                    "has_external_supplier": not bool(node.predecessors()),
                    "serves_external_customer": not bool(node.successors()),
                },
            }
        )

    edges = []
    ordered_nodes = sorted(network.nodes, key=lambda item: item.index)
    for upstream, downstream in zip(ordered_nodes[:-1], ordered_nodes[1:], strict=True):
        edges.append(
            {
                "edge_id": f"edge:{upstream.index}->{downstream.index}",
                "source": f"node:{upstream.index}",
                "target": f"node:{downstream.index}",
                "edge_type": "replenishment",
                "features": {
                    "downstream_shipment_lead_time": float(downstream.shipment_lead_time or 0.0)
                },
            }
        )
    return {
        "topology_id": f"stockpyl:{world_id}:{stable_digest(nodes, edges, length=12)}",
        "nodes": nodes,
        "edges": edges,
    }


def _requested_actions(spec: dict[str, Any], branch_id: str | None, t: int) -> dict[int, float]:
    if branch_id is None or t < spec["branch_at"]:
        quantity = spec["prefix_action"]
    elif branch_id == "low":
        quantity = spec["low_action"]
    elif branch_id == "high":
        quantity = spec["high_action"]
    else:
        raise ValueError(f"Unsupported Stockpyl branch: {branch_id}")
    return {index: float(quantity) for index in range(spec["num_nodes"])}


def _override(actions: dict[int, float]) -> dict[int, dict[None, dict[None, float]]]:
    return {node_index: {None: {None: quantity}} for node_index, quantity in actions.items()}


def _executed_quantity(node: Any, t: int) -> float:
    return _sum_nested(node.state_vars[t].order_quantity)


def _state(network: Any, t: int) -> dict[str, Any]:
    entities = []
    events = []
    for node in sorted(network.nodes, key=lambda item: item.index):
        state = node.state_vars[t]
        inventory = _sum_nested(state.inventory_level)
        inbound_orders = _sum_nested(state.inbound_order)
        outbound_shipments = _sum_nested(state.outbound_shipment)
        inbound_shipments = _sum_nested(state.inbound_shipment)
        pipeline = _sum_nested(state.inbound_shipment_pipeline)
        backlog = _sum_nested(state.backorders_by_successor)
        order_quantity = _sum_nested(state.order_quantity)
        entities.append(
            make_entity(
                f"node:{node.index}",
                "inventory_node_state",
                node_id=f"node:{node.index}",
                product_id="product:aggregate",
                features={
                    "inventory_level": inventory,
                    "inventory_on_hand": max(inventory, 0.0),
                    "inventory_backlog": max(-inventory, 0.0),
                    "inbound_orders": inbound_orders,
                    "outbound_shipments": outbound_shipments,
                    "inbound_shipments": inbound_shipments,
                    "inbound_pipeline": pipeline,
                    "backorders": backlog,
                    "replenishment_order": order_quantity,
                    "holding_cost": float(state.holding_cost_incurred),
                    "stockout_cost": float(state.stockout_cost_incurred),
                    "total_cost": float(state.total_cost_incurred),
                },
            )
        )
        events.append(
            make_event(
                f"order:{t}:node:{node.index}",
                "replenishment_order",
                f"node:{node.index}",
                node_id=f"node:{node.index}",
                event_time=float(t),
                report_time=float(t),
                observed=True,
                features={"quantity": order_quantity},
            )
        )

    edge_states = []
    ordered_nodes = sorted(network.nodes, key=lambda item: item.index)
    for upstream, downstream in zip(ordered_nodes[:-1], ordered_nodes[1:], strict=True):
        downstream_state = downstream.state_vars[t]
        edge_states.append(
            make_edge_state(
                f"edge:{upstream.index}->{downstream.index}",
                {
                    "order_quantity": _sum_nested(
                        downstream_state.order_quantity.get(upstream.index, {})
                    ),
                    "shipment_received": _sum_nested(
                        downstream_state.inbound_shipment.get(upstream.index, {})
                    ),
                    "shipment_pipeline": _sum_nested(
                        downstream_state.inbound_shipment_pipeline.get(upstream.index, {})
                    ),
                },
            )
        )

    sink = ordered_nodes[-1]
    sink_state = sink.state_vars[t]
    demand = _sum_nested(sink_state.inbound_order.get(None, {}))
    sales = _sum_nested(sink_state.outbound_shipment.get(None, {}))
    events.append(
        make_event(
            f"demand:{t}:node:{sink.index}",
            "external_customer_demand",
            f"node:{sink.index}",
            node_id=f"node:{sink.index}",
            event_time=float(t),
            report_time=float(t),
            observed=True,
            features={"demand": demand, "sales": sales},
        )
    )
    return {
        "entities": entities,
        "edges": edge_states,
        "global": {
            "period": float(t),
            "inventory_total": float(sum(item["features"]["inventory_level"] for item in entities)),
            "pipeline_total": float(sum(item["features"]["inbound_pipeline"] for item in entities)),
            "backlog_total": float(sum(item["features"]["backorders"] for item in entities)),
            "demand_total": demand,
            "sales_total": sales,
            "period_cost_total": float(sum(item["features"]["total_cost"] for item in entities)),
        },
        "events": events,
    }


def _run_rollout(
    serial_system: Any,
    sim_api: tuple[Any, Any, Any],
    spec: dict[str, Any],
    *,
    branch_id: str | None,
    simulator_seed: int,
) -> tuple[list[dict[str, Any]], float]:
    initialize, step, close = sim_api
    network = _build_network(serial_system, spec)
    initialize(network, len(spec["demand"]), rand_seed=simulator_seed)
    steps = []
    for t, demand in enumerate(spec["demand"]):
        requested = _requested_actions(spec, branch_id, t)
        step(network, order_quantity_override=_override(requested), consistency_checks="E")

        actions = []
        for node in sorted(network.nodes, key=lambda item: item.index):
            executed = _executed_quantity(node, t)
            expected = requested[node.index]
            if not np.isclose(executed, expected, rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    "Stockpyl did not execute the requested action exactly: "
                    f"t={t}, node={node.index}, requested={expected}, "
                    f"executed={executed}"
                )
            predecessor = node.predecessors()
            target_id = f"node:{predecessor[0].index}" if predecessor else "external_supplier"
            actions.append(
                make_action(
                    "replenishment_order",
                    f"node:{node.index}",
                    target_id=target_id,
                    quantity=executed,
                    features={
                        "requested_quantity": expected,
                        "executed_quantity": executed,
                        "execution_validated": True,
                        "interventional": branch_id is not None,
                    },
                )
            )

        latent = _state(network, t)
        if not np.isclose(latent["global"]["demand_total"], demand, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                "Stockpyl did not execute the generated demand stream exactly: "
                f"t={t}, generated={demand}, "
                f"executed={latent['global']['demand_total']}"
            )
        observation = deepcopy(latent)
        observation["status"] = []
        steps.append(
            {
                "t": t,
                "clock": float(t),
                "latent_state": latent,
                "observation": observation,
                "actions": actions,
                "outcomes": {
                    "demand": float(demand),
                    "sales": float(latent["global"]["sales_total"]),
                    "backlog": float(latent["global"]["backlog_total"]),
                    "inventory": float(latent["global"]["inventory_total"]),
                    "period_cost": float(latent["global"]["period_cost_total"]),
                },
                "exogenous_digest": stable_digest(
                    "stockpyl-generated-demand", simulator_seed, t, float(demand)
                ),
                "terminal": t == len(spec["demand"]) - 1,
            }
        )
    return steps, float(close(network))


def _record(
    *,
    world_id: str,
    split: str,
    seed: int,
    spec: dict[str, Any],
    topology: dict[str, Any],
    branch_id: str | None,
    steps: list[dict[str, Any]],
    total_cost: float,
    module_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    is_dynamics = branch_id is not None
    branch_group_id = f"{world_id}:branch-at-{spec['branch_at']}"
    demand_stream_id = stable_digest(
        "stockpyl-v2-generated-demand", seed, spec["demand"], length=20
    )
    return json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "record_id": make_record_id("stockpyl", world_id, branch_id),
            "simulator": {
                "family": "stockpyl",
                "name": "Stockpyl",
                "version": SOURCE_TAG,
                "source_repository": SOURCE_REPOSITORY,
                "source_commit": SOURCE_COMMIT,
                "license": SOURCE_LICENSE,
            },
            "split": split,
            "episode": {
                "episode_id": f"{world_id}:{branch_id or 'state'}",
                "world_id": world_id,
                "config_id": stable_digest(
                    "stockpyl-v2-config",
                    spec["num_nodes"],
                    spec["lead_times"],
                    spec["holding_costs"],
                    spec["base_stocks"],
                    length=16,
                ),
                "seed": int(seed),
                "time_unit": "period",
                "horizon": len(steps),
                "exogenous_stream_id": demand_stream_id,
                "domain_parameters": {
                    "num_nodes": float(spec["num_nodes"]),
                    "mean_demand": float(spec["mean_demand"]),
                    "season_amplitude": float(spec["season_amplitude"]),
                    "branch_at": float(spec["branch_at"]),
                    "generated_inputs_only": True,
                    "downstream_lineage_used": False,
                    "total_cost": float(total_cost),
                },
            },
            "branch": (
                {
                    "branch_group_id": branch_group_id,
                    "branch_id": branch_id,
                    "branch_at": float(spec["branch_at"]),
                    "common_random_numbers": True,
                }
                if is_dynamics
                else None
            ),
            "objective_eligibility": {
                "state_jepa": True,
                "dynamics_jepa": is_dynamics,
                "decision_heads": is_dynamics,
            },
            "topology": topology,
            "steps": steps,
            "provenance": {
                "adapter_version": ADAPTER_VERSION,
                "generator": GENERATOR_NAME,
                "causal_semantics": (
                    "Observational generated rollout; not eligible for action effects."
                    if not is_dynamics
                    else "Exact low/high intervention pair with identical generated "
                    "demand stream, simulator seed, world parameters, and action prefix."
                ),
                "observation_semantics": (
                    "Complete simulator state is exposed as the observation; structured "
                    "masking is applied later by the JEPA corpus/training pipeline."
                ),
                "source_files": [str(module_path.relative_to(source_root.resolve()))],
                "runtime_dependencies": {
                    "stockpyl_revision": SOURCE_COMMIT,
                    "stockpyl_tag": SOURCE_TAG,
                    "source_import_mode": "pinned_checkout",
                },
            },
        }
    )


def _validate_pair(low: dict[str, Any], high: dict[str, Any]) -> None:
    if low["episode"]["exogenous_stream_id"] != high["episode"]["exogenous_stream_id"]:
        raise RuntimeError("Dynamics pair does not share one exogenous stream")
    branch_at = int(low["branch"]["branch_at"])
    if low["branch"]["branch_group_id"] != high["branch"]["branch_group_id"]:
        raise RuntimeError("Dynamics records do not share one branch group")
    for low_step, high_step in zip(low["steps"], high["steps"], strict=True):
        if low_step["exogenous_digest"] != high_step["exogenous_digest"]:
            raise RuntimeError("Dynamics pair has mismatched exogenous demand")
        low_actions = [action["quantity"] for action in low_step["actions"]]
        high_actions = [action["quantity"] for action in high_step["actions"]]
        if low_step["t"] < branch_at and low_actions != high_actions:
            raise RuntimeError("Dynamics pair action prefixes are not identical")
        if low_step["t"] < branch_at and low_step["latent_state"] != high_step["latent_state"]:
            raise RuntimeError("Dynamics pair state prefixes are not identical")
        if low_step["t"] >= branch_at and low_actions == high_actions:
            raise RuntimeError("Dynamics pair does not intervene after branch time")

    low_future = [step["latent_state"]["global"] for step in low["steps"][branch_at:]]
    high_future = [step["latent_state"]["global"] for step in high["steps"][branch_at:]]
    if low_future == high_future:
        raise RuntimeError("Dynamics actions did not produce divergent future states")


def generate_stockpyl_records(
    source_root: Path,
    worlds: int = 8,
    horizon: int = 32,
    seed: int = 20260730,
    include_state: bool = True,
    include_dynamics: bool = True,
) -> list[dict[str, Any]]:
    """Generate canonical Stockpyl State rollouts and CRN Dynamics pairs."""
    if worlds < 1:
        raise ValueError("worlds must be at least 1")
    if horizon < 4:
        raise ValueError("horizon must be at least 4")
    if not include_state and not include_dynamics:
        return []

    source_root = source_root.resolve()
    serial_system, sim_api, module_path = _load_stockpyl(source_root)
    records: list[dict[str, Any]] = []
    for world_index in range(worlds):
        world_seed = int(np.random.SeedSequence([seed, world_index]).generate_state(1)[0])
        world_id = f"stockpyl-v2-seed-{seed}-world-{world_index:06d}"
        split = _split_for(world_id)
        spec = _world_spec(seed, world_index, horizon)
        topology = _topology(_build_network(serial_system, spec), world_id)

        if include_state:
            steps, total_cost = _run_rollout(
                serial_system,
                sim_api,
                spec,
                branch_id=None,
                simulator_seed=world_seed,
            )
            records.append(
                _record(
                    world_id=world_id,
                    split=split,
                    seed=world_seed,
                    spec=spec,
                    topology=topology,
                    branch_id=None,
                    steps=steps,
                    total_cost=total_cost,
                    module_path=module_path,
                    source_root=source_root,
                )
            )

        if include_dynamics:
            pair = []
            for branch_id in ("low", "high"):
                steps, total_cost = _run_rollout(
                    serial_system,
                    sim_api,
                    spec,
                    branch_id=branch_id,
                    simulator_seed=world_seed,
                )
                pair.append(
                    _record(
                        world_id=world_id,
                        split=split,
                        seed=world_seed,
                        spec=spec,
                        topology=topology,
                        branch_id=branch_id,
                        steps=steps,
                        total_cost=total_cost,
                        module_path=module_path,
                        source_root=source_root,
                    )
                )
            _validate_pair(pair[0], pair[1])
            records.extend(pair)
    return records


class StockpylAdapter(SourceAdapter):
    """Convert a pinned Stockpyl checkout into generated canonical episodes."""

    source_id = "stockpyl"
    source_kind = "simulator"
    supports_state = True
    supports_dynamics = True

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        include_dynamics = self.resolved_dynamics_flag(request)
        worlds = request.worlds
        if request.limit is not None:
            worlds = min(worlds, request.limit)
        if worlds == 0 or (not request.include_state and not include_dynamics):
            return []
        if request.source_root is None:
            raise ValueError("stockpyl requires AdapterRequest.source_root")
        return generate_stockpyl_records(
            Path(request.source_root),
            worlds=worlds,
            horizon=request.horizon,
            seed=request.seed,
            include_state=request.include_state,
            include_dynamics=include_dynamics,
        )


__all__ = ["StockpylAdapter", "generate_stockpyl_records"]
