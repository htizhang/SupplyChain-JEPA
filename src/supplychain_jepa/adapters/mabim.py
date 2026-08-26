"""Generated-input MABIM adapter for canonical pretraining episodes."""

from __future__ import annotations

import importlib
import math
import random
import subprocess
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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

SOURCE_COMMIT = "e667565615461ecd4102a60ad1ecd6b772e357d6"
SOURCE_VERSION = "1.0"
SOURCE_REPOSITORY = "https://github.com/VictorYXL/ReplenishmentEnv"
SOURCE_LICENSE = "Apache-2.0"
ADAPTER_VERSION = "v2-mabim-code-only-v1"
DEFAULT_WORLDS = 1
DEFAULT_HORIZON = 24
DEFAULT_SEED = 0
LOOKBACK = 7
WAREHOUSES = ("warehouse_0", "warehouse_1")
SKUS = ("sku_0", "sku_1")
_STATE_ITEMS = (
    "selling_price",
    "procurement_cost",
    "demand",
    "sale",
    "vlt",
    "volume",
    "unit_order_cost",
    "basic_holding_cost",
    "in_stock",
    "replenish",
    "excess",
    "overflow_cost_ratio",
    "in_transit",
    "arrived",
    "accepted",
    "backlog_ratio",
)


def _checkout_commit(root: Path) -> str:
    """Read a checkout revision from Git or its immutable runtime sidecar."""
    if (root / ".git").exists():
        try:
            return subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Unable to verify pinned MABIM checkout: {root}") from exc
    pin_file = root / ".scjepa-source-commit"
    if pin_file.is_file():
        return pin_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"Unable to verify pinned MABIM checkout: {root}")


def _checkout_root(source_root: Path) -> Path:
    root = source_root.resolve()
    required = root / "ReplenishmentEnv" / "env" / "replenishment_env.py"
    if not required.is_file():
        raise FileNotFoundError(f"Pinned MABIM checkout not found: {root}")
    commit = _checkout_commit(root)
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"MABIM checkout is {commit}; expected pinned commit {SOURCE_COMMIT}")
    return root


def _install_gym_compatibility() -> None:
    """Install the tiny subset of legacy Gym required by MABIM, if absent."""
    try:
        importlib.import_module("gym")
        return
    except ImportError:
        pass

    gym = types.ModuleType("gym")
    spaces = types.ModuleType("gym.spaces")

    class Env:
        pass

    class Wrapper:
        def __init__(self, env: Any) -> None:
            self.env = env

        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)

    class Box:
        def __init__(
            self,
            low: Any,
            high: Any,
            shape: tuple[int, ...] | None = None,
            dtype: Any = np.float32,
        ) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class Discrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

    gym.Env = Env
    gym.Wrapper = Wrapper
    gym.spaces = spaces
    spaces.Box = Box
    spaces.Discrete = Discrete
    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = spaces


def _install_pyecharts_compatibility() -> None:
    """Provide import-only visualizer stubs when optional pyecharts is absent."""
    try:
        importlib.import_module("pyecharts")
        return
    except ImportError:
        pass

    package = types.ModuleType("pyecharts")
    package.__path__ = []
    options = types.ModuleType("pyecharts.options")
    charts = types.ModuleType("pyecharts.charts")

    class _NoOp:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def __getattr__(self, _: str) -> Any:
            return lambda *args, **kwargs: self

        def render(self, path: str | None = None) -> str:
            return path or ""

    options.__getattr__ = lambda _: _NoOp
    charts.Line = _NoOp
    charts.Timeline = _NoOp
    charts.Grid = _NoOp
    package.options = options
    package.charts = charts
    sys.modules["pyecharts"] = package
    sys.modules["pyecharts.options"] = options
    sys.modules["pyecharts.charts"] = charts


def _load_environment_class(source_root: Path) -> type:
    """Load ReplenishmentEnv directly from the requested pinned checkout."""
    root = _checkout_root(source_root)
    package_root = root / "ReplenishmentEnv"
    _install_gym_compatibility()
    _install_pyecharts_compatibility()

    existing = sys.modules.get("ReplenishmentEnv")
    if existing is None:
        package = types.ModuleType("ReplenishmentEnv")
        package.__path__ = [str(package_root)]
        package.__package__ = "ReplenishmentEnv"
        sys.modules["ReplenishmentEnv"] = package
    else:
        locations = [Path(item).resolve() for item in getattr(existing, "__path__", [])]
        module_file = getattr(existing, "__file__", None)
        if module_file is not None:
            locations.append(Path(module_file).resolve())
        if not any(
            location == package_root or package_root in location.parents for location in locations
        ):
            raise ImportError("ReplenishmentEnv is already loaded outside the pinned checkout")

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module("ReplenishmentEnv.env.replenishment_env")
    finally:
        sys.dont_write_bytecode = previous
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise ImportError(f"MABIM resolved outside the pinned checkout: {module_path}")
    return module.ReplenishmentEnv


def _write_matrix(path: Path, dates: pd.DatetimeIndex, values: np.ndarray) -> None:
    frame = pd.DataFrame(values, index=dates, columns=SKUS)
    frame.index = frame.index.strftime("%Y/%m/%d")
    frame.index.name = "Date"
    frame.to_csv(path)


def _sample_inputs(world_seed: int, horizon: int) -> dict[str, Any]:
    """Sample mutually independent demand, market, lead-time, and policy streams."""
    streams = np.random.SeedSequence(world_seed).spawn(5)
    demand_rng = np.random.default_rng(streams[0])
    market_rng = np.random.default_rng(streams[1])
    lead_rng = np.random.default_rng(streams[2])
    state_policy_rng = np.random.default_rng(streams[3])
    branch_policy_rng = np.random.default_rng(streams[4])

    periods = horizon + LOOKBACK
    t = np.arange(periods, dtype=float)[:, None]
    base = demand_rng.uniform(8.0, 18.0, size=(1, len(SKUS)))
    weekly = 1.0 + 0.22 * np.sin(2.0 * np.pi * t / 7.0 + demand_rng.uniform(0, 2))
    demand = demand_rng.poisson(np.maximum(1.0, base * weekly)).astype(float)
    shock_start = LOOKBACK + max(2, horizon // 2)
    shock_width = min(3, max(0, periods - shock_start))
    if shock_width:
        demand[shock_start : shock_start + shock_width] += demand_rng.integers(
            5, 13, size=(shock_width, len(SKUS))
        )

    supplier_cost = market_rng.uniform(3.0, 6.0, size=(1, len(SKUS)))
    transfer_price = supplier_cost + market_rng.uniform(1.0, 2.0, size=(1, len(SKUS)))
    retail_price = transfer_price + market_rng.uniform(2.0, 4.0, size=(1, len(SKUS)))
    drift = np.cumsum(market_rng.normal(0.0, 0.012, size=(periods, len(SKUS))), axis=0)
    supplier_cost_series = np.maximum(0.5, supplier_cost * (1.0 + drift))
    transfer_price_series = np.maximum(
        supplier_cost_series + 0.5, transfer_price * (1.0 + 0.5 * drift)
    )
    retail_price_series = np.maximum(
        transfer_price_series + 1.0, retail_price * (1.0 + 0.35 * drift)
    )

    lead_times = lead_rng.integers(1, 4, size=(len(WAREHOUSES), len(SKUS)))
    initial_stock = np.ceil(np.vstack((base * (lead_times[0] + 3), base * (lead_times[1] + 3))))
    state_actions = np.maximum(
        0.0,
        base[None, :, :]
        * state_policy_rng.uniform(0.65, 1.45, size=(horizon, len(WAREHOUSES), len(SKUS))),
    )
    shared_actions = np.maximum(
        0.0,
        base[None, :, :]
        * branch_policy_rng.uniform(0.8, 1.2, size=(horizon, len(WAREHOUSES), len(SKUS))),
    )
    state_actions = np.rint(state_actions)
    shared_actions = np.rint(shared_actions)

    exogenous_digest = stable_digest(
        "mabim-generated-exogenous-v1",
        world_seed,
        demand.tolist(),
        supplier_cost_series.tolist(),
        transfer_price_series.tolist(),
        retail_price_series.tolist(),
        lead_times.tolist(),
        initial_stock.tolist(),
    )
    action_stream_digest = stable_digest(
        "mabim-generated-policy-streams-v1",
        world_seed,
        state_actions.tolist(),
        shared_actions.tolist(),
    )
    input_digest = stable_digest(
        "mabim-generated-inputs-v1", exogenous_digest, action_stream_digest
    )
    return {
        "demand": demand,
        "supplier_cost": supplier_cost_series,
        "transfer_price": transfer_price_series,
        "retail_price": retail_price_series,
        "lead_times": lead_times,
        "initial_stock": initial_stock,
        "state_actions": state_actions,
        "shared_actions": shared_actions,
        "exogenous_digest": exogenous_digest,
        "action_stream_digest": action_stream_digest,
        "input_digest": input_digest,
    }


def _write_generated_inputs(
    directory: Path,
    sampled: dict[str, Any],
    horizon: int,
) -> tuple[Path, pd.DatetimeIndex]:
    """Materialize every MABIM input in a temporary, generated-only directory."""
    start = pd.Timestamp("2024-01-08")
    dates = pd.date_range(start - pd.Timedelta(days=LOOKBACK), periods=horizon + LOOKBACK)
    sku_list_path = directory / "sku_list.csv"
    pd.DataFrame({"SKU": SKUS}).to_csv(sku_list_path, index=False)

    for warehouse_index, warehouse in enumerate(WAREHOUSES):
        static_path = directory / f"{warehouse}_static.csv"
        pd.DataFrame(
            {
                "SKU": SKUS,
                "init_stock": sampled["initial_stock"][warehouse_index],
                "vlt": sampled["lead_times"][warehouse_index],
            }
        ).to_csv(static_path, index=False)

    _write_matrix(directory / "head_procurement.csv", dates, sampled["supplier_cost"])
    _write_matrix(directory / "head_selling.csv", dates, sampled["transfer_price"])
    _write_matrix(directory / "tail_procurement.csv", dates, sampled["transfer_price"])
    _write_matrix(directory / "tail_selling.csv", dates, sampled["retail_price"])
    _write_matrix(directory / "tail_demand.csv", dates, sampled["demand"])

    def warehouse_config(
        index: int,
        *,
        upstream: str,
        downstream: str,
        procurement: str,
        selling: str,
        include_demand: bool,
    ) -> dict[str, Any]:
        dynamic_data = [
            {"name": "procurement_cost", "file": str(directory / procurement)},
            {"name": "selling_price", "file": str(directory / selling)},
        ]
        if include_demand:
            dynamic_data.append({"name": "demand", "file": str(directory / "tail_demand.csv")})
        return {
            "name": WAREHOUSES[index],
            "initial_balance": 0,
            "capacity": 600.0,
            "unit_storage_cost": 0.002,
            "accept_sku": "equal_accept",
            "upstream": upstream,
            "downstream": downstream,
            "sku": {
                "dynamic_data": dynamic_data,
                "static_data": str(directory / f"{WAREHOUSES[index]}_static.csv"),
                "shared_data": {
                    "unit_order_cost": 2.0,
                    "volume": 1.0,
                    "basic_holding_cost": 0.001,
                    "backlog_ratio": 0.2,
                    "overflow_cost_ratio": 0.5,
                },
            },
        }

    config = {
        "env": {
            "mode": [
                {
                    "name": "train",
                    "start_date": start.strftime("%Y/%m/%d"),
                    "end_date": (start + pd.Timedelta(days=horizon - 1)).strftime("%Y/%m/%d"),
                    "random_interception": False,
                }
            ],
            "horizon": horizon,
            "lookback_len": LOOKBACK,
            "integerization_sku": True,
            "warmup": "replenish_by_last_demand",
            "sku_list": str(sku_list_path),
        },
        "warehouse": [
            warehouse_config(
                0,
                upstream="super_vendor",
                downstream=WAREHOUSES[1],
                procurement="head_procurement.csv",
                selling="head_selling.csv",
                include_demand=False,
            ),
            warehouse_config(
                1,
                upstream=WAREHOUSES[0],
                downstream="consumer",
                procurement="tail_procurement.csv",
                selling="tail_selling.csv",
                include_demand=True,
            ),
        ],
        "profit_function": "reward2",
        "reward_function": "reward2",
        "action": {"mode": "continuous"},
        "output_state": {
            "current_state": list(_STATE_ITEMS),
            "lookback_state": ["demand"],
        },
        "visualization": {"state": ["demand", "sale", "in_stock", "in_transit"]},
    }
    config_path = directory / "generated_config.yml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path, dates


def _number(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"MABIM emitted non-finite {label}: {number!r}")
    return number


def _topology(sampled: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    for index, warehouse in enumerate(WAREHOUSES):
        nodes.append(
            {
                "node_id": warehouse,
                "node_type": "upstream_warehouse" if index == 0 else "retail_warehouse",
                "features": {
                    "capacity": 600.0,
                    "unit_storage_cost": 0.002,
                    "echelon": float(index),
                    "sku_count": float(len(SKUS)),
                    "mean_lead_time": float(np.mean(sampled["lead_times"][index])),
                },
            }
        )
    return {
        "topology_id": f"mabim:2w2sku:{sampled['exogenous_digest'][:12]}",
        "nodes": nodes,
        "edges": [
            {
                "edge_id": f"{WAREHOUSES[0]}->{WAREHOUSES[1]}",
                "source": WAREHOUSES[0],
                "target": WAREHOUSES[1],
                "edge_type": "replenishment_flow",
                "features": {"sku_count": float(len(SKUS))},
            }
        ],
    }


def _state_at(env: Any, t: int) -> dict[str, Any]:
    entities = []
    for warehouse in WAREHOUSES:
        for sku in SKUS:
            features = {
                item: _number(
                    env.agent_states[warehouse, item, t, sku],
                    label=f"{warehouse}.{sku}.{item}@{t}",
                )
                for item in _STATE_ITEMS
            }
            features["stockout"] = bool(features["sale"] + 1e-9 < features["demand"])
            entities.append(
                make_entity(
                    f"{warehouse}:{sku}",
                    "sku_warehouse_state",
                    node_id=warehouse,
                    product_id=sku,
                    features=features,
                )
            )

    head_shipments = sum(
        _number(env.agent_states[WAREHOUSES[0], "sale", t, sku], label="shipment") for sku in SKUS
    )
    tail_orders = sum(
        _number(env.agent_states[WAREHOUSES[1], "replenish", t, sku], label="tail_order")
        for sku in SKUS
    )
    edge_states = [
        make_edge_state(
            f"{WAREHOUSES[0]}->{WAREHOUSES[1]}",
            {
                "order_requested": tail_orders,
                "shipment": head_shipments,
                "unfilled_order": max(0.0, tail_orders - head_shipments),
            },
        )
    ]
    events = []
    for warehouse in WAREHOUSES:
        upstream = "super_vendor" if warehouse == WAREHOUSES[0] else WAREHOUSES[0]
        for sku in SKUS:
            quantity = _number(
                env.agent_states[warehouse, "replenish", t, sku],
                label="replenish",
            )
            events.append(
                make_event(
                    f"order:{t}:{warehouse}:{sku}",
                    "replenishment_order",
                    f"{warehouse}:{sku}",
                    node_id=warehouse,
                    event_time=float(t),
                    report_time=float(t),
                    observed=True,
                    features={"quantity": quantity, "upstream": upstream},
                )
            )

    demand_total = sum(
        _number(env.agent_states[WAREHOUSES[1], "demand", t, sku], label="demand") for sku in SKUS
    )
    sales_total = sum(
        _number(env.agent_states[WAREHOUSES[1], "sale", t, sku], label="sale") for sku in SKUS
    )
    return {
        "entities": entities,
        "edges": edge_states,
        "global": {
            "inventory_total": sum(entity["features"]["in_stock"] for entity in entities),
            "in_transit_total": sum(entity["features"]["in_transit"] for entity in entities),
            "demand_total": demand_total,
            "sales_total": sales_total,
            "fill_rate": sales_total / demand_total if demand_total > 0 else 1.0,
            "balance_total": _number(np.sum(env.balance), label="balance"),
        },
        "events": events,
    }


def _observation_from_state(latent: dict[str, Any]) -> dict[str, Any]:
    entities = []
    statuses = []
    for entity in latent["entities"]:
        truth = entity["features"]
        is_tail = entity["node_id"] == WAREHOUSES[1]
        censored = bool(is_tail and truth["stockout"])
        demand_reported = truth["sale"] if censored else truth["demand"]
        entities.append(
            make_entity(
                entity["entity_id"],
                "sku_warehouse_report",
                node_id=entity["node_id"],
                product_id=entity["product_id"],
                features={
                    "inventory_reported": truth["in_stock"],
                    "in_transit_reported": truth["in_transit"],
                    "sales_observed": truth["sale"],
                    "demand_reported": demand_reported,
                    "demand_censored": censored,
                    "selling_price": truth["selling_price"],
                    "procurement_cost": truth["procurement_cost"],
                    "lead_time": truth["vlt"],
                    "report_available": True,
                },
            )
        )
        statuses.append(
            observation_status(
                f"observation.entities.{entity['entity_id']}.demand_reported",
                "censored" if censored else "observed",
                reason=(
                    "consumer demand is sales-censored during a stockout"
                    if censored
                    else "demand is fully observed"
                ),
            )
        )
    events = [deepcopy(event) for event in latent["events"]]
    return {
        "entities": entities,
        "edges": deepcopy(latent["edges"]),
        "global": {
            "inventory_total_reported": latent["global"]["inventory_total"],
            "in_transit_total_reported": latent["global"]["in_transit_total"],
            "sales_total_observed": latent["global"]["sales_total"],
            "balance_total_reported": latent["global"]["balance_total"],
        },
        "events": events,
        "status": statuses,
    }


def _actions_for_step(
    actions: np.ndarray,
    *,
    t: int,
    policy_id: str,
) -> list[dict[str, Any]]:
    result = []
    for warehouse_index, warehouse in enumerate(WAREHOUSES):
        target = "super_vendor" if warehouse_index == 0 else WAREHOUSES[0]
        for sku_index, sku in enumerate(SKUS):
            result.append(
                make_action(
                    "replenish",
                    warehouse,
                    target_id=target,
                    quantity=float(actions[warehouse_index, sku_index]),
                    features={"sku_id": sku, "policy_id": policy_id, "period": float(t)},
                )
            )
    return result


def _exogenous_digest(sampled: dict[str, Any], t: int) -> str:
    row = t + LOOKBACK
    return stable_digest(
        "mabim-exogenous-period-v1",
        sampled["exogenous_digest"],
        t,
        sampled["demand"][row].tolist(),
        sampled["supplier_cost"][row].tolist(),
        sampled["transfer_price"][row].tolist(),
        sampled["retail_price"][row].tolist(),
        sampled["lead_times"].tolist(),
    )


def _run_policy(
    environment_class: type,
    config_path: Path,
    sampled: dict[str, Any],
    *,
    actions: np.ndarray,
    policy_id: str,
    horizon: int,
    world_seed: int,
) -> list[dict[str, Any]]:
    random.seed(world_seed)
    np.random.seed(world_seed % (2**32))
    env = environment_class(str(config_path), mode="train")
    env.reset()
    steps = []
    for t in range(horizon):
        selected = np.asarray(actions[t], dtype=float)
        _, rewards, done, info = env.step(selected)
        latent = _state_at(env, t)
        reward_info = info["reward_info"]
        demand_total = latent["global"]["demand_total"]
        sales_total = latent["global"]["sales_total"]
        steps.append(
            {
                "t": t,
                "clock": float(t),
                "latent_state": latent,
                "observation": _observation_from_state(latent),
                "actions": _actions_for_step(selected, t=t, policy_id=policy_id),
                "outcomes": {
                    "reward": _number(np.sum(rewards), label="reward"),
                    "profit": _number(np.sum(reward_info.get("profit", 0.0)), label="profit"),
                    "holding_cost": _number(
                        np.sum(reward_info["holding_cost"]), label="holding_cost"
                    ),
                    "backlog_cost": _number(
                        np.sum(reward_info["backlog_cost"]), label="backlog_cost"
                    ),
                    "order_cost": _number(np.sum(reward_info["order_cost"]), label="order_cost"),
                    "demand_total": demand_total,
                    "sales_total": sales_total,
                    "fill_rate": sales_total / demand_total if demand_total > 0 else 1.0,
                },
                "exogenous_digest": _exogenous_digest(sampled, t),
                "terminal": bool(done),
            }
        )
    return steps


def _split_for_world(world_id: str) -> str:
    bucket = int(stable_digest("mabim-v2-split", world_id, length=8), 16) % 20
    if bucket == 18:
        return "pretrain_dev"
    if bucket == 19:
        return "pretrain_test"
    return "pretrain_train"


def _make_record(
    *,
    world_id: str,
    world_seed: int,
    horizon: int,
    sampled: dict[str, Any],
    steps: list[dict[str, Any]],
    branch_id: str | None,
    branch_at: int | None,
) -> dict[str, Any]:
    dynamics = branch_id is not None
    branch_group_id = f"{world_id}:action-branch-at-{branch_at}"
    generated_input_digest = sampled["input_digest"]
    exogenous_stream_digest = sampled["exogenous_digest"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": make_record_id("mabim", world_id, branch_id),
        "simulator": {
            "family": "mabim",
            "name": "MABIM ReplenishmentEnv",
            "version": SOURCE_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "license": SOURCE_LICENSE,
        },
        "split": _split_for_world(world_id),
        "episode": {
            "episode_id": f"{world_id}:{branch_id or 'state'}",
            "world_id": world_id,
            "config_id": "generated-code-only-2w2sku-v1",
            "seed": int(world_seed),
            "time_unit": "day",
            "horizon": horizon,
            "exogenous_stream_id": exogenous_stream_digest[:20],
            "domain_parameters": {
                "warehouse_count": float(len(WAREHOUSES)),
                "sku_count": float(len(SKUS)),
                "lookback": float(LOOKBACK),
                "branch_at": None if branch_at is None else float(branch_at),
                "inputs": "independently_generated_demand_price_leadtime_inventory",
                "bundled_mabim_data_used": False,
                "generated_input_digest": generated_input_digest,
                "action_stream_digest": sampled["action_stream_digest"],
            },
        },
        "branch": (
            None
            if not dynamics
            else {
                "branch_group_id": branch_group_id,
                "branch_id": branch_id,
                "branch_at": float(branch_at),
                "common_random_numbers": True,
            }
        ),
        "objective_eligibility": {
            "state_jepa": True,
            "dynamics_jepa": dynamics,
            "decision_heads": True,
        },
        "topology": _topology(sampled),
        "steps": steps,
        "provenance": {
            "adapter_version": ADAPTER_VERSION,
            "generator": "supplychain_jepa.adapters.mabim.generate_mabim_records",
            "causal_semantics": (
                "State trajectory is observational under one generated policy."
                if not dynamics
                else (
                    "Low/high replenishment branches use identical generated "
                    "inputs, simulator seed, and exact pre-branch actions; only "
                    "post-branch replenishment quantities differ."
                )
            ),
            "observation_semantics": (
                "Inventory, transit, sales, prices, costs, and lead time are "
                "reported; downstream demand is sales-censored on stockout."
            ),
            "source_files": [
                "ReplenishmentEnv/env/replenishment_env.py",
                "ReplenishmentEnv/env/agent_states.py",
                "ReplenishmentEnv/env/supply_chain.py",
                f"generated://mabim/{world_id}/csv-and-yaml/{generated_input_digest}",
            ],
            "runtime_dependencies": {
                "mabim_code_only": True,
                "mabim_revision": SOURCE_COMMIT,
                "source_import_mode": "pinned_checkout",
                "bundled_data_used": False,
                "generated_inputs_temporary": True,
                "warehouse_count": float(len(WAREHOUSES)),
                "sku_count": float(len(SKUS)),
            },
        },
    }
    return json_safe(record)


def _validate_pair(low: dict[str, Any], high: dict[str, Any]) -> None:
    if low["episode"]["exogenous_stream_id"] != high["episode"]["exogenous_stream_id"]:
        raise RuntimeError("Dynamics pair does not share one exogenous stream")
    if low["branch"]["branch_group_id"] != high["branch"]["branch_group_id"]:
        raise RuntimeError("Dynamics records do not share one branch group")
    branch_at = int(low["branch"]["branch_at"])
    for low_step, high_step in zip(low["steps"], high["steps"], strict=True):
        if low_step["exogenous_digest"] != high_step["exogenous_digest"]:
            raise RuntimeError("Dynamics pair has mismatched exogenous inputs")
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


def generate_mabim_records(
    source_root: Path,
    worlds: int = DEFAULT_WORLDS,
    horizon: int = DEFAULT_HORIZON,
    seed: int = DEFAULT_SEED,
    include_state: bool = True,
    include_dynamics: bool = True,
) -> list[dict[str, Any]]:
    """Generate canonical MABIM State rollouts and two-branch CRN pairs."""
    if worlds < 1:
        raise ValueError("worlds must be at least 1")
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if include_dynamics and horizon < 4:
        raise ValueError("Dynamics generation requires horizon >= 4")
    if not include_state and not include_dynamics:
        return []

    environment_class = _load_environment_class(source_root)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="scjepa-mabim-v2-") as temporary:
        root = Path(temporary)
        for world_index in range(worlds):
            world_seed = int(np.random.SeedSequence([int(seed), world_index]).generate_state(1)[0])
            world_id = f"mabim-v2-seed-{seed}-world-{world_index:05d}"
            sampled = _sample_inputs(world_seed, horizon)
            world_dir = root / f"world-{world_index:05d}"
            world_dir.mkdir()
            config_path, _ = _write_generated_inputs(world_dir, sampled, horizon)

            if include_state:
                state_steps = _run_policy(
                    environment_class,
                    config_path,
                    sampled,
                    actions=sampled["state_actions"],
                    policy_id="state_generated_policy",
                    horizon=horizon,
                    world_seed=world_seed,
                )
                records.append(
                    _make_record(
                        world_id=world_id,
                        world_seed=world_seed,
                        horizon=horizon,
                        sampled=sampled,
                        steps=state_steps,
                        branch_id=None,
                        branch_at=None,
                    )
                )

            if include_dynamics:
                branch_at = max(2, horizon // 3)
                prefix = sampled["shared_actions"].copy()
                low_actions = prefix.copy()
                high_actions = prefix.copy()
                low_actions[branch_at:] = np.rint(0.25 * prefix[branch_at:])
                high_actions[branch_at:] = np.rint(1.75 * prefix[branch_at:] + 2.0)
                pair = []
                for branch_id, branch_actions in (
                    ("low", low_actions),
                    ("high", high_actions),
                ):
                    steps = _run_policy(
                        environment_class,
                        config_path,
                        sampled,
                        actions=branch_actions,
                        policy_id="dynamics_crn_policy",
                        horizon=horizon,
                        world_seed=world_seed,
                    )
                    pair.append(
                        _make_record(
                            world_id=world_id,
                            world_seed=world_seed,
                            horizon=horizon,
                            sampled=sampled,
                            steps=steps,
                            branch_id=branch_id,
                            branch_at=branch_at,
                        )
                    )
                _validate_pair(pair[0], pair[1])
                records.extend(pair)
    return records


class MABIMAdapter(SourceAdapter):
    """Convert a pinned MABIM checkout into generated canonical episodes."""

    source_id = "mabim"
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
            raise ValueError("mabim requires AdapterRequest.source_root")
        return generate_mabim_records(
            Path(request.source_root),
            worlds=worlds,
            horizon=request.horizon,
            seed=request.seed,
            include_state=request.include_state,
            include_dynamics=include_dynamics,
        )


__all__ = ["MABIMAdapter", "generate_mabim_records"]
