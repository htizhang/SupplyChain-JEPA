"""Deterministic clean-room reconstruction of the Pool-B6 decision contract.

The historical executable for the selected action-selection experiment was not
retained.  This module therefore reconstructs the documented mechanics without
claiming byte-for-byte identity with that runner: semi-Markov supplier quality,
MNAR public reports, quantity-matched regular/split orders, common-random-number
future branches, and a no-further-orders continuation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

import numpy as np

from .contracts import (
    HISTORY_FEATURES,
    ActionPlan,
    DecisionWorld,
    Order,
    PairedFutureCost,
    ReconstructionConfig,
    WorldSplit,
)


def _digest(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semi_markov_path(
    rng: np.random.Generator,
    *,
    length: int,
    initial: int,
    dwell_scale: float,
) -> np.ndarray:
    """Generate reliability states 0=disrupted, 1=strained, 2=reliable."""

    path = np.empty(length, dtype=np.int8)
    state = int(initial)
    remaining = 0
    transition = np.asarray(
        ((0.70, 0.27, 0.03), (0.12, 0.70, 0.18), (0.03, 0.17, 0.80)),
        dtype=np.float64,
    )
    for time_index in range(length):
        if remaining <= 0:
            state = int(rng.choice(3, p=transition[state]))
            base_dwell = int(rng.integers(4, 10))
            remaining = max(2, round(base_dwell * dwell_scale))
        path[time_index] = state
        remaining -= 1
    return path


def _report_probability(
    reliability_state: int,
    congestion: bool,
    config: ReconstructionConfig,
) -> float:
    impairment = (2 - reliability_state) / 2.0
    missing = (
        config.report_independent_dropout
        + config.report_impairment_penalty * impairment
        + config.report_congestion_penalty * float(congestion)
    )
    return min(max(missing, 0.0), 0.95)


def _public_history(
    rng: np.random.Generator,
    *,
    config: ReconstructionConfig,
    demand_level: float,
    demand_trend: float,
    primary_capacity: float,
    secondary_capacity: float,
    primary_states: np.ndarray,
    secondary_states: np.ndarray,
    congestion: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    inventory = float(rng.uniform(1.5, 4.5) * demand_level)
    backlog = float(rng.uniform(0.0, 0.3) * demand_level)
    pipeline = float(rng.uniform(0.5, 2.0) * demand_level)
    reported_capacity = [primary_capacity, secondary_capacity]
    report_age = [0.0, 0.0]
    rows: list[tuple[float, ...]] = []

    for time_index in range(config.lookback):
        seasonal = 1.0 + 0.16 * math.sin(2.0 * math.pi * time_index / 7.0)
        latent_demand = max(
            0.1,
            demand_level * seasonal
            + demand_trend * time_index
            + float(rng.normal(0.0, 0.16 * demand_level)),
        )
        delivered = min(
            pipeline,
            demand_level
            * (0.25 + 0.20 * primary_states[time_index])
            * (0.72 if congestion[time_index] else 1.0),
        )
        pipeline = max(0.0, pipeline - delivered)
        available = max(0.0, inventory + delivered)
        sales = min(latent_demand + backlog, available)
        backlog = max(0.0, backlog + latent_demand - sales)
        inventory = max(0.0, available - sales)
        if time_index % 3 == 0:
            pipeline += max(0.0, 1.8 * demand_level - inventory - pipeline)

        availabilities: list[bool] = []
        capacities = (primary_capacity, secondary_capacity)
        states = (primary_states[time_index], secondary_states[time_index])
        for supplier_index, (capacity, state) in enumerate(
            zip(capacities, states, strict=True)
        ):
            available_report = bool(
                rng.random()
                >= _report_probability(int(state), bool(congestion[time_index]), config)
            )
            availabilities.append(available_report)
            if available_report:
                state_multiplier = (0.34, 0.68, 1.0)[int(state)]
                reported_capacity[supplier_index] = max(
                    0.0, capacity * state_multiplier + rng.normal(0.0, 0.03 * capacity)
                )
                report_age[supplier_index] = 0.0
            else:
                report_age[supplier_index] += 1.0

        # Sales are the public demand proxy when stock is unavailable.
        demand_reported = sales if inventory <= 1e-8 else latent_demand
        row = (
            inventory,
            backlog,
            demand_reported,
            sales,
            pipeline,
            reported_capacity[0],
            reported_capacity[1],
            report_age[0],
            report_age[1],
            float(availabilities[0]),
            float(availabilities[1]),
            float(congestion[time_index]),
        )
        if len(row) != len(HISTORY_FEATURES):
            raise AssertionError("public-history feature contract changed")
        rows.append(tuple(float(value) for value in row))
    return tuple(rows)


def _candidate_plans(history: tuple[tuple[float, ...], ...]) -> tuple[ActionPlan, ActionPlan]:
    recent = np.asarray(history[-4:], dtype=np.float64)
    demand = float(recent[:, 2].mean())
    inventory, backlog, pipeline = history[-1][0], history[-1][1], history[-1][4]
    quantity = float(np.clip(4.0 * demand + backlog - inventory - pipeline, 4.0, 120.0))
    regular = ActionPlan(
        name="regular",
        orders=(Order("primary", quantity, 1),),
    )
    primary_quantity = 0.50 * quantity
    split = ActionPlan(
        name="split",
        orders=(
            Order("primary", primary_quantity, 1),
            Order("secondary", quantity - primary_quantity, 1),
        ),
    )
    if not math.isclose(regular.total_quantity, split.total_quantity, abs_tol=1e-9):
        raise AssertionError("candidate plans are not quantity matched")
    return regular, split


def _future_cost(
    plan: ActionPlan,
    *,
    history: tuple[tuple[float, ...], ...],
    demand_noise: np.ndarray,
    congestion: np.ndarray,
    primary_states: np.ndarray,
    secondary_states: np.ndarray,
    fill_uniforms: np.ndarray,
    horizon: int,
) -> float:
    inventory = float(history[-1][0])
    backlog = float(history[-1][1])
    recent_demand = float(np.mean([row[2] for row in history[-4:]]))
    arrivals = np.zeros(horizon, dtype=np.float64)
    state_paths = {"primary": primary_states, "secondary": secondary_states}
    supplier_index = {"primary": 0, "secondary": 1}
    for order in plan.orders:
        index = supplier_index[order.supplier]
        state = int(state_paths[order.supplier][0])
        delay = int(congestion[0]) + int(state == 0)
        arrival_period = min(horizon - 1, order.nominal_lead_time - 1 + delay)
        base_fill = (0.45, 0.75, 0.98)[state]
        fill = float(np.clip(base_fill + 0.10 * (fill_uniforms[index] - 0.5), 0.2, 1.0))
        arrivals[arrival_period] += order.quantity * fill

    cost = 0.30 * max(0, len(plan.orders) - 1)
    for period in range(horizon):
        demand = max(0.0, recent_demand + demand_noise[period])
        available = inventory + arrivals[period]
        fulfilled = min(available, demand + backlog)
        backlog = max(0.0, backlog + demand - fulfilled)
        inventory = max(0.0, available - fulfilled)
        cost += 0.11 * inventory + 1.35 * backlog
        if backlog > recent_demand:
            cost += 0.18 * (backlog - recent_demand)
    return float(cost)


def generate_reconstruction_worlds(
    config: ReconstructionConfig,
) -> tuple[DecisionWorld, ...]:
    """Generate deterministic decision worlds without external data."""

    config.validate()
    worlds: list[DecisionWorld] = []
    full_length = config.lookback + config.horizon
    for world_index in range(config.worlds):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, world_index]))
        world_id = f"world-{world_index:05d}-{_digest(config.seed, world_index)[:10]}"
        demand_level = float(rng.uniform(8.0, 28.0))
        demand_trend = float(rng.normal(0.0, 0.12))
        primary_capacity = float(rng.uniform(1.0, 2.5) * demand_level)
        secondary_capacity = float(rng.uniform(0.8, 2.2) * demand_level)
        primary_states = _semi_markov_path(
            rng,
            length=full_length,
            initial=int(rng.integers(0, 3)),
            dwell_scale=config.semi_markov_dwell_scale,
        )
        secondary_states = _semi_markov_path(
            rng,
            length=full_length,
            initial=int(rng.integers(0, 3)),
            dwell_scale=config.semi_markov_dwell_scale,
        )
        congestion = rng.random(full_length) < 0.16
        history = _public_history(
            rng,
            config=config,
            demand_level=demand_level,
            demand_trend=demand_trend,
            primary_capacity=primary_capacity,
            secondary_capacity=secondary_capacity,
            primary_states=primary_states,
            secondary_states=secondary_states,
            congestion=congestion,
        )
        regular, split = _candidate_plans(history)
        paired_costs: list[PairedFutureCost] = []
        for replication in range(config.future_replications):
            future_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, world_index, replication, 8849])
            )
            demand_noise = future_rng.normal(
                0.0, 0.20 * demand_level, size=config.horizon
            )
            future_congestion = future_rng.random(config.horizon) < 0.18
            fill_uniforms = future_rng.random(2)
            noise_digest = _digest(
                world_id,
                replication,
                demand_noise.tolist(),
                future_congestion.tolist(),
                fill_uniforms.tolist(),
            )
            common = {
                "history": history,
                "demand_noise": demand_noise,
                "congestion": future_congestion,
                "primary_states": primary_states[config.lookback :],
                "secondary_states": secondary_states[config.lookback :],
                "fill_uniforms": fill_uniforms,
                "horizon": config.horizon,
            }
            paired_costs.append(
                PairedFutureCost(
                    replication=replication,
                    noise_digest=noise_digest,
                    regular_cost=_future_cost(regular, **common),
                    split_cost=_future_cost(split, **common),
                )
            )
        worlds.append(
            DecisionWorld(
                world_id=world_id,
                public_history=history,
                regular_plan=regular,
                split_plan=split,
                paired_future_costs=tuple(paired_costs),
            )
        )
    return tuple(worlds)


def build_world_splits(world_ids: Iterable[str], *, salt: int) -> WorldSplit:
    """Hash-rank worlds into disjoint 60/20/20 fit/select/report partitions."""

    unique = tuple(sorted(set(world_ids)))
    if len(unique) < 10:
        raise ValueError("at least ten worlds are required for three-way splitting")
    ranked = sorted(unique, key=lambda item: (_digest(salt, item), item))
    fit_end = max(1, int(0.60 * len(ranked)))
    select_end = max(fit_end + 1, int(0.80 * len(ranked)))
    split = WorldSplit(
        fit=tuple(ranked[:fit_end]),
        select=tuple(ranked[fit_end:select_end]),
        report=tuple(ranked[select_end:]),
    )
    if not split.report:
        raise ValueError("report partition is empty")
    return split
