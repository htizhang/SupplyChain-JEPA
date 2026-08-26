"""Self-contained State-only milestone-flow adapter.

MilestoneFlow generates cargo-process trajectories with latent physical events,
delayed reports, and permanently missing milestones.  Its records support State
JEPA only; no generated policy action is labeled as a counterfactual intervention.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
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

_EVENT_TYPES = ("BKD", "RCS", "ARR_HUB", "DEP", "RCF", "DLV")
ADAPTER_VERSION = "v2-milestone-flow-state-v1"
SOURCE_COMMIT = "release-adapters-v1"
BASE_SOURCE_COMMIT = "62d62dbc94943dd1b728324b0f2f3044bee9de8c"
GENERATOR_NAME = "supplychain_jepa.adapters.milestone_flow.MilestoneFlowAdapter"
_SOURCE_FILE = "src/supplychain_jepa/adapters/milestone_flow.py"


def _require_simpy() -> Any:
    try:
        import simpy
    except ImportError as exc:
        raise RuntimeError(
            "MilestoneFlow conversion requires the adapter extra: uv sync --extra adapters"
        ) from exc
    return simpy


@dataclass(frozen=True)
class _BranchPolicy:
    branch_id: str
    mode: str
    action_cost: float


class _MilestoneWorld:
    """One deterministic replay of a pre-sampled cargo world."""

    def __init__(
        self,
        plan: dict[str, Any],
        *,
        policy: _BranchPolicy,
        branch_at: float,
        horizon: int,
    ) -> None:
        self.plan = plan
        self.policy = policy
        self.branch_at = float(branch_at)
        self.horizon = int(horizon)
        simpy = _require_simpy()
        self.env = simpy.Environment()

        self.origin_resource = simpy.Resource(self.env, capacity=2)
        self.hub_resource = simpy.Resource(self.env, capacity=1)
        self.reserved_hub_resource = simpy.Resource(self.env, capacity=1)
        self.alternate_hub_resource = simpy.Resource(self.env, capacity=1)
        self.destination_resource = simpy.Resource(self.env, capacity=1)

        self.events: list[dict[str, Any]] = []
        self.states: dict[str, dict[str, Any]] = {}
        self.processing_cost = 0.0
        self.transport_cost = 0.0
        self.action_cost = 0.0
        self.total_queue_wait = 0.0

        for shipment in plan["shipments"]:
            shipment_id = shipment["shipment_id"]
            self.states[shipment_id] = {
                "status": "booked",
                "node_id": "ORIGIN",
                "route": "main",
                "service_class": "standard",
                "released": False,
                "delivered": False,
                "queue_wait": 0.0,
                "last_event": "NONE",
                "last_event_time": -1.0,
            }
            self.env.process(self._shipment_process(shipment))
        self.env.process(self._activate_policy())

    def _activate_policy(self):
        yield self.env.timeout(self.branch_at)
        self.action_cost += self.policy.action_cost

    def _policy_is_active(self) -> bool:
        return self.env.now >= self.branch_at and self.policy.mode != "normal"

    def _emit(
        self,
        shipment: dict[str, Any],
        event_type: str,
        node_id: str,
        **features: Any,
    ) -> None:
        report = shipment["reports"][event_type]
        report_time = None if report["missing"] else round(float(self.env.now + report["delay"]), 6)
        event = make_event(
            f"{shipment['shipment_id']}:{event_type}",
            event_type,
            shipment["shipment_id"],
            node_id=node_id,
            event_time=round(float(self.env.now), 6),
            report_time=report_time,
            observed=False,
            features={
                "quantity": shipment["quantity"],
                "service_class": self.states[shipment["shipment_id"]]["service_class"],
                **features,
            },
        )
        self.events.append(event)
        state = self.states[shipment["shipment_id"]]
        state["last_event"] = event_type
        state["last_event_time"] = round(float(self.env.now), 6)
        state["node_id"] = node_id

    def _shipment_process(self, shipment: dict[str, Any]):
        shipment_id = shipment["shipment_id"]
        state = self.states[shipment_id]

        yield self.env.timeout(shipment["release_time"])
        state["released"] = True
        state["status"] = "origin_queue"
        self._emit(shipment, "BKD", "ORIGIN")

        queue_enter = self.env.now
        with self.origin_resource.request() as request:
            yield request
            wait = float(self.env.now - queue_enter)
            state["queue_wait"] += wait
            self.total_queue_wait += wait
            state["status"] = "origin_processing"
            yield self.env.timeout(shipment["origin_process"])
        self.processing_cost += 0.7 * shipment["quantity"]
        state["status"] = "origin_released"
        self._emit(shipment, "RCS", "ORIGIN")

        reroute = self._policy_is_active() and self.policy.mode == "reroute"
        if reroute:
            state["route"] = "alternate"
            state["service_class"] = "rerouted"
            state["status"] = "alternate_transit"
            yield self.env.timeout(0.65 * shipment["origin_to_hub"])
            hub_node = "ALT_HUB"
            hub_resource = self.alternate_hub_resource
        else:
            state["status"] = "mainline_transit"
            yield self.env.timeout(shipment["origin_to_hub"])
            hub_node = "HUB"
            reserve = self._policy_is_active() and self.policy.mode == "expedite_reserved"
            hub_resource = self.reserved_hub_resource if reserve else self.hub_resource
            if reserve:
                state["service_class"] = "reserved_expedite"

        state["status"] = "hub_queue"
        self._emit(shipment, "ARR_HUB", hub_node, route=state["route"])
        queue_enter = self.env.now
        with hub_resource.request() as request:
            yield request
            wait = float(self.env.now - queue_enter)
            state["queue_wait"] += wait
            self.total_queue_wait += wait
            state["status"] = "hub_processing"
            service_factor = (
                0.55
                if self._policy_is_active() and self.policy.mode == "expedite_reserved"
                else 1.0
            )
            yield self.env.timeout(service_factor * shipment["hub_service"])

        if self._policy_is_active() and self.policy.mode == "expedite_reserved":
            consolidation_wait = 0.2
        elif reroute:
            consolidation_wait = 0.35
        else:
            interval = self.plan["dispatch_interval"]
            next_wave = math.ceil(self.env.now / interval) * interval
            consolidation_wait = max(0.0, next_wave - self.env.now)
        state["status"] = "consolidating"
        yield self.env.timeout(consolidation_wait)
        self._emit(
            shipment,
            "DEP",
            hub_node,
            route=state["route"],
            consolidation_wait=round(float(consolidation_wait), 6),
        )

        state["status"] = "destination_transit"
        base_transit = shipment["alternate_transit"] if reroute else shipment["main_transit"]
        if self._policy_is_active() and self.policy.mode == "expedite_reserved":
            base_transit *= 0.7
        failure_delay = (
            shipment["failure_delay"]
            if shipment["reliability_draw"] < self.plan["failure_threshold"]
            else 0.0
        )
        if reroute:
            failure_delay *= 0.35
        yield self.env.timeout(base_transit + failure_delay)
        self.transport_cost += shipment["quantity"] * (1.8 if reroute else 1.0)

        queue_enter = self.env.now
        with self.destination_resource.request() as request:
            yield request
            wait = float(self.env.now - queue_enter)
            state["queue_wait"] += wait
            self.total_queue_wait += wait
            state["status"] = "destination_processing"
            yield self.env.timeout(shipment["destination_process"])
        state["status"] = "received"
        self._emit(
            shipment,
            "RCF",
            "DESTINATION",
            route=state["route"],
            failure_delay=round(float(failure_delay), 6),
        )

        state["status"] = "last_mile"
        yield self.env.timeout(shipment["last_mile"])
        state["status"] = "delivered"
        state["delivered"] = True
        self._emit(
            shipment,
            "DLV",
            "CUSTOMER",
            due_time=shipment["due_time"],
            late=bool(self.env.now > shipment["due_time"]),
        )

    @staticmethod
    def _resource_state(resource: Any) -> dict[str, float]:
        return {
            "busy": float(resource.count),
            "queue": float(len(resource.queue)),
            "capacity": float(resource.capacity),
        }

    def _event_snapshot(self, clock: float) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for base_event in self.events:
            if base_event["event_time"] > clock:
                continue
            event = deepcopy(base_event)
            event["observed"] = bool(
                event["report_time"] is not None and event["report_time"] <= clock
            )
            snapshot.append(event)
        return sorted(snapshot, key=lambda item: (item["event_time"], item["event_id"]))

    def _latent_entities(self) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        shipment_by_id = {item["shipment_id"]: item for item in self.plan["shipments"]}
        for shipment_id in sorted(self.states):
            state = self.states[shipment_id]
            shipment = shipment_by_id[shipment_id]
            entities.append(
                make_entity(
                    shipment_id,
                    "shipment",
                    node_id=state["node_id"],
                    product_id="GENERAL_CARGO",
                    features={
                        "quantity": shipment["quantity"],
                        "release_time": shipment["release_time"],
                        "due_time": shipment["due_time"],
                        "status": state["status"],
                        "route": state["route"],
                        "service_class": state["service_class"],
                        "released": state["released"],
                        "delivered": state["delivered"],
                        "queue_wait": round(float(state["queue_wait"]), 6),
                        "last_event": state["last_event"],
                        "last_event_time": state["last_event_time"],
                    },
                )
            )
        return entities

    def _observed_entities(
        self, clock: float, observed_events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in observed_events:
            previous = latest.get(event["entity_id"])
            if previous is None or (
                event["event_time"],
                event["event_id"],
            ) > (
                previous["event_time"],
                previous["event_id"],
            ):
                latest[event["entity_id"]] = event

        entities: list[dict[str, Any]] = []
        for shipment in self.plan["shipments"]:
            shipment_id = shipment["shipment_id"]
            event = latest.get(shipment_id)
            entities.append(
                make_entity(
                    shipment_id,
                    "shipment_report",
                    node_id=None if event is None else event["node_id"],
                    product_id="GENERAL_CARGO",
                    features={
                        "quantity": shipment["quantity"],
                        "latest_reported_milestone": (
                            "NONE" if event is None else event["event_type"]
                        ),
                        "latest_physical_event_time": (
                            -1.0 if event is None else event["event_time"]
                        ),
                        "report_age": (
                            -1.0 if event is None else round(float(clock - event["event_time"]), 6)
                        ),
                        "delivery_reported": bool(
                            event is not None and event["event_type"] == "DLV"
                        ),
                    },
                )
            )
        return entities

    def _edge_states(self, *, observed: bool) -> list[dict[str, Any]]:
        main_active = sum(
            state["status"] in {"mainline_transit", "destination_transit"}
            and state["route"] == "main"
            for state in self.states.values()
        )
        alternate_active = sum(
            state["status"] in {"alternate_transit", "destination_transit"}
            and state["route"] == "alternate"
            for state in self.states.values()
        )
        if observed:
            return [
                make_edge_state(
                    "ORIGIN_MAIN_HUB",
                    {"mode": "air", "published_capacity": 1.0},
                ),
                make_edge_state(
                    "MAIN_HUB_DESTINATION",
                    {"mode": "air", "published_capacity": 1.0},
                ),
                make_edge_state(
                    "ORIGIN_ALT_HUB",
                    {"mode": "truck_air", "published_capacity": 1.0},
                ),
                make_edge_state(
                    "ALT_HUB_DESTINATION",
                    {"mode": "truck_air", "published_capacity": 1.0},
                ),
            ]
        return [
            make_edge_state(
                "ORIGIN_MAIN_HUB",
                {"active_shipments": main_active, "congested": main_active > 1},
            ),
            make_edge_state(
                "MAIN_HUB_DESTINATION",
                {"active_shipments": main_active, "congested": main_active > 1},
            ),
            make_edge_state(
                "ORIGIN_ALT_HUB",
                {
                    "active_shipments": alternate_active,
                    "congested": alternate_active > 1,
                },
            ),
            make_edge_state(
                "ALT_HUB_DESTINATION",
                {
                    "active_shipments": alternate_active,
                    "congested": alternate_active > 1,
                },
            ),
        ]

    def _step_actions(self, clock: float) -> list[dict[str, Any]]:
        if not math.isclose(clock, self.branch_at):
            return []
        if self.policy.mode == "normal":
            return [
                make_action(
                    "continue_standard_dispatch",
                    "CARGO_CONTROLLER",
                    target_id="HUB",
                    quantity=0.0,
                    features={"policy": "normal"},
                )
            ]
        if self.policy.mode == "expedite_reserved":
            return [
                make_action(
                    "expedite",
                    "CARGO_CONTROLLER",
                    target_id="HUB",
                    quantity=5.0,
                    features={"transit_time_multiplier": 0.7},
                ),
                make_action(
                    "capacity_reservation",
                    "CARGO_CONTROLLER",
                    target_id="HUB",
                    quantity=1.0,
                    features={"reserved_server_count": 1.0},
                ),
            ]
        if self.policy.mode == "reroute":
            return [
                make_action(
                    "reroute",
                    "CARGO_CONTROLLER",
                    target_id="ALT_HUB",
                    quantity=5.0,
                    features={"route": "alternate"},
                )
            ]
        raise ValueError(f"Unknown branch policy: {self.policy.mode}")

    def snapshot(self, t: int) -> dict[str, Any]:
        clock = float(t)
        latent_events = self._event_snapshot(clock)
        observed_events = [event for event in latent_events if event["observed"]]
        missing_count = sum(event["report_time"] is None for event in latent_events)
        delayed_count = sum(
            event["report_time"] is not None and event["report_time"] > clock
            for event in latent_events
        )
        delivered = sum(state["delivered"] for state in self.states.values())
        released = sum(state["released"] for state in self.states.values())
        late = sum(
            event["event_type"] == "DLV" and bool(event["features"].get("late", False))
            for event in latent_events
        )
        total_cost = self.processing_cost + self.transport_cost + self.action_cost
        resources = {
            "origin": self._resource_state(self.origin_resource),
            "hub": self._resource_state(self.hub_resource),
            "reserved_hub": self._resource_state(self.reserved_hub_resource),
            "alternate_hub": self._resource_state(self.alternate_hub_resource),
            "destination": self._resource_state(self.destination_resource),
        }
        status = [
            observation_status(
                "latent_state.global.congestion_regime",
                "missing",
                reason="operational congestion regime is hidden",
            ),
            observation_status(
                "latent_state.global.reliability_regime",
                "missing",
                reason="carrier reliability regime is hidden",
            ),
            observation_status(
                "latent_state.events",
                "delayed" if delayed_count else "observed",
                age_steps=float(delayed_count),
                reason="milestones become visible only after their report time",
            ),
            observation_status(
                "latent_state.events",
                "missing" if missing_count else "observed",
                age_steps=0.0,
                reason="selected milestone reports are permanently absent",
            ),
        ]

        latent_global = {
            "hidden_regime": (
                f"{self.plan['congestion_regime']}|{self.plan['reliability_regime']}"
            ),
            "congestion_regime": self.plan["congestion_regime"],
            "reliability_regime": self.plan["reliability_regime"],
            "released_shipments": released,
            "delivered_shipments": delivered,
            "late_shipments": late,
            "pipeline_in_transit": max(0, released - delivered),
            "queue_length": sum(resource_state["queue"] for resource_state in resources.values()),
            "capacity_available": sum(
                resource_state["capacity"] - resource_state["busy"]
                for resource_state in resources.values()
            ),
            "origin_busy": resources["origin"]["busy"],
            "origin_queue": resources["origin"]["queue"],
            "hub_busy": resources["hub"]["busy"],
            "hub_queue": resources["hub"]["queue"],
            "reserved_hub_busy": resources["reserved_hub"]["busy"],
            "reserved_hub_queue": resources["reserved_hub"]["queue"],
            "alternate_hub_busy": resources["alternate_hub"]["busy"],
            "alternate_hub_queue": resources["alternate_hub"]["queue"],
            "destination_busy": resources["destination"]["busy"],
            "destination_queue": resources["destination"]["queue"],
            "total_queue_wait": round(float(self.total_queue_wait), 6),
            "processing_cost": round(float(self.processing_cost), 6),
            "transport_cost": round(float(self.transport_cost), 6),
            "action_cost": round(float(self.action_cost), 6),
            "total_cost": round(float(total_cost), 6),
        }
        observation_global = {
            "reported_events": len(observed_events),
            "reported_deliveries": sum(event["event_type"] == "DLV" for event in observed_events),
            "released_orders": released,
            "published_hub_capacity": 1.0,
            "clock": clock,
        }
        return {
            "t": t,
            "clock": clock,
            "latent_state": {
                "entities": self._latent_entities(),
                "edges": self._edge_states(observed=False),
                "global": latent_global,
                "events": latent_events,
            },
            "observation": {
                "entities": self._observed_entities(clock, observed_events),
                "edges": self._edge_states(observed=True),
                "global": observation_global,
                "events": observed_events,
                "status": status,
            },
            "actions": self._step_actions(clock),
            "outcomes": {
                "delivered_shipments": delivered,
                "late_shipments": late,
                "on_time_rate": (
                    0.0 if delivered == 0 else round(float((delivered - late) / delivered), 6)
                ),
                "total_cost": round(float(total_cost), 6),
                "total_queue_wait": round(float(self.total_queue_wait), 6),
            },
            "exogenous_digest": stable_digest("milestone-flow-exogenous-v1", self.plan),
            "terminal": t == self.horizon - 1,
        }

    def run(self) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for t in range(self.horizon):
            if t:
                self.env.run(until=float(t))
            steps.append(self.snapshot(t))
        return steps


def _topology() -> dict[str, Any]:
    return {
        "topology_id": "milestone-flow-four-node-two-route-v1",
        "nodes": [
            {
                "node_id": "ORIGIN",
                "node_type": "cargo_origin",
                "features": {"processing_capacity": 2.0},
            },
            {
                "node_id": "HUB",
                "node_type": "consolidation_hub",
                "features": {
                    "standard_capacity": 1.0,
                    "reservable_capacity": 1.0,
                },
            },
            {
                "node_id": "ALT_HUB",
                "node_type": "alternate_hub",
                "features": {"processing_capacity": 1.0},
            },
            {
                "node_id": "DESTINATION",
                "node_type": "cargo_destination",
                "features": {"processing_capacity": 1.0},
            },
            {
                "node_id": "CUSTOMER",
                "node_type": "delivery_endpoint",
                "features": {"acceptance_capacity": 5.0},
            },
        ],
        "edges": [
            {
                "edge_id": "ORIGIN_MAIN_HUB",
                "source": "ORIGIN",
                "target": "HUB",
                "edge_type": "transport_leg",
                "features": {"mode": "air", "route": "main"},
            },
            {
                "edge_id": "MAIN_HUB_DESTINATION",
                "source": "HUB",
                "target": "DESTINATION",
                "edge_type": "transport_leg",
                "features": {"mode": "air", "route": "main"},
            },
            {
                "edge_id": "ORIGIN_ALT_HUB",
                "source": "ORIGIN",
                "target": "ALT_HUB",
                "edge_type": "transport_leg",
                "features": {"mode": "truck_air", "route": "alternate"},
            },
            {
                "edge_id": "ALT_HUB_DESTINATION",
                "source": "ALT_HUB",
                "target": "DESTINATION",
                "edge_type": "transport_leg",
                "features": {"mode": "truck_air", "route": "alternate"},
            },
            {
                "edge_id": "DESTINATION_CUSTOMER",
                "source": "DESTINATION",
                "target": "CUSTOMER",
                "edge_type": "last_mile",
                "features": {"mode": "truck", "route": "shared"},
            },
        ],
    }


def _episode_record(
    *,
    plan: dict[str, Any],
    policy: _BranchPolicy,
    branch_group_id: str,
    branch_at: float,
    split: str,
    horizon: int,
    config_id: str = "milestone-flow-smoke-v1",
    domain_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    world = _MilestoneWorld(
        plan,
        policy=policy,
        branch_at=branch_at,
        horizon=horizon,
    )
    world_id = plan["world_id"]
    episode_domain_parameters = (
        {
            "shipment_count": len(plan["shipments"]),
            "origin_capacity": 2.0,
            "standard_hub_capacity": 1.0,
            "reserved_hub_capacity": 1.0,
            "alternate_hub_capacity": 1.0,
            "destination_capacity": 1.0,
            "observation_model": "delayed_and_missing_milestones",
        }
        if domain_parameters is None
        else dict(domain_parameters)
    )
    return json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "record_id": make_record_id("milestone_flow", world_id, policy.branch_id),
            "simulator": {
                "family": "milestone_flow",
                "name": "MilestoneFlow",
                "version": ADAPTER_VERSION,
                "source_repository": "local Stage 2 implementation",
                "source_commit": ADAPTER_VERSION,
                "license": "project code",
            },
            "split": split,
            "episode": {
                "episode_id": f"{world_id}:{policy.branch_id}",
                "world_id": world_id,
                "config_id": config_id,
                "seed": plan["seed"],
                "time_unit": "hour",
                "horizon": horizon,
                "exogenous_stream_id": stable_digest("milestone-flow-stream-v1", plan, length=20),
                "domain_parameters": episode_domain_parameters,
            },
            "branch": {
                "branch_group_id": branch_group_id,
                "branch_id": policy.branch_id,
                "branch_at": branch_at,
                "common_random_numbers": True,
            },
            "objective_eligibility": {
                "state_jepa": True,
                "dynamics_jepa": True,
                "decision_heads": True,
            },
            "topology": _topology(),
            "steps": world.run(),
            "provenance": {
                "adapter_version": ADAPTER_VERSION,
                "generator": "MilestoneFlow deterministic SimPy DES",
                "causal_semantics": (
                    "Actions alter service, route, or reserved capacity while "
                    "paired branches reuse the exact pre-sampled exogenous plan."
                ),
                "observation_semantics": (
                    "Latent state contains physical milestones and hidden "
                    "regimes; observations contain only non-missing reports "
                    "whose report_time has elapsed."
                ),
                "source_files": [_SOURCE_FILE],
                "runtime_dependencies": {
                    "simpy": "4.1.2",
                    "simpy_repository": "https://gitlab.com/team-simpy/simpy",
                    "simpy_license": "MIT",
                },
            },
        }
    )


def _sample_stage3_world(
    *,
    physical_seed: int,
    observation_seed: int,
    world_id: str,
    congestion_regime: str,
    reliability_regime: str,
    continuous: dict[str, Any],
    observation_config: dict[str, Any],
) -> dict[str, Any]:
    """Sample physical and reporting processes from independent RNG streams."""
    physical_rng = np.random.default_rng(physical_seed)
    observation_rng = np.random.default_rng(observation_seed)
    congestion_factor = {"normal": 1.0, "peak": 1.55}[congestion_regime]
    reliability_factor = {"stable": 0.7, "fragile": 1.45}[reliability_regime]
    shipment_count = int(round(float(continuous["shipment_count"])))
    release_span = float(continuous["release_span"])
    releases = np.linspace(0.0, release_span, num=shipment_count)
    service_scale = float(continuous["service_scale"])
    failure_delay_scale = float(continuous["failure_delay_scale"])
    report_delay_scale = float(observation_config.get("report_delay_scale", 1.0))
    missing_probability = float(observation_config.get("missing_probability", 0.05))
    shipments: list[dict[str, Any]] = []
    for index, release_time in enumerate(releases):
        event_reports: dict[str, dict[str, Any]] = {}
        for event_index, event_type in enumerate(_EVENT_TYPES):
            report_delay = float(
                (
                    observation_rng.uniform(0.15, 0.85)
                    + (0.9 if event_type in {"DEP", "RCF"} else 0.0)
                    + (0.45 if congestion_regime == "peak" else 0.0)
                )
                * report_delay_scale
            )
            sampled_missing = bool(
                observation_rng.random() < missing_probability * reliability_factor
                and event_index not in {0, len(_EVENT_TYPES) - 1}
            )
            event_reports[event_type] = {
                "delay": round(report_delay, 6),
                "missing": sampled_missing,
            }
        shipments.append(
            {
                "shipment_id": f"{world_id}:shipment-{index:02d}",
                "quantity": int(physical_rng.integers(8, 19)),
                "release_time": round(float(release_time), 6),
                "due_time": round(
                    12.0 + index * 0.8 + float(physical_rng.uniform(-0.4, 0.7)),
                    6,
                ),
                "origin_process": round(float(physical_rng.uniform(0.7, 1.5) * service_scale), 6),
                "origin_to_hub": round(float(physical_rng.uniform(1.0, 2.0)), 6),
                "hub_service": round(
                    float(physical_rng.uniform(0.9, 1.8) * congestion_factor * service_scale),
                    6,
                ),
                "main_transit": round(float(physical_rng.uniform(3.2, 5.0)), 6),
                "alternate_transit": round(float(physical_rng.uniform(2.6, 4.1)), 6),
                "destination_process": round(
                    float(physical_rng.uniform(0.5, 1.2) * service_scale), 6
                ),
                "last_mile": round(float(physical_rng.uniform(0.4, 1.0)), 6),
                "reliability_draw": round(float(physical_rng.random()), 8),
                "failure_delay": round(
                    float(
                        physical_rng.uniform(1.2, 3.2) * reliability_factor * failure_delay_scale
                    ),
                    6,
                ),
                "reports": event_reports,
            }
        )
    return {
        "world_id": world_id,
        "seed": int(physical_seed),
        "congestion_regime": congestion_regime,
        "reliability_regime": reliability_regime,
        "dispatch_interval": (3.0 if congestion_regime == "peak" else 2.5) * service_scale,
        "failure_threshold": 0.22 if reliability_regime == "fragile" else 0.08,
        "shipments": shipments,
    }


def _physical_plan(plan: dict[str, Any]) -> dict[str, Any]:
    physical = deepcopy(plan)
    for shipment in physical["shipments"]:
        shipment.pop("reports", None)
    return physical


_FORBIDDEN_LINEAGE_MARKERS = (
    "freshretailnet",
    "cargo2000",
    "gym-invmgmt",
    "gym_invmgmt",
    "downstream-application",
)


def _split_for(world_id: str) -> str:
    bucket = int(stable_digest("milestone-flow-v2-split", world_id, length=8), 16) % 20
    if bucket == 18:
        return "pretrain_dev"
    if bucket == 19:
        return "pretrain_test"
    return "pretrain_train"


def _world_parameters(seed: int, world_index: int) -> tuple[int, int, dict[str, Any]]:
    sequence = np.random.SeedSequence([seed, world_index])
    physical_sequence, observation_sequence, parameter_sequence = sequence.spawn(3)
    physical_seed = int(physical_sequence.generate_state(1)[0])
    observation_seed = int(observation_sequence.generate_state(1)[0])
    rng = np.random.default_rng(parameter_sequence)
    congestion_regime = "peak" if rng.random() < 0.5 else "normal"
    reliability_regime = "fragile" if rng.random() < 0.5 else "stable"
    parameters = {
        "congestion_regime": congestion_regime,
        "reliability_regime": reliability_regime,
        "shipment_count": int(rng.integers(6, 11)),
        "release_span": float(round(rng.uniform(5.0, 9.0), 6)),
        "service_scale": float(round(rng.uniform(0.8, 1.2), 6)),
        "failure_delay_scale": float(round(rng.uniform(0.75, 1.4), 6)),
        "report_delay_scale": float(round(rng.uniform(0.8, 1.4), 6)),
        "missing_probability": float(round(rng.uniform(0.03, 0.12), 6)),
    }
    return physical_seed, observation_seed, parameters


def _force_observation_challenges(plan: dict[str, Any]) -> None:
    """Guarantee meaningful delayed and missing reports in every world."""
    first = plan["shipments"][0]["reports"]
    first["BKD"]["delay"] = max(float(first["BKD"]["delay"]), 2.25)
    first["RCS"]["missing"] = True
    if len(plan["shipments"]) > 1:
        plan["shipments"][1]["reports"]["ARR_HUB"]["missing"] = True


def _release_record(
    *,
    seed: int,
    world_index: int,
    horizon: int,
) -> dict[str, Any]:
    physical_seed, observation_seed, parameters = _world_parameters(seed, world_index)
    world_id = f"milestone-flow-v2-seed-{seed}-world-{world_index:06d}"
    observation_config = {
        "name": "generated_delayed_and_missing_milestones",
        "observation_cell_id": f"{world_id}:observation",
        "report_delay_scale": parameters["report_delay_scale"],
        "missing_probability": parameters["missing_probability"],
    }
    plan = _sample_stage3_world(
        physical_seed=physical_seed,
        observation_seed=observation_seed,
        world_id=world_id,
        congestion_regime=parameters["congestion_regime"],
        reliability_regime=parameters["reliability_regime"],
        continuous={
            "shipment_count": parameters["shipment_count"],
            "release_span": parameters["release_span"],
            "service_scale": parameters["service_scale"],
            "failure_delay_scale": parameters["failure_delay_scale"],
        },
        observation_config=observation_config,
    )
    _force_observation_challenges(plan)
    physical_plan = _physical_plan(plan)
    config_id = stable_digest("milestone-flow-v2-config", parameters, horizon, length=16)
    record = _episode_record(
        plan=plan,
        policy=_BranchPolicy(branch_id="state", mode="normal", action_cost=0.0),
        branch_group_id=f"{world_id}:unused-state-branch",
        branch_at=float(max(1, min(horizon - 2, horizon // 3))),
        split=_split_for(world_id),
        horizon=horizon,
        config_id=config_id,
        domain_parameters={
            "shipment_count": float(parameters["shipment_count"]),
            "origin_capacity": 2.0,
            "standard_hub_capacity": 1.0,
            "reserved_hub_capacity": 1.0,
            "alternate_hub_capacity": 1.0,
            "destination_capacity": 1.0,
            "congestion_regime": parameters["congestion_regime"],
            "reliability_regime": parameters["reliability_regime"],
            "release_span": parameters["release_span"],
            "service_scale": parameters["service_scale"],
            "failure_delay_scale": parameters["failure_delay_scale"],
            "report_delay_scale": parameters["report_delay_scale"],
            "missing_probability": parameters["missing_probability"],
            "generated_inputs_only": True,
            "downstream_lineage_used": False,
        },
    )

    record["record_id"] = make_record_id("milestone_flow", world_id, None)
    record["simulator"] = {
        "family": "milestone_flow",
        "name": "MilestoneFlow",
        "version": ADAPTER_VERSION,
        "source_repository": "project-authored",
        "source_commit": SOURCE_COMMIT,
        "license": "project code",
    }
    record["episode"]["episode_id"] = f"{world_id}:state"
    record["episode"]["seed"] = physical_seed
    record["episode"]["exogenous_stream_id"] = stable_digest(
        "milestone-flow-v2-physical-process", physical_plan, length=20
    )
    record["branch"] = None
    record["objective_eligibility"] = {
        "state_jepa": True,
        "dynamics_jepa": False,
        "decision_heads": False,
    }
    for step in record["steps"]:
        step["actions"] = []
        step["exogenous_digest"] = stable_digest(
            "milestone-flow-v2-physical-process", physical_plan, step["t"]
        )
    record["provenance"] = {
        "adapter_version": ADAPTER_VERSION,
        "generator": GENERATOR_NAME,
        "causal_semantics": (
            "Observational generated process rollout; no intervention pair or "
            "action-effect eligibility."
        ),
        "observation_semantics": (
            "Latent state follows the generated physical logistics process. "
            "Observations expose milestones only after generated report delays, "
            "with selected reports deterministically absent."
        ),
        "source_files": [_SOURCE_FILE],
        "runtime_dependencies": {
            "simpy": "4.1.2",
            "base_source_commit": BASE_SOURCE_COMMIT,
            "input_mode": "generated_processes_only",
        },
    }
    record = json_safe(record)
    serialized = repr(record).lower()
    found = [marker for marker in _FORBIDDEN_LINEAGE_MARKERS if marker in serialized]
    if found:
        raise RuntimeError(f"Forbidden downstream lineage marker emitted: {found}")
    return record


def generate_milestone_flow_records(
    *,
    worlds: int = 8,
    horizon: int = 32,
    seed: int = 20260730,
    include_state: bool = True,
    include_dynamics: bool = False,
) -> list[dict[str, Any]]:
    """Generate deterministic, State-only milestone-flow episodes."""
    if worlds < 0:
        raise ValueError("worlds must be non-negative")
    if horizon < 4:
        raise ValueError("horizon must be at least 4")
    if include_dynamics:
        raise ValueError("MilestoneFlow adapter is State-only")
    if not include_state or worlds == 0:
        return []
    return [
        _release_record(seed=seed, world_index=world_index, horizon=horizon)
        for world_index in range(worlds)
    ]


class MilestoneFlowAdapter(SourceAdapter):
    """Generate project-authored milestone-flow worlds without an external checkout."""

    source_id = "milestone_flow"
    source_kind = "simulator"
    supports_state = True
    supports_dynamics = False

    def convert(self, request: AdapterRequest) -> list[dict[str, Any]]:
        include_dynamics = self.resolved_dynamics_flag(request)
        if not request.include_state:
            return []
        worlds = request.worlds
        if request.limit is not None:
            worlds = min(worlds, request.limit)
        return generate_milestone_flow_records(
            worlds=worlds,
            horizon=request.horizon,
            seed=request.seed,
            include_state=request.include_state,
            include_dynamics=include_dynamics,
        )


__all__ = ["ADAPTER_VERSION", "MilestoneFlowAdapter", "SOURCE_COMMIT"]
