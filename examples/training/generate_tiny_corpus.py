"""Generate a deterministic canonical corpus for the training smoke recipe.

The records are deliberately small and project-authored. They exercise partial
observations, structured topology, paired actions, and counterfactual futures;
they are not intended to reproduce the released model's empirical quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from supplychain_jepa.schemas import canonical_episode_schema

DEMAND = (3.0, 4.0, 5.0, 3.0, 6.0, 4.0, 5.0)
BRANCH_AT = 2
ARRIVAL_AT = 4


def _digest(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _topology() -> dict[str, Any]:
    return {
        "topology_id": "two-echelon-line",
        "nodes": [
            {
                "node_id": "supplier",
                "node_type": "supplier",
                "features": {"capacity": 20.0, "region": "north"},
            },
            {
                "node_id": "store",
                "node_type": "retailer",
                "features": {"storage_capacity": 30.0, "region": "south"},
            },
        ],
        "edges": [
            {
                "edge_id": "supplier-to-store",
                "source": "supplier",
                "target": "store",
                "edge_type": "replenishment_lane",
                "features": {"nominal_lead_time": 2.0},
            }
        ],
    }


def _entity(
    entity_id: str,
    entity_type: str,
    node_id: str,
    *,
    features: dict[str, Any],
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "node_id": node_id,
        "product_id": "sku-1",
        "features": features,
    }


def _event(t: int, *, observed: bool) -> dict[str, Any]:
    return {
        "event_id": f"shipment-report-{t}",
        "event_type": "shipment_report",
        "entity_id": "sku-1-at-store",
        "node_id": "store",
        "event_time": float(t),
        "report_time": float(t if observed else t + 1),
        "observed": observed,
        "features": {"progress": min(t / (len(DEMAND) - 1), 1.0)},
    }


def _inventory_state(t: int, quantity: float) -> tuple[float, float, float]:
    cumulative_demand = sum(DEMAND[: t + 1])
    delivered = quantity if t >= ARRIVAL_AT else 0.0
    net_inventory = 16.0 + delivered - cumulative_demand
    on_hand = max(net_inventory, 0.0)
    backlog = max(-net_inventory, 0.0)
    in_transit = quantity if BRANCH_AT <= t < ARRIVAL_AT else 0.0
    return on_hand, backlog, in_transit


def _step(t: int, quantity: float, *, stream_id: str) -> dict[str, Any]:
    on_hand, backlog, in_transit = _inventory_state(t, quantity)
    delayed_inventory = t == 3
    observed_on_hand = _inventory_state(t - 1, quantity)[0] if delayed_inventory else on_hand
    sales = min(DEMAND[t], observed_on_hand)
    latent_event = _event(t, observed=True)
    observation_event = _event(t, observed=not delayed_inventory)
    actions = []
    if t == BRANCH_AT:
        actions.append(
            {
                "action_type": "replenish",
                "actor_id": "store",
                "target_id": "supplier",
                "quantity": quantity,
                "features": {"arrival_day": ARRIVAL_AT, "item_id": "sku-1"},
            }
        )

    latent_entities = [
        _entity(
            "sku-1-at-store",
            "product_location",
            "store",
            features={
                "on_hand": on_hand,
                "backlog": backlog,
                "demand": DEMAND[t],
                "inventory_position": on_hand + in_transit - backlog,
                "service_level": float(backlog == 0.0),
            },
        ),
        _entity(
            "supplier-capacity",
            "supplier_state",
            "supplier",
            features={"available_capacity": 20.0 - in_transit, "lead_time": 2.0},
        ),
    ]
    observed_entities = deepcopy(latent_entities)
    observed_entities[0]["features"].update(
        {"on_hand": observed_on_hand, "demand": sales}
    )
    statuses = []
    if delayed_inventory:
        statuses.extend(
            [
                {
                    "path": "observation.entities.sku-1-at-store.on_hand",
                    "status": "delayed",
                    "age_steps": 1.0,
                    "reason": "one-period reporting lag",
                },
                {
                    "path": "observation.events.shipment-report-3",
                    "status": "delayed",
                    "age_steps": 1.0,
                    "reason": "event not yet reported",
                },
            ]
        )

    holding_cost = 0.2 * on_hand
    backlog_cost = 2.0 * backlog
    order_cost = 0.05 * quantity if t == BRANCH_AT else 0.0
    return {
        "t": t,
        "clock": float(t),
        "latent_state": {
            "entities": latent_entities,
            "edges": [
                {
                    "edge_id": "supplier-to-store",
                    "features": {"in_transit": in_transit, "lead_time": 2.0},
                }
            ],
            "global": {
                "period_cost": holding_cost + backlog_cost + order_cost,
                "fill_rate": float(backlog == 0.0),
            },
            "events": [latent_event],
        },
        "observation": {
            "entities": observed_entities,
            "edges": [
                {
                    "edge_id": "supplier-to-store",
                    "features": {"in_transit": in_transit, "lead_time": 2.0},
                }
            ],
            "global": {
                "period_cost": holding_cost + backlog_cost + order_cost,
                "fill_rate": float(backlog == 0.0),
            },
            "events": [observation_event] if observation_event["observed"] else [],
            "status": statuses,
        },
        "actions": actions,
        "outcomes": {
            "holding_cost": holding_cost,
            "backlog_cost": backlog_cost,
            "order_cost": order_cost,
        },
        "exogenous_digest": _digest(stream_id, t, DEMAND[t]),
        "terminal": t == len(DEMAND) - 1,
    }


def _record(split: str, branch_id: str, quantity: float) -> dict[str, Any]:
    group = f"{split}-paired-actions"
    stream_id = f"{split}-demand-stream"
    return {
        "schema_version": "supplychain-jepa.synthetic-episode.v1",
        "record_id": f"{group}-{branch_id}",
        "simulator": {
            "family": "tiny_inventory",
            "name": "project-authored deterministic inventory example",
            "version": "1",
            "source_repository": "this repository",
            "source_commit": "working-tree",
            "license": "repository terms",
        },
        "split": split,
        "episode": {
            "episode_id": f"{group}-{branch_id}",
            "world_id": group,
            "config_id": "two-echelon-toy-v1",
            "seed": 17 if split == "pretrain_train" else 23,
            "time_unit": "period",
            "horizon": len(DEMAND),
            "exogenous_stream_id": stream_id,
            "domain_parameters": {
                "initial_inventory": 16.0,
                "arrival_lag": ARRIVAL_AT - BRANCH_AT,
            },
        },
        "branch": {
            "branch_group_id": group,
            "branch_id": branch_id,
            "branch_at": float(BRANCH_AT),
            "common_random_numbers": True,
        },
        "objective_eligibility": {
            "state_jepa": True,
            "dynamics_jepa": True,
            "decision_heads": False,
        },
        "topology": _topology(),
        "steps": [_step(t, quantity, stream_id=stream_id) for t in range(len(DEMAND))],
        "provenance": {
            "adapter_version": "tiny-corpus-v1",
            "generator": "examples/training/generate_tiny_corpus.py",
            "causal_semantics": "paired actions share one pre-action state and demand stream",
            "observation_semantics": "sales are censored and one period has delayed reports",
            "source_files": [],
        },
    }


def records() -> list[dict[str, Any]]:
    return [
        _record("pretrain_train", "low-order", 4.0),
        _record("pretrain_train", "high-order", 14.0),
        _record("pretrain_dev", "low-order", 5.0),
        _record("pretrain_dev", "high-order", 13.0),
    ]


def write_corpus(output: Path) -> None:
    schema = json.loads(canonical_episode_schema().read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    payload = records()
    for record in payload:
        validator.validate(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in payload),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_corpus(args.output)
    print(json.dumps({"output": str(args.output), "records": len(records())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
