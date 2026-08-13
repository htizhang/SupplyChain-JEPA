"""Typed contracts for the clean action-selection reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

HISTORY_FEATURES = (
    "inventory_reported",
    "backlog_reported",
    "demand_reported",
    "sales_observed",
    "pipeline_reported",
    "primary_capacity_reported",
    "secondary_capacity_reported",
    "primary_report_age",
    "secondary_report_age",
    "primary_report_available",
    "secondary_report_available",
    "congestion_reported",
)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for the self-contained Pool-B6-style reconstruction."""

    worlds: int = 256
    lookback: int = 16
    horizon: int = 4
    future_replications: int = 8
    seed: int = 9601
    report_impairment_penalty: float = 0.16
    report_congestion_penalty: float = 0.07
    report_independent_dropout: float = 0.10
    semi_markov_dwell_scale: float = 1.0

    def validate(self) -> None:
        if self.worlds < 1:
            raise ValueError("worlds must be positive")
        if self.lookback != 16:
            raise ValueError("the retained contract requires a 16-step lookback")
        if self.horizon != 4:
            raise ValueError("the retained contract requires a four-period horizon")
        if self.future_replications < 2:
            raise ValueError("future_replications must be at least 2")
        rates = (
            self.report_impairment_penalty,
            self.report_congestion_penalty,
            self.report_independent_dropout,
        )
        if any(not 0.0 <= value < 1.0 for value in rates):
            raise ValueError("report impairment rates must lie in [0, 1)")
        if sum(rates) >= 1.0:
            raise ValueError("combined report impairment must remain below one")
        if self.semi_markov_dwell_scale <= 0.0:
            raise ValueError("semi_markov_dwell_scale must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed downstream adaptation and evaluation recipe."""

    reconstruction: ReconstructionConfig = ReconstructionConfig()
    split_salts: tuple[int, ...] = (17011, 17029, 17047, 17071, 17093)
    learning_rates: tuple[float, ...] = (3e-4, 1e-4)
    epochs: int = 60
    batch_size: int = 128
    weight_decay: float = 1e-4
    extra_trees_estimators: int = 300
    device: str = "cpu"
    threads: int = 1

    def validate(self) -> None:
        self.reconstruction.validate()
        if self.reconstruction.worlds < 10:
            raise ValueError("evaluation requires at least ten worlds")
        if len(self.split_salts) != 5 or len(set(self.split_salts)) != 5:
            raise ValueError("exactly five unique world-split salts are required")
        if self.learning_rates != (3e-4, 1e-4):
            raise ValueError("the retained learning-rate screen is {3e-4, 1e-4}")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.extra_trees_estimators < 1:
            raise ValueError("extra_trees_estimators must be positive")
        if self.threads < 1:
            raise ValueError("threads must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Order:
    supplier: str
    quantity: float
    nominal_lead_time: int


@dataclass(frozen=True)
class ActionPlan:
    name: str
    orders: tuple[Order, ...]

    @property
    def total_quantity(self) -> float:
        return sum(order.quantity for order in self.orders)


@dataclass(frozen=True)
class PairedFutureCost:
    replication: int
    noise_digest: str
    regular_cost: float
    split_cost: float


@dataclass(frozen=True)
class DecisionWorld:
    world_id: str
    public_history: tuple[tuple[float, ...], ...]
    regular_plan: ActionPlan
    split_plan: ActionPlan
    paired_future_costs: tuple[PairedFutureCost, ...]
    continuation_policy: str = "no_further_orders"

    def expected_cost(self, plan: str) -> float:
        if plan not in {"regular", "split"}:
            raise ValueError(f"unknown plan {plan!r}")
        values = [
            item.regular_cost if plan == "regular" else item.split_cost
            for item in self.paired_future_costs
        ]
        return sum(values) / len(values)


@dataclass(frozen=True)
class WorldSplit:
    fit: tuple[str, ...]
    select: tuple[str, ...]
    report: tuple[str, ...]
