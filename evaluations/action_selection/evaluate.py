"""Bounded five-split action-selection evaluation and JSON reporting."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backends import ActionSelectionBackbone
from .contracts import ActionPlan, DecisionWorld, EvaluationConfig
from .reconstruction import build_world_splits, generate_reconstruction_worlds

RETAINED_REFERENCE = {
    "provenance": "retained_prose_and_summary_only",
    "pretrained_regret_mean": 5.618,
    "pretrained_regret_sd": 0.559,
    "matched_random_regret_mean": 5.972,
    "relative_regret_reduction": 0.059,
    "pretrained_split_wins": 4,
    "total_splits": 5,
    "extra_trees_regret_mean": 9.274,
    "extra_trees_regret_sd": 2.547,
}


class PlanQualityModel(nn.Module):
    """Adaptable Dynamics predictor followed by a scalar plan-cost readout."""

    def __init__(self, dynamics: nn.Module, state_dim: int) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.readout = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(
        self,
        state: Tensor,
        actions: Tensor,
        horizons: Tensor,
        action_padding_mask: Tensor,
    ) -> Tensor:
        future = self.dynamics(
            state,
            actions,
            horizons,
            action_padding_mask=action_padding_mask,
        )
        return self.readout(future[:, -1]).squeeze(-1)


def _set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, threads))


def _history_matrix(worlds: Sequence[DecisionWorld]) -> np.ndarray:
    return np.asarray([world.public_history for world in worlds], dtype=np.float32)


def _cost_matrix(worlds: Sequence[DecisionWorld]) -> np.ndarray:
    return np.asarray(
        [[world.expected_cost("regular"), world.expected_cost("split")] for world in worlds],
        dtype=np.float32,
    )


def _plans(worlds: Sequence[DecisionWorld], action_index: int) -> list[ActionPlan]:
    if action_index == 0:
        return [world.regular_plan for world in worlds]
    if action_index == 1:
        return [world.split_plan for world in worlds]
    raise ValueError("action_index must be 0 or 1")


def _paired_rows(
    state: Tensor,
    worlds: Sequence[DecisionWorld],
    backend: ActionSelectionBackbone,
    *,
    horizon: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    rows = len(worlds)
    repeated_state = state[:, None].expand(-1, 2, -1, -1).reshape(
        rows * 2, state.shape[1], state.shape[2]
    )
    flattened_plans = [
        plan
        for world in worlds
        for plan in (world.regular_plan, world.split_plan)
    ]
    actions, action_padding = backend.serialize_plans(
        flattened_plans,
        decision_time=16,
        device=device,
    )
    horizons = torch.full((rows * 2, 1), float(horizon), device=device)
    costs = torch.tensor(_cost_matrix(worlds), device=device).reshape(-1)
    pair_index = torch.arange(rows, device=device).repeat_interleave(2)
    return repeated_state, actions, horizons, action_padding, costs, pair_index


def _train_candidate(
    *,
    dynamics: nn.Module,
    state_dim: int,
    state: Tensor,
    worlds: Sequence[DecisionWorld],
    backend: ActionSelectionBackbone,
    config: EvaluationConfig,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> PlanQualityModel:
    _set_seed(seed, config.threads)
    model = PlanQualityModel(dynamics, state_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    repeated_state, actions, horizons, padding, costs, _ = _paired_rows(
        state,
        worlds,
        backend,
        horizon=config.reconstruction.horizon,
        device=device,
    )
    target_center = costs.view(-1, 2).mean(dim=1, keepdim=True).repeat(1, 2).reshape(-1)
    centered = costs - target_center
    row_count = costs.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(config.epochs):
        permutation = torch.randperm(row_count, generator=generator)
        for start in range(0, row_count, config.batch_size):
            indices = permutation[start : start + config.batch_size].to(device)
            prediction = model(
                repeated_state.index_select(0, indices),
                actions.index_select(0, indices),
                horizons.index_select(0, indices),
                padding.index_select(0, indices),
            )
            loss = F.smooth_l1_loss(prediction, centered.index_select(0, indices))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


@torch.no_grad()
def _model_predictions(
    model: PlanQualityModel,
    state: Tensor,
    worlds: Sequence[DecisionWorld],
    backend: ActionSelectionBackbone,
    config: EvaluationConfig,
    device: torch.device,
) -> np.ndarray:
    repeated_state, actions, horizons, padding, _, _ = _paired_rows(
        state,
        worlds,
        backend,
        horizon=config.reconstruction.horizon,
        device=device,
    )
    return (
        model(repeated_state, actions, horizons, padding)
        .reshape(len(worlds), 2)
        .cpu()
        .numpy()
    )


def _regret(costs: np.ndarray, prediction: np.ndarray) -> tuple[float, list[int]]:
    choices = np.asarray(prediction).argmin(axis=1)
    realized = costs[np.arange(len(costs)), choices]
    oracle = costs.min(axis=1)
    return float(np.mean(realized - oracle)), choices.astype(int).tolist()


def _fit_extra_trees(
    fit_worlds: Sequence[DecisionWorld],
    *,
    estimators: int,
    seed: int,
) -> Any:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "the public-history comparator requires the project 'eval' extra"
        ) from error
    model = ExtraTreesRegressor(
        n_estimators=estimators,
        min_samples_leaf=max(2, len(fit_worlds) // 40),
        max_features=0.75,
        n_jobs=1,
        random_state=seed,
    )
    history = _history_matrix(fit_worlds).reshape(len(fit_worlds), -1)
    action = np.tile(np.eye(2, dtype=np.float32), (len(fit_worlds), 1))
    features = np.concatenate((np.repeat(history, 2, axis=0), action), axis=1)
    model.fit(features, _cost_matrix(fit_worlds).reshape(-1))
    return model


def _extra_trees_predictions(model: Any, worlds: Sequence[DecisionWorld]) -> np.ndarray:
    history = _history_matrix(worlds).reshape(len(worlds), -1)
    action = np.tile(np.eye(2, dtype=np.float32), (len(worlds), 1))
    features = np.concatenate((np.repeat(history, 2, axis=0), action), axis=1)
    return np.asarray(model.predict(features), dtype=np.float64).reshape(len(worlds), 2)


def _selected_model(
    *,
    pretrained: bool,
    fit_worlds: Sequence[DecisionWorld],
    select_worlds: Sequence[DecisionWorld],
    fit_state: Tensor,
    select_state: Tensor,
    backend: ActionSelectionBackbone,
    config: EvaluationConfig,
    seed: int,
    device: torch.device,
) -> tuple[PlanQualityModel, float, dict[str, float]]:
    candidates: list[tuple[float, float, PlanQualityModel]] = []
    dev_regrets: dict[str, float] = {}
    for learning_rate in config.learning_rates:
        dynamics = backend.new_dynamics(
            pretrained=pretrained,
            device=device,
            seed=seed,
        )
        model = _train_candidate(
            dynamics=dynamics,
            state_dim=backend.state_dim,
            state=fit_state,
            worlds=fit_worlds,
            backend=backend,
            config=config,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        prediction = _model_predictions(
            model, select_state, select_worlds, backend, config, device
        )
        regret, _ = _regret(_cost_matrix(select_worlds), prediction)
        dev_regrets[f"{learning_rate:.0e}"] = regret
        candidates.append((regret, learning_rate, deepcopy(model).cpu()))
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, learning_rate, selected = candidates[0]
    return selected.to(device).eval(), learning_rate, dev_regrets


def _split_payload(
    *,
    split_index: int,
    salt: int,
    split: Any,
    pretrained_regret: float,
    random_regret: float,
    extra_trees_regret: float,
    pretrained_choice: list[int],
    random_choice: list[int],
    extra_trees_choice: list[int],
    pretrained_lr: float,
    random_lr: float,
    pretrained_dev: dict[str, float],
    random_dev: dict[str, float],
) -> dict[str, Any]:
    return {
        "format": "supplychain-jepa.action-selection.reconstruction.v1",
        "provenance": {
            "kind": "clean_reconstruction",
            "historical_runner_retained": False,
            "historical_artifacts_reused": False,
        },
        "split_index": split_index,
        "split_salt": salt,
        "worlds": asdict(split),
        "selection": {
            "pretrained_learning_rate": pretrained_lr,
            "matched_random_learning_rate": random_lr,
            "pretrained_dev_regret_by_learning_rate": pretrained_dev,
            "matched_random_dev_regret_by_learning_rate": random_dev,
        },
        "regret": {
            "pretrained": pretrained_regret,
            "matched_random": random_regret,
            "extra_trees": extra_trees_regret,
        },
        "choices": {
            "pretrained": pretrained_choice,
            "matched_random": random_choice,
            "extra_trees": extra_trees_choice,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_evaluation(
    config: EvaluationConfig,
    *,
    backend: ActionSelectionBackbone,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the bounded reconstruction and write one JSON file per world split."""

    config.validate()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    worlds = generate_reconstruction_worlds(config.reconstruction)
    world_by_id = {world.world_id: world for world in worlds}
    world_index = {world.world_id: index for index, world in enumerate(worlds)}
    pretrained_all = backend.encode_worlds(worlds, pretrained=True, device=device)
    random_all = backend.encode_worlds(worlds, pretrained=False, device=device)
    split_payloads: list[dict[str, Any]] = []

    for split_index, salt in enumerate(config.split_salts):
        split = build_world_splits(world_by_id, salt=salt)
        fit = [world_by_id[world_id] for world_id in split.fit]
        select = [world_by_id[world_id] for world_id in split.select]
        report = [world_by_id[world_id] for world_id in split.report]
        seed = config.reconstruction.seed + 1009 * split_index

        def state_rows(values: Tensor, ids: tuple[str, ...]) -> Tensor:
            indices = torch.tensor(
                [world_index[world_id] for world_id in ids],
                dtype=torch.long,
                device=values.device,
            )
            return values.index_select(0, indices)

        pretrained_fit = state_rows(pretrained_all, split.fit)
        pretrained_select = state_rows(pretrained_all, split.select)
        pretrained_report = state_rows(pretrained_all, split.report)
        random_fit = state_rows(random_all, split.fit)
        random_select = state_rows(random_all, split.select)
        random_report = state_rows(random_all, split.report)

        pretrained_model, pretrained_lr, pretrained_dev = _selected_model(
            pretrained=True,
            fit_worlds=fit,
            select_worlds=select,
            fit_state=pretrained_fit,
            select_state=pretrained_select,
            backend=backend,
            config=config,
            seed=seed,
            device=device,
        )
        random_model, random_lr, random_dev = _selected_model(
            pretrained=False,
            fit_worlds=fit,
            select_worlds=select,
            fit_state=random_fit,
            select_state=random_select,
            backend=backend,
            config=config,
            seed=seed,
            device=device,
        )
        report_costs = _cost_matrix(report)
        pretrained_prediction = _model_predictions(
            pretrained_model,
            pretrained_report,
            report,
            backend,
            config,
            device,
        )
        random_prediction = _model_predictions(
            random_model, random_report, report, backend, config, device
        )
        extra_trees = _fit_extra_trees(
            fit, estimators=config.extra_trees_estimators, seed=seed
        )
        extra_trees_prediction = _extra_trees_predictions(extra_trees, report)
        pretrained_regret, pretrained_choice = _regret(
            report_costs, pretrained_prediction
        )
        random_regret, random_choice = _regret(report_costs, random_prediction)
        extra_trees_regret, extra_trees_choice = _regret(
            report_costs, extra_trees_prediction
        )
        payload = _split_payload(
            split_index=split_index,
            salt=salt,
            split=split,
            pretrained_regret=pretrained_regret,
            random_regret=random_regret,
            extra_trees_regret=extra_trees_regret,
            pretrained_choice=pretrained_choice,
            random_choice=random_choice,
            extra_trees_choice=extra_trees_choice,
            pretrained_lr=pretrained_lr,
            random_lr=random_lr,
            pretrained_dev=pretrained_dev,
            random_dev=random_dev,
        )
        _write_json(output_root / f"split_{split_index:02d}.json", payload)
        split_payloads.append(payload)

    pretrained = [item["regret"]["pretrained"] for item in split_payloads]
    random_values = [item["regret"]["matched_random"] for item in split_payloads]
    extra_trees_values = [item["regret"]["extra_trees"] for item in split_payloads]
    pretrained_mean = mean(pretrained)
    random_mean = mean(random_values)
    extra_trees_mean = mean(extra_trees_values)
    relative_reduction = (
        (random_mean - pretrained_mean) / random_mean if random_mean > 0.0 else 0.0
    )
    wins = sum(
        left < right for left, right in zip(pretrained, random_values, strict=True)
    )
    gates = {
        "pretrained_regret_within_0_6_of_retained_reference": abs(
            pretrained_mean - RETAINED_REFERENCE["pretrained_regret_mean"]
        )
        <= 0.6,
        "matched_random_regret_within_0_6_of_retained_reference": abs(
            random_mean - RETAINED_REFERENCE["matched_random_regret_mean"]
        )
        <= 0.6,
        "relative_reduction_at_least_5_percent": relative_reduction >= 0.05,
        "pretrained_wins_at_least_4_of_5_splits": wins >= 4,
        "extra_trees_within_2_6_of_retained_reference": abs(
            extra_trees_mean - RETAINED_REFERENCE["extra_trees_regret_mean"]
        )
        <= 2.6,
    }
    summary = {
        "format": "supplychain-jepa.action-selection.reconstruction.v1",
        "provenance": {
            "kind": "clean_reconstruction",
            "historical_runner_retained": False,
            "historical_artifacts_reused": False,
            "interpretation": (
                "Fresh results from a documented-contract reconstruction; not an "
                "exact rerun of the missing historical executable."
            ),
        },
        "contract": {
            "lookback": config.reconstruction.lookback,
            "horizon": config.reconstruction.horizon,
            "candidate_plans": ["regular", "split"],
            "quantity_matched": True,
            "common_random_numbers": True,
            "continuation_policy": "no_further_orders",
            "world_split_salts": list(config.split_salts),
        },
        "training": {
            "state_encoder": "frozen",
            "dynamics_predictor": "supervised_adaptation",
            "readout": "zero_initialized_scalar_plan_quality",
            "optimizer": "AdamW",
            "learning_rates": list(config.learning_rates),
            "epochs": config.epochs,
        },
        "metrics": {
            "pretrained_regret_mean": pretrained_mean,
            "pretrained_regret_sd": stdev(pretrained),
            "matched_random_regret_mean": random_mean,
            "matched_random_regret_sd": stdev(random_values),
            "extra_trees_regret_mean": extra_trees_mean,
            "extra_trees_regret_sd": stdev(extra_trees_values),
            "relative_regret_reduction": relative_reduction,
            "pretrained_split_wins": wins,
            "total_splits": len(split_payloads),
        },
        "retained_reference": RETAINED_REFERENCE,
        "reproduction_gates": gates,
        "all_reproduction_gates_passed": all(gates.values()),
        "splits": split_payloads,
    }
    normalized = json.loads(json.dumps(summary, sort_keys=True))
    _write_json(output_root / "summary.json", normalized)
    return normalized
