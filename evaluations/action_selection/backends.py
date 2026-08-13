"""Frozen State and adaptable Dynamics backends for action selection."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from supplychain_jepa import ModelBundle, load_pretrained
from supplychain_jepa.model import DynamicsPredictor, SequenceEncoder

from .contracts import HISTORY_FEATURES, ActionPlan, DecisionWorld


class ActionSelectionBackbone(ABC):
    """Interface that keeps State encoding frozen during downstream fitting."""

    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @abstractmethod
    def encode_worlds(
        self,
        worlds: Sequence[DecisionWorld],
        *,
        pretrained: bool,
        device: torch.device,
    ) -> Tensor: ...

    @abstractmethod
    def new_dynamics(
        self,
        *,
        pretrained: bool,
        device: torch.device,
        seed: int,
    ) -> nn.Module: ...

    @abstractmethod
    def serialize_plans(
        self,
        plans: Sequence[ActionPlan],
        *,
        decision_time: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]: ...


class _TinyStateEncoder(nn.Module):
    def __init__(self, feature_count: int, state_dim: int, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.projection = nn.Linear(feature_count, state_dim, bias=False)
        with torch.no_grad():
            self.projection.weight.normal_(0.0, 0.12, generator=generator)

    def forward(self, histories: Tensor) -> Tensor:
        scale = histories.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
        normalized = histories / scale
        return torch.tanh(self.projection(normalized)).mean(dim=1, keepdim=True)


class _TinyDynamics(nn.Module):
    def __init__(self, state_dim: int, *, seed: int, warm: bool) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.state = nn.Linear(state_dim, state_dim)
        self.action = nn.Linear(5, state_dim)
        self.output = nn.Sequential(nn.GELU(), nn.Linear(state_dim, state_dim))
        for parameter in self.parameters():
            parameter.data.normal_(0.0, 0.08, generator=generator)
        if warm:
            with torch.no_grad():
                diagonal = min(state_dim, 5)
                self.action.weight[:diagonal, :diagonal] += torch.eye(diagonal)

    def forward(
        self,
        state_latents: Tensor,
        actions: Tensor,
        horizons: Tensor,
        *,
        state_padding_mask: Tensor | None = None,
        action_padding_mask: Tensor | None = None,
    ) -> Tensor:
        del state_padding_mask
        valid = (
            torch.ones(actions.shape[:2], device=actions.device, dtype=actions.dtype)
            if action_padding_mask is None
            else (~action_padding_mask).to(actions.dtype)
        )
        quantity = (actions[..., 0] * valid).sum(dim=1)
        supplier_mix = (
            (actions[..., 4] * actions[..., 0] * valid).sum(dim=1)
            / quantity.clamp_min(1e-6)
        )
        order_count = valid.sum(dim=1)
        lead = (torch.expm1(actions[..., 7]) * valid).sum(dim=1) / order_count.clamp_min(
            1.0
        )
        action_features = torch.stack(
            (
                torch.log1p(quantity),
                supplier_mix,
                order_count,
                lead,
                torch.log1p(horizons[:, 0].float()),
            ),
            dim=-1,
        )
        future = self.state(state_latents[:, 0]) + self.action(action_features)
        return self.output(future).unsqueeze(1)


class TinyBackbone(ActionSelectionBackbone):
    """Small deterministic backend used by unit and synthetic smoke tests."""

    def __init__(self, history_features: int, state_dim: int, seed: int) -> None:
        if history_features != len(HISTORY_FEATURES):
            raise ValueError("history feature count does not match the contract")
        self._state_dim = state_dim
        self.seed = seed
        self.pretrained_state_encoder = _TinyStateEncoder(
            history_features, state_dim, seed=seed
        )
        self.random_state_encoder = _TinyStateEncoder(
            history_features, state_dim, seed=seed + 1
        )
        for encoder in (self.pretrained_state_encoder, self.random_state_encoder):
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def encode_worlds(
        self,
        worlds: Sequence[DecisionWorld],
        *,
        pretrained: bool,
        device: torch.device,
    ) -> Tensor:
        histories = torch.tensor(
            [world.public_history for world in worlds], dtype=torch.float32, device=device
        )
        encoder = (
            self.pretrained_state_encoder if pretrained else self.random_state_encoder
        ).to(device)
        with torch.no_grad():
            return encoder(histories).detach()

    def new_dynamics(
        self,
        *,
        pretrained: bool,
        device: torch.device,
        seed: int,
    ) -> nn.Module:
        return _TinyDynamics(
            self.state_dim,
            seed=self.seed + seed + (0 if pretrained else 100_000),
            warm=pretrained,
        ).to(device)

    def serialize_plans(
        self,
        plans: Sequence[ActionPlan],
        *,
        decision_time: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        return _serialize_plans(
            plans,
            decision_time=decision_time,
            action_type_id=75,
            device=device,
        )


class ModelBundleBackbone(ActionSelectionBackbone):
    """Adapter over the clean, verified :class:`supplychain_jepa.ModelBundle`."""

    def __init__(self, bundle: ModelBundle, *, seed: int = 9601) -> None:
        self.bundle = bundle
        self.seed = seed
        self.tensorizer = bundle.tensorizer_config
        self.model_config = bundle.model.config
        self.pretrained_encoder = deepcopy(bundle.model.online_encoder).cpu().eval()
        self.pretrained_dynamics = deepcopy(bundle.model.dynamics_predictor).cpu()

        self._set_seed(seed + 1)
        self.random_encoder = SequenceEncoder(self.model_config).cpu().eval()
        self.random_dynamics = DynamicsPredictor(self.model_config).cpu()
        for encoder in (self.pretrained_encoder, self.random_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

        feature_names = list(self.tensorizer["feature_names"])
        self.feature_index = {name: index for index, name in enumerate(feature_names)}
        self.metadata = self.tensorizer["metadata_vocabs"]
        self.status_ids = {
            "observed": 1,
            "delayed": 3,
            "missing": 4,
        }

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path | None = None,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        seed: int = 9601,
    ) -> ModelBundleBackbone:
        bundle = load_pretrained(
            source,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            device="cpu",
        )
        return cls(bundle, seed=seed)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def state_dim(self) -> int:
        return int(self.model_config.d_model)

    def _numeric(
        self,
        values: Tensor,
        present: Tensor,
        status: Tensor,
        age: Tensor,
        *,
        row: int,
        feature: str,
        value: float,
        status_name: str = "observed",
        feature_age: float = 0.0,
    ) -> None:
        column = self.feature_index[feature]
        mean = float(self.tensorizer["numeric_mean"][column])
        scale = max(float(self.tensorizer["numeric_std"][column]), 1e-6)
        values[row, column] = (float(value) - mean) / scale
        present[row, column] = True
        status[row, column] = self.status_ids[status_name]
        age[row, column] = float(feature_age)

    def _categorical(
        self,
        categorical: Tensor,
        present: Tensor,
        status: Tensor,
        age: Tensor,
        *,
        row: int,
        feature: str,
        text: str,
        status_name: str = "observed",
        feature_age: float = 0.0,
    ) -> None:
        column = self.feature_index[feature]
        categorical[row, column] = self.tensorizer["categorical_vocabs"][feature].get(
            text, 1
        )
        present[row, column] = True
        status[row, column] = self.status_ids[status_name]
        age[row, column] = float(feature_age)

    def _batch(self, worlds: Sequence[DecisionWorld], device: torch.device) -> dict[str, Tensor]:
        feature_count = int(self.model_config.feature_count)
        sequence_length = len(worlds[0].public_history) * 3
        matrices: dict[str, list[Tensor]] = {
            "values": [],
            "categorical_ids": [],
            "feature_present": [],
            "status_ids": [],
            "age": [],
        }
        vectors: dict[str, list[Tensor]] = {
            key: []
            for key in (
                "time_index",
                "clock",
                "family_id",
                "token_kind_id",
                "type_id",
                "entity_index",
                "product_index",
                "node_index",
                "source_index",
                "target_index",
                "alignment_index",
            )
        }
        family_id = self.metadata["family"].get("stockpyl", 1)
        kind_global = self.metadata["token_kind"].get("global", 1)
        kind_entity = self.metadata["token_kind"].get("entity", 1)
        type_global = self.metadata["token_type"].get("global_state", 1)
        type_supplier = self.metadata["token_type"].get("supplier_state", 1)

        for world in worlds:
            values = torch.zeros(sequence_length, feature_count)
            categorical = torch.zeros(sequence_length, feature_count, dtype=torch.long)
            present = torch.zeros(sequence_length, feature_count, dtype=torch.bool)
            status = torch.zeros(sequence_length, feature_count, dtype=torch.long)
            age = torch.zeros(sequence_length, feature_count)
            time_index: list[int] = []
            token_kind: list[int] = []
            token_type: list[int] = []
            entity_index: list[int] = []
            node_index: list[int] = []

            for time, public in enumerate(world.public_history):
                global_row = 3 * time
                numeric_global = {
                    "global::inventory_total_reported": public[0],
                    "global::backlog_total": public[1],
                    "global::demand_reported_this_period": public[2],
                    "global::sales_observed": public[3],
                    "global::pipeline_total": public[4],
                }
                for feature, value in numeric_global.items():
                    self._numeric(
                        values,
                        present,
                        status,
                        age,
                        row=global_row,
                        feature=feature,
                        value=value,
                    )
                self._categorical(
                    categorical,
                    present,
                    status,
                    age,
                    row=global_row,
                    feature="global::congestion_regime",
                    text="str:peak" if public[11] else "str:normal",
                )
                time_index.append(time)
                token_kind.append(kind_global)
                token_type.append(type_global)
                entity_index.append(0)
                node_index.append(0)

                for supplier_offset, supplier_row in enumerate((global_row + 1, global_row + 2)):
                    report_available = bool(public[9 + supplier_offset])
                    report_age = float(public[7 + supplier_offset])
                    self._numeric(
                        values,
                        present,
                        status,
                        age,
                        row=supplier_row,
                        feature="entity::capacity",
                        value=public[5 + supplier_offset],
                        status_name="observed" if report_available else "delayed",
                        feature_age=report_age,
                    )
                    self._numeric(
                        values,
                        present,
                        status,
                        age,
                        row=supplier_row,
                        feature="entity::report_age",
                        value=report_age,
                        status_name="observed" if report_available else "delayed",
                        feature_age=report_age,
                    )
                    self._categorical(
                        categorical,
                        present,
                        status,
                        age,
                        row=supplier_row,
                        feature="entity::report_available",
                        text="bool:true" if report_available else "bool:false",
                        status_name="observed" if report_available else "delayed",
                        feature_age=report_age,
                    )
                    time_index.append(time)
                    token_kind.append(kind_entity)
                    token_type.append(type_supplier)
                    entity_index.append(2 + supplier_offset)
                    node_index.append(2 + supplier_offset)

            matrices["values"].append(values)
            matrices["categorical_ids"].append(categorical)
            matrices["feature_present"].append(present)
            matrices["status_ids"].append(status)
            matrices["age"].append(age)
            vectors["time_index"].append(torch.tensor(time_index, dtype=torch.long))
            vectors["clock"].append(torch.tensor(time_index, dtype=torch.float32))
            vectors["family_id"].append(
                torch.full((sequence_length,), family_id, dtype=torch.long)
            )
            vectors["token_kind_id"].append(torch.tensor(token_kind, dtype=torch.long))
            vectors["type_id"].append(torch.tensor(token_type, dtype=torch.long))
            vectors["entity_index"].append(torch.tensor(entity_index, dtype=torch.long))
            vectors["node_index"].append(torch.tensor(node_index, dtype=torch.long))
            for key in ("product_index", "source_index", "target_index", "alignment_index"):
                vectors[key].append(torch.zeros(sequence_length, dtype=torch.long))

        batch = {
            f"context_{key}": torch.stack(items).to(device)
            for key, items in {**matrices, **vectors}.items()
        }
        batch["context_padding_mask"] = torch.zeros(
            len(worlds), sequence_length, dtype=torch.bool, device=device
        )
        return batch

    def encode_worlds(
        self,
        worlds: Sequence[DecisionWorld],
        *,
        pretrained: bool,
        device: torch.device,
    ) -> Tensor:
        encoder = self.pretrained_encoder if pretrained else self.random_encoder
        encoder = encoder.to(device).eval()
        batch = self._batch(worlds, device)
        with torch.no_grad():
            encoding = encoder.encode_with_special_tokens(batch, "context")
            valid = (~encoding.token_padding_mask).to(encoding.token_latents.dtype)
            pooled = (encoding.token_latents * valid.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        encoder.cpu()
        return pooled.detach().unsqueeze(1)

    def new_dynamics(
        self,
        *,
        pretrained: bool,
        device: torch.device,
        seed: int,
    ) -> nn.Module:
        self._set_seed(seed)
        template = self.pretrained_dynamics if pretrained else self.random_dynamics
        dynamics = deepcopy(template).to(device)
        dynamics.train()
        for parameter in dynamics.parameters():
            parameter.requires_grad_(True)
        return dynamics

    def serialize_plans(
        self,
        plans: Sequence[ActionPlan],
        *,
        decision_time: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        action_type = self.metadata["token_type"].get("replenishment", 1)
        return _serialize_plans(
            plans,
            decision_time=decision_time,
            action_type_id=action_type,
            device=device,
        )


def _serialize_plans(
    plans: Sequence[ActionPlan],
    *,
    decision_time: int,
    action_type_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    maximum = max(len(plan.orders) for plan in plans)
    actions = torch.zeros(len(plans), maximum, 8, dtype=torch.float32, device=device)
    padding = torch.ones(len(plans), maximum, dtype=torch.bool, device=device)
    for row, plan in enumerate(plans):
        for column, order in enumerate(plan.orders):
            target_index = 2 if order.supplier == "primary" else 3
            actions[row, column] = torch.tensor(
                (
                    order.quantity,
                    float(action_type_id),
                    float(decision_time),
                    4.0,
                    float(target_index),
                    5.0,
                    1.0,
                    float(np.log1p(order.nominal_lead_time)),
                ),
                device=device,
            )
            padding[row, column] = False
    return actions, padding
