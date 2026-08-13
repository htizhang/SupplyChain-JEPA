from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from supplychain_jepa.losses import masked_latent_loss, paired_difference_loss
from supplychain_jepa.model import ModelConfig, SupplyChainJEPACore
from supplychain_jepa.schemas import canonical_episode_schema
from supplychain_jepa.training import _training_config, build_parser

ROOT = Path(__file__).resolve().parents[1]
LOCAL_INDEX_FIELDS = (
    "entity_index",
    "product_index",
    "node_index",
    "source_index",
    "target_index",
    "alignment_index",
)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(9)
    batch: dict[str, torch.Tensor] = {}
    for prefix, length in (("context", 5), ("target", 4)):
        shape = (2, length, 3)
        batch[f"{prefix}_values"] = torch.randn(shape)
        batch[f"{prefix}_categorical_ids"] = torch.randint(0, 8, shape)
        batch[f"{prefix}_feature_present"] = torch.ones(shape, dtype=torch.bool)
        batch[f"{prefix}_legal_observation_mask"] = torch.ones(shape, dtype=torch.bool)
        batch[f"{prefix}_status_ids"] = torch.ones(shape, dtype=torch.long)
        batch[f"{prefix}_age"] = torch.zeros(shape)
        batch[f"{prefix}_time_index"] = torch.arange(length).repeat(2, 1)
        batch[f"{prefix}_clock"] = torch.arange(length).float().repeat(2, 1)
        batch[f"{prefix}_family_id"] = torch.ones((2, length), dtype=torch.long)
        batch[f"{prefix}_token_kind_id"] = torch.ones((2, length), dtype=torch.long)
        batch[f"{prefix}_type_id"] = torch.ones((2, length), dtype=torch.long)
        for name in LOCAL_INDEX_FIELDS:
            batch[f"{prefix}_{name}"] = torch.ones((2, length), dtype=torch.long)
        batch[f"{prefix}_padding_mask"] = torch.zeros((2, length), dtype=torch.bool)
    return batch


def _model() -> SupplyChainJEPACore:
    return SupplyChainJEPACore(
        ModelConfig(
            feature_count=3,
            categorical_vocab_size=8,
            family_vocab_size=3,
            token_kind_vocab_size=4,
            type_vocab_size=4,
            d_model=16,
            nhead=4,
            encoder_layers=1,
            state_predictor_layers=1,
            dynamics_predictor_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            action_dim=3,
            local_index_capacity=32,
        )
    )


def test_split_configs_compose_without_objective_overriding_architecture() -> None:
    state = _training_config(ROOT / "configs/model.json", ROOT / "configs/state.json")
    dynamics = _training_config(
        ROOT / "configs/model.json", ROOT / "configs/dynamics.json"
    )
    assert state["model"] == dynamics["model"]
    assert "state_training" in state and "dynamics_training" not in state
    assert "dynamics_training" in dynamics and "state_training" not in dynamics
    state_args = build_parser().parse_args(
        ["state", "--data", "episodes.jsonl", "--output-dir", "output"]
    )
    dynamics_args = build_parser().parse_args(
        [
            "dynamics",
            "--data",
            "episodes.jsonl",
            "--output-dir",
            "output",
            "--checkpoint",
            "state.pt",
        ]
    )
    assert Path(state_args.config).name == "state.json"
    assert Path(dynamics_args.config).name == "dynamics.json"
    assert Path(state_args.model_config).name == "model.json"


def test_module_training_entrypoint_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "supplychain_jepa.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{state,dynamics}" in completed.stdout


def test_raw_episode_schema_is_packaged() -> None:
    schema = json.loads(canonical_episode_schema().read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == (
        "supplychain-jepa.synthetic-episode.v1"
    )
    assert {"episode", "topology", "steps", "objective_eligibility"}.issubset(
        schema["required"]
    )


def test_state_then_frozen_dynamics_objectives_backpropagate() -> None:
    model = _model().train()
    batch = _batch()
    predicted = model.predict_state(batch)
    target, _ = model.encode_target(batch)
    state_loss = masked_latent_loss(predicted, target, kind="smooth_l1")
    state_loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.online_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    model.zero_grad(set_to_none=True)
    model.freeze_encoders_for_dynamics(source="target")
    state_memory = model.canonical_state_latents(batch)
    actions = torch.randn(2, 2, 3)
    horizons = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    future = model.dynamics_predictor(state_memory, actions, horizons)
    paired = paired_difference_loss(future[0], future[1], target[0, :2], target[1, :2])
    paired.backward()
    assert all(
        parameter.grad is None for parameter in model.online_encoder.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in model.dynamics_predictor.parameters()
    )
