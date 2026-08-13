from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from supplychain_jepa import DEFAULT_REVISION, load_pretrained
from supplychain_jepa.loading import verify_release
from supplychain_jepa.model import ModelConfig, SupplyChainJEPACore

pytestmark = pytest.mark.private_model

_RELEASE_DIRECTORY = os.environ.get("SUPPLYCHAIN_JEPA_RELEASE_DIR")
if not _RELEASE_DIRECTORY:
    pytest.skip(
        "SUPPLYCHAIN_JEPA_RELEASE_DIR is required for private model parity tests",
        allow_module_level=True,
    )

RELEASE = Path(_RELEASE_DIRECTORY).expanduser().resolve()
RELEASE_MODEL = RELEASE / "modeling_supplychain_jepa.py"


def _released_module():
    spec = importlib.util.spec_from_file_location("released_supplychain_jepa", RELEASE_MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_config(cls):
    return cls(
        feature_count=5,
        categorical_vocab_size=12,
        family_vocab_size=5,
        token_kind_vocab_size=9,
        type_vocab_size=11,
        d_model=24,
        nhead=4,
        encoder_layers=1,
        state_predictor_layers=1,
        dynamics_predictor_layers=1,
        dim_feedforward=48,
        dropout=0.0,
        action_dim=3,
        local_index_capacity=128,
    )


def test_clean_model_preserves_released_architecture_exactly() -> None:
    released = _released_module()
    torch.manual_seed(17)
    expected = released.SupplyChainJEPACore(_tiny_config(released.ModelConfig))
    torch.manual_seed(17)
    actual = SupplyChainJEPACore(_tiny_config(ModelConfig))
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in actual.state_dict().items():
        assert tensor.shape == expected.state_dict()[name].shape, name
        assert torch.equal(tensor, expected.state_dict()[name]), name


def test_verified_local_release_loads_strictly_and_frozen() -> None:
    verify_release(RELEASE)
    bundle = load_pretrained(RELEASE)
    assert sum(parameter.numel() for parameter in bundle.model.parameters()) == 73_629_696
    assert bundle.model.config.d_model == 384
    assert bundle.model.config.encoder_layers == 6
    assert bundle.model.config.dynamics_predictor_layers == 4
    assert bundle.tensorizer.state.feature_count == 309
    assert bundle.release_manifest["parameter_counts"]["core"] == 73_629_696
    assert bundle.revision is None
    assert DEFAULT_REVISION == "493786fda0855b72f61235068c62cc7b2a11fbec"
    for module in (
        bundle.model,
        bundle.state_pool_predictor,
        bundle.semantic_concept_slot_predictor,
        bundle.grounding_head,
    ):
        assert module is not None
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())

    released_core = load_file(RELEASE / "model.safetensors", device="cpu")
    assert bundle.model.state_dict().keys() == released_core.keys()
    for name, value in bundle.model.state_dict().items():
        assert torch.equal(value.cpu(), released_core[name]), name

    released_aux = load_file(RELEASE / "state_auxiliary.safetensors", device="cpu")
    loaded_aux = {
        **{
            f"state_pool_predictor.{name}": value
            for name, value in bundle.state_pool_predictor.state_dict().items()
        },
        **{
            f"semantic_concept_slot_predictor.{name}": value
            for name, value in bundle.semantic_concept_slot_predictor.state_dict().items()
        },
        **{
            f"grounding_head.{name}": value
            for name, value in bundle.grounding_head.state_dict().items()
        },
    }
    assert loaded_aux.keys() == released_aux.keys()
    for name, value in loaded_aux.items():
        assert torch.equal(value.cpu(), released_aux[name]), name

    explicit = load_pretrained(RELEASE, revision=DEFAULT_REVISION)
    assert explicit.revision == DEFAULT_REVISION
