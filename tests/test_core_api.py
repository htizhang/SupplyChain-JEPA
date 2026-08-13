from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest
import torch

from supplychain_jepa import (
    ModelBundle,
    compare_actions,
    encode_state,
    loading,
    predict_dynamics,
    predict_masked_state,
)
from supplychain_jepa.data import (
    TENSORIZER_FORMAT,
    EpisodeTensorizer,
    FittedTensorizerState,
)
from supplychain_jepa.model import ModelConfig, SupplyChainJEPACore

LOCAL_INDEX_FIELDS = (
    "entity_index",
    "product_index",
    "node_index",
    "source_index",
    "target_index",
    "alignment_index",
)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    batch: dict[str, torch.Tensor] = {}
    for prefix, length in (("context", 6), ("target", 7)):
        shape = (2, length, 5)
        batch[f"{prefix}_values"] = torch.randn(shape)
        batch[f"{prefix}_categorical_ids"] = torch.randint(0, 12, shape)
        batch[f"{prefix}_feature_present"] = torch.rand(shape) > 0.2
        batch[f"{prefix}_legal_observation_mask"] = batch[
            f"{prefix}_feature_present"
        ].clone()
        batch[f"{prefix}_status_ids"] = torch.randint(0, 8, shape)
        batch[f"{prefix}_age"] = torch.rand(shape)
        batch[f"{prefix}_time_index"] = torch.arange(length).repeat(2, 1)
        batch[f"{prefix}_clock"] = torch.arange(length).float().repeat(2, 1)
        batch[f"{prefix}_family_id"] = torch.ones((2, length), dtype=torch.long)
        batch[f"{prefix}_token_kind_id"] = torch.randint(1, 9, (2, length))
        batch[f"{prefix}_type_id"] = torch.randint(1, 11, (2, length))
        for name in LOCAL_INDEX_FIELDS:
            batch[f"{prefix}_{name}"] = torch.randint(0, 32, (2, length))
        padding = torch.zeros((2, length), dtype=torch.bool)
        padding[0, -2:] = True
        batch[f"{prefix}_padding_mask"] = padding
    return batch


def _bundle() -> ModelBundle:
    config = ModelConfig(
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
    model = SupplyChainJEPACore(config).eval()
    state = FittedTensorizerState(
        state_version=TENSORIZER_FORMAT,
        train_splits=(),
        feature_names=tuple(f"f{index}" for index in range(5)),
        feature_kinds=("numeric",) * 5,
        numeric_mean=(0.0,) * 5,
        numeric_std=(1.0,) * 5,
        categorical_vocabs={},
        target_categorical_vocabs={},
        metadata_vocabs={
            "family": {"<PAD>": 0, "<UNK>": 1},
            "time_unit": {"<PAD>": 0, "<UNK>": 1},
            "token_kind": {"<PAD>": 0, "<UNK>": 1},
            "token_type": {"<PAD>": 0, "<UNK>": 1},
        },
    )
    return ModelBundle(
        model=model,
        state_pool_predictor=None,
        semantic_concept_slot_predictor=None,
        grounding_head=torch.nn.Linear(24, 2),
        tensorizer=EpisodeTensorizer(state),
        config={"model": config.to_dict()},
        tensorizer_config=state.to_dict(),
        schema={},
        normalization={},
        release_manifest={},
        source=Path("."),
        revision=None,
    )


def test_shape_explicit_state_and_dynamics_wrappers() -> None:
    bundle = _bundle()
    batch = _batch()
    encoding = encode_state(bundle, batch)
    assert encoding.token_latents.shape == (2, 6, 24)
    assert encoding.belief_latent.shape == (2, 24)
    prediction = predict_masked_state(bundle, batch)
    assert prediction.shape == (2, 7, 24)

    actions = torch.randn(2, 4, 3)
    horizons = torch.tensor([[1.0, 4.0], [1.0, 4.0]])
    future = predict_dynamics(
        bundle,
        encoding.token_latents,
        actions,
        horizons,
        state_padding_mask=encoding.token_padding_mask,
    )
    assert future.shape == (2, 2, 24)


def test_compare_actions_returns_latent_contrasts_not_fake_preferences() -> None:
    bundle = _bundle()
    encoding = encode_state(bundle, _batch())
    candidates = torch.randn(2, 3, 4, 3)
    comparison = compare_actions(
        bundle,
        encoding.token_latents,
        candidates,
        torch.tensor([[1.0, 4.0], [1.0, 4.0]]),
        state_padding_mask=encoding.token_padding_mask,
    )
    assert comparison.predicted_latents.shape == (2, 3, 2, 24)
    assert comparison.pairwise_distances.shape == (2, 3, 3, 2)
    diagonal = comparison.pairwise_distances.diagonal(dim1=1, dim2=2)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal))
    assert torch.allclose(
        comparison.pairwise_distances,
        comparison.pairwise_distances.transpose(1, 2),
    )

    with pytest.raises(ValueError, match="action dimension"):
        compare_actions(
            bundle,
            encoding.token_latents,
            torch.randn(2, 3, 4, 2),
            torch.ones(2, 1),
        )


def test_hub_snapshot_download_includes_every_checksum_declared_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    fake_hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    fake_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved = loading._resolve_snapshot(
        None,
        repo_id="owner/model",
        revision=loading.DEFAULT_REVISION,
        token=None,
        local_files_only=False,
    )
    assert resolved == tmp_path.resolve()
    patterns = set(captured["allow_patterns"])
    assert patterns == loading._SNAPSHOT_FILES
    assert {
        "README.md",
        "load_model.py",
        "modeling_supplychain_jepa.py",
        "CHECKSUMS.sha256",
    }.issubset(patterns)


def test_release_verification_accepts_content_addressed_snapshot_symlinks(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    declarations: list[str] = []
    for name in sorted(loading._REQUIRED_FILES - {"CHECKSUMS.sha256"}):
        payload = name.encode()
        (snapshot / name).write_bytes(payload)
        declarations.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
    readme = b"model card"
    (blobs / "readme").write_bytes(readme)
    (snapshot / "README.md").symlink_to(blobs / "readme")
    declarations.append(f"{hashlib.sha256(readme).hexdigest()}  README.md")
    (snapshot / "CHECKSUMS.sha256").write_text(
        "\n".join(declarations) + "\n", encoding="utf-8"
    )

    loading.verify_release(snapshot)
