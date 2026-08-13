"""Verified loading from a local release directory or a pinned Hub snapshot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .bundle import ModelBundle
from .data import TENSORIZER_FORMAT, EpisodeTensorizer, FittedTensorizerState
from .model import (
    ModelConfig,
    SemanticConceptSlotPredictor,
    StatePoolPredictor,
    SupplyChainJEPACore,
)

DEFAULT_MODEL_ID = "htizhang/SupplyChain-JEPA"
DEFAULT_REVISION = "493786fda0855b72f61235068c62cc7b2a11fbec"
_REQUIRED_FILES = frozenset(
    {
        "CHECKSUMS.sha256",
        "config.json",
        "model.safetensors",
        "normalization.json",
        "release_manifest.json",
        "schema.json",
        "state_auxiliary.safetensors",
        "tensorizer.json",
    }
)
_CHECKSUM_DECLARED_FILES = frozenset(
    {
        "README.md",
        "config.json",
        "load_model.py",
        "model.safetensors",
        "modeling_supplychain_jepa.py",
        "normalization.json",
        "release_manifest.json",
        "requirements.txt",
        "schema.json",
        "state_auxiliary.safetensors",
        "tensorizer.json",
    }
)
_SNAPSHOT_FILES = _REQUIRED_FILES | _CHECKSUM_DECLARED_FILES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(directory: str | Path) -> None:
    """Verify every declared file and the release's component parameter counts."""

    root = Path(directory).expanduser().resolve()
    missing = sorted(name for name in _REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete model release; missing {missing}")
    declarations: dict[str, str] = {}
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, separator, name = line.partition("  ")
        if not separator or not expected or not name:
            raise ValueError(f"Invalid checksum declaration: {line!r}")
        relative = Path(name)
        candidate = root / relative
        # Hub snapshots commonly store files as symlinks into a content-addressed
        # blob cache. Validate the declared path lexically, then follow the
        # symlink only to hash its contents. Requiring the resolved target to
        # remain below ``root`` would reject otherwise valid Hub snapshots.
        if relative.is_absolute() or ".." in relative.parts or not candidate.is_file():
            raise ValueError(f"Unsafe or missing checksum path: {name!r}")
        declarations[name] = expected
        actual = _sha256(candidate)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}: {actual} != {expected}")
    undeclared = sorted(_REQUIRED_FILES - {"CHECKSUMS.sha256"} - declarations.keys())
    if undeclared:
        raise ValueError(f"Required release files lack checksums: {undeclared}")


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _tensorizer(config: Mapping[str, Any]) -> EpisodeTensorizer:
    state = FittedTensorizerState(
        state_version=str(config.get("format", TENSORIZER_FORMAT)),
        train_splits=(),
        feature_names=tuple(str(value) for value in config["feature_names"]),
        feature_kinds=tuple(str(value) for value in config["feature_kinds"]),
        numeric_mean=tuple(float(value) for value in config["numeric_mean"]),
        numeric_std=tuple(float(value) for value in config["numeric_std"]),
        categorical_vocabs={
            str(feature): {str(value): int(index) for value, index in vocab.items()}
            for feature, vocab in config["categorical_vocabs"].items()
        },
        target_categorical_vocabs={
            str(feature): {str(value): int(index) for value, index in vocab.items()}
            for feature, vocab in config["target_categorical_vocabs"].items()
        },
        metadata_vocabs={
            str(name): {str(value): int(index) for value, index in vocab.items()}
            for name, vocab in config["metadata_vocabs"].items()
        },
    )
    return EpisodeTensorizer(state)


def _resolve_snapshot(
    source: str | Path | None,
    *,
    repo_id: str,
    revision: str,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    if source is not None:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Model release directory does not exist: {root}")
        return root
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise ImportError(
            "huggingface_hub is required when no local release directory is supplied"
        ) from error
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=sorted(_SNAPSHOT_FILES),
        )
    ).resolve()


def load_pretrained(
    source: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    verify: bool = True,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> ModelBundle:
    """Load a frozen release, pinned to the immutable published revision by default."""

    effective_revision = revision or DEFAULT_REVISION
    root = _resolve_snapshot(
        source,
        repo_id=repo_id,
        revision=effective_revision,
        token=token,
        local_files_only=local_files_only,
    )
    if verify:
        verify_release(root)
    config = _json(root / "config.json")
    tensorizer_config = _json(root / "tensorizer.json")
    model_config = ModelConfig(**config["model"])
    model = SupplyChainJEPACore(model_config)
    model.load_state_dict(
        load_file(root / config["weights"]["core"], device="cpu"), strict=True
    )

    raw_aux = load_file(root / config["weights"]["state_auxiliary"], device="cpu")
    pool_cfg = config["state_pool_predictor"]
    state_pool_predictor = None
    if pool_cfg["enabled"]:
        state_pool_predictor = StatePoolPredictor(
            model_config.d_model,
            query_vocab_size=max(pool_cfg["query_vocabulary"].values()) + 1,
            mask_vocab_size=max(pool_cfg["mask_vocabulary"].values()) + 1,
            hidden_multiplier=pool_cfg["hidden_multiplier"],
            dropout=pool_cfg["dropout"],
        )
        state_pool_predictor.load_state_dict(
            {
                key.split(".", 1)[1]: value
                for key, value in raw_aux.items()
                if key.startswith("state_pool_predictor.")
            },
            strict=True,
        )

    slot_cfg = config["semantic_concept_slot_predictor"]
    semantic_concept_slot_predictor = None
    if slot_cfg["enabled"]:
        slot_state = {
            key.split(".", 1)[1]: value
            for key, value in raw_aux.items()
            if key.startswith("semantic_concept_slot_predictor.")
        }
        semantic_concept_slot_predictor = SemanticConceptSlotPredictor(
            model_config.d_model,
            concept_count=slot_state["concept_queries"].shape[1],
            nhead=model_config.nhead,
            decoder_layers=slot_cfg["decoder_layers"],
            dropout=slot_cfg["dropout"],
        )
        semantic_concept_slot_predictor.load_state_dict(slot_state, strict=True)

    grounding_state = {
        key.split(".", 1)[1]: value
        for key, value in raw_aux.items()
        if key.startswith("grounding_head.")
    }
    grounding_head = torch.nn.Linear(
        model_config.d_model, grounding_state["weight"].shape[0]
    )
    grounding_head.load_state_dict(grounding_state, strict=True)

    modules = (
        model,
        state_pool_predictor,
        semantic_concept_slot_predictor,
        grounding_head,
    )
    for module in modules:
        if module is not None:
            module.to(device).eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    manifest = _json(root / "release_manifest.json")
    expected_core_parameters = int(manifest["parameter_counts"]["core"])
    actual_core_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_core_parameters != expected_core_parameters:
        raise ValueError(
            "Loaded architecture does not match the release manifest: "
            f"{actual_core_parameters} != {expected_core_parameters}"
        )

    return ModelBundle(
        model=model,
        state_pool_predictor=state_pool_predictor,
        semantic_concept_slot_predictor=semantic_concept_slot_predictor,
        grounding_head=grounding_head,
        tensorizer=_tensorizer(tensorizer_config),
        config=config,
        tensorizer_config=tensorizer_config,
        schema=_json(root / "schema.json"),
        normalization=_json(root / "normalization.json"),
        release_manifest=manifest,
        source=root,
        revision=effective_revision if source is None else revision,
    )
