"""Source-agnostic training for one continuous SupplyChain-JEPA weight chain.

The implementation trains:

``State JEPA -> frozen-encoder action-conditioned Dynamics JEPA``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import windows as windows_module
from .data import (
    TENSORIZER_FORMAT,
    EpisodeTensorizer,
    FittedTensorizerState,
    fit_tensorizer_state,
    iter_episode_records,
)
from .losses import (
    collapse_diagnostics,
    masked_latent_loss,
    paired_difference_loss,
    sigreg_loss,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)
from .masking import StructuredMaskSampler
from .model import (
    ModelConfig,
    SemanticConceptSlotPredictor,
    StatePoolPredictor,
    SupplyChainJEPACore,
)
from .semantic_features import FEATURE_CONCEPTS, feature_concept_ids
from .windows import (
    DynamicsTrainingExample,
    StateTrainingExample,
    WindowSpec,
    build_dynamics_examples,
    build_state_example,
    collate_dynamics_examples,
    collate_state_examples,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_MODEL_CONFIG = CONFIG_ROOT / "model.json"
DEFAULT_STATE_CONFIG = CONFIG_ROOT / "state.json"
DEFAULT_DYNAMICS_CONFIG = CONFIG_ROOT / "dynamics.json"
TRAIN_SPLIT = "pretrain_train"
DEV_SPLIT = "pretrain_dev"
LOCAL_INDEX_NAMES = (
    "entity_index",
    "product_index",
    "node_index",
    "source_index",
    "target_index",
    "alignment_index",
)
GROUNDING_PATTERNS = (
    "inventory",
    "backlog",
    "demand",
    "lead_time",
    "remaining_time",
    "remaining_container",
    "delay",
    "stockout",
)
STATE_QUERY_TYPES = ("hidden_current", "action_omitted_future")


def _read_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Training config must be a JSON object")
    return payload


def _training_config(
    model_path: str | Path, objective_path: str | Path
) -> dict[str, Any]:
    """Compose an immutable architecture config with one objective config."""

    model_payload = _read_config(model_path)
    objective_payload = _read_config(objective_path)
    if "model" in model_payload:
        model_payload = model_payload["model"]
    if not isinstance(model_payload, Mapping):
        raise TypeError("Model config must be a JSON object")
    if "model" in objective_payload:
        raise ValueError("Objective config must not redefine the model architecture")
    return {**objective_payload, "model": dict(model_payload)}


def _normal_name(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _string_leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _string_leaves(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            yield from _string_leaves(nested)


def _enforce_forbidden_sources(
    records: Sequence[Mapping[str, Any]],
    source_paths: Sequence[str | Path],
    forbidden: Sequence[str],
) -> None:
    blocked = {_normal_name(value): str(value) for value in forbidden}
    for source in source_paths:
        normalized = _normal_name(Path(source).name)
        for alias, display in blocked.items():
            if alias and alias in normalized:
                raise ValueError(f"Forbidden downstream source {display!r} in {source}")
    for record in records:
        for value in _string_leaves(record):
            normalized = _normal_name(value)
            for alias, display in blocked.items():
                if alias and alias in normalized:
                    record_id = record.get("record_id", "<unknown>")
                    raise ValueError(
                        f"Forbidden downstream source {display!r} in record {record_id}"
                    )


def _load_records(
    sources: Sequence[str | Path], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = list(iter_episode_records(list(sources)))
    if not records:
        raise ValueError("No canonical episode records were loaded")
    _enforce_forbidden_sources(
        records,
        sources,
        tuple(config.get("forbidden_downstream_sources", ())),
    )
    return records


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _max_vocab_id(vocabs: Mapping[str, Mapping[str, int]]) -> int:
    return max(
        (int(index) for vocab in vocabs.values() for index in vocab.values()),
        default=1,
    )


def _apply_model_overrides(
    config: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    model = payload.setdefault("model", {})
    if args.small_model:
        model.update(
            {
                "d_model": 32,
                "n_heads": 4,
                "encoder_layers": 1,
                "state_predictor_layers": 1,
                "dynamics_predictor_layers": 1,
                "encoder_ffn_dim": 64,
                "predictor_ffn_dim": 64,
                "dropout": 0.0,
            }
        )
    override_names = {
        "d_model": "d_model",
        "nhead": "n_heads",
        "encoder_layers": "encoder_layers",
        "state_predictor_layers": "state_predictor_layers",
        "dynamics_predictor_layers": "dynamics_predictor_layers",
        "dim_feedforward": "encoder_ffn_dim",
    }
    for argument, key in override_names.items():
        value = getattr(args, argument, None)
        if value is not None:
            model[key] = value
            if argument == "dim_feedforward":
                model["predictor_ffn_dim"] = value
    return payload


def _model_config(
    training_config: Mapping[str, Any], tensorizer_state: FittedTensorizerState
) -> ModelConfig:
    metadata = tensorizer_state.metadata_vocabs
    model_payload = training_config.get("model", {})
    semantic_mode = str(
        model_payload.get("feature_semantic_mode", "feature_specific")
    )
    concept_ids = (
        feature_concept_ids(tensorizer_state.feature_names)
        if semantic_mode == "concept_residual"
        else ()
    )
    return ModelConfig.from_training_config(
        training_config,
        feature_count=tensorizer_state.feature_count,
        categorical_vocab_size=max(
            _max_vocab_id(tensorizer_state.categorical_vocabs),
            _max_vocab_id(tensorizer_state.target_categorical_vocabs),
        )
        + 1,
        family_vocab_size=max(metadata["family"].values(), default=1) + 1,
        token_kind_vocab_size=max(metadata["token_kind"].values(), default=1) + 1,
        type_vocab_size=max(metadata["token_type"].values(), default=1) + 1,
        action_dim=8,
        feature_concept_ids=concept_ids,
        concept_vocab_size=(
            len(FEATURE_CONCEPTS) if semantic_mode == "concept_residual" else 1
        ),
    )


@torch.no_grad()
def _migrate_semantic_state_checkpoint(
    checkpoint_path: str | Path,
    *,
    tensorizer_state: FittedTensorizerState,
    model: SupplyChainJEPACore,
    grounding_head: nn.Linear,
) -> Mapping[str, Any]:
    """Warm-start shared concepts while preserving the old embedding exactly.

    The old raw-feature tables are decomposed into a trainable concept mean and
    a frozen feature residual.  With a schema gate of one, their sum reproduces
    the parent feature/value embedding before semantic migration training.
    """

    path = Path(checkpoint_path)
    parent = torch.load(path, map_location="cpu", weights_only=False)
    if parent.get("stage") != "state":
        raise ValueError("Semantic State initialization requires a State checkpoint")
    parent_tensorizer = FittedTensorizerState.from_dict(parent["tensorizer_state"])
    if parent_tensorizer.feature_names != tensorizer_state.feature_names:
        raise ValueError("Semantic initialization requires the same fitted feature schema")
    if parent_tensorizer.metadata_vocabs != tensorizer_state.metadata_vocabs:
        raise ValueError("Semantic initialization requires the same metadata vocabularies")
    if model.config.feature_semantic_mode != "concept_residual":
        raise ValueError("Semantic initialization requires concept_residual mode")

    # A State checkpoint never trains the Dynamics predictor.  Its action
    # encoder may legitimately change between State generations, so do not
    # make a random, frozen predictor part of the semantic migration contract.
    # Encoders and the State predictor remain strict below.
    parent_state = {
        key: value
        for key, value in parent["model_state"].items()
        if not key.startswith("dynamics_predictor.")
    }
    incompatible = model.load_state_dict(parent_state, strict=False)
    expected_missing = {
        key
        for key in model.state_dict()
        if key.startswith("dynamics_predictor.")
        or ".concept_" in key
        or key.endswith(".feature_concept_ids")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Semantic initialization checkpoint mismatch outside concept parameters: "
            f"missing={sorted(set(incompatible.missing_keys) - expected_missing)!r}, "
            f"unexpected={sorted(incompatible.unexpected_keys)!r}"
        )

    embedding = model.online_encoder.embedding
    assert embedding.feature_concept_ids is not None
    assert embedding.concept_numeric_weight is not None
    assert embedding.concept_embedding is not None
    concept_ids = embedding.feature_concept_ids
    raw_numeric = embedding.numeric_weight.detach().clone()
    raw_keys = embedding.feature_embedding.weight.detach().clone()
    for concept_id in range(model.config.concept_vocab_size):
        members = concept_ids.eq(concept_id)
        if not bool(members.any()):
            continue
        embedding.concept_numeric_weight[concept_id].copy_(raw_numeric[members].mean(0))
        embedding.concept_embedding.weight[concept_id].copy_(raw_keys[members].mean(0))
    embedding.numeric_weight.copy_(
        raw_numeric - embedding.concept_numeric_weight[concept_ids]
    )
    embedding.feature_embedding.weight.copy_(
        raw_keys - embedding.concept_embedding.weight[concept_ids]
    )
    model.hard_sync_target()
    model.freeze_source_schema_residuals()
    grounding_head.load_state_dict(parent["grounding_head_state"], strict=True)
    return parent


@torch.no_grad()
def _initialize_state_checkpoint(
    checkpoint_path: str | Path,
    *,
    tensorizer_state: FittedTensorizerState,
    model: SupplyChainJEPACore,
    grounding_head: nn.Linear,
) -> Mapping[str, Any]:
    """Warm-start a State generation without changing its deployed interface."""

    if model.config.feature_semantic_mode == "concept_residual":
        return _migrate_semantic_state_checkpoint(
            checkpoint_path,
            tensorizer_state=tensorizer_state,
            model=model,
            grounding_head=grounding_head,
        )

    path = Path(checkpoint_path)
    parent = torch.load(path, map_location="cpu", weights_only=False)
    if parent.get("stage") != "state":
        raise ValueError("State initialization requires a State checkpoint")
    parent_tensorizer = FittedTensorizerState.from_dict(parent["tensorizer_state"])
    if parent_tensorizer.feature_names != tensorizer_state.feature_names:
        raise ValueError("State initialization requires the same fitted feature schema")
    if parent_tensorizer.metadata_vocabs != tensorizer_state.metadata_vocabs:
        raise ValueError("State initialization requires the same metadata vocabularies")
    parent_state = {
        key: value
        for key, value in parent["model_state"].items()
        if not key.startswith("dynamics_predictor.")
    }
    incompatible = model.load_state_dict(parent_state, strict=False)
    expected_missing = {
        key for key in model.state_dict() if key.startswith("dynamics_predictor.")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "State initialization checkpoint mismatch outside Dynamics: "
            f"missing={sorted(set(incompatible.missing_keys) - expected_missing)!r}, "
            f"unexpected={sorted(incompatible.unexpected_keys)!r}"
        )
    grounding_head.load_state_dict(parent["grounding_head_state"], strict=True)
    return parent


def _state_tensorizer_state(
    records: Sequence[Mapping[str, Any]],
    *,
    init_checkpoint: str | Path | None,
    reuse_parent_tensorizer: bool,
) -> FittedTensorizerState:
    """Fit a fresh schema or retain a State checkpoint's deployed interface.

    Continuation training must not silently re-index features or categorical
    values merely because its replay corpus is smaller than the original
    pretraining corpus. Reusing the parent tensorizer keeps the encoder input
    contract fixed; unseen categorical values use the existing UNK path and
    unknown feature names are ignored by ``EpisodeTensorizer``.
    """

    if not reuse_parent_tensorizer:
        return fit_tensorizer_state(
            records,
            train_splits=(TRAIN_SPLIT,),
            state_version=TENSORIZER_FORMAT,
        )
    if init_checkpoint is None:
        raise ValueError("--reuse-parent-tensorizer requires --init-checkpoint")
    parent = torch.load(
        Path(init_checkpoint), map_location="cpu", weights_only=False
    )
    if parent.get("stage") != "state":
        raise ValueError("Parent tensorizer reuse requires a State checkpoint")
    payload = parent.get("tensorizer_state")
    if not isinstance(payload, Mapping):
        raise ValueError("State checkpoint lacks tensorizer_state")
    return FittedTensorizerState.from_dict(payload)


def _required_local_index_capacity(episodes: Sequence[Any]) -> int:
    maximum = 1
    for episode in episodes:
        for prefix in ("context", "target"):
            for name in LOCAL_INDEX_NAMES:
                values = getattr(episode, f"{prefix}_{name}")
                if values.numel():
                    maximum = max(maximum, int(values.max()))
    return maximum + 1


def _expand_local_index_capacity(
    config: dict[str, Any], episodes: Sequence[Any]
) -> tuple[int, int]:
    configured = int(config["model"]["local_index_capacity"])
    required = _required_local_index_capacity(episodes)
    mode = str(config["model"].get("local_index_mode", "absolute"))
    if mode == "absolute" and required > configured:
        config["model"]["local_index_capacity"] = 1 << (required - 1).bit_length()
    return required, int(config["model"]["local_index_capacity"])


def _amp_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=enabled)


def _cosine_learning_rate_scale(
    update: int,
    max_updates: int,
    *,
    warmup_fraction: float,
    minimum_ratio: float,
) -> float:
    """Warm up, then cosine-decay, using optimizer-update coordinates."""

    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum learning-rate ratio must be in [0, 1]")
    step = update + 1
    warmup_updates = int(max_updates * warmup_fraction)
    if warmup_updates > 0 and step <= warmup_updates:
        return step / warmup_updates
    decay_updates = max(1, max_updates - warmup_updates)
    progress = min(1.0, max(0.0, (step - warmup_updates) / decay_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _write_outputs(
    output_dir: Path,
    checkpoint_name: str,
    checkpoint: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / checkpoint_name
    torch.save(dict(checkpoint), checkpoint_path)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def _cuda_peak_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    gib = 1024.0**3
    return {
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def _source_families(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(record["simulator"]["family"]) for record in records})


def _state_episode_pools(
    episodes: Sequence[Any],
    source_weights: Mapping[str, float],
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        grouped[str(episode.family)][str(episode.world_id)].append(episode)
    missing_weights = sorted(set(grouped) - set(source_weights))
    if missing_weights:
        raise ValueError(
            "State source weights are missing corpus families: "
            + ", ".join(missing_weights)
        )
    pools: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for family, worlds in sorted(grouped.items()):
        if float(source_weights[family]) <= 0.0:
            continue
        pools[family] = tuple(
            tuple(sorted(members, key=lambda episode: episode.record_id))
            for _, members in sorted(worlds.items())
        )
    if not pools:
        raise ValueError("No positively weighted State source families are available")
    return pools


def _draw_state_episode(
    pools: Mapping[str, Sequence[Sequence[Any]]],
    source_weights: Mapping[str, float],
    rng: random.Random,
) -> Any:
    families = sorted(pools)
    weights = [float(source_weights[family]) for family in families]
    family = rng.choices(families, weights=weights, k=1)[0]
    world = rng.choice(pools[family])
    return rng.choice(world)


def _draw_state_episode_from_family(
    pools: Mapping[str, Sequence[Sequence[Any]]],
    family: str,
    rng: random.Random,
) -> Any:
    if family not in pools:
        raise ValueError(f"State source family {family!r} is unavailable")
    world = rng.choice(pools[family])
    return rng.choice(world)


def _state_examples(
    episode_pools: Mapping[str, Sequence[Sequence[Any]]],
    *,
    batch_size: int,
    update: int,
    seed: int,
    source_weights: Mapping[str, float],
    sampler: StructuredMaskSampler,
    window_spec: WindowSpec,
    current_kinds: Sequence[str],
    minimum_rows_per_family: int = 0,
) -> list[StateTrainingExample]:
    if minimum_rows_per_family < 0:
        raise ValueError("minimum_rows_per_family cannot be negative")
    required_families: list[str | None] = []
    families = sorted(episode_pools)
    required_rows = minimum_rows_per_family * len(families)
    if minimum_rows_per_family and batch_size >= required_rows:
        required_families = [
            family for family in families for _ in range(minimum_rows_per_family)
        ] + [None] * (batch_size - required_rows)
    examples: list[StateTrainingExample] = []
    attempts = 0
    maximum_attempts = max(32, batch_size * 16)
    rng = random.Random(f"{seed}:state-source:{update}")
    if required_families:
        rng.shuffle(required_families)
    while len(examples) < batch_size and attempts < maximum_attempts:
        ordinal = update * batch_size + attempts
        required_family = (
            required_families[len(examples)] if required_families else None
        )
        episode = (
            _draw_state_episode(episode_pools, source_weights, rng)
            if required_family is None
            else _draw_state_episode_from_family(
                episode_pools,
                required_family,
                rng,
            )
        )
        # Two future queries in every five examples; the rest are structured
        # current-state masks.
        kind = (
            "future_horizon"
            if ordinal % 5 < 2
            else current_kinds[(ordinal // 5 + ordinal) % len(current_kinds)]
        )
        try:
            plan = sampler.sample(episode, epoch=ordinal, kind=kind)
            examples.append(
                build_state_example(
                    episode,
                    plan,
                    spec=window_spec,
                    epoch=ordinal,
                )
            )
        except ValueError:
            pass
        attempts += 1
    if len(examples) < batch_size:
        raise ValueError("Could not construct a full State batch from the corpus")
    return examples


def _state_objective(
    model: SupplyChainJEPACore,
    grounding_head: nn.Linear,
    regularization_projector: nn.Module | None,
    state_pool_predictor: StatePoolPredictor | None,
    batch: Mapping[str, Any],
    *,
    lambda_variance: float,
    lambda_covariance: float,
    lambda_grounding: float,
    pooled_state_weight: float,
    anti_collapse: Mapping[str, Any],
    canonical_prediction: Mapping[str, Any],
    query_vocabulary: Mapping[str, int],
    mask_vocabulary: Mapping[str, int],
    grounding_feature_mask: Tensor,
    semantic_slot_predictor: SemanticConceptSlotPredictor | None = None,
    semantic_concept_matrix: Tensor | None = None,
    semantic_active_concepts: Tensor | None = None,
    semantic_slots: Mapping[str, Any] | None = None,
    regularization_token_samples: int = 0,
    anti_collapse_scale: float = 1.0,
    canonical_prediction_scale: float = 1.0,
    semantic_slot_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    context = model.encode_context_with_special_tokens(batch)
    target, target_padding_mask = model.encode_target_with_special_tokens(batch)
    prediction = model.state_predictor(
        context.token_latents,
        model.target_query_metadata(batch),
        context_padding_mask=batch.get("context_padding_mask"),
        target_padding_mask=target_padding_mask,
        role_ids=batch.get("role_ids"),
        horizons=batch.get("horizons"),
    )
    token_mask = batch["objective_target_token_mask"] & ~batch["target_padding_mask"]
    latent = masked_latent_loss(
        prediction,
        target.token_latents,
        mask=token_mask,
    )
    pooled_prediction = _masked_token_mean(prediction, token_mask)
    pooled_target = _masked_token_mean(target.token_latents, token_mask)
    pooled_state = F.smooth_l1_loss(pooled_prediction, pooled_target)
    canonical_state = model.select_canonical_state(context)
    # The deployable state must predict the representation of the actual JEPA
    # objective (hidden current tokens or future tokens), not a mean over the
    # full target window that also contains visible/easy tokens.
    canonical_target = pooled_target
    predictive_state = pooled_prediction

    zero = prediction.new_zeros(())
    canonical_prediction_loss = zero
    canonical_prediction_absolute = zero
    canonical_prediction_centered_mse = zero
    canonical_prediction_centered_cosine = zero
    canonical_prediction_matched_contrastive = zero
    canonical_prediction_matched_accuracy = zero
    canonical_prediction_matched_rows = 0
    canonical_prediction_shuffled = zero
    canonical_state_prediction = canonical_state
    shuffled_state_prediction = canonical_state
    shuffled_rows = torch.zeros(
        canonical_state.shape[0], dtype=torch.bool, device=canonical_state.device
    )
    row_horizons: Tensor | None = None
    if state_pool_predictor is not None:
        row_horizons, query_ids, mask_ids = _state_pool_metadata(
            batch,
            query_vocabulary=query_vocabulary,
            mask_vocabulary=mask_vocabulary,
            device=canonical_state.device,
        )
        predictor_input = (
            canonical_state
            if bool(canonical_prediction.get("propagate_encoder_gradients", True))
            else canonical_state.detach()
        )
        canonical_state_prediction = state_pool_predictor(
            predictor_input,
            row_horizons,
            query_ids,
            mask_ids,
        )
        canonical_prediction_loss, canonical_parts = _canonical_prediction_loss(
            canonical_state_prediction,
            canonical_target,
            batch["family"],
            canonical_prediction,
        )
        canonical_prediction_absolute = canonical_parts["absolute"]
        canonical_prediction_centered_mse = canonical_parts["centered_mse"]
        canonical_prediction_centered_cosine = canonical_parts["centered_cosine"]
        matched_contrastive_weight = float(
            canonical_prediction.get("matched_contrastive_weight", 0.0)
        )
        if matched_contrastive_weight < 0.0:
            raise ValueError("matched_contrastive_weight must be nonnegative")
        if matched_contrastive_weight > 0.0:
            (
                canonical_prediction_matched_contrastive,
                canonical_prediction_matched_accuracy,
                canonical_prediction_matched_rows,
            ) = _canonical_matched_contrastive_loss(
                canonical_state_prediction,
                canonical_target,
                batch["family"],
                batch["query_type"],
                batch["mask_kind"],
                row_horizons,
                temperature=float(
                    canonical_prediction.get("matched_contrastive_temperature", 0.1)
                ),
                variance_floor=float(
                    canonical_prediction.get("matched_contrastive_variance_floor", 1e-6)
                ),
            )
            canonical_prediction_loss = (
                canonical_prediction_loss
                + matched_contrastive_weight
                * canonical_prediction_matched_contrastive
            )
        with torch.no_grad():
            shuffled_state, shuffled_rows = _rotate_rows_within_matched_strata(
                canonical_state,
                batch["family"],
                batch["query_type"],
                batch["mask_kind"],
                row_horizons,
            )
            shuffled_state_prediction = state_pool_predictor(
                shuffled_state,
                row_horizons,
                query_ids,
                mask_ids,
            )
            if bool(shuffled_rows.any()):
                matched_families = [
                    str(family)
                    for family, matched in zip(
                        batch["family"], shuffled_rows.detach().cpu().tolist(), strict=False
                    )
                    if matched
                ]
                canonical_prediction_shuffled, _ = _canonical_prediction_loss(
                    shuffled_state_prediction[shuffled_rows],
                    canonical_target[shuffled_rows],
                    matched_families,
                    canonical_prediction,
                )

    semantic_slot_loss = zero
    semantic_slot_smooth_l1 = zero
    semantic_slot_cosine = zero
    semantic_slot_prediction = prediction.new_zeros(
        (prediction.shape[0], 0, prediction.shape[-1])
    )
    semantic_slot_target = semantic_slot_prediction.float()
    semantic_slot_valid = torch.zeros(
        (prediction.shape[0], 0), dtype=torch.bool, device=prediction.device
    )
    if semantic_slot_predictor is not None:
        if semantic_concept_matrix is None or semantic_active_concepts is None:
            raise ValueError("Semantic slot training requires concept metadata")
        if semantic_slots is None:
            raise ValueError("Semantic slot training requires its loss config")
        if row_horizons is None:
            row_horizons, _, _ = _state_pool_metadata(
                batch,
                query_vocabulary=query_vocabulary,
                mask_vocabulary=mask_vocabulary,
                device=canonical_state.device,
            )
        semantic_context = (
            context.token_latents
            if bool(semantic_slots.get("propagate_encoder_gradients", True))
            else context.token_latents.detach()
        )
        semantic_slot_prediction = semantic_slot_predictor(
            semantic_context,
            row_horizons,
            context_padding_mask=context.token_padding_mask,
        )
        semantic_slot_target, semantic_slot_valid = _semantic_target_slots(
            target.token_latents,
            batch["objective_target_feature_mask"],
            semantic_concept_matrix,
            objective_token_mask=batch["objective_target_token_mask"],
            target_padding_mask=batch["target_padding_mask"],
            active_concept_mask=semantic_active_concepts,
        )
        (
            semantic_slot_loss,
            semantic_slot_smooth_l1,
            semantic_slot_cosine,
        ) = _semantic_concept_slot_loss(
            semantic_slot_prediction,
            semantic_slot_target,
            semantic_slot_valid,
            smooth_l1_weight=float(semantic_slots.get("smooth_l1_weight", 0.8)),
            cosine_weight=float(semantic_slots.get("cosine_weight", 0.2)),
        )

    method = str(anti_collapse.get("method", "legacy_mixed_vicreg"))
    variance = zero
    covariance = zero
    sigreg = zero
    projected_state = canonical_state
    if method == "legacy_mixed_vicreg":
        regularization_parts = [context.belief_latent, context.reg_latent]
        if regularization_token_samples > 0:
            valid_token_latents = context.token_latents[~context.token_padding_mask]
            if valid_token_latents.shape[0] > regularization_token_samples:
                indices = (
                    torch.linspace(
                        0,
                        valid_token_latents.shape[0] - 1,
                        steps=regularization_token_samples,
                        device=valid_token_latents.device,
                    )
                    .round()
                    .long()
                )
                valid_token_latents = valid_token_latents.index_select(0, indices)
            regularization_parts.append(valid_token_latents)
        projected_state = torch.cat(regularization_parts, dim=0)
        variance = vicreg_variance_loss(projected_state)
        covariance = vicreg_covariance_loss(
            projected_state,
            normalization="standard_width",
        )
    elif method in {"vicreg", "sigreg"}:
        projection = str(anti_collapse.get("projection", "mlp"))
        if projection == "identity":
            projected_branches = [canonical_state, predictive_state]
        elif projection == "mlp":
            if regularization_projector is None:
                raise ValueError(f"{method} requires a regularization projector")
            projected_branches = [
                regularization_projector(canonical_state),
                regularization_projector(predictive_state),
            ]
        else:
            raise ValueError(f"Unsupported anti-collapse projection {projection!r}")
        if bool(anti_collapse.get("family_centered", False)):
            projected_branches = [
                _center_rows_by_family(rows, batch["family"])
                for rows in projected_branches
            ]
        projected_state = torch.cat(projected_branches, dim=0)
        if method == "vicreg":
            variance = torch.stack(
                [vicreg_variance_loss(rows) for rows in projected_branches]
            ).mean()
            covariance = torch.stack(
                [
                    vicreg_covariance_loss(
                        rows,
                        normalization=str(
                            anti_collapse.get(
                                "covariance_normalization", "standard_width"
                            )
                        ),
                    )
                    for rows in projected_branches
                ]
            )
            covariance = covariance.mean()
        else:
            sigreg = torch.stack(
                [
                    sigreg_loss(
                        rows,
                        num_projections=int(anti_collapse.get("num_projections", 256)),
                        knots=int(anti_collapse.get("knots", 17)),
                        max_frequency=float(anti_collapse.get("max_frequency", 3.0)),
                    )
                    for rows in projected_branches
                ]
            ).mean()
    elif method != "none":
        raise ValueError(f"Unsupported anti-collapse method {method!r}")
    endpoints = (
        batch["target_time_index"]
        .masked_fill(batch["target_padding_mask"], -1)
        .max(dim=1)
        .values
    )
    contemporaneous = batch["target_time_index"] == endpoints[:, None]
    feature_mask = (
        batch["objective_target_feature_mask"]
        & grounding_feature_mask.view(1, 1, -1)
        & contemporaneous.unsqueeze(-1)
        & ~batch["target_padding_mask"].unsqueeze(-1)
    )
    if bool(feature_mask.any()):
        grounded = grounding_head(prediction)
        grounding = F.smooth_l1_loss(
            grounded,
            batch["target_values"],
            reduction="none",
        )[feature_mask].mean()
    else:
        grounding = prediction.new_zeros(())
    total = (
        latent
        + pooled_state_weight * pooled_state
        + canonical_prediction_scale
        * float(canonical_prediction.get("weight", 0.0))
        * canonical_prediction_loss
        + anti_collapse_scale * lambda_variance * variance
        + anti_collapse_scale * lambda_covariance * covariance
        + lambda_grounding * grounding
        + anti_collapse_scale * float(anti_collapse.get("weight", 0.0)) * sigreg
        + semantic_slot_scale
        * float((semantic_slots or {}).get("weight", 0.0))
        * semantic_slot_loss
    )
    return (
        total,
        {
            "latent": latent,
            "pooled_state": pooled_state,
            "canonical_prediction": canonical_prediction_loss,
            "canonical_prediction_absolute": canonical_prediction_absolute,
            "canonical_prediction_centered_mse": canonical_prediction_centered_mse,
            "canonical_prediction_centered_cosine": canonical_prediction_centered_cosine,
            "canonical_prediction_matched_contrastive": (
                canonical_prediction_matched_contrastive
            ),
            "canonical_prediction_matched_accuracy": (
                canonical_prediction_matched_accuracy
            ),
            "canonical_prediction_matched_rows": prediction.new_tensor(
                float(canonical_prediction_matched_rows)
            ),
            "canonical_prediction_shuffled": canonical_prediction_shuffled,
            "variance": variance,
            "covariance": covariance,
            "sigreg": sigreg,
            "grounding": grounding,
            "semantic_slot": semantic_slot_loss,
            "semantic_slot_smooth_l1": semantic_slot_smooth_l1,
            "semantic_slot_cosine": semantic_slot_cosine,
        },
        {
            "belief": context.belief_latent,
            "reg": context.reg_latent,
            "canonical": canonical_state,
            "canonical_target": canonical_target,
            "canonical_prediction": canonical_state_prediction,
            "canonical_shuffled_prediction": shuffled_state_prediction,
            "canonical_shuffle_mask": shuffled_rows,
            "predictive": predictive_state,
            "prediction": prediction,
            "target": target.token_latents,
            "target_mask": token_mask,
            "pooled_target": pooled_target,
            "regularization": projected_state,
            "semantic_slot_prediction": semantic_slot_prediction,
            "semantic_slot_target": semantic_slot_target,
            "semantic_slot_valid": semantic_slot_valid,
        },
    )


def _masked_token_mean(latents: Tensor, mask: Tensor) -> Tensor:
    if mask.shape != latents.shape[:-1]:
        raise ValueError("token mean mask does not match latent rows")
    counts = mask.sum(dim=1, keepdim=True)
    if counts.eq(0).any():
        raise ValueError("Every pooled state requires at least one token")
    weights = mask.to(latents.dtype).unsqueeze(-1)
    return (latents * weights).sum(dim=1) / counts.to(latents.dtype)


def _feature_concept_matrix(
    feature_names: Sequence[str],
    *,
    device: torch.device,
) -> Tensor:
    """Return a fixed feature-to-concept map with no learned schema path."""

    concept_ids = torch.tensor(
        feature_concept_ids(feature_names),
        dtype=torch.long,
        device=device,
    )
    return F.one_hot(concept_ids, num_classes=len(FEATURE_CONCEPTS)).float()


def _semantic_target_slots(
    target_latents: Tensor,
    objective_feature_mask: Tensor,
    concept_matrix: Tensor,
    *,
    objective_token_mask: Tensor,
    target_padding_mask: Tensor,
    active_concept_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pool target tokens into concept slots using binary semantic support.

    Multiple source aliases for the same concept on one token contribute one
    vote rather than allowing wide schemas to dominate the auxiliary target.
    """

    if target_latents.ndim != 3:
        raise ValueError("target_latents must be [B,L,D]")
    if objective_feature_mask.ndim != 3:
        raise ValueError("objective_feature_mask must be [B,L,F]")
    batch_size, target_length, _ = target_latents.shape
    if objective_feature_mask.shape[:2] != (batch_size, target_length):
        raise ValueError("Feature mask does not align with target tokens")
    if concept_matrix.shape[0] != objective_feature_mask.shape[-1]:
        raise ValueError("Concept matrix does not align with fitted features")
    concept_count = concept_matrix.shape[1]
    expected_tokens = (batch_size, target_length)
    if objective_token_mask.shape != expected_tokens:
        raise ValueError("objective_token_mask must be [B,L]")
    if target_padding_mask.shape != expected_tokens:
        raise ValueError("target_padding_mask must be [B,L]")
    if active_concept_mask.shape != (concept_count,):
        raise ValueError("active_concept_mask must be [K]")
    if any(
        mask.dtype != torch.bool
        for mask in (
            objective_feature_mask,
            objective_token_mask,
            target_padding_mask,
            active_concept_mask,
        )
    ):
        raise TypeError("Semantic slot masks must be torch.bool")

    support = torch.matmul(
        objective_feature_mask.float(), concept_matrix.float()
    ).gt(0)
    valid_tokens = objective_token_mask & ~target_padding_mask
    support = support & valid_tokens.unsqueeze(-1)
    weights = support.transpose(1, 2).float()
    counts = weights.sum(dim=-1, keepdim=True)
    slots = torch.bmm(weights, target_latents.float()) / counts.clamp_min(1.0)
    valid_slots = counts.squeeze(-1).gt(0) & active_concept_mask.view(1, -1)
    slots = slots.masked_fill(~valid_slots.unsqueeze(-1), 0.0)
    return slots, valid_slots


def _semantic_concept_slot_loss(
    prediction: Tensor,
    target: Tensor,
    valid_slots: Tensor,
    *,
    smooth_l1_weight: float = 0.8,
    cosine_weight: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compare valid concept slots after parameter-free latent normalization."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Semantic slot prediction and target must be [B,K,D]")
    if valid_slots.shape != prediction.shape[:2] or valid_slots.dtype != torch.bool:
        raise ValueError("valid_slots must be a boolean [B,K] tensor")
    if smooth_l1_weight < 0.0 or cosine_weight < 0.0:
        raise ValueError("Semantic slot component weights must be nonnegative")
    if smooth_l1_weight + cosine_weight <= 0.0:
        raise ValueError("Semantic slot loss needs a positive component weight")
    if not bool(valid_slots.any()):
        zero = prediction.new_zeros((), dtype=torch.float32)
        return zero, zero, zero
    normalized_shape = (prediction.shape[-1],)
    normalized_prediction = F.layer_norm(prediction.float(), normalized_shape)
    normalized_target = F.layer_norm(target.detach().float(), normalized_shape)
    selected_prediction = normalized_prediction[valid_slots]
    selected_target = normalized_target[valid_slots]
    smooth_l1 = F.smooth_l1_loss(selected_prediction, selected_target)
    cosine = (
        1.0
        - F.cosine_similarity(
            selected_prediction,
            selected_target,
            dim=-1,
            eps=1e-6,
        )
    ).mean()
    total = smooth_l1_weight * smooth_l1 + cosine_weight * cosine
    return total, smooth_l1, cosine


def _center_rows_by_family(rows: Tensor, families: Sequence[str]) -> Tensor:
    if rows.ndim != 2 or rows.shape[0] != len(families):
        raise ValueError("Family centering requires one row per family label")
    centered = torch.empty_like(rows)
    for family in sorted(set(families)):
        indices = [index for index, value in enumerate(families) if value == family]
        index_tensor = torch.tensor(indices, device=rows.device)
        selected = rows.index_select(0, index_tensor)
        centered.index_copy_(
            0, index_tensor, selected - selected.mean(dim=0, keepdim=True)
        )
    return centered


def _rotate_rows_within_matched_strata(
    rows: Tensor,
    families: Sequence[str],
    queries: Sequence[str],
    masks: Sequence[str],
    horizons: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotate context rows only inside exactly matched task strata.

    Matching family, query, mask kind, and horizon prevents the negative control
    from becoming harder merely because it received incompatible metadata.
    Singleton strata are intentionally left untouched and excluded from the
    reported shuffle comparison.
    """

    row_count = rows.shape[0]
    if rows.ndim != 2 or not (
        row_count == len(families) == len(queries) == len(masks) == horizons.shape[0]
    ):
        raise ValueError("Matched rotation metadata does not align with rows")
    rotated = rows.clone()
    matched = torch.zeros(row_count, dtype=torch.bool, device=rows.device)
    strata: dict[tuple[str, str, str, float], list[int]] = defaultdict(list)
    horizon_values = horizons.detach().float().cpu().tolist()
    for index, (family, query, mask, horizon) in enumerate(
        zip(families, queries, masks, horizon_values, strict=False)
    ):
        strata[(str(family), str(query), str(mask), round(float(horizon), 6))].append(
            index
        )
    for indices in strata.values():
        if len(indices) < 2:
            continue
        destination = torch.tensor(indices, device=rows.device, dtype=torch.long)
        source = torch.tensor(
            indices[1:] + indices[:1],
            device=rows.device,
            dtype=torch.long,
        )
        rotated.index_copy_(0, destination, rows.index_select(0, source))
        matched[destination] = True
    return rotated, matched


def _state_pool_metadata(
    batch: Mapping[str, Any],
    *,
    query_vocabulary: Mapping[str, int],
    mask_vocabulary: Mapping[str, int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert non-model query labels to bounded predictor descriptors."""

    queries = [str(value) for value in batch["query_type"]]
    masks = [str(value) for value in batch["mask_kind"]]
    unknown_queries = sorted(set(queries) - set(query_vocabulary))
    unknown_masks = sorted(set(masks) - set(mask_vocabulary))
    if unknown_queries:
        raise ValueError(f"Unknown State query types: {unknown_queries}")
    if unknown_masks:
        raise ValueError(f"Unknown State mask kinds: {unknown_masks}")
    horizons = batch["horizons"]
    padding = batch["target_padding_mask"]
    if horizons.shape != padding.shape:
        raise ValueError("State horizon rows do not match target padding")
    row_horizons = horizons.masked_fill(padding, 0.0).amax(dim=1).to(device)
    query_ids = torch.tensor(
        [query_vocabulary[value] for value in queries],
        dtype=torch.long,
        device=device,
    )
    mask_ids = torch.tensor(
        [mask_vocabulary[value] for value in masks],
        dtype=torch.long,
        device=device,
    )
    return row_horizons, query_ids, mask_ids


def _canonical_prediction_loss(
    prediction: Tensor,
    target: Tensor,
    families: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Align operational states while removing family-constant shortcuts."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Canonical prediction and target must have shape [B,D]")
    if prediction.shape[0] != len(families):
        raise ValueError("Canonical prediction family labels do not align")
    # Normalize in fp32 even under CUDA autocast.  Within-family variance can be
    # much smaller than the global family separation and is easy to underflow in
    # fp16, exactly the signal this branch is meant to preserve.
    prediction_float = prediction.float()
    target_float = target.float()
    absolute = F.smooth_l1_loss(prediction_float, target_float)
    centered_prediction = _center_rows_by_family(prediction_float, families)
    centered_target = _center_rows_by_family(target_float, families)
    variance_floor = float(config.get("variance_floor", 1e-3))
    if variance_floor <= 0.0:
        raise ValueError("canonical prediction variance_floor must be positive")
    family_mse: list[Tensor] = []
    family_cosine: list[Tensor] = []
    cosine_floor = float(config.get("cosine_norm_floor", 1e-6))
    for family in sorted(set(families)):
        indices = [index for index, value in enumerate(families) if value == family]
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, device=prediction.device)
        family_prediction = centered_prediction.index_select(0, index_tensor)
        family_target = centered_target.index_select(0, index_tensor)
        target_energy = family_target.detach().square().mean().clamp_min(variance_floor)
        family_mse.append(
            (family_prediction - family_target).square().mean() / target_energy
        )
        valid_cosine = family_target.norm(dim=-1) > cosine_floor
        if bool(valid_cosine.any()):
            family_cosine.append(
                (
                    1.0
                    - F.cosine_similarity(
                        family_prediction[valid_cosine],
                        family_target[valid_cosine],
                        dim=-1,
                        eps=1e-6,
                    )
                ).mean()
            )
    centered_mse = (
        torch.stack(family_mse).mean() if family_mse else prediction_float.new_zeros(())
    )
    centered_cosine = (
        torch.stack(family_cosine).mean()
        if family_cosine
        else prediction_float.new_zeros(())
    )
    weights = {
        "absolute": float(config.get("absolute_weight", 0.25)),
        "centered_mse": float(config.get("centered_mse_weight", 0.5)),
        "centered_cosine": float(config.get("centered_cosine_weight", 0.25)),
    }
    if any(value < 0.0 for value in weights.values()) or not any(weights.values()):
        raise ValueError("canonical prediction component weights must be nonnegative")
    total = (
        weights["absolute"] * absolute
        + weights["centered_mse"] * centered_mse
        + weights["centered_cosine"] * centered_cosine
    )
    return total, {
        "absolute": absolute,
        "centered_mse": centered_mse,
        "centered_cosine": centered_cosine,
    }


def _canonical_matched_contrastive_loss(
    prediction: Tensor,
    target: Tensor,
    families: Sequence[str],
    queries: Sequence[str],
    masks: Sequence[str],
    horizons: Tensor,
    *,
    temperature: float,
    variance_floor: float,
) -> tuple[Tensor, Tensor, int]:
    """Match operational states only against valid same-task negatives.

    Family, query type, mask kind, and horizon are held fixed inside every
    contrastive group.  The loss therefore cannot be reduced by identifying a
    simulator or the requested prediction task.  Centering within each group
    removes its shared latent offset and makes sample-specific operational
    variation determine the match.
    """

    row_count = prediction.shape[0]
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Matched contrastive rows must have shape [B,D]")
    if not (
        row_count
        == len(families)
        == len(queries)
        == len(masks)
        == horizons.shape[0]
    ):
        raise ValueError("Matched contrastive metadata does not align with rows")
    if temperature <= 0.0:
        raise ValueError("Matched contrastive temperature must be positive")
    if variance_floor <= 0.0:
        raise ValueError("Matched contrastive variance floor must be positive")

    strata: dict[tuple[str, str, str, float], list[int]] = defaultdict(list)
    horizon_values = horizons.detach().float().cpu().tolist()
    for index, (family, query, mask, horizon) in enumerate(
        zip(families, queries, masks, horizon_values, strict=False)
    ):
        strata[(str(family), str(query), str(mask), round(float(horizon), 6))].append(
            index
        )

    losses: list[Tensor] = []
    correct = prediction.new_zeros((), dtype=torch.float32)
    compared_rows = 0
    for indices in strata.values():
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, device=prediction.device, dtype=torch.long)
        group_prediction = prediction.float().index_select(0, index_tensor)
        group_target = target.float().index_select(0, index_tensor)
        centered_prediction = group_prediction - group_prediction.mean(
            dim=0, keepdim=True
        )
        centered_target = group_target - group_target.mean(dim=0, keepdim=True)
        if float(centered_target.detach().square().mean()) <= variance_floor:
            continue
        normalized_prediction = F.normalize(centered_prediction, dim=-1, eps=1e-6)
        normalized_target = F.normalize(
            centered_target.detach(), dim=-1, eps=1e-6
        )
        logits = normalized_prediction @ normalized_target.transpose(0, 1)
        logits = logits / temperature
        labels = torch.arange(len(indices), device=prediction.device)
        losses.append(F.cross_entropy(logits, labels))
        correct = correct + (logits.argmax(dim=1) == labels).float().sum()
        compared_rows += len(indices)

    if not losses:
        zero = prediction.new_zeros((), dtype=torch.float32)
        return zero, zero, 0
    return torch.stack(losses).mean(), correct / compared_rows, compared_rows


class _StateRegularizationProjector(nn.Module):
    """Training-only expander; its isotropy constraints are not deployed."""

    def __init__(
        self,
        width: int,
        hidden_multiplier: int,
        *,
        hidden_width: int | None = None,
        output_width: int | None = None,
    ):
        super().__init__()
        hidden = (
            int(hidden_width)
            if hidden_width is not None
            else max(width, int(width) * int(hidden_multiplier))
        )
        output = int(output_width) if output_width is not None else int(width)
        if hidden < 1 or output < 1:
            raise ValueError("Regularization projector widths must be positive")
        self.network = nn.Sequential(
            nn.Linear(width, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, output, bias=False),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


@torch.no_grad()
def _evaluate_state(
    model: SupplyChainJEPACore,
    grounding_head: nn.Linear,
    regularization_projector: nn.Module | None,
    state_pool_predictor: StatePoolPredictor | None,
    episode_pools: Mapping[str, Sequence[Sequence[Any]]],
    *,
    sampler: StructuredMaskSampler,
    window_spec: WindowSpec,
    batch_size: int,
    current_kinds: Sequence[str],
    seed: int,
    source_weights: Mapping[str, float],
    device: torch.device,
    weights: Mapping[str, float],
    pooled_state_weight: float,
    anti_collapse: Mapping[str, Any],
    canonical_prediction: Mapping[str, Any],
    query_vocabulary: Mapping[str, int],
    mask_vocabulary: Mapping[str, int],
    grounding_feature_mask: Tensor,
    evaluation_samples: int,
    regularization_token_samples: int,
    minimum_rows_per_family: int,
    semantic_slot_predictor: SemanticConceptSlotPredictor | None = None,
    semantic_concept_matrix: Tensor | None = None,
    semantic_active_concepts: Tensor | None = None,
    semantic_slots: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    grounding_head.eval()
    if regularization_projector is not None:
        regularization_projector.eval()
    if state_pool_predictor is not None:
        state_pool_predictor.eval()
    if semantic_slot_predictor is not None:
        semantic_slot_predictor.eval()
    requested = max(batch_size, int(evaluation_samples))
    total_weight = 0
    totals: Counter[str] = Counter()
    belief_rows: list[Tensor] = []
    reg_rows: list[Tensor] = []
    canonical_rows: list[Tensor] = []
    target_canonical_rows: list[Tensor] = []
    canonical_by_family: dict[str, list[Tensor]] = defaultdict(list)
    target_canonical_by_family: dict[str, list[Tensor]] = defaultdict(list)
    token_error_sum = 0.0
    token_element_count = 0
    token_target_sum: Tensor | None = None
    token_target_square_sum: Tensor | None = None
    token_target_row_count = 0
    pooled_error_sum = 0.0
    pooled_element_count = 0
    pooled_target_sum: Tensor | None = None
    pooled_target_square_sum: Tensor | None = None
    pooled_target_row_count = 0
    canonical_prediction_rows: list[Tensor] = []
    canonical_prediction_target_rows: list[Tensor] = []
    canonical_matched_prediction_rows: list[Tensor] = []
    canonical_shuffled_prediction_rows: list[Tensor] = []
    canonical_shuffled_target_rows: list[Tensor] = []
    canonical_prediction_families: list[str] = []
    semantic_slot_prediction_rows: list[Tensor] = []
    semantic_slot_target_rows: list[Tensor] = []
    semantic_slot_valid_count = 0
    semantic_slot_candidate_count = 0
    for batch_index, start in enumerate(range(0, requested, batch_size)):
        current_batch_size = min(batch_size, requested - start)
        examples = _state_examples(
            episode_pools,
            batch_size=current_batch_size,
            update=10_000 + batch_index,
            seed=seed,
            source_weights=source_weights,
            sampler=sampler,
            window_spec=window_spec,
            current_kinds=current_kinds,
            minimum_rows_per_family=minimum_rows_per_family,
        )
        batch = _move_batch(collate_state_examples(examples), device)
        total, pieces, latents = _state_objective(
            model,
            grounding_head,
            regularization_projector,
            state_pool_predictor,
            batch,
            lambda_variance=float(weights["variance"]),
            lambda_covariance=float(weights["covariance"]),
            lambda_grounding=float(weights["grounding"]),
            pooled_state_weight=pooled_state_weight,
            anti_collapse=anti_collapse,
            canonical_prediction=canonical_prediction,
            query_vocabulary=query_vocabulary,
            mask_vocabulary=mask_vocabulary,
            grounding_feature_mask=grounding_feature_mask,
            semantic_slot_predictor=semantic_slot_predictor,
            semantic_concept_matrix=semantic_concept_matrix,
            semantic_active_concepts=semantic_active_concepts,
            semantic_slots=semantic_slots,
            regularization_token_samples=regularization_token_samples,
        )
        total_weight += current_batch_size
        totals["loss"] += float(total) * current_batch_size
        for name, value in pieces.items():
            totals[f"{name}_loss"] += float(value) * current_batch_size
        belief_rows.append(latents["belief"].detach().float().cpu())
        reg_rows.append(latents["reg"].detach().float().cpu())
        canonical_rows.append(latents["canonical"].detach().float().cpu())
        if state_pool_predictor is not None:
            canonical_prediction_rows.append(
                latents["canonical_prediction"].detach().float().cpu()
            )
            canonical_prediction_target_rows.append(
                latents["canonical_target"].detach().float().cpu()
            )
            shuffle_mask = latents["canonical_shuffle_mask"].detach()
            if bool(shuffle_mask.any()):
                canonical_matched_prediction_rows.append(
                    latents["canonical_prediction"][shuffle_mask].detach().float().cpu()
                )
                canonical_shuffled_prediction_rows.append(
                    latents["canonical_shuffled_prediction"][shuffle_mask]
                    .detach()
                    .float()
                    .cpu()
                )
                canonical_shuffled_target_rows.append(
                    latents["canonical_target"][shuffle_mask].detach().float().cpu()
                )
            canonical_prediction_families.extend(
                str(value) for value in batch["family"]
            )
        valid_prediction = latents["prediction"][latents["target_mask"]].float()
        valid_target = latents["target"][latents["target_mask"]].float()
        token_error_sum += float((valid_prediction - valid_target).square().sum())
        token_element_count += valid_target.numel()
        batch_target_sum = valid_target.sum(dim=0).cpu()
        batch_target_square_sum = valid_target.square().sum(dim=0).cpu()
        token_target_sum = (
            batch_target_sum
            if token_target_sum is None
            else token_target_sum + batch_target_sum
        )
        token_target_square_sum = (
            batch_target_square_sum
            if token_target_square_sum is None
            else token_target_square_sum + batch_target_square_sum
        )
        token_target_row_count += valid_target.shape[0]
        pooled_prediction = latents["predictive"].float()
        pooled_target = latents["pooled_target"].float()
        pooled_error_sum += float((pooled_prediction - pooled_target).square().sum())
        pooled_element_count += pooled_target.numel()
        batch_pooled_sum = pooled_target.sum(dim=0).cpu()
        batch_pooled_square_sum = pooled_target.square().sum(dim=0).cpu()
        pooled_target_sum = (
            batch_pooled_sum
            if pooled_target_sum is None
            else pooled_target_sum + batch_pooled_sum
        )
        pooled_target_square_sum = (
            batch_pooled_square_sum
            if pooled_target_square_sum is None
            else pooled_target_square_sum + batch_pooled_square_sum
        )
        pooled_target_row_count += pooled_target.shape[0]
        if semantic_slot_predictor is not None:
            valid_slots = latents["semantic_slot_valid"]
            semantic_slot_valid_count += int(valid_slots.sum())
            if semantic_active_concepts is not None:
                semantic_slot_candidate_count += current_batch_size * int(
                    semantic_active_concepts.sum()
                )
            if bool(valid_slots.any()):
                semantic_slot_prediction_rows.append(
                    latents["semantic_slot_prediction"][valid_slots]
                    .detach()
                    .float()
                    .cpu()
                )
                semantic_slot_target_rows.append(
                    latents["semantic_slot_target"][valid_slots]
                    .detach()
                    .float()
                    .cpu()
                )
        target_context = model.target_encoder.encode_with_special_tokens(
            batch,
            "context",
        )
        target_canonical = model.select_canonical_state(target_context)
        target_canonical_rows.append(target_canonical.detach().float().cpu())
        for row, family in enumerate(batch["family"]):
            canonical_by_family[str(family)].append(
                latents["canonical"][row : row + 1].detach().float().cpu()
            )
            target_canonical_by_family[str(family)].append(
                target_canonical[row : row + 1].detach().float().cpu()
            )
    canonical_diagnostics = collapse_diagnostics(torch.cat(canonical_rows, dim=0))
    belief_diagnostics = collapse_diagnostics(torch.cat(belief_rows, dim=0))
    reg_diagnostics = collapse_diagnostics(torch.cat(reg_rows, dim=0))
    target_canonical_diagnostics = collapse_diagnostics(
        torch.cat(target_canonical_rows, dim=0)
    )
    assert token_target_sum is not None and token_target_square_sum is not None
    assert pooled_target_sum is not None and pooled_target_square_sum is not None
    token_variance = (
        (
            token_target_square_sum / token_target_row_count
            - (token_target_sum / token_target_row_count).square()
        )
        .clamp_min(0.0)
        .mean()
    )
    pooled_variance = (
        (
            pooled_target_square_sum / pooled_target_row_count
            - (pooled_target_sum / pooled_target_row_count).square()
        )
        .clamp_min(0.0)
        .mean()
    )
    token_rmse = math.sqrt(token_error_sum / token_element_count)
    pooled_rmse = math.sqrt(pooled_error_sum / pooled_element_count)
    canonical_prediction_metrics: dict[str, float] = {}
    if canonical_prediction_rows:
        state_prediction = torch.cat(canonical_prediction_rows, dim=0)
        state_target = torch.cat(canonical_prediction_target_rows, dim=0)
        prediction_mse = float((state_prediction - state_target).square().mean())
        target_variance = float(state_target.var(dim=0, unbiased=False).mean())
        centered_prediction = _center_rows_by_family(
            state_prediction,
            canonical_prediction_families,
        )
        centered_target = _center_rows_by_family(
            state_target,
            canonical_prediction_families,
        )
        centered_mse = float((centered_prediction - centered_target).square().mean())
        centered_variance = float(centered_target.square().mean())
        matched_rows = 0
        matched_prediction_mse = prediction_mse
        shuffled_mse = prediction_mse
        if canonical_shuffled_prediction_rows:
            matched_prediction = torch.cat(canonical_matched_prediction_rows, dim=0)
            shuffled_prediction = torch.cat(canonical_shuffled_prediction_rows, dim=0)
            shuffled_target = torch.cat(canonical_shuffled_target_rows, dim=0)
            matched_rows = int(shuffled_target.shape[0])
            matched_prediction_mse = float(
                (matched_prediction - shuffled_target).square().mean()
            )
            shuffled_mse = float(
                (shuffled_prediction - shuffled_target).square().mean()
            )
        canonical_prediction_metrics = {
            "canonical_prediction_rmse": math.sqrt(prediction_mse),
            "canonical_prediction_target_variance": target_variance,
            "canonical_prediction_relative_rmse": math.sqrt(
                prediction_mse / max(target_variance, 1e-12)
            ),
            "canonical_prediction_within_family_target_variance": centered_variance,
            "canonical_prediction_within_family_relative_rmse": math.sqrt(
                centered_mse / max(centered_variance, 1e-12)
            ),
            "canonical_prediction_matched_shuffle_rows": matched_rows,
            "canonical_prediction_matched_rmse": math.sqrt(matched_prediction_mse),
            "canonical_prediction_shuffled_rmse": math.sqrt(shuffled_mse),
            "canonical_prediction_shuffled_mse_ratio": shuffled_mse
            / max(matched_prediction_mse, 1e-12),
        }
    semantic_slot_metrics: dict[str, float] = {}
    if semantic_slot_prediction_rows:
        slot_prediction = torch.cat(semantic_slot_prediction_rows, dim=0)
        slot_target = torch.cat(semantic_slot_target_rows, dim=0)
        slot_mse = float((slot_prediction - slot_target).square().mean())
        slot_variance = float(slot_target.var(dim=0, unbiased=False).mean())
        semantic_slot_metrics = {
            "semantic_slot_valid_count": semantic_slot_valid_count,
            "semantic_slot_valid_fraction": semantic_slot_valid_count
            / max(semantic_slot_candidate_count, 1),
            "semantic_slot_rmse": math.sqrt(slot_mse),
            "semantic_slot_target_variance": slot_variance,
            "semantic_slot_relative_rmse": math.sqrt(
                slot_mse / max(slot_variance, 1e-12)
            ),
            **{
                f"semantic_slot_prediction_{name}": value
                for name, value in collapse_diagnostics(slot_prediction).items()
            },
            **{
                f"semantic_slot_target_{name}": value
                for name, value in collapse_diagnostics(slot_target).items()
            },
        }
    elif semantic_slot_predictor is not None:
        semantic_slot_metrics = {
            "semantic_slot_valid_count": 0,
            "semantic_slot_valid_fraction": 0.0,
        }
    family_diagnostics: dict[str, float] = {}
    for family, rows in sorted(canonical_by_family.items()):
        slug = _normal_name(family)
        online = collapse_diagnostics(torch.cat(rows, dim=0))
        target_rows = collapse_diagnostics(
            torch.cat(target_canonical_by_family[family], dim=0)
        )
        family_diagnostics.update(
            {f"family_{slug}_{name}": value for name, value in online.items()}
        )
        family_diagnostics.update(
            {
                f"target_family_{slug}_{name}": value
                for name, value in target_rows.items()
            }
        )
    return {
        **{name: value / total_weight for name, value in totals.items()},
        # Backward-compatible aliases refer to the configured deployed state.
        **canonical_diagnostics,
        **{f"canonical_{name}": value for name, value in canonical_diagnostics.items()},
        **{f"belief_{name}": value for name, value in belief_diagnostics.items()},
        **{f"reg_{name}": value for name, value in reg_diagnostics.items()},
        "token_rmse": token_rmse,
        "token_relative_rmse": token_rmse
        / math.sqrt(max(float(token_variance), 1e-12)),
        "pooled_rmse": pooled_rmse,
        "pooled_relative_rmse": pooled_rmse
        / math.sqrt(max(float(pooled_variance), 1e-12)),
        **canonical_prediction_metrics,
        **semantic_slot_metrics,
        **{
            f"target_canonical_{name}": value
            for name, value in target_canonical_diagnostics.items()
        },
        **family_diagnostics,
    }


def run_state(args: argparse.Namespace) -> dict[str, Any]:
    raw_config = _training_config(args.model_config, args.config)
    config = _apply_model_overrides(raw_config, args)
    seed = int(args.seed if args.seed is not None else config["seed"])
    _seed_everything(seed)
    device = _device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    preprocessing_started = time.perf_counter()
    records = _load_records(args.data, config)
    train_records = [
        record
        for record in records
        if record["split"] == TRAIN_SPLIT
        and bool(record["objective_eligibility"].get("state_jepa"))
    ]
    dev_records = [
        record
        for record in records
        if record["split"] == DEV_SPLIT
        and bool(record["objective_eligibility"].get("state_jepa"))
    ]
    if not train_records:
        raise ValueError("State training requires pretrain_train records")
    train_record_count = len(train_records)
    dev_record_count = len(dev_records)
    print(
        json.dumps(
            {
                "event": "state_preprocessing",
                "phase": "records_loaded",
                "elapsed_seconds": time.perf_counter() - preprocessing_started,
                "train_records": train_record_count,
                "dev_records": dev_record_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    state_sources = _source_families(train_records)
    tensorizer_started = time.perf_counter()
    tensorizer_state = _state_tensorizer_state(
        records,
        init_checkpoint=args.init_checkpoint,
        reuse_parent_tensorizer=bool(args.reuse_parent_tensorizer),
    )
    print(
        json.dumps(
            {
                "event": "state_preprocessing",
                "phase": (
                    "tensorizer_reused_from_parent"
                    if args.reuse_parent_tensorizer
                    else "tensorizer_fitted"
                ),
                "elapsed_seconds": time.perf_counter() - tensorizer_started,
                "feature_count": tensorizer_state.feature_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tensorizer = EpisodeTensorizer(tensorizer_state)
    # Release canonical dictionaries incrementally while tensorizing. Keeping
    # both the full JSON object graph and every tensorized corpus record
    # episode alive at once needlessly pushes a host machine toward host OOM.
    del records
    train_queue = deque(train_records)
    del train_records
    train_episodes = []
    while train_queue:
        train_episodes.append(tensorizer.transform(train_queue.popleft()))
    dev_queue = deque(dev_records)
    del dev_records
    if dev_queue:
        dev_episodes = []
        while dev_queue:
            dev_episodes.append(tensorizer.transform(dev_queue.popleft()))
    else:
        dev_episodes = train_episodes
    gc.collect()
    print(
        json.dumps(
            {
                "event": "state_preprocessing",
                "phase": "episodes_tensorized",
                "elapsed_seconds": time.perf_counter() - preprocessing_started,
                "train_episodes": len(train_episodes),
                "dev_episodes": len(dev_episodes),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    required_local_index_capacity, actual_local_index_capacity = (
        _expand_local_index_capacity(config, [*train_episodes, *dev_episodes])
    )
    training = config["state_training"]
    source_weights = {
        str(family): float(weight)
        for family, weight in training["source_sampling_weights"].items()
    }
    train_episode_pools = _state_episode_pools(train_episodes, source_weights)
    dev_episode_pools = _state_episode_pools(dev_episodes, source_weights)
    model_config = _model_config(config, tensorizer_state)
    model = SupplyChainJEPACore(model_config).to(device)
    for parameter in model.dynamics_predictor.parameters():
        parameter.requires_grad_(False)
    model.dynamics_predictor.eval()
    grounding_head = nn.Linear(model_config.d_model, model_config.feature_count).to(
        device
    )
    initialization_parent: Mapping[str, Any] | None = None
    if args.init_checkpoint is not None:
        initialization_parent = _initialize_state_checkpoint(
            args.init_checkpoint,
            tensorizer_state=tensorizer_state,
            model=model,
            grounding_head=grounding_head,
        )
    anti_collapse = dict(
        training.get("anti_collapse", {"method": "legacy_mixed_vicreg"})
    )
    anti_collapse_method = str(anti_collapse.get("method", "legacy_mixed_vicreg"))
    if anti_collapse_method not in {
        "legacy_mixed_vicreg",
        "vicreg",
        "sigreg",
        "none",
    }:
        raise ValueError(f"Unsupported anti-collapse method {anti_collapse_method!r}")
    anti_collapse_projection = str(anti_collapse.get("projection", "mlp"))
    if anti_collapse_projection not in {"identity", "mlp"}:
        raise ValueError(
            f"Unsupported anti-collapse projection {anti_collapse_projection!r}"
        )
    regularization_projector = None
    if (
        anti_collapse_method in {"vicreg", "sigreg"}
        and anti_collapse_projection == "mlp"
    ):
        regularization_projector = _StateRegularizationProjector(
            model_config.d_model,
            int(anti_collapse.get("projector_hidden_multiplier", 2)),
            hidden_width=(
                int(anti_collapse["projector_hidden_width"])
                if "projector_hidden_width" in anti_collapse
                else None
            ),
            output_width=(
                int(anti_collapse["projector_output_width"])
                if "projector_output_width" in anti_collapse
                else None
            ),
        ).to(device)
    canonical_prediction = dict(training.get("canonical_prediction", {}))
    canonical_prediction_enabled = bool(canonical_prediction.get("enabled", False))
    canonical_prediction_weight = float(canonical_prediction.get("weight", 0.0))
    if canonical_prediction_enabled and canonical_prediction_weight <= 0.0:
        raise ValueError("Enabled canonical prediction requires a positive weight")
    canonical_prediction_warmup_updates = int(
        canonical_prediction.get("warmup_updates", 0)
    )
    if canonical_prediction_warmup_updates < 0:
        raise ValueError("canonical prediction warmup cannot be negative")
    query_labels = tuple(
        str(value)
        for value in canonical_prediction.get("query_types", STATE_QUERY_TYPES)
    )
    mask_labels = tuple(
        str(value)
        for value in canonical_prediction.get(
            "mask_kinds",
            (*config["masking"]["current_kinds"], "future_horizon"),
        )
    )
    if len(set(query_labels)) != len(query_labels) or not query_labels:
        raise ValueError("canonical prediction query types must be unique")
    if len(set(mask_labels)) != len(mask_labels) or not mask_labels:
        raise ValueError("canonical prediction mask kinds must be unique")
    query_vocabulary = {value: index for index, value in enumerate(query_labels)}
    mask_vocabulary = {value: index for index, value in enumerate(mask_labels)}
    state_pool_predictor = None
    if canonical_prediction_enabled:
        state_pool_predictor = StatePoolPredictor(
            model_config.d_model,
            query_vocab_size=len(query_vocabulary),
            mask_vocab_size=len(mask_vocabulary),
            hidden_multiplier=int(canonical_prediction.get("hidden_multiplier", 2)),
            dropout=float(canonical_prediction.get("dropout", 0.0)),
        ).to(device)
        if initialization_parent is not None:
            parent_pool_state = initialization_parent.get("state_pool_predictor_state")
            if parent_pool_state is None:
                raise ValueError(
                    "State initialization checkpoint lacks the State pool predictor"
                )
            state_pool_predictor.load_state_dict(parent_pool_state, strict=True)
    semantic_slots = dict(training.get("semantic_concept_slots", {}))
    semantic_slots_enabled = bool(semantic_slots.get("enabled", False))
    semantic_slot_weight = float(semantic_slots.get("weight", 0.0))
    if semantic_slots_enabled and semantic_slot_weight <= 0.0:
        raise ValueError("Enabled semantic concept slots require a positive weight")
    semantic_slot_warmup_updates = int(semantic_slots.get("warmup_updates", 0))
    if semantic_slot_warmup_updates < 0:
        raise ValueError("semantic concept slot warmup cannot be negative")
    excluded_concepts = {
        str(value) for value in semantic_slots.get("exclude_concepts", ())
    }
    unknown_exclusions = sorted(excluded_concepts - set(FEATURE_CONCEPTS))
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded semantic concepts: {unknown_exclusions}")
    semantic_active_concepts = torch.tensor(
        [name not in excluded_concepts for name in FEATURE_CONCEPTS],
        dtype=torch.bool,
        device=device,
    )
    semantic_concept_matrix = _feature_concept_matrix(
        tensorizer_state.feature_names,
        device=device,
    )
    semantic_slot_predictor = None
    if semantic_slots_enabled:
        if not bool(semantic_active_concepts.any()):
            raise ValueError("Semantic concept slots require an active concept")
        semantic_slot_predictor = SemanticConceptSlotPredictor(
            model_config.d_model,
            concept_count=len(FEATURE_CONCEPTS),
            nhead=model_config.nhead,
            decoder_layers=int(semantic_slots.get("decoder_layers", 1)),
            ffn_multiplier=int(semantic_slots.get("ffn_multiplier", 2)),
            dropout=float(semantic_slots.get("dropout", 0.0)),
        ).to(device)
    optimized_parameters = [
        *(
            parameter
            for parameter in model.online_encoder.parameters()
            if parameter.requires_grad
        ),
        *(parameter for parameter in model.state_predictor.parameters() if parameter.requires_grad),
        *(parameter for parameter in grounding_head.parameters() if parameter.requires_grad),
    ]
    if regularization_projector is not None:
        optimized_parameters.extend(
            parameter
            for parameter in regularization_projector.parameters()
            if parameter.requires_grad
        )
    if state_pool_predictor is not None:
        optimized_parameters.extend(
            parameter
            for parameter in state_pool_predictor.parameters()
            if parameter.requires_grad
        )
    if semantic_slot_predictor is not None:
        optimized_parameters.extend(
            parameter
            for parameter in semantic_slot_predictor.parameters()
            if parameter.requires_grad
        )
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(training["learning_rate"]),
        betas=tuple(training.get("betas", (0.9, 0.95))),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = _scaler(device, amp_enabled)
    mask_config = config["masking"]
    current_kinds = tuple(mask_config["current_kinds"])
    sampler = StructuredMaskSampler(
        seed=seed,
        min_time_span=int(mask_config["min_time_span"]),
        max_time_span=int(mask_config["max_time_span"]),
        max_entity_group=int(mask_config["max_entity_group"]),
        neighborhood_radius=int(mask_config["neighborhood_radius"]),
        future_horizons=tuple(mask_config["future_horizons"]),
    )
    window_spec = WindowSpec(
        max_tokens=int(config["tensorization"]["max_tokens"]),
        target_window_steps=int(config["tensorization"]["target_window_steps"]),
        seed=seed,
    )
    weights = training["loss_weights"]
    pooled_state_weight = float(training.get("pooled_state_weight", 0.0))
    grounding_feature_mask = torch.tensor(
        [
            kind == "numeric"
            and any(pattern in name.lower() for pattern in GROUNDING_PATTERNS)
            for name, kind in zip(
                tensorizer_state.feature_names,
                tensorizer_state.feature_kinds, strict=False,
            )
        ],
        dtype=torch.bool,
        device=device,
    )
    last: dict[str, float] = {}
    last_gradient_norm = 0.0
    train_source_counts: Counter[str] = Counter()
    max_updates = int(args.max_updates)
    accumulation_steps = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else training.get("gradient_accumulation_steps", 1)
    )
    evaluation_samples = int(
        args.eval_samples
        if args.eval_samples is not None
        else training.get("evaluation_samples", args.batch_size)
    )
    evaluation_batch_size = int(
        training.get("evaluation_batch_size", min(int(args.batch_size), 8))
    )
    deployment_evaluation_samples = int(
        training.get(
            "deployment_evaluation_samples",
            min(evaluation_samples, 256),
        )
    )
    minimum_rows_per_family = int(training.get("minimum_family_rows_per_batch", 0))
    regularization_token_samples = int(training.get("regularization_token_samples", 0))
    schema_residual_warmup_updates = int(
        training.get("schema_residual_warmup_updates", 0)
    )
    anti_collapse_warmup_updates = int(anti_collapse.get("warmup_updates", 0))
    if accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    if evaluation_samples < 1:
        raise ValueError("State evaluation sample count must be positive")
    if evaluation_batch_size < 1:
        raise ValueError("State evaluation batch size must be positive")
    if deployment_evaluation_samples < 1:
        raise ValueError("State deployment evaluation sample count must be positive")
    if regularization_token_samples < 0:
        raise ValueError("regularization_token_samples cannot be negative")
    if schema_residual_warmup_updates < 0:
        raise ValueError("schema_residual_warmup_updates cannot be negative")
    base_learning_rate = float(training["learning_rate"])
    warmup_fraction = float(training.get("warmup_fraction", 0.0))
    minimum_lr_ratio = float(training.get("minimum_lr_ratio", 1.0))
    explicit_ema_updates = training.get("ema_updates_before_freeze")
    ema_freeze_fraction = training.get("ema_freeze_fraction")
    if explicit_ema_updates is not None and ema_freeze_fraction is not None:
        raise ValueError(
            "Specify only one of ema_updates_before_freeze and ema_freeze_fraction"
        )
    if explicit_ema_updates is not None:
        ema_update_limit = min(max_updates, max(0, int(explicit_ema_updates)))
    elif ema_freeze_fraction is not None:
        fraction = float(ema_freeze_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("ema_freeze_fraction must be in (0, 1]")
        ema_update_limit = min(max_updates, max(1, math.ceil(max_updates * fraction)))
    else:
        ema_update_limit = max_updates
    ema_updates_executed = 0
    progress_interval = max(1, max_updates // 10)
    for update in range(max_updates):
        if model_config.feature_semantic_mode == "concept_residual":
            migration_progress = (
                1.0
                if schema_residual_warmup_updates == 0
                else min(1.0, update / schema_residual_warmup_updates)
            )
            configured_keep_probability = float(
                model_config.schema_residual_keep_probability
            )
            online_keep_probability = 1.0 - (
                1.0 - configured_keep_probability
            ) * migration_progress
            target_schema_scale = 1.0 - migration_progress
            model.set_semantic_schema_schedule(
                online_keep_probability=online_keep_probability,
                target_scale=target_schema_scale,
            )
        else:
            online_keep_probability = 1.0
            target_schema_scale = 1.0
        model.train()
        grounding_head.train()
        if regularization_projector is not None:
            regularization_projector.train()
        if state_pool_predictor is not None:
            state_pool_predictor.train()
        if semantic_slot_predictor is not None:
            semantic_slot_predictor.train()
        learning_rate = base_learning_rate * _cosine_learning_rate_scale(
            update,
            max_updates,
            warmup_fraction=warmup_fraction,
            minimum_ratio=minimum_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        anti_collapse_scale = (
            1.0
            if anti_collapse_warmup_updates <= 0
            else min(1.0, (update + 1) / anti_collapse_warmup_updates)
        )
        canonical_prediction_scale = (
            1.0
            if canonical_prediction_warmup_updates <= 0
            else min(
                1.0,
                (update + 1) / canonical_prediction_warmup_updates,
            )
        )
        semantic_slot_scale = (
            1.0
            if semantic_slot_warmup_updates <= 0
            else min(1.0, (update + 1) / semantic_slot_warmup_updates)
        )
        accumulated_total = 0.0
        accumulated_pieces: Counter[str] = Counter()
        latest_latents: dict[str, Tensor] | None = None
        for micro_step in range(accumulation_steps):
            sampling_update = update * accumulation_steps + micro_step
            batch_examples = _state_examples(
                train_episode_pools,
                batch_size=int(args.batch_size),
                update=sampling_update,
                seed=seed,
                source_weights=source_weights,
                sampler=sampler,
                window_spec=window_spec,
                current_kinds=current_kinds,
                minimum_rows_per_family=minimum_rows_per_family,
            )
            train_source_counts.update(example.family for example in batch_examples)
            batch = _move_batch(collate_state_examples(batch_examples), device)
            with _amp_context(device, amp_enabled):
                total, pieces, latest_latents = _state_objective(
                    model,
                    grounding_head,
                    regularization_projector,
                    state_pool_predictor,
                    batch,
                    lambda_variance=float(weights["variance"]),
                    lambda_covariance=float(weights["covariance"]),
                    lambda_grounding=float(weights["grounding"]),
                    pooled_state_weight=pooled_state_weight,
                    anti_collapse=anti_collapse,
                    canonical_prediction=canonical_prediction,
                    query_vocabulary=query_vocabulary,
                    mask_vocabulary=mask_vocabulary,
                    grounding_feature_mask=grounding_feature_mask,
                    semantic_slot_predictor=semantic_slot_predictor,
                    semantic_concept_matrix=semantic_concept_matrix,
                    semantic_active_concepts=semantic_active_concepts,
                    semantic_slots=semantic_slots,
                    regularization_token_samples=regularization_token_samples,
                    anti_collapse_scale=anti_collapse_scale,
                    canonical_prediction_scale=canonical_prediction_scale,
                    semantic_slot_scale=semantic_slot_scale,
                )
            scaler.scale(total / accumulation_steps).backward()
            accumulated_total += float(total.detach())
            for name, value in pieces.items():
                accumulated_pieces[name] += float(value.detach())
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ],
            float(training["gradient_clip"]),
        )
        last_gradient_norm = float(gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        momentum: float | None = None
        if update < ema_update_limit:
            progress = (update + 1) / max(ema_update_limit, 1)
            momentum = float(training["ema_momentum_start"]) + progress * (
                float(training["ema_momentum_end"])
                - float(training["ema_momentum_start"])
            )
            model.update_target(momentum)
            ema_updates_executed += 1
        last = {
            "train_loss": accumulated_total / accumulation_steps,
            **{
                f"train_{name}_loss": value / accumulation_steps
                for name, value in accumulated_pieces.items()
            },
            "ema_momentum": momentum,
            "learning_rate": learning_rate,
            "anti_collapse_scale": anti_collapse_scale,
            "canonical_prediction_scale": canonical_prediction_scale,
            "semantic_slot_scale": semantic_slot_scale,
            "online_schema_keep_probability": online_keep_probability,
            "target_schema_residual_scale": target_schema_scale,
        }
        if update == 0 or (update + 1) % progress_interval == 0:
            assert latest_latents is not None
            batch_geometry = collapse_diagnostics(
                latest_latents["canonical"].detach().float()
            )
            last.update(
                {
                    f"train_batch_canonical_{name}": value
                    for name, value in batch_geometry.items()
                }
            )
            print(
                json.dumps(
                    {
                        "event": "state_progress",
                        "update": update + 1,
                        **last,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    # Adam states, gradients, and the AMP scaler are no longer needed.  Free
    # them before evaluation so the target pass has predictable headroom on a
    # 15 GB T4 even when the largest training micro-batch was near capacity.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dev_metrics = _evaluate_state(
        model,
        grounding_head,
        regularization_projector,
        state_pool_predictor,
        dev_episode_pools,
        sampler=sampler,
        window_spec=window_spec,
        batch_size=evaluation_batch_size,
        current_kinds=current_kinds,
        seed=seed,
        source_weights=source_weights,
        device=device,
        weights=weights,
        pooled_state_weight=pooled_state_weight,
        anti_collapse=anti_collapse,
        canonical_prediction=canonical_prediction,
        query_vocabulary=query_vocabulary,
        mask_vocabulary=mask_vocabulary,
        grounding_feature_mask=grounding_feature_mask,
        evaluation_samples=evaluation_samples,
        regularization_token_samples=regularization_token_samples,
        minimum_rows_per_family=minimum_rows_per_family,
        semantic_slot_predictor=semantic_slot_predictor,
        semantic_concept_matrix=semantic_concept_matrix,
        semantic_active_concepts=semantic_active_concepts,
        semantic_slots=semantic_slots,
    )
    state_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in [*model.parameters(), *grounding_head.parameters()]
        if parameter.requires_grad
    )
    state_trainable_parameter_count += sum(
        parameter.numel()
        for parameter in (
            regularization_projector.parameters()
            if regularization_projector is not None
            else ()
        )
        if parameter.requires_grad
    )
    state_trainable_parameter_count += sum(
        parameter.numel()
        for parameter in (
            semantic_slot_predictor.parameters()
            if semantic_slot_predictor is not None
            else ()
        )
        if parameter.requires_grad
    )
    state_trainable_parameter_count += sum(
        parameter.numel()
        for parameter in (
            state_pool_predictor.parameters()
            if state_pool_predictor is not None
            else ()
        )
        if parameter.requires_grad
    )
    deployment_encoder_source = str(training.get("deployment_encoder_source", "online"))
    model.canonicalize_encoder(source=deployment_encoder_source)
    finalization_updates = int(training.get("finalization_updates", 0))
    if finalization_updates < 0:
        raise ValueError("State predictor finalization updates cannot be negative")
    finalization_learning_rate = float(
        training.get("finalization_learning_rate", base_learning_rate * 0.5)
    )
    if finalization_learning_rate <= 0.0:
        raise ValueError("State predictor finalization learning rate must be positive")
    finalization_source_counts: Counter[str] = Counter()
    finalization_last: dict[str, float] = {}
    finalization_gradient_norm = 0.0
    if finalization_updates > 0:
        # The State predictors were trained against EMA targets.  Once the
        # selected deployment encoder is hard-synchronized, briefly realign
        # only the lightweight prediction heads to that final latent space.
        # Encoder geometry is frozen and therefore cannot drift or collapse.
        for encoder in (model.online_encoder, model.target_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            encoder.eval()
        if regularization_projector is not None:
            for parameter in regularization_projector.parameters():
                parameter.requires_grad_(False)
            regularization_projector.eval()
        finalization_parameters = [
            *model.state_predictor.parameters(),
            *grounding_head.parameters(),
        ]
        if state_pool_predictor is not None:
            finalization_parameters.extend(state_pool_predictor.parameters())
        if semantic_slot_predictor is not None:
            finalization_parameters.extend(semantic_slot_predictor.parameters())
        finalization_optimizer = torch.optim.AdamW(
            finalization_parameters,
            lr=finalization_learning_rate,
            betas=tuple(training.get("betas", (0.9, 0.95))),
            weight_decay=float(training["weight_decay"]),
        )
        finalization_scaler = _scaler(device, amp_enabled)
        finalization_interval = max(1, finalization_updates // 5)
        for finalization_update in range(finalization_updates):
            model.train()
            grounding_head.train()
            if state_pool_predictor is not None:
                state_pool_predictor.train()
            if semantic_slot_predictor is not None:
                semantic_slot_predictor.train()
            learning_rate = finalization_learning_rate * _cosine_learning_rate_scale(
                finalization_update,
                finalization_updates,
                warmup_fraction=0.1,
                minimum_ratio=0.2,
            )
            for group in finalization_optimizer.param_groups:
                group["lr"] = learning_rate
            finalization_optimizer.zero_grad(set_to_none=True)
            accumulated_total = 0.0
            accumulated_pieces: Counter[str] = Counter()
            for micro_step in range(accumulation_steps):
                sampling_update = (
                    max_updates * accumulation_steps
                    + finalization_update * accumulation_steps
                    + micro_step
                )
                batch_examples = _state_examples(
                    train_episode_pools,
                    batch_size=int(args.batch_size),
                    update=sampling_update,
                    seed=seed,
                    source_weights=source_weights,
                    sampler=sampler,
                    window_spec=window_spec,
                    current_kinds=current_kinds,
                    minimum_rows_per_family=minimum_rows_per_family,
                )
                finalization_source_counts.update(
                    example.family for example in batch_examples
                )
                batch = _move_batch(collate_state_examples(batch_examples), device)
                with _amp_context(device, amp_enabled):
                    total, pieces, _ = _state_objective(
                        model,
                        grounding_head,
                        None,
                        state_pool_predictor,
                        batch,
                        lambda_variance=0.0,
                        lambda_covariance=0.0,
                        lambda_grounding=float(weights["grounding"]),
                        pooled_state_weight=pooled_state_weight,
                        anti_collapse={"method": "none"},
                        canonical_prediction=canonical_prediction,
                        query_vocabulary=query_vocabulary,
                        mask_vocabulary=mask_vocabulary,
                        grounding_feature_mask=grounding_feature_mask,
                        semantic_slot_predictor=semantic_slot_predictor,
                        semantic_concept_matrix=semantic_concept_matrix,
                        semantic_active_concepts=semantic_active_concepts,
                        semantic_slots=semantic_slots,
                        canonical_prediction_scale=1.0,
                        semantic_slot_scale=1.0,
                    )
                finalization_scaler.scale(total / accumulation_steps).backward()
                accumulated_total += float(total.detach())
                for name, value in pieces.items():
                    accumulated_pieces[name] += float(value.detach())
            finalization_scaler.unscale_(finalization_optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                finalization_parameters,
                float(training["gradient_clip"]),
            )
            finalization_gradient_norm = float(gradient_norm)
            finalization_scaler.step(finalization_optimizer)
            finalization_scaler.update()
            finalization_last = {
                "train_loss": accumulated_total / accumulation_steps,
                **{
                    f"train_{name}_loss": value / accumulation_steps
                    for name, value in accumulated_pieces.items()
                },
                "learning_rate": learning_rate,
            }
            if (
                finalization_update == 0
                or (finalization_update + 1) % finalization_interval == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "state_predictor_finalization_progress",
                            "update": finalization_update + 1,
                            **finalization_last,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finalization_optimizer.zero_grad(set_to_none=True)
        del finalization_optimizer, finalization_scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    deployed_dev_metrics = _evaluate_state(
        model,
        grounding_head,
        regularization_projector,
        state_pool_predictor,
        dev_episode_pools,
        sampler=sampler,
        window_spec=window_spec,
        batch_size=evaluation_batch_size,
        current_kinds=current_kinds,
        seed=seed,
        source_weights=source_weights,
        device=device,
        weights=weights,
        pooled_state_weight=pooled_state_weight,
        anti_collapse=anti_collapse,
        canonical_prediction=canonical_prediction,
        query_vocabulary=query_vocabulary,
        mask_vocabulary=mask_vocabulary,
        grounding_feature_mask=grounding_feature_mask,
        evaluation_samples=deployment_evaluation_samples,
        regularization_token_samples=regularization_token_samples,
        minimum_rows_per_family=minimum_rows_per_family,
        semantic_slot_predictor=semantic_slot_predictor,
        semantic_concept_matrix=semantic_concept_matrix,
        semantic_active_concepts=semantic_active_concepts,
        semantic_slots=semantic_slots,
    )
    metrics: dict[str, Any] = {
        "stage": "state",
        "updates": max_updates,
        "micro_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": int(args.batch_size) * accumulation_steps,
        "evaluation_samples": evaluation_samples,
        "deployment_evaluation_samples": deployment_evaluation_samples,
        "evaluation_batch_size": evaluation_batch_size,
        "minimum_family_rows_per_batch": minimum_rows_per_family,
        "regularization_token_samples": regularization_token_samples,
        "anti_collapse_warmup_updates": anti_collapse_warmup_updates,
        "anti_collapse_method": anti_collapse_method,
        "anti_collapse_projection": anti_collapse_projection,
        "canonical_state_pooling": model_config.canonical_state_pooling,
        "local_index_mode": model_config.local_index_mode,
        "feature_semantic_mode": model_config.feature_semantic_mode,
        "concept_vocab_size": model_config.concept_vocab_size,
        "schema_residual_warmup_updates": schema_residual_warmup_updates,
        "schema_residual_keep_probability": (
            model_config.schema_residual_keep_probability
        ),
        "transfer_schema_residual_scale": (
            model_config.transfer_schema_residual_scale
        ),
        "initialization_checkpoint": (
            str(Path(args.init_checkpoint).resolve())
            if args.init_checkpoint is not None
            else None
        ),
        "initialization_checkpoint_sha256": (
            _file_sha256(Path(args.init_checkpoint))
            if args.init_checkpoint is not None
            else None
        ),
        "pooled_state_weight": pooled_state_weight,
        "canonical_prediction_enabled": canonical_prediction_enabled,
        "canonical_prediction_weight": canonical_prediction_weight,
        "canonical_prediction_warmup_updates": canonical_prediction_warmup_updates,
        "canonical_prediction_propagates_encoder_gradients": bool(
            canonical_prediction.get("propagate_encoder_gradients", True)
        ),
        "semantic_concept_slots_enabled": semantic_slots_enabled,
        "semantic_concept_slot_weight": semantic_slot_weight,
        "semantic_concept_slot_warmup_updates": semantic_slot_warmup_updates,
        "semantic_concept_slot_active_count": int(
            semantic_active_concepts.sum().item()
        ),
        "semantic_concept_slot_excluded": sorted(excluded_concepts),
        "semantic_concept_slot_propagates_encoder_gradients": bool(
            semantic_slots.get("propagate_encoder_gradients", True)
        ),
        "semantic_concept_slot_predictor_parameter_count": sum(
            parameter.numel()
            for parameter in (
                semantic_slot_predictor.parameters()
                if semantic_slot_predictor is not None
                else ()
            )
        ),
        "state_pool_predictor_parameter_count": sum(
            parameter.numel()
            for parameter in (
                state_pool_predictor.parameters()
                if state_pool_predictor is not None
                else ()
            )
        ),
        "grounding_feature_count": int(grounding_feature_mask.sum().item()),
        "target_policy": (
            "ema_then_frozen" if ema_update_limit < max_updates else "ema_all_updates"
        ),
        "ema_updates_executed": ema_updates_executed,
        "ema_update_limit": ema_update_limit,
        "deployment_encoder_source": deployment_encoder_source,
        "finalization_updates": finalization_updates,
        "finalization_learning_rate": finalization_learning_rate,
        "finalization_gradient_norm": finalization_gradient_norm,
        "finalization_source_counts": dict(sorted(finalization_source_counts.items())),
        **{f"finalization_{name}": value for name, value in finalization_last.items()},
        "regularization_projector_parameter_count": sum(
            parameter.numel()
            for parameter in (
                regularization_projector.parameters()
                if regularization_projector is not None
                else ()
            )
        ),
        "required_local_index_capacity": required_local_index_capacity,
        "actual_local_index_capacity": actual_local_index_capacity,
        "train_record_count": train_record_count,
        "dev_record_count": dev_record_count,
        "device": str(device),
        "amp": amp_enabled,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": state_trainable_parameter_count,
        "gradient_norm": last_gradient_norm,
        "train_source_counts": dict(sorted(train_source_counts.items())),
        **last,
        **{f"dev_{key}": value for key, value in dev_metrics.items()},
        **{f"deployed_dev_{key}": value for key, value in deployed_dev_metrics.items()},
        **_cuda_peak_metrics(device),
    }
    output_dir = Path(args.output_dir)
    checkpoint = {
        "format_version": "supplychain-jepa-checkpoint",
        "stage": "state",
        "updates": max_updates,
        "model_state": model.state_dict(),
        "grounding_head_state": grounding_head.state_dict(),
        "regularization_projector_state": (
            regularization_projector.state_dict()
            if regularization_projector is not None
            else None
        ),
        "state_pool_predictor_state": (
            state_pool_predictor.state_dict()
            if state_pool_predictor is not None
            else None
        ),
        "state_pool_predictor_config": {
            **canonical_prediction,
            "query_vocabulary": query_vocabulary,
            "mask_vocabulary": mask_vocabulary,
        },
        "semantic_concept_slot_predictor_state": (
            semantic_slot_predictor.state_dict()
            if semantic_slot_predictor is not None
            else None
        ),
        "semantic_concept_slot_predictor_config": {
            **semantic_slots,
            "concept_names": FEATURE_CONCEPTS,
            "active_concepts": [
                name for name in FEATURE_CONCEPTS if name not in excluded_concepts
            ],
        },
        "tensorizer_state": tensorizer_state.to_dict(),
        "model_config": model_config.to_dict(),
        "training_config": config,
        "sources": {"state": state_sources},
        "initialization_parent": (
            {
                "path": str(Path(args.init_checkpoint).resolve()),
                "sha256": _file_sha256(Path(args.init_checkpoint)),
                "stage": initialization_parent.get("stage"),
                "updates": initialization_parent.get("updates"),
            }
            if initialization_parent is not None
            else None
        ),
        "metrics": metrics,
    }
    checkpoint_path = _write_outputs(
        output_dir, "state_checkpoint.pt", checkpoint, metrics
    )
    result = {**metrics, "checkpoint": str(checkpoint_path)}
    return result


def _dynamics_groups(
    records: Sequence[Mapping[str, Any]], split: str
) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        branch = record.get("branch")
        if (
            record.get("split") == split
            and bool(record["objective_eligibility"].get("dynamics_jepa"))
            and isinstance(branch, Mapping)
        ):
            grouped[str(branch["branch_group_id"])].append(record)
    return [members for _, members in sorted(grouped.items()) if len(members) >= 2]


def _build_dynamics_source_agnostic(
    records: Sequence[Mapping[str, Any]],
    tensorizer: EpisodeTensorizer,
    *,
    horizons: Sequence[int],
    window_spec: WindowSpec,
    include_controls: bool,
) -> list[DynamicsTrainingExample]:
    """Use the canonical builder while admitting any branch-valid source family."""

    families = {str(record["simulator"]["family"]) for record in records}
    original = windows_module.DYNAMICS_FAMILIES
    windows_module.DYNAMICS_FAMILIES = frozenset(set(original) | families)
    try:
        return build_dynamics_examples(
            records,
            tensorizer,
            horizons=horizons,
            spec=window_spec,
            include_controls=include_controls,
        )
    finally:
        windows_module.DYNAMICS_FAMILIES = original


def _dynamics_units(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    tensorizer: EpisodeTensorizer,
    *,
    horizons: Sequence[int],
    window_spec: WindowSpec,
    include_controls: bool = False,
) -> list[list[DynamicsTrainingExample]]:
    units: list[list[DynamicsTrainingExample]] = []
    for group in groups:
        examples = _build_dynamics_source_agnostic(
            group,
            tensorizer,
            horizons=horizons,
            window_spec=window_spec,
            include_controls=include_controls,
        )
        bucketed: dict[tuple[int, str], list[DynamicsTrainingExample]] = defaultdict(
            list
        )
        for example in examples:
            variant_key = (
                "all_controls" if include_controls else example.control_variant
            )
            bucketed[(example.horizon, variant_key)].append(example)
        for _, members in sorted(bucketed.items()):
            branch_positions = {
                example.branch_position
                for example in members
                if example.control_variant == "correct"
            }
            if len(branch_positions) >= 2:
                units.append(members)
    if not units:
        raise ValueError("No branch pair supports the requested Dynamics horizons")
    return units


def _select_dynamics_batch(
    unit_pools: Mapping[
        str,
        Sequence[Sequence[DynamicsTrainingExample]],
    ],
    *,
    update: int,
    batch_size: int,
    seed: int,
    source_weights: Mapping[str, float],
) -> list[DynamicsTrainingExample]:
    families = sorted(unit_pools)
    if not families:
        raise ValueError("No positively weighted Dynamics source is available")
    weights = [float(source_weights[family]) for family in families]
    rng = random.Random(f"{seed}:dynamics-source:{update}")
    selected: list[DynamicsTrainingExample] = []
    while len(selected) < max(2, batch_size):
        family = rng.choices(families, weights=weights, k=1)[0]
        unit = rng.choice(unit_pools[family])
        selected.extend(unit)
    return selected


def _dynamics_unit_pools(
    units: Sequence[Sequence[DynamicsTrainingExample]],
    source_weights: Mapping[str, float],
) -> dict[str, list[Sequence[DynamicsTrainingExample]]]:
    pools: dict[str, list[Sequence[DynamicsTrainingExample]]] = defaultdict(list)
    for unit in units:
        if not unit:
            continue
        family = str(unit[0].family)
        if float(source_weights.get(family, 0.0)) > 0.0:
            pools[family].append(unit)
    if not pools:
        raise ValueError("No positively weighted Dynamics source is available")
    return dict(pools)


def _paired_loss(
    prediction: Tensor,
    target: Tensor,
    batch: Mapping[str, Any],
    *,
    scale_floor: float | None = None,
) -> Tensor:
    grouped: dict[tuple[str, float], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, metadata in enumerate(batch["batching_metadata"]):
        if batch["control_variant"][index] != "correct":
            continue
        grouped[
            (
                str(metadata["branch_group_id"]),
                float(batch["horizons"][index, 0]),
            )
        ][int(metadata["branch_position"])].append(index)
    losses = []
    for by_position in grouped.values():
        if len(by_position) < 2:
            continue
        ordered = [sorted(rows) for _, rows in sorted(by_position.items())]
        for left, right in zip(ordered[0], ordered[1], strict=False):
            prediction_left = prediction[left : left + 1]
            prediction_right = prediction[right : right + 1]
            target_left = target[left : left + 1]
            target_right = target[right : right + 1]
            if scale_floor is not None:
                target_difference = target_right - target_left
                scale = (
                    target_difference.detach()
                    .square()
                    .mean(dim=tuple(range(1, target_difference.ndim)), keepdim=True)
                    .sqrt()
                    .clamp_min(scale_floor)
                )
                prediction_left = prediction_left / scale
                prediction_right = prediction_right / scale
                target_left = target_left / scale
                target_right = target_right / scale
            losses.append(
                paired_difference_loss(
                    prediction_left,
                    prediction_right,
                    target_left,
                    target_right,
                )
            )
    if not losses:
        raise ValueError("Dynamics batch contains no complete branch pair")
    return torch.stack(losses).mean()


def _branch_contrastive_loss(
    prediction: Tensor,
    target: Tensor,
    batch: Mapping[str, Any],
    *,
    temperature: float,
    mode: str = "absolute",
) -> tuple[Tensor, Tensor, int]:
    """Match each action-conditioned prediction to its sibling branch future.

    Negatives come only from other feasible actions applied to the same
    pre-branch state and horizon.  ``effect_centered`` removes each group's
    shared future component before cosine matching, so the contrast is driven
    by the action-induced difference rather than the common initial state.
    """

    if temperature <= 0.0:
        raise ValueError("branch contrastive temperature must be positive")
    if mode not in {"absolute", "effect_centered"}:
        raise ValueError("branch contrastive mode must be absolute or effect_centered")
    grouped: dict[tuple[str, float], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, metadata in enumerate(batch["batching_metadata"]):
        if batch["control_variant"][index] != "correct":
            continue
        grouped[
            (
                str(metadata["branch_group_id"]),
                float(batch["horizons"][index, 0]),
            )
        ][int(metadata["branch_position"])].append(index)

    losses: list[Tensor] = []
    accuracies: list[Tensor] = []
    comparison_count = 0
    for by_position in grouped.values():
        if len(by_position) < 2:
            continue
        ordered = [sorted(rows) for _, rows in sorted(by_position.items())]
        for occurrence in range(min(len(rows) for rows in ordered)):
            rows = [position_rows[occurrence] for position_rows in ordered]
            predicted = prediction[rows].float().flatten(start_dim=1)
            future = target[rows].float().flatten(start_dim=1)
            if mode == "effect_centered":
                predicted = predicted - predicted.mean(dim=0, keepdim=True)
                future = future - future.mean(dim=0, keepdim=True)
            predicted = F.normalize(predicted, dim=-1, eps=1e-6)
            future = F.normalize(future, dim=-1, eps=1e-6)
            logits = predicted @ future.transpose(0, 1) / float(temperature)
            labels = torch.arange(len(rows), device=logits.device)
            losses.append(
                0.5
                * (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.transpose(0, 1), labels)
                )
            )
            accuracies.append(logits.argmax(dim=1).eq(labels).float().mean())
            comparison_count += len(rows)
    if not losses:
        zero = prediction.new_zeros((), dtype=torch.float32)
        return zero, zero, 0
    return (
        torch.stack(losses).mean(),
        torch.stack(accuracies).mean(),
        comparison_count,
    )


def _control_margin_loss(
    prediction: Tensor,
    target: Tensor,
    batch: Mapping[str, Any],
    *,
    margin: float,
) -> tuple[Tensor, Tensor, int]:
    """Prefer the registered action over no-op/shuffled/wrong-entity controls."""

    grouped: dict[tuple[str, float, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, metadata in enumerate(batch["batching_metadata"]):
        grouped[
            (
                str(metadata["branch_group_id"]),
                float(batch["horizons"][index, 0]),
                int(metadata["branch_position"]),
            )
        ][str(batch["control_variant"][index])].append(index)

    losses: list[Tensor] = []
    wins: list[Tensor] = []
    for variants in grouped.values():
        correct_rows = variants.get("correct", [])
        for occurrence, correct_row in enumerate(correct_rows):
            correct_error = F.smooth_l1_loss(
                prediction[correct_row], target[correct_row], reduction="mean"
            )
            for variant in ("noop", "wrong_entity"):
                control_rows = variants.get(variant, [])
                if occurrence >= len(control_rows):
                    continue
                control_row = control_rows[occurrence]
                control_error = F.smooth_l1_loss(
                    prediction[control_row], target[correct_row], reduction="mean"
                )
                losses.append(F.relu(float(margin) + correct_error - control_error))
                wins.append((correct_error < control_error).to(prediction.dtype))
    if not losses:
        zero = prediction.new_zeros(())
        return zero, zero, 0
    return torch.stack(losses).mean(), torch.stack(wins).mean(), len(losses)


def _shuffled_counterfactual_loss(
    prediction: Tensor,
    target: Tensor,
    batch: Mapping[str, Any],
) -> Tensor:
    """Map each shuffled action to the cyclically next registered future."""

    grouped: dict[tuple[str, float], dict[int, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for index, metadata in enumerate(batch["batching_metadata"]):
        grouped[
            (
                str(metadata["branch_group_id"]),
                float(batch["horizons"][index, 0]),
            )
        ][int(metadata["branch_position"])][
            str(batch["control_variant"][index])
        ].append(index)

    losses: list[Tensor] = []
    for by_position in grouped.values():
        positions = sorted(by_position)
        if len(positions) < 2 or positions != list(range(len(positions))):
            continue
        occurrence_counts = {
            len(by_position[position].get(variant, []))
            for position in positions
            for variant in ("correct", "shuffled")
        }
        if (
            len(occurrence_counts) != 1
            or not occurrence_counts
            or 0 in occurrence_counts
        ):
            continue
        occurrence_count = occurrence_counts.pop()
        for position_index, position in enumerate(positions):
            target_position = positions[(position_index + 1) % len(positions)]
            shuffled_rows = by_position[position]["shuffled"]
            cyclic_targets = by_position[target_position]["correct"]
            for occurrence in range(occurrence_count):
                losses.append(
                    F.smooth_l1_loss(
                        prediction[shuffled_rows[occurrence]],
                        target[cyclic_targets[occurrence]],
                        reduction="mean",
                    )
                )
    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def _rollout_action_padding_mask(
    batch: Mapping[str, Any], *, step_offset: int
) -> Tensor:
    action_times = batch["action_times"]
    padding = batch["action_padding_mask"]
    decision_times = batch["decision_times"].to(action_times.device)
    return padding | action_times.ne((decision_times + int(step_offset))[:, None])


def _two_step_rollout_loss(
    model: SupplyChainJEPACore,
    state: Tensor,
    target: Tensor,
    batch: Mapping[str, Any],
) -> Tensor:
    correct = torch.tensor(
        [variant == "correct" for variant in batch["control_variant"]],
        dtype=torch.bool,
        device=state.device,
    )
    rows = correct & batch["horizons"][:, 0].eq(2)
    if not bool(rows.any()):
        return state.new_zeros(())
    first = model.dynamics_predictor(
        state[rows],
        batch["actions"][rows],
        torch.ones_like(batch["horizons"][rows]),
        action_padding_mask=_rollout_action_padding_mask(batch, step_offset=0)[rows],
    )
    second = model.dynamics_predictor(
        first,
        batch["actions"][rows],
        torch.ones_like(batch["horizons"][rows]),
        action_padding_mask=_rollout_action_padding_mask(batch, step_offset=1)[rows],
    )
    return masked_latent_loss(second, target[rows])


def _dynamics_objective(
    model: SupplyChainJEPACore,
    batch: Mapping[str, Any],
    *,
    paired_weight: float,
    paired_scale_floor: float | None = None,
    control_margin_weight: float = 0.0,
    control_margin: float = 0.05,
    shuffled_counterfactual_weight: float = 0.0,
    rollout_two_step_weight: float = 0.0,
    branch_contrastive_weight: float = 0.0,
    branch_contrastive_temperature: float = 0.1,
    branch_contrastive_mode: str = "absolute",
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    int,
    Tensor,
    Tensor,
    int,
]:
    state = model.canonical_belief_latent(batch, prefix="context")
    with torch.no_grad():
        target_encoding = model.canonical_state_encoding(batch, prefix="target")
        target = model.select_canonical_state(target_encoding).unsqueeze(1)
    prediction = model.dynamics_predictor(
        state,
        batch["actions"],
        batch["horizons"],
        action_padding_mask=batch.get("action_padding_mask"),
        exogenous=batch.get("known_exogenous"),
    )
    correct_rows = torch.tensor(
        [variant == "correct" for variant in batch["control_variant"]],
        dtype=torch.bool,
        device=prediction.device,
    )
    if not bool(correct_rows.any()):
        raise ValueError("Dynamics batch contains no registered action rows")
    latent = masked_latent_loss(prediction[correct_rows], target[correct_rows])
    paired = _paired_loss(
        prediction,
        target,
        batch,
        scale_floor=paired_scale_floor,
    )
    control, control_accuracy, control_count = _control_margin_loss(
        prediction,
        target,
        batch,
        margin=control_margin,
    )
    shuffled_counterfactual = _shuffled_counterfactual_loss(
        prediction,
        target,
        batch,
    )
    rollout = _two_step_rollout_loss(model, state, target, batch)
    branch_contrastive, branch_accuracy, branch_count = _branch_contrastive_loss(
        prediction,
        target,
        batch,
        temperature=branch_contrastive_temperature,
        mode=branch_contrastive_mode,
    )
    total = (
        latent
        + paired_weight * paired
        + control_margin_weight * control
        + shuffled_counterfactual_weight * shuffled_counterfactual
        + rollout_two_step_weight * rollout
        + branch_contrastive_weight * branch_contrastive
    )
    return (
        total,
        latent,
        paired,
        control,
        shuffled_counterfactual,
        rollout,
        control_accuracy,
        control_count,
        branch_contrastive,
        branch_accuracy,
        branch_count,
    )


@torch.no_grad()
def _evaluate_dynamics(
    model: SupplyChainJEPACore,
    unit_pools: Mapping[
        str,
        Sequence[Sequence[DynamicsTrainingExample]],
    ],
    *,
    batch_size: int,
    device: torch.device,
    paired_weight: float,
    paired_scale_floor: float | None,
    control_margin_weight: float,
    control_margin: float,
    shuffled_counterfactual_weight: float,
    rollout_two_step_weight: float,
    branch_contrastive_weight: float,
    branch_contrastive_temperature: float,
    branch_contrastive_mode: str,
    seed: int,
    source_weights: Mapping[str, float],
) -> dict[str, float]:
    model.eval()
    examples = _select_dynamics_batch(
        unit_pools,
        update=10_000,
        batch_size=batch_size,
        seed=seed,
        source_weights=source_weights,
    )
    batch = _move_batch(collate_dynamics_examples(examples), device)
    (
        total,
        latent,
        paired,
        control,
        shuffled_counterfactual,
        rollout,
        control_accuracy,
        control_count,
        branch_contrastive,
        branch_accuracy,
        branch_count,
    ) = _dynamics_objective(
        model,
        batch,
        paired_weight=paired_weight,
        paired_scale_floor=paired_scale_floor,
        control_margin_weight=control_margin_weight,
        control_margin=control_margin,
        shuffled_counterfactual_weight=shuffled_counterfactual_weight,
        rollout_two_step_weight=rollout_two_step_weight,
        branch_contrastive_weight=branch_contrastive_weight,
        branch_contrastive_temperature=branch_contrastive_temperature,
        branch_contrastive_mode=branch_contrastive_mode,
    )
    return {
        "loss": float(total),
        "latent_loss": float(latent),
        "paired_difference_loss": float(paired),
        "control_margin_loss": float(control),
        "shuffled_counterfactual_loss": float(shuffled_counterfactual),
        "control_ranking_accuracy": float(control_accuracy),
        "control_comparison_count": float(control_count),
        "rollout_two_step_loss": float(rollout),
        "branch_contrastive_loss": float(branch_contrastive),
        "branch_ranking_accuracy": float(branch_accuracy),
        "branch_comparison_count": float(branch_count),
    }


@torch.no_grad()
def _action_sensitivity(
    model: SupplyChainJEPACore,
    groups: Sequence[Sequence[Mapping[str, Any]]],
    tensorizer: EpisodeTensorizer,
    *,
    horizons: Sequence[int],
    window_spec: WindowSpec,
    device: torch.device,
) -> float:
    differences: list[Tensor] = []
    for group in groups[:4]:
        examples = _build_dynamics_source_agnostic(
            group,
            tensorizer,
            horizons=horizons,
            window_spec=window_spec,
            include_controls=True,
        )
        by_key: dict[tuple[str, int, int], dict[str, DynamicsTrainingExample]] = (
            defaultdict(dict)
        )
        for example in examples:
            by_key[(example.record_id, example.branch_position, example.horizon)][
                example.control_variant
            ] = example
        for variants in by_key.values():
            if "correct" not in variants or "noop" not in variants:
                continue
            batch = _move_batch(
                collate_dynamics_examples([variants["correct"], variants["noop"]]),
                device,
            )
            state = model.canonical_belief_latent(batch, prefix="context")
            prediction = model.dynamics_predictor(
                state,
                batch["actions"],
                batch["horizons"],
                action_padding_mask=batch.get("action_padding_mask"),
                exogenous=batch.get("known_exogenous"),
            )
            differences.append((prediction[0] - prediction[1]).square().mean().sqrt())
    if not differences:
        return 0.0
    return float(torch.stack(differences).mean())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dynamics(args: argparse.Namespace) -> dict[str, Any]:
    raw_config = _training_config(args.model_config, args.config)
    config = _apply_model_overrides(raw_config, args)
    seed = int(args.seed if args.seed is not None else config["seed"])
    _seed_everything(seed)
    device = _device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    preprocessing_started = time.perf_counter()
    records = _load_records(args.data, config)
    checkpoint_path = Path(args.checkpoint)
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if parent.get("stage") != "state":
        raise ValueError("Dynamics requires a State-stage checkpoint")
    tensorizer_state = FittedTensorizerState.from_dict(parent["tensorizer_state"])
    tensorizer = EpisodeTensorizer(tensorizer_state)
    model_config_payload = dict(parent["model_config"])
    model_config_payload["structured_action_encoding"] = bool(
        config.get("model", {}).get(
            "structured_action_encoding",
            model_config_payload.get("structured_action_encoding", False),
        )
    )
    model_config = ModelConfig(**model_config_payload)
    model = SupplyChainJEPACore(model_config)
    parent_state = {
        key: value
        for key, value in parent["model_state"].items()
        if not key.startswith("dynamics_predictor.")
    }
    incompatible = model.load_state_dict(parent_state, strict=False)
    expected_missing = {
        key
        for key in model.state_dict()
        if key.startswith("dynamics_predictor.")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "State checkpoint mismatch outside the freshly initialized "
            "Dynamics predictor"
        )
    training = config["dynamics_training"]
    encoder_source = str(training.get("encoder_source", "target"))
    model.freeze_encoders_for_dynamics(source=encoder_source)
    model.to(device)
    source_weights = {
        str(family): float(weight)
        for family, weight in training["source_sampling_weights"].items()
    }
    horizons = tuple(int(value) for value in training["horizons"])
    dynamics_target_window_steps = int(
        training.get(
            "target_window_steps",
            config["tensorization"]["target_window_steps"],
        )
    )
    window_spec = WindowSpec(
        max_tokens=int(config["tensorization"]["max_tokens"]),
        target_window_steps=dynamics_target_window_steps,
        seed=seed,
    )
    train_groups = _dynamics_groups(records, TRAIN_SPLIT)
    dev_groups = _dynamics_groups(records, DEV_SPLIT) or train_groups
    if not train_groups:
        raise ValueError("Dynamics training requires pretrain_train branch pairs")
    train_branch_group_count = len(train_groups)
    dev_branch_group_count = len(dev_groups)
    print(
        json.dumps(
            {
                "event": "dynamics_preprocessing",
                "phase": "groups_loaded",
                "elapsed_seconds": time.perf_counter() - preprocessing_started,
                "train_branch_groups": train_branch_group_count,
                "dev_branch_groups": dev_branch_group_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dynamics_sources = _source_families(
        [record for group in train_groups for record in group]
    )
    include_action_controls = bool(training.get("include_action_controls", False))
    train_units = _dynamics_units(
        train_groups,
        tensorizer,
        horizons=horizons,
        window_spec=window_spec,
        include_controls=include_action_controls,
    )
    dev_units = _dynamics_units(
        dev_groups,
        tensorizer,
        horizons=horizons,
        window_spec=window_spec,
        include_controls=include_action_controls,
    )
    train_unit_pools = _dynamics_unit_pools(train_units, source_weights)
    dev_unit_pools = _dynamics_unit_pools(dev_units, source_weights)
    del records, train_groups, train_units, dev_units
    gc.collect()
    print(
        json.dumps(
            {
                "event": "dynamics_preprocessing",
                "phase": "units_built",
                "elapsed_seconds": time.perf_counter() - preprocessing_started,
                "train_units": sum(len(units) for units in train_unit_pools.values()),
                "dev_units": sum(len(units) for units in dev_unit_pools.values()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.dynamics_predictor.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(training.get("betas", (0.9, 0.95))),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = _scaler(device, amp_enabled)
    paired_weight = float(training["paired_difference_weight"])
    paired_scale_floor_raw = training.get("paired_difference_scale_floor")
    paired_scale_floor = (
        None if paired_scale_floor_raw is None else float(paired_scale_floor_raw)
    )
    control_margin_weight = float(training.get("control_margin_weight", 0.0))
    control_margin = float(training.get("control_margin", 0.05))
    shuffled_counterfactual_weight = float(
        training.get("shuffled_counterfactual_weight", 0.0)
    )
    rollout_two_step_weight = float(training.get("rollout_two_step_weight", 0.0))
    branch_contrastive_weight = float(training.get("branch_contrastive_weight", 0.0))
    branch_contrastive_temperature = float(
        training.get("branch_contrastive_temperature", 0.1)
    )
    branch_contrastive_mode = str(
        training.get("branch_contrastive_mode", "absolute")
    )
    if branch_contrastive_weight < 0.0:
        raise ValueError("branch contrastive weight cannot be negative")
    if branch_contrastive_temperature <= 0.0:
        raise ValueError("branch contrastive temperature must be positive")
    if branch_contrastive_mode not in {"absolute", "effect_centered"}:
        raise ValueError(
            "branch contrastive mode must be absolute or effect_centered"
        )
    last: dict[str, float] = {}
    last_gradient_norm = 0.0
    train_source_counts: Counter[str] = Counter()
    max_updates = int(args.max_updates)
    accumulation_steps = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else training.get("gradient_accumulation_steps", 1)
    )
    if accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    base_learning_rate = float(training["learning_rate"])
    warmup_fraction = float(training.get("warmup_fraction", 0.0))
    minimum_lr_ratio = float(training.get("minimum_lr_ratio", 1.0))
    progress_interval = max(1, max_updates // 10)
    for update in range(max_updates):
        model.train()
        learning_rate = base_learning_rate * _cosine_learning_rate_scale(
            update,
            max_updates,
            warmup_fraction=warmup_fraction,
            minimum_ratio=minimum_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_total = 0.0
        accumulated_latent = 0.0
        accumulated_paired = 0.0
        accumulated_control = 0.0
        accumulated_shuffled_counterfactual = 0.0
        accumulated_rollout = 0.0
        accumulated_control_accuracy = 0.0
        accumulated_control_count = 0
        accumulated_branch_contrastive = 0.0
        accumulated_branch_accuracy = 0.0
        accumulated_branch_count = 0
        for micro_step in range(accumulation_steps):
            sampling_update = update * accumulation_steps + micro_step
            examples = _select_dynamics_batch(
                train_unit_pools,
                update=sampling_update,
                batch_size=int(args.batch_size),
                seed=seed,
                source_weights=source_weights,
            )
            train_source_counts.update(example.family for example in examples)
            batch = _move_batch(collate_dynamics_examples(examples), device)
            with _amp_context(device, amp_enabled):
                (
                    total,
                    latent,
                    paired,
                    control,
                    shuffled_counterfactual,
                    rollout,
                    control_accuracy,
                    control_count,
                    branch_contrastive,
                    branch_accuracy,
                    branch_count,
                ) = _dynamics_objective(
                    model,
                    batch,
                    paired_weight=paired_weight,
                    paired_scale_floor=paired_scale_floor,
                    control_margin_weight=control_margin_weight,
                    control_margin=control_margin,
                    shuffled_counterfactual_weight=shuffled_counterfactual_weight,
                    rollout_two_step_weight=rollout_two_step_weight,
                    branch_contrastive_weight=branch_contrastive_weight,
                    branch_contrastive_temperature=branch_contrastive_temperature,
                    branch_contrastive_mode=branch_contrastive_mode,
                )
            scaler.scale(total / accumulation_steps).backward()
            accumulated_total += float(total.detach())
            accumulated_latent += float(latent.detach())
            accumulated_paired += float(paired.detach())
            accumulated_control += float(control.detach())
            accumulated_shuffled_counterfactual += float(
                shuffled_counterfactual.detach()
            )
            accumulated_rollout += float(rollout.detach())
            accumulated_control_accuracy += float(control_accuracy.detach())
            accumulated_control_count += int(control_count)
            accumulated_branch_contrastive += float(branch_contrastive.detach())
            accumulated_branch_accuracy += float(branch_accuracy.detach())
            accumulated_branch_count += int(branch_count)
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.dynamics_predictor.parameters(),
            float(training["gradient_clip"]),
        )
        last_gradient_norm = float(gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        last = {
            "train_loss": accumulated_total / accumulation_steps,
            "train_latent_loss": accumulated_latent / accumulation_steps,
            "train_paired_difference_loss": accumulated_paired / accumulation_steps,
            "train_control_margin_loss": accumulated_control / accumulation_steps,
            "train_shuffled_counterfactual_loss": (
                accumulated_shuffled_counterfactual / accumulation_steps
            ),
            "train_control_ranking_accuracy": accumulated_control_accuracy
            / accumulation_steps,
            "train_control_comparison_count": accumulated_control_count,
            "train_rollout_two_step_loss": accumulated_rollout / accumulation_steps,
            "train_branch_contrastive_loss": accumulated_branch_contrastive
            / accumulation_steps,
            "train_branch_ranking_accuracy": accumulated_branch_accuracy
            / accumulation_steps,
            "train_branch_comparison_count": accumulated_branch_count,
            "learning_rate": learning_rate,
        }
        if update == 0 or (update + 1) % progress_interval == 0:
            print(
                json.dumps(
                    {
                        "event": "dynamics_progress",
                        "update": update + 1,
                        **last,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dev_metrics = _evaluate_dynamics(
        model,
        dev_unit_pools,
        batch_size=int(args.batch_size),
        device=device,
        paired_weight=paired_weight,
        paired_scale_floor=paired_scale_floor,
        control_margin_weight=control_margin_weight,
        control_margin=control_margin,
        shuffled_counterfactual_weight=shuffled_counterfactual_weight,
        rollout_two_step_weight=rollout_two_step_weight,
        branch_contrastive_weight=branch_contrastive_weight,
        branch_contrastive_temperature=branch_contrastive_temperature,
        branch_contrastive_mode=branch_contrastive_mode,
        seed=seed,
        source_weights=source_weights,
    )
    sensitivity = _action_sensitivity(
        model,
        dev_groups,
        tensorizer,
        horizons=horizons,
        window_spec=window_spec,
        device=device,
    )
    metrics: dict[str, Any] = {
        "stage": "dynamics",
        "updates": max_updates,
        "micro_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": int(args.batch_size) * accumulation_steps,
        "train_branch_group_count": train_branch_group_count,
        "dev_branch_group_count": dev_branch_group_count,
        "device": str(device),
        "amp": amp_enabled,
        "gradient_norm": last_gradient_norm,
        "action_sensitivity": sensitivity,
        "paired_difference_weight": paired_weight,
        "paired_difference_scale_floor": paired_scale_floor,
        "include_action_controls": include_action_controls,
        "control_margin_weight": control_margin_weight,
        "control_margin": control_margin,
        "shuffled_counterfactual_weight": shuffled_counterfactual_weight,
        "rollout_two_step_weight": rollout_two_step_weight,
        "branch_contrastive_weight": branch_contrastive_weight,
        "branch_contrastive_temperature": branch_contrastive_temperature,
        "branch_contrastive_mode": branch_contrastive_mode,
        "structured_action_encoding": model_config.structured_action_encoding,
        "dynamics_target_window_steps": dynamics_target_window_steps,
        "encoder_source": encoder_source,
        "train_source_counts": dict(sorted(train_source_counts.items())),
        **last,
        **{f"dev_{key}": value for key, value in dev_metrics.items()},
        **_cuda_peak_metrics(device),
    }
    sources = dict(parent.get("sources", {}))
    sources["dynamics"] = dynamics_sources
    checkpoint = {
        **{
            key: value
            for key, value in parent.items()
            if key not in {"stage", "updates", "model_state", "metrics", "sources"}
        },
        "stage": "dynamics",
        "updates": max_updates,
        "model_config": model_config.to_dict(),
        "training_config": config,
        "model_state": model.state_dict(),
        "sources": sources,
        "metrics": metrics,
        "parent_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _file_sha256(checkpoint_path),
            "stage": parent["stage"],
        },
    }
    output_path = _write_outputs(
        Path(args.output_dir),
        "dynamics_checkpoint.pt",
        checkpoint,
        metrics,
    )
    return {**metrics, "checkpoint": str(output_path)}


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, default_config: Path
) -> None:
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--small-model", action="store_true")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--state-predictor-layers", type=int)
    parser.add_argument("--dynamics-predictor-layers", type=int)
    parser.add_argument("--dim-feedforward", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    state = subparsers.add_parser("state", help="train predictive State JEPA")
    _add_common_arguments(state, default_config=DEFAULT_STATE_CONFIG)
    state.add_argument("--init-checkpoint")
    state.add_argument(
        "--reuse-parent-tensorizer",
        action="store_true",
        help=(
            "keep the initialization checkpoint's fitted feature/vocabulary "
            "interface during continuation training"
        ),
    )
    dynamics = subparsers.add_parser(
        "dynamics", help="train frozen-encoder action-conditioned Dynamics JEPA"
    )
    _add_common_arguments(dynamics, default_config=DEFAULT_DYNAMICS_CONFIG)
    dynamics.add_argument("--checkpoint", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.max_updates < 1:
        raise ValueError("--max-updates must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if (
        args.gradient_accumulation_steps is not None
        and args.gradient_accumulation_steps < 1
    ):
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.eval_samples is not None and args.eval_samples < 1:
        raise ValueError("--eval-samples must be positive")
    result = run_state(args) if args.command == "state" else run_dynamics(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
