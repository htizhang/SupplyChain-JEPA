"""Neural core for the continuous SupplyChain-JEPA weight chain.

The module intentionally contains no training loop.  It provides one online
encoder, one structurally identical EMA target encoder, a state predictor, and
an action-conditioned dynamics predictor.  Identity enters only through
bounded canonical episode-local indices; raw global IDs/hashes are unsupported.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    feature_count: int
    categorical_vocab_size: int
    family_vocab_size: int
    token_kind_vocab_size: int
    type_vocab_size: int
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 4
    state_predictor_layers: int = 2
    dynamics_predictor_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.0
    status_vocab_size: int = 8
    role_vocab_size: int = 16
    action_dim: int = 8
    exogenous_dim: int = 0
    local_index_capacity: int = 2048
    local_index_mode: str = "absolute"
    canonical_state_pooling: str = "belief_token"
    structured_action_encoding: bool = False
    family_embedding_mode: str = "explicit"
    feature_semantic_mode: str = "feature_specific"
    feature_concept_ids: tuple[int, ...] = ()
    concept_vocab_size: int = 1
    feature_residual_scale: float = 1.0
    target_feature_residual_scale: float = 1.0
    schema_residual_keep_probability: float = 1.0
    transfer_schema_residual_scale: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "feature_count": self.feature_count,
            "categorical_vocab_size": self.categorical_vocab_size,
            "family_vocab_size": self.family_vocab_size,
            "token_kind_vocab_size": self.token_kind_vocab_size,
            "type_vocab_size": self.type_vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "encoder_layers": self.encoder_layers,
            "state_predictor_layers": self.state_predictor_layers,
            "dynamics_predictor_layers": self.dynamics_predictor_layers,
            "dim_feedforward": self.dim_feedforward,
            "status_vocab_size": self.status_vocab_size,
            "role_vocab_size": self.role_vocab_size,
            "action_dim": self.action_dim,
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"Model dimensions must be positive: {invalid}")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.exogenous_dim < 0:
            raise ValueError("Optional dimensions cannot be negative")
        if self.local_index_capacity < 2:
            raise ValueError("local_index_capacity must include PAD=0 and UNK=1")
        if self.local_index_mode not in {"absolute", "bucketed", "compositional"}:
            raise ValueError(
                "local_index_mode must be 'absolute', 'bucketed', or 'compositional'"
            )
        if self.canonical_state_pooling not in {"belief_token", "masked_mean"}:
            raise ValueError(
                "canonical_state_pooling must be 'belief_token' or 'masked_mean'"
            )
        if self.family_embedding_mode not in {"explicit", "unknown"}:
            raise ValueError("family_embedding_mode must be explicit or unknown")
        if self.feature_semantic_mode not in {
            "feature_specific",
            "concept_residual",
        }:
            raise ValueError(
                "feature_semantic_mode must be feature_specific or concept_residual"
            )
        concept_ids = tuple(int(value) for value in self.feature_concept_ids)
        object.__setattr__(self, "feature_concept_ids", concept_ids)
        if self.concept_vocab_size < 1:
            raise ValueError("concept_vocab_size must be positive")
        if self.feature_residual_scale < 0.0:
            raise ValueError("feature_residual_scale must be nonnegative")
        if self.target_feature_residual_scale < 0.0:
            raise ValueError("target_feature_residual_scale must be nonnegative")
        if not 0.0 <= self.schema_residual_keep_probability <= 1.0:
            raise ValueError("schema_residual_keep_probability must be in [0, 1]")
        if self.transfer_schema_residual_scale < 0.0:
            raise ValueError("transfer_schema_residual_scale must be nonnegative")
        if self.feature_semantic_mode == "concept_residual":
            if len(concept_ids) != self.feature_count:
                raise ValueError(
                    "feature_concept_ids must align with every fitted feature"
                )
            if concept_ids and (
                min(concept_ids) < 0 or max(concept_ids) >= self.concept_vocab_size
            ):
                raise ValueError("feature_concept_ids exceed concept_vocab_size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_max_local_index(self, maximum: int) -> None:
        """Reject a tensorizer/model capacity mismatch before model execution."""

        if maximum < 0:
            raise ValueError("maximum local index cannot be negative")
        if maximum >= self.local_index_capacity:
            raise ValueError(
                f"Local index {maximum} exceeds configured capacity "
                f"{self.local_index_capacity}; increase local_index_capacity"
            )

    @classmethod
    def from_training_config(
        cls,
        training_config: Mapping[str, Any],
        *,
        feature_count: int,
        categorical_vocab_size: int,
        family_vocab_size: int,
        token_kind_vocab_size: int,
        type_vocab_size: int,
        action_dim: int = 8,
        exogenous_dim: int = 0,
        local_index_capacity: int | None = None,
        max_local_index: int | None = None,
        feature_concept_ids: tuple[int, ...] = (),
        concept_vocab_size: int = 1,
    ) -> ModelConfig:
        """Construct a model from a model configuration mapping style mappings.

        Tensorizer-derived vocabulary sizes remain explicit because the training
        protocol cannot know them before fitting.  A supplied maximum local
        index is validated immediately; every forward pass validates again.
        """

        model_payload = training_config.get("model", training_config)
        if not isinstance(model_payload, Mapping):
            raise TypeError("training_config['model'] must be a mapping")
        activation = str(model_payload.get("activation", "gelu")).lower()
        if activation != "gelu":
            raise ValueError("Only gelu training configurations are supported")
        encoder_ffn = int(model_payload.get("encoder_ffn_dim", 512))
        predictor_ffn = int(model_payload.get("predictor_ffn_dim", encoder_ffn))
        if encoder_ffn != predictor_ffn:
            raise ValueError(
                "Current core requires encoder_ffn_dim == predictor_ffn_dim"
            )
        capacity = (
            int(local_index_capacity)
            if local_index_capacity is not None
            else int(model_payload.get("local_index_capacity", 2048))
        )
        config = cls(
            feature_count=int(feature_count),
            categorical_vocab_size=int(categorical_vocab_size),
            family_vocab_size=int(family_vocab_size),
            token_kind_vocab_size=int(token_kind_vocab_size),
            type_vocab_size=int(type_vocab_size),
            d_model=int(model_payload.get("d_model", 128)),
            nhead=int(model_payload.get("n_heads", model_payload.get("nhead", 4))),
            encoder_layers=int(model_payload.get("encoder_layers", 4)),
            state_predictor_layers=int(model_payload.get("state_predictor_layers", 2)),
            dynamics_predictor_layers=int(
                model_payload.get("dynamics_predictor_layers", 3)
            ),
            dim_feedforward=encoder_ffn,
            dropout=float(model_payload.get("dropout", 0.0)),
            action_dim=int(action_dim),
            exogenous_dim=int(exogenous_dim),
            local_index_capacity=capacity,
            local_index_mode=str(model_payload.get("local_index_mode", "absolute")),
            canonical_state_pooling=str(
                model_payload.get("canonical_state_pooling", "belief_token")
            ),
            structured_action_encoding=bool(
                model_payload.get("structured_action_encoding", False)
            ),
            family_embedding_mode=str(
                model_payload.get("family_embedding_mode", "explicit")
            ),
            feature_semantic_mode=str(
                model_payload.get("feature_semantic_mode", "feature_specific")
            ),
            feature_concept_ids=tuple(feature_concept_ids),
            concept_vocab_size=int(concept_vocab_size),
            feature_residual_scale=float(
                model_payload.get("feature_residual_scale", 1.0)
            ),
            target_feature_residual_scale=float(
                model_payload.get("target_feature_residual_scale", 1.0)
            ),
            schema_residual_keep_probability=float(
                model_payload.get("schema_residual_keep_probability", 1.0)
            ),
            transfer_schema_residual_scale=float(
                model_payload.get("transfer_schema_residual_scale", 1.0)
            ),
        )
        if max_local_index is not None and config.local_index_mode in {
            "absolute",
            "compositional",
        }:
            config.validate_max_local_index(int(max_local_index))
        return config


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in evaluation mode and disable all parameter gradients."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def unfreeze_module(module: nn.Module, *, train: bool = True) -> nn.Module:
    """Enable all parameter gradients, optionally restoring training mode."""

    module.train(train)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    return module


def _assert_same_structure(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("Source and target modules do not have identical state keys")
    mismatched = [
        name
        for name in source_state
        if source_state[name].shape != target_state[name].shape
        or source_state[name].dtype != target_state[name].dtype
    ]
    if mismatched:
        raise ValueError(f"Source and target tensor structures differ: {mismatched}")


@torch.no_grad()
def hard_sync(source: nn.Module, target: nn.Module) -> None:
    """Exactly copy a source module into a structurally identical target."""

    _assert_same_structure(source, target)
    target.load_state_dict(source.state_dict(), strict=True)


@torch.no_grad()
def ema_update(source: nn.Module, target: nn.Module, momentum: float) -> None:
    """EMA-update target parameters and exactly copy non-floating buffers."""

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    _assert_same_structure(source, target)
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, target_value in target_state.items():
        source_value = source_state[name].detach()
        if torch.is_floating_point(target_value):
            target_value.mul_(momentum).add_(
                source_value.to(device=target_value.device), alpha=1.0 - momentum
            )
        else:
            target_value.copy_(source_value.to(device=target_value.device))


def build_local_target_window(
    time_index: Tensor,
    anchor_time: Tensor,
    window_size: int,
    *,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Expose the endpoint-causal interval ``[anchor-window_size+1, anchor]``.

    ``anchor_time`` has shape ``[B]``.  Topology tokens with negative time and
    all post-endpoint tokens are hidden.  ``window_size`` counts discrete time
    indices, including the endpoint.  The return value follows PyTorch
    key-padding semantics: ``True`` means hidden/padded.
    """

    if time_index.ndim != 2 or anchor_time.ndim != 1:
        raise ValueError("time_index must be [B,L] and anchor_time must be [B]")
    if time_index.shape[0] != anchor_time.shape[0]:
        raise ValueError("Batch dimensions do not match")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    endpoint = anchor_time[:, None]
    start = endpoint - int(window_size) + 1
    keep = (time_index >= start) & (time_index <= endpoint) & (time_index >= 0)
    mask = ~keep
    if padding_mask is not None:
        if padding_mask.shape != time_index.shape:
            raise ValueError("padding_mask shape does not match time_index")
        mask = mask | padding_mask
    if (~mask).sum(dim=1).eq(0).any():
        raise ValueError("Each local target window must contain at least one token")
    return mask


def _safe_ids(values: Tensor, embedding: nn.Embedding) -> Tensor:
    return values.clamp(min=0, max=embedding.num_embeddings - 1)


class FactorizedTokenEmbedding(nn.Module):
    """Combine values, feature presence, status, and token metadata.

    Identity is represented only by bounded, canonical episode-local indices.
    Raw global IDs and hashes have no accepted field or configuration path.
    """

    _LOCAL_INDEX_FIELDS = (
        "entity_index",
        "product_index",
        "node_index",
        "source_index",
        "target_index",
        "alignment_index",
    )
    _OBJECT_INDEX_FIELDS = {
        "entity_index",
        "node_index",
        "source_index",
        "target_index",
    }

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        feature_count = config.feature_count
        self.numeric_weight = nn.Parameter(torch.empty(feature_count, d_model))
        self.feature_embedding = nn.Embedding(feature_count, d_model)
        self.feature_residual_scale = float(config.feature_residual_scale)
        self.schema_residual_keep_probability = float(
            config.schema_residual_keep_probability
        )
        self.transfer_schema_residual_scale = float(
            config.transfer_schema_residual_scale
        )
        self._schema_residual_scale_override: float | None = None
        if config.feature_semantic_mode == "concept_residual":
            self.register_buffer(
                "feature_concept_ids",
                torch.tensor(config.feature_concept_ids, dtype=torch.long),
            )
            self.concept_numeric_weight = nn.Parameter(
                torch.empty(config.concept_vocab_size, d_model)
            )
            self.concept_embedding = nn.Embedding(config.concept_vocab_size, d_model)
        else:
            self.feature_concept_ids = None
            self.concept_numeric_weight = None
            self.concept_embedding = None
        self.categorical_embedding = nn.Embedding(
            config.categorical_vocab_size, d_model, padding_idx=0
        )
        self.status_embedding = nn.Embedding(
            config.status_vocab_size, d_model, padding_idx=0
        )
        self.family_embedding = nn.Embedding(
            config.family_vocab_size, d_model, padding_idx=0
        )
        self.kind_embedding = nn.Embedding(
            config.token_kind_vocab_size, d_model, padding_idx=0
        )
        self.type_embedding = nn.Embedding(
            config.type_vocab_size, d_model, padding_idx=0
        )
        self.age_projection = nn.Linear(1, d_model, bias=False)
        self.time_projection = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if config.local_index_mode == "compositional":
            self.local_index_low_embedding = nn.Embedding(258, d_model, padding_idx=0)
            high_capacity = 2 + math.ceil((config.local_index_capacity - 2) / 256)
            self.local_index_high_embedding = nn.Embedding(
                high_capacity,
                d_model,
                padding_idx=0,
            )
            self.object_index_embedding = None
            self.product_index_embedding = None
            self.alignment_index_embedding = None
        else:
            self.local_index_low_embedding = None
            self.local_index_high_embedding = None
            self.object_index_embedding = nn.Embedding(
                config.local_index_capacity, d_model, padding_idx=0
            )
            self.product_index_embedding = nn.Embedding(
                config.local_index_capacity, d_model, padding_idx=0
            )
            self.alignment_index_embedding = nn.Embedding(
                config.local_index_capacity, d_model, padding_idx=0
            )
        self.local_role_embedding = nn.Embedding(len(self._LOCAL_INDEX_FIELDS), d_model)
        self.normalization = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.numeric_weight, std=self.config.d_model**-0.5)
        nn.init.normal_(self.feature_embedding.weight, std=0.02)
        if self.concept_numeric_weight is not None:
            nn.init.normal_(
                self.concept_numeric_weight,
                std=self.config.d_model**-0.5,
            )
        if self.concept_embedding is not None:
            nn.init.normal_(self.concept_embedding.weight, std=0.02)

    def set_feature_residual_scale(self, scale: float) -> None:
        """Set the non-persistent raw-schema residual strength for this encoder."""

        if scale < 0.0:
            raise ValueError("feature residual scale must be nonnegative")
        self.feature_residual_scale = float(scale)

    def set_schema_residual_schedule(
        self,
        *,
        keep_probability: float | None = None,
        scale_override: float | None = None,
    ) -> None:
        """Set the transient source-schema gate used during migration training."""

        if keep_probability is not None:
            if not 0.0 <= keep_probability <= 1.0:
                raise ValueError("schema residual keep probability must be in [0, 1]")
            self.schema_residual_keep_probability = float(keep_probability)
        if scale_override is not None and scale_override < 0.0:
            raise ValueError("schema residual scale override must be nonnegative")
        self._schema_residual_scale_override = (
            None if scale_override is None else float(scale_override)
        )

    def _schema_residual_gate(
        self,
        family: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return one source-schema gate per example, shared by every token."""

        batch_size = family.shape[0]
        if self.config.feature_semantic_mode != "concept_residual":
            return torch.ones((batch_size, 1, 1), device=family.device, dtype=dtype)
        if self._schema_residual_scale_override is not None:
            return torch.full(
                (batch_size, 1, 1),
                self._schema_residual_scale_override,
                device=family.device,
                dtype=dtype,
            )
        if self.training:
            return torch.rand(
                (batch_size, 1, 1), device=family.device, dtype=dtype
            ).lt(self.schema_residual_keep_probability).to(dtype)
        return torch.full(
            (batch_size, 1, 1),
            self.transfer_schema_residual_scale,
            device=family.device,
            dtype=dtype,
        )

    @staticmethod
    def _field(batch: Mapping[str, Any], prefix: str, name: str) -> Tensor:
        key = f"{prefix}_{name}"
        if key not in batch:
            raise KeyError(f"Missing batch tensor {key!r}")
        value = batch[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Batch field {key!r} is not a tensor")
        return value

    def metadata(
        self,
        batch: Mapping[str, Any],
        prefix: str,
        *,
        schema_residual_gate: Tensor | None = None,
    ) -> Tensor:
        family = self._field(batch, prefix, "family_id")
        if self.config.family_embedding_mode == "unknown":
            family = torch.where(family.eq(0), family, torch.ones_like(family))
        kind = self._field(batch, prefix, "token_kind_id")
        token_type = self._field(batch, prefix, "type_id")
        time_index = self._field(batch, prefix, "time_index").float()
        clock = self._field(batch, prefix, "clock").float()
        time_features = torch.stack(
            (
                torch.sign(time_index) * torch.log1p(time_index.abs()),
                torch.sign(clock) * torch.log1p(clock.abs()),
                torch.sin(time_index / 16.0),
                torch.cos(time_index / 16.0),
            ),
            dim=-1,
        )
        family_values = self.family_embedding(_safe_ids(family, self.family_embedding))
        if self.config.feature_semantic_mode == "concept_residual":
            if schema_residual_gate is None:
                schema_residual_gate = self._schema_residual_gate(
                    family, dtype=family_values.dtype
                )
            family_values = family_values * schema_residual_gate
        embedded = (
            family_values
            + self.kind_embedding(_safe_ids(kind, self.kind_embedding))
            + self.type_embedding(_safe_ids(token_type, self.type_embedding))
            + self.time_projection(time_features)
        )
        local_sum = torch.zeros_like(embedded)
        local_count = torch.zeros_like(family, dtype=embedded.dtype).unsqueeze(-1)
        for role, name in enumerate(self._LOCAL_INDEX_FIELDS):
            indices = self._bounded_local_indices(batch, prefix, name)
            if self.config.local_index_mode == "compositional":
                assert self.local_index_low_embedding is not None
                assert self.local_index_high_embedding is not None
                ordinary = (indices - 2).clamp_min(0)
                low = torch.where(indices <= 1, indices, 2 + ordinary.remainder(256))
                high = torch.where(
                    indices <= 1, indices, 2 + ordinary.div(256, rounding_mode="floor")
                )
                index_embedding = self.local_index_low_embedding(
                    low
                ) + self.local_index_high_embedding(high)
            elif name in self._OBJECT_INDEX_FIELDS:
                assert self.object_index_embedding is not None
                index_embedding = self.object_index_embedding(indices)
            elif name == "product_index":
                assert self.product_index_embedding is not None
                index_embedding = self.product_index_embedding(indices)
            else:
                assert self.alignment_index_embedding is not None
                index_embedding = self.alignment_index_embedding(indices)
            present = indices.ne(0).unsqueeze(-1)
            role_ids = torch.full_like(indices, role)
            local_sum = (
                local_sum
                + (index_embedding + self.local_role_embedding(role_ids)) * present
            )
            local_count = local_count + present.to(local_count.dtype)
        embedded = embedded + local_sum / local_count.clamp_min(1.0)
        return embedded

    def _bounded_local_indices(
        self, batch: Mapping[str, Any], prefix: str, name: str
    ) -> Tensor:
        indices = self._field(batch, prefix, name)
        if indices.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(f"{prefix}_{name} must contain integer indices")
        if indices.numel():
            minimum = int(indices.min())
            maximum = int(indices.max())
            if minimum < 0:
                raise ValueError(f"{prefix}_{name} contains negative local indices")
            if self.config.local_index_mode in {"absolute", "compositional"}:
                self.config.validate_max_local_index(maximum)
        indices = indices.long()
        if self.config.local_index_mode == "bucketed":
            # Preserve PAD=0 and UNK=1 exactly.  Remaining episode-local IDs are
            # equality keys, not globally meaningful identities, so bounded
            # modulo buckets retain within-episode matching without giant tables.
            bucket_count = self.config.local_index_capacity - 2
            if bucket_count < 1:
                return indices.clamp(max=1)
            indices = torch.where(
                indices <= 1,
                indices,
                2 + (indices - 2).remainder(bucket_count),
            )
        return indices

    def forward(self, batch: Mapping[str, Any], prefix: str) -> Tensor:
        values = self._field(batch, prefix, "values")
        categorical = self._field(batch, prefix, "categorical_ids")
        present = self._field(batch, prefix, "feature_present")
        status = self._field(batch, prefix, "status_ids")
        age = self._field(batch, prefix, "age")
        if values.ndim != 3 or values.shape[-1] != self.config.feature_count:
            raise ValueError(
                f"{prefix}_values must be [B,L,{self.config.feature_count}]"
            )
        if any(
            item.shape != values.shape for item in (categorical, present, status, age)
        ):
            raise ValueError("All token feature tensors must have identical shapes")

        present_float = present.to(values.dtype)
        count = present_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        batch_size, sequence_length, feature_count = values.shape
        schema_residual_gate = self._schema_residual_gate(
            self._field(batch, prefix, "family_id"), dtype=values.dtype
        )

        def weighted_embedding_sum(ids: Tensor, embedding: nn.Embedding) -> Tensor:
            safe = _safe_ids(ids, embedding).reshape(-1)
            offsets = torch.arange(
                0,
                safe.numel(),
                feature_count,
                device=safe.device,
                dtype=torch.long,
            )
            weights = present_float.reshape(-1).to(embedding.weight.dtype)
            return F.embedding_bag(
                safe,
                embedding.weight,
                offsets,
                mode="sum",
                per_sample_weights=weights,
                include_last_offset=False,
                padding_idx=embedding.padding_idx,
            ).reshape(batch_size, sequence_length, self.config.d_model)

        # Algebraically equivalent reductions avoid materializing five
        # [B, L, F, D] tensors.  large corpora have hundreds of union-schema
        # features, so the direct broadcast formulation consumed several GB
        # per component and prevented useful batch sizes on a 15 GB T4.
        numeric_sum = torch.matmul(values * present_float, self.numeric_weight)
        categorical_sum = weighted_embedding_sum(
            categorical, self.categorical_embedding
        )
        feature_sum = torch.matmul(present_float, self.feature_embedding.weight)
        status_sum = weighted_embedding_sum(status, self.status_embedding)
        if self.config.feature_semantic_mode == "concept_residual":
            assert self.feature_concept_ids is not None
            assert self.concept_numeric_weight is not None
            assert self.concept_embedding is not None
            concept_numeric = self.concept_numeric_weight[self.feature_concept_ids]
            concept_keys = self.concept_embedding.weight[self.feature_concept_ids]
            semantic_sum = torch.matmul(
                values * present_float, concept_numeric
            ) + torch.matmul(present_float, concept_keys)
            schema_residual = self.feature_residual_scale * (
                numeric_sum + feature_sum
            ) / count * schema_residual_gate
            value_embedding = (
                semantic_sum / count
                + schema_residual
                + (categorical_sum + status_sum) / count
                + self.age_projection(
                    (age * present_float).sum(dim=-1, keepdim=True) / count
                )
            )
            return self.normalization(
                value_embedding
                + self.metadata(
                    batch,
                    prefix,
                    schema_residual_gate=schema_residual_gate,
                )
            )
        age_average = (age * present_float).sum(dim=-1, keepdim=True) / count
        value_embedding = (
            numeric_sum + categorical_sum + feature_sum + status_sum
        ) / count + self.age_projection(age_average)
        return self.normalization(value_embedding + self.metadata(batch, prefix))


@dataclass
class SequenceEncoding:
    """Aligned token outputs plus learned sequence-level special latents."""

    token_latents: Tensor
    belief_latent: Tensor
    reg_latent: Tensor
    token_padding_mask: Tensor


class SequenceEncoder(nn.Module):
    """Batch-first Transformer encoder over factorized supply-chain tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = FactorizedTokenEmbedding(config)
        self.belief_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.reg_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.belief_token, std=0.02)
        nn.init.normal_(self.reg_token, std=0.02)

    def encode_with_special_tokens(
        self,
        batch: Mapping[str, Any],
        prefix: str,
        *,
        padding_mask: Tensor | None = None,
    ) -> SequenceEncoding:
        """Encode ``[BELIEF, REG, token_1, ...]`` with explicit alignment.

        ``token_latents`` always corresponds one-for-one to the input episode
        tokens.  BELIEF is a pooled state for Dynamics; REG is a dedicated
        representation on which anti-collapse regularization may be applied.
        """

        if padding_mask is None:
            padding_mask = batch.get(f"{prefix}_padding_mask")
        tokens = self.embedding(batch, prefix)
        if padding_mask is None:
            padding_mask = torch.zeros(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
        if padding_mask.shape != tokens.shape[:2]:
            raise ValueError("padding_mask does not match token sequence")
        if (~padding_mask).sum(dim=1).eq(0).any():
            raise ValueError("Every sequence must contain a non-padding token")
        special = torch.cat(
            (
                self.belief_token.expand(tokens.shape[0], -1, -1),
                self.reg_token.expand(tokens.shape[0], -1, -1),
            ),
            dim=1,
        )
        transformer_input = torch.cat((special, tokens), dim=1)
        special_padding = torch.zeros(
            (tokens.shape[0], 2), dtype=torch.bool, device=tokens.device
        )
        full_padding_mask = torch.cat((special_padding, padding_mask), dim=1)
        encoded = self.transformer(
            transformer_input, src_key_padding_mask=full_padding_mask
        )
        token_latents = encoded[:, 2:].masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return SequenceEncoding(
            token_latents=token_latents,
            belief_latent=encoded[:, 0],
            reg_latent=encoded[:, 1],
            token_padding_mask=padding_mask,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        prefix: str,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Stable token-only API; special latents never alter token alignment."""

        return self.encode_with_special_tokens(
            batch, prefix, padding_mask=padding_mask
        ).token_latents


class StatePredictor(nn.Module):
    """Lightweight role/horizon-conditioned latent target predictor."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.role_embedding = nn.Embedding(config.role_vocab_size, config.d_model)
        self.horizon_projection = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=config.state_predictor_layers,
            norm=nn.LayerNorm(config.d_model),
        )

    def forward(
        self,
        context_latents: Tensor,
        target_metadata: Tensor,
        *,
        context_padding_mask: Tensor | None = None,
        target_padding_mask: Tensor | None = None,
        role_ids: Tensor | None = None,
        horizons: Tensor | None = None,
    ) -> Tensor:
        if target_metadata.ndim != 3:
            raise ValueError("target_metadata must be [B,Q,D]")
        batch_size, query_count, _ = target_metadata.shape
        device = target_metadata.device
        if role_ids is None:
            role_ids = torch.zeros(
                (batch_size, query_count), dtype=torch.long, device=device
            )
        if horizons is None:
            horizons = torch.zeros(
                (batch_size, query_count), dtype=target_metadata.dtype, device=device
            )
        if role_ids.shape != (batch_size, query_count):
            raise ValueError("role_ids must be [B,Q]")
        if horizons.shape != (batch_size, query_count):
            raise ValueError("horizons must be [B,Q]")
        horizon_features = torch.stack(
            (torch.log1p(horizons.float().clamp_min(0.0)), horizons.float().sign()),
            dim=-1,
        )
        queries = (
            target_metadata
            + self.role_embedding(
                role_ids.clamp(min=0, max=self.config.role_vocab_size - 1)
            )
            + self.horizon_projection(horizon_features)
        )
        predicted = self.decoder(
            tgt=queries,
            memory=context_latents,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=context_padding_mask,
        )
        if target_padding_mask is not None:
            predicted = predicted.masked_fill(target_padding_mask.unsqueeze(-1), 0.0)
        return predicted


class StatePoolPredictor(nn.Module):
    """Predict a target operational state from the deployed context state.

    The token decoder can use every context token, so success there does not
    guarantee that the single state vector exported to downstream tasks is
    predictively sufficient.  This deliberately small branch receives only the
    canonical context state plus leakage-safe query descriptors.  Its target is
    an EMA-encoded pooled operational state.

    The module intentionally lives outside :class:`SupplyChainJEPACore` when it
    is used by the training code.  Older core checkpoints therefore retain an
    identical strict-loading surface.
    """

    def __init__(
        self,
        d_model: int,
        *,
        query_vocab_size: int,
        mask_vocab_size: int,
        hidden_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or query_vocab_size < 1 or mask_vocab_size < 1:
            raise ValueError("StatePoolPredictor dimensions must be positive")
        if hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.query_embedding = nn.Embedding(query_vocab_size, d_model)
        self.mask_embedding = nn.Embedding(mask_vocab_size, d_model)
        self.horizon_projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.state_norm = nn.LayerNorm(d_model)
        self.state_skip = nn.Linear(d_model, d_model, bias=False)
        hidden_width = int(d_model) * int(hidden_multiplier)
        self.residual = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, d_model),
        )

    def forward(
        self,
        canonical_state: Tensor,
        horizons: Tensor,
        query_ids: Tensor,
        mask_ids: Tensor,
    ) -> Tensor:
        if canonical_state.ndim != 2 or canonical_state.shape[-1] != self.d_model:
            raise ValueError("canonical_state must be [B,D]")
        batch_size = canonical_state.shape[0]
        expected = (batch_size,)
        if horizons.shape != expected:
            raise ValueError("horizons must be [B]")
        if query_ids.shape != expected or mask_ids.shape != expected:
            raise ValueError("query_ids and mask_ids must be [B]")
        if query_ids.dtype != torch.long or mask_ids.dtype != torch.long:
            raise TypeError("query_ids and mask_ids must be torch.long")
        horizon = horizons.float().clamp_min(0.0)
        horizon_features = torch.stack(
            (torch.log1p(horizon), horizon.gt(0).to(horizon.dtype)),
            dim=-1,
        )
        normalized_state = self.state_norm(canonical_state)
        conditioned = (
            normalized_state
            + self.query_embedding(query_ids)
            + self.mask_embedding(mask_ids)
            + self.horizon_projection(horizon_features)
        )
        return self.state_skip(canonical_state) + self.residual(conditioned)


class SemanticConceptSlotPredictor(nn.Module):
    """Predict schema-agnostic operational concepts from context tokens.

    Each learned query names one supply-chain concept, while the decoder memory
    retains the full feature-specific pretrained representation.  The query interface
    deliberately excludes simulator, family, raw-feature, and mask identity so
    the auxiliary objective cannot be solved by selecting a source schema.
    """

    def __init__(
        self,
        d_model: int,
        *,
        concept_count: int,
        nhead: int,
        decoder_layers: int = 1,
        ffn_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or concept_count < 1 or nhead < 1:
            raise ValueError("Semantic concept slot dimensions must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if decoder_layers < 1 or ffn_multiplier < 1:
            raise ValueError("Semantic concept slot depths must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.concept_count = int(concept_count)
        self.concept_queries = nn.Parameter(
            torch.empty(1, self.concept_count, self.d_model)
        )
        self.horizon_projection = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=self.d_model * int(ffn_multiplier),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=int(decoder_layers),
            norm=nn.LayerNorm(self.d_model),
        )
        nn.init.normal_(self.concept_queries, std=0.02)

    def forward(
        self,
        context_latents: Tensor,
        horizons: Tensor,
        *,
        context_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if context_latents.ndim != 3 or context_latents.shape[-1] != self.d_model:
            raise ValueError("context_latents must be [B,L,D]")
        batch_size, sequence_length, _ = context_latents.shape
        if horizons.shape != (batch_size,):
            raise ValueError("horizons must be [B]")
        if context_padding_mask is not None:
            if context_padding_mask.shape != (batch_size, sequence_length):
                raise ValueError("context_padding_mask must be [B,L]")
            if context_padding_mask.dtype != torch.bool:
                raise TypeError("context_padding_mask must be torch.bool")
            if context_padding_mask.all(dim=1).any():
                raise ValueError("Every context row requires a visible token")
        horizon = horizons.float().clamp_min(0.0)
        horizon_features = torch.stack(
            (torch.log1p(horizon), horizon.gt(0).to(horizon.dtype)),
            dim=-1,
        )
        queries = self.concept_queries.expand(batch_size, -1, -1)
        queries = queries + self.horizon_projection(horizon_features).unsqueeze(1)
        return self.decoder(
            tgt=queries,
            memory=context_latents,
            memory_key_padding_mask=context_padding_mask,
        )


class DynamicsPredictor(nn.Module):
    """Action-conditioned latent predictor trained over frozen encoder states."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.structured_action_encoding = bool(
            config.structured_action_encoding and config.action_dim >= 6
        )
        if self.structured_action_encoding:
            continuous_width = config.action_dim - 5
            self.action_projection = None
            self.action_continuous_projection = nn.Sequential(
                nn.Linear(continuous_width, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.action_type_embedding = nn.Embedding(
                config.type_vocab_size,
                config.d_model,
                padding_idx=0,
            )
            self.action_time_projection = nn.Sequential(
                nn.Linear(3, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            if config.local_index_mode == "compositional":
                self.action_local_low_embedding = nn.Embedding(
                    258,
                    config.d_model,
                    padding_idx=0,
                )
                high_capacity = 2 + math.ceil(
                    (config.local_index_capacity - 2) / 256
                )
                self.action_local_high_embedding = nn.Embedding(
                    high_capacity,
                    config.d_model,
                    padding_idx=0,
                )
                self.action_local_index_embedding = None
            else:
                self.action_local_low_embedding = None
                self.action_local_high_embedding = None
                self.action_local_index_embedding = nn.Embedding(
                    config.local_index_capacity,
                    config.d_model,
                    padding_idx=0,
                )
            self.action_local_role_embedding = nn.Embedding(3, config.d_model)
            self.action_normalization = nn.LayerNorm(config.d_model)
            self.action_fusion = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.action_projection = nn.Sequential(
                nn.Linear(config.action_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.action_continuous_projection = None
            self.action_type_embedding = None
            self.action_time_projection = None
            self.action_local_low_embedding = None
            self.action_local_high_embedding = None
            self.action_local_index_embedding = None
            self.action_local_role_embedding = None
            self.action_normalization = None
            self.action_fusion = None
        self.exogenous_projection = (
            nn.Linear(config.exogenous_dim, config.d_model)
            if config.exogenous_dim
            else None
        )
        self.query = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.horizon_projection = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=config.dynamics_predictor_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        nn.init.normal_(self.query, std=0.02)

    def _normalize_action_features(self, actions: Tensor) -> Tensor:
        """Compress heterogeneous action magnitudes before linear projection.

        Quantity must remain visible beside large episode-local entity indices.
        A signed log transform keeps intervention magnitude ordered, while the
        shared local-index scale maps type/time/entity coordinates near [0, 1].
        The final reserved indicator/exogenous slots retain their raw scale.
        This transform has no parameters, so State checkpoints remain loadable.
        """

        normalized = torch.sign(actions) * torch.log1p(actions.abs())
        structured_end = min(actions.shape[-1], 6)
        if structured_end > 1:
            normalized = normalized.clone()
            normalized[..., 1:structured_end] /= math.log1p(
                self.config.local_index_capacity
            )
        if actions.shape[-1] > 6:
            normalized[..., 6:] = actions[..., 6:]
        return normalized

    def _bounded_action_indices(self, values: Tensor) -> Tensor:
        indices = values.round().long()
        if values.numel() and not torch.allclose(
            values.float(),
            indices.to(values.dtype),
            atol=1e-4,
            rtol=0.0,
        ):
            raise ValueError("Structured action indices must be integer-valued")
        if indices.numel():
            minimum = int(indices.min())
            maximum = int(indices.max())
            if minimum < 0:
                raise ValueError("Structured action indices cannot be negative")
            if self.config.local_index_mode in {"absolute", "compositional"}:
                self.config.validate_max_local_index(maximum)
        if self.config.local_index_mode == "bucketed":
            bucket_count = self.config.local_index_capacity - 2
            indices = torch.where(
                indices <= 1,
                indices,
                2 + (indices - 2).remainder(bucket_count),
            )
        return indices

    def _action_index_embedding(self, values: Tensor) -> Tensor:
        indices = self._bounded_action_indices(values)
        if self.config.local_index_mode == "compositional":
            assert self.action_local_low_embedding is not None
            assert self.action_local_high_embedding is not None
            ordinary = (indices - 2).clamp_min(0)
            low = torch.where(indices <= 1, indices, 2 + ordinary.remainder(256))
            high = torch.where(
                indices <= 1,
                indices,
                2 + ordinary.div(256, rounding_mode="floor"),
            )
            return self.action_local_low_embedding(
                low
            ) + self.action_local_high_embedding(high)
        assert self.action_local_index_embedding is not None
        return self.action_local_index_embedding(indices)

    def _encode_actions(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        if not self.structured_action_encoding:
            assert self.action_projection is not None
            normalized = self._normalize_action_features(actions)
            return (
                self.action_projection(normalized),
                self.action_projection[0](normalized),
            )

        assert self.action_continuous_projection is not None
        assert self.action_type_embedding is not None
        assert self.action_time_projection is not None
        assert self.action_local_role_embedding is not None
        assert self.action_normalization is not None
        assert self.action_fusion is not None
        quantity = torch.sign(actions[..., :1]) * torch.log1p(
            actions[..., :1].abs()
        )
        continuous = torch.cat((quantity, actions[..., 6:]), dim=-1)
        action_type = actions[..., 1].round().long()
        time_index = actions[..., 2]
        time_features = torch.stack(
            (
                torch.sign(time_index) * torch.log1p(time_index.abs()),
                torch.sin(time_index / 16.0),
                torch.cos(time_index / 16.0),
            ),
            dim=-1,
        )
        token = (
            self.action_continuous_projection(continuous)
            + self.action_type_embedding(
                _safe_ids(action_type, self.action_type_embedding)
            )
            + self.action_time_projection(time_features)
        )
        local_sum = torch.zeros_like(token)
        local_count = torch.zeros_like(actions[..., :1])
        for role, column in enumerate((3, 4, 5)):
            indices = self._bounded_action_indices(actions[..., column])
            present = indices.ne(0).unsqueeze(-1)
            role_ids = torch.full_like(indices, role)
            local_sum = local_sum + (
                self._action_index_embedding(actions[..., column])
                + self.action_local_role_embedding(role_ids)
            ) * present
            local_count = local_count + present.to(local_count.dtype)
        token = token + local_sum / local_count.clamp_min(1.0)
        normalized = self.action_normalization(token)
        encoded = token + self.action_fusion(normalized)
        return encoded, normalized

    def forward(
        self,
        state_latents: Tensor,
        actions: Tensor,
        horizons: Tensor,
        *,
        state_padding_mask: Tensor | None = None,
        action_padding_mask: Tensor | None = None,
        exogenous: Tensor | None = None,
    ) -> Tensor:
        if state_latents.ndim != 3 or actions.ndim != 3 or horizons.ndim != 2:
            raise ValueError(
                "state_latents/actions/horizons must be [B,S,D], [B,A,U], [B,Q]"
            )
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected action dimension {self.config.action_dim}")
        if (
            state_latents.shape[0] != actions.shape[0]
            or actions.shape[0] != horizons.shape[0]
        ):
            raise ValueError("Batch dimensions do not match")
        action_tokens, direct_action_tokens = self._encode_actions(actions)
        if action_padding_mask is None:
            direct_action_summary = direct_action_tokens.mean(dim=1)
        else:
            action_valid = (
                (~action_padding_mask).unsqueeze(-1).to(direct_action_tokens.dtype)
            )
            direct_action_summary = (direct_action_tokens * action_valid).sum(
                dim=1
            ) / action_valid.sum(dim=1).clamp_min(1.0)
        memory = torch.cat((state_latents, action_tokens), dim=1)
        memory_padding_mask = None
        if state_padding_mask is not None or action_padding_mask is not None:
            if state_padding_mask is None:
                state_padding_mask = torch.zeros(
                    state_latents.shape[:2],
                    dtype=torch.bool,
                    device=state_latents.device,
                )
            if action_padding_mask is None:
                action_padding_mask = torch.zeros(
                    actions.shape[:2], dtype=torch.bool, device=actions.device
                )
            memory_padding_mask = torch.cat(
                (state_padding_mask, action_padding_mask), dim=1
            )
        queries = self.query.expand(horizons.shape[0], horizons.shape[1], -1)
        queries = (
            queries
            + self.horizon_projection(
                torch.log1p(horizons.float().clamp_min(0.0)).unsqueeze(-1)
            )
            + 0.1 * direct_action_summary.unsqueeze(1)
        )
        if self.exogenous_projection is not None:
            if exogenous is None:
                raise ValueError("exogenous is required by this model configuration")
            if (
                exogenous.shape[:2] != horizons.shape
                or exogenous.shape[-1] != self.config.exogenous_dim
            ):
                raise ValueError("exogenous must be [B,Q,exogenous_dim]")
            queries = queries + self.exogenous_projection(exogenous)
        return self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_padding_mask,
        )


class SupplyChainJEPACore(nn.Module):
    """One shared encoder chain with State and Dynamics predictors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.online_encoder = SequenceEncoder(config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        if config.feature_semantic_mode == "concept_residual":
            self.target_encoder.embedding.set_feature_residual_scale(
                config.target_feature_residual_scale
            )
        freeze_module(self.target_encoder)
        self.state_predictor = StatePredictor(config)
        self.dynamics_predictor = DynamicsPredictor(config)

    def train(self, mode: bool = True) -> SupplyChainJEPACore:
        """Set train mode while keeping the stop-gradient target deterministic."""

        super().train(mode)
        self.target_encoder.eval()
        if not any(
            parameter.requires_grad for parameter in self.online_encoder.parameters()
        ):
            self.online_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        ema_update(self.online_encoder, self.target_encoder, momentum)

    def set_semantic_schema_schedule(
        self,
        *,
        online_keep_probability: float,
        target_scale: float,
    ) -> None:
        """Route source-schema information during semantic checkpoint migration."""

        if self.config.feature_semantic_mode != "concept_residual":
            return
        self.online_encoder.embedding.set_schema_residual_schedule(
            keep_probability=online_keep_probability,
            scale_override=None,
        )
        self.target_encoder.embedding.set_schema_residual_schedule(
            scale_override=target_scale,
        )

    def freeze_source_schema_residuals(self) -> None:
        """Freeze migrated raw-feature and family tables that encode source identity."""

        if self.config.feature_semantic_mode != "concept_residual":
            return
        for encoder in (self.online_encoder, self.target_encoder):
            embedding = encoder.embedding
            embedding.numeric_weight.requires_grad_(False)
            embedding.feature_embedding.weight.requires_grad_(False)
            embedding.family_embedding.weight.requires_grad_(False)

    @torch.no_grad()
    def hard_sync_target(self) -> None:
        """Copy the online encoder into the EMA target encoder."""

        hard_sync(self.online_encoder, self.target_encoder)

    @torch.no_grad()
    def hard_sync_online_from_target(self) -> None:
        """Copy the EMA target encoder into the online encoder."""

        hard_sync(self.target_encoder, self.online_encoder)

    @torch.no_grad()
    def canonicalize_encoder(self, source: str = "target") -> nn.Module:
        """Hard-sync one canonical encoder space from ``target`` or ``online``.

        EMA target is the default state-pretraining deployment selection.  Selecting it
        copies target -> online and therefore never overwrites the chosen EMA
        weights with the noisier online encoder.
        """

        if source == "target":
            self.hard_sync_online_from_target()
        elif source == "online":
            self.hard_sync_target()
        else:
            raise ValueError("source must be 'target' or 'online'")
        return self.online_encoder

    def freeze_encoders_for_dynamics(self, *, source: str = "target") -> None:
        """Canonicalize the selected source, then freeze both dynamics encoders."""

        self.canonicalize_encoder(source=source)
        freeze_module(self.online_encoder)
        freeze_module(self.target_encoder)
        unfreeze_module(self.dynamics_predictor)

    def encode_context(self, batch: Mapping[str, Any]) -> Tensor:
        return self.online_encoder(batch, "context")

    def encode_context_with_special_tokens(
        self, batch: Mapping[str, Any]
    ) -> SequenceEncoding:
        """Expose BELIEF/REG latents during State JEPA training."""

        return self.online_encoder.encode_with_special_tokens(batch, "context")

    @torch.no_grad()
    def encode_target_with_special_tokens(
        self,
        batch: Mapping[str, Any],
        *,
        anchor_time: Tensor | None = None,
        window_size: int | None = None,
    ) -> tuple[SequenceEncoding, Tensor | None]:
        """Encode target tokens and expose target BELIEF/REG without misalignment."""

        padding_mask = batch.get("target_padding_mask")
        if (anchor_time is None) != (window_size is None):
            raise ValueError("anchor_time and window_size must be supplied together")
        if anchor_time is not None:
            padding_mask = build_local_target_window(
                batch["target_time_index"],
                anchor_time,
                int(window_size),
                padding_mask=padding_mask,
            )
        return (
            self.target_encoder.encode_with_special_tokens(
                batch, "target", padding_mask=padding_mask
            ),
            padding_mask,
        )

    @torch.no_grad()
    def encode_target(
        self,
        batch: Mapping[str, Any],
        *,
        anchor_time: Tensor | None = None,
        window_size: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Stable target-token API with endpoint-causal optional windows."""

        encoding, padding_mask = self.encode_target_with_special_tokens(
            batch,
            anchor_time=anchor_time,
            window_size=window_size,
        )
        return encoding.token_latents, padding_mask

    def target_query_metadata(self, batch: Mapping[str, Any]) -> Tensor:
        """Build leakage-safe target queries from metadata only, never values."""

        # Semantic migration deliberately removes raw feature/family identity
        # from target queries.  The target embedding owns that deterministic
        # schema gate, whereas the online context gate is stochastic.
        return self.target_encoder.embedding.metadata(batch, "target")

    def select_canonical_state(self, encoding: SequenceEncoding) -> Tensor:
        """Select the checkpoint-declared state vector used by Dynamics/transfer."""

        if self.config.canonical_state_pooling == "belief_token":
            return encoding.belief_latent
        valid = (~encoding.token_padding_mask).to(encoding.token_latents.dtype)
        return (encoding.token_latents * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def predict_state(
        self,
        batch: Mapping[str, Any],
        *,
        role_ids: Tensor | None = None,
        horizons: Tensor | None = None,
        target_padding_mask: Tensor | None = None,
    ) -> Tensor:
        context = self.encode_context(batch)
        if target_padding_mask is None:
            target_padding_mask = batch.get("target_padding_mask")
        return self.state_predictor(
            context,
            self.target_query_metadata(batch),
            context_padding_mask=batch.get("context_padding_mask"),
            target_padding_mask=target_padding_mask,
            role_ids=role_ids,
            horizons=horizons,
        )

    @torch.no_grad()
    def canonical_state_latents(
        self, batch: Mapping[str, Any], *, prefix: str = "context"
    ) -> Tensor:
        """Encode dynamics inputs without constructing encoder gradients."""

        if any(
            parameter.requires_grad for parameter in self.online_encoder.parameters()
        ):
            raise RuntimeError(
                "Freeze encoders with freeze_encoders_for_dynamics() before dynamics post-training"
            )
        return self.canonical_state_encoding(batch, prefix=prefix).token_latents

    @torch.no_grad()
    def canonical_state_encoding(
        self, batch: Mapping[str, Any], *, prefix: str = "context"
    ) -> SequenceEncoding:
        """Return frozen token, BELIEF, and REG latents for dynamics post-training."""

        if any(
            parameter.requires_grad for parameter in self.online_encoder.parameters()
        ):
            raise RuntimeError(
                "Freeze encoders with freeze_encoders_for_dynamics() before dynamics post-training"
            )
        return self.online_encoder.encode_with_special_tokens(batch, prefix)

    @torch.no_grad()
    def canonical_belief_latent(
        self, batch: Mapping[str, Any], *, prefix: str = "context"
    ) -> Tensor:
        """Return the configured canonical state as Dynamics memory ``[B,1,D]``."""

        encoding = self.canonical_state_encoding(batch, prefix=prefix)
        return self.select_canonical_state(encoding).unsqueeze(1)
