"""Map FreshRetailNet observations to a frozen SupplyChain-JEPA state space."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import benchmark

DEFAULT_MODEL_ID = "htizhang/SupplyChain-JEPA"
DEFAULT_REVISION = "493786fda0855b72f61235068c62cc7b2a11fbec"
STATUS_TO_ID = {
    "observed": 1, "censored": 2, "delayed": 3, "missing": 4,
    "outage": 5, "not_applicable": 6,
}
HISTORY_DAYS = benchmark.HISTORY_DAYS
TOKEN_DAYS = 29
TOKENS_PER_DAY = 2
TOKEN_COUNT = TOKEN_DAYS * TOKENS_PER_DAY

_FRESH_FEATURE_CANDIDATES = {
    "edge::sales_observed": ("edge::sales_observed", "entity::sales_observed", "entity::sales", "event::sales"),
    "edge::demand_reported": ("edge::demand_reported", "entity::demand_reported", "entity::demand", "event::demand"),
    "edge::current_demand_lower_bound": ("edge::current_demand_lower_bound", "entity::sales_observed", "entity::demand_reported"),
    "edge::customer_shipments_observed": ("edge::customer_shipments_observed", "entity::outbound_shipments", "entity::delivery_reported", "entity::sales_observed"),
    "global::sales_observed_total": ("global::sales_observed_total", "global::sales_total_observed", "global::sales_this_period"),
    "global::demand_reported_total": ("global::demand_reported_total", "global::demand_reported_this_period", "global::demand_total"),
    "global::current_demand_lower_bound_total": ("global::current_demand_lower_bound_total", "global::sales_total_observed", "global::sales_this_period"),
    "global::customer_shipments_observed_total": ("global::customer_shipments_observed_total", "global::delivered_shipments", "global::sales_total_observed"),
    "global::reported_events": ("global::reported_events",),
    "global::clock": ("global::clock",),
    "global::day": ("global::day", "global::calendar_day"),
    "global::period": ("global::period", "action::period"),
    "edge::stockout_indicator": ("edge::stockout_indicator", "entity::stockout"),
    "edge::demand_censored_indicator": ("edge::demand_censored_indicator", "entity::demand_censored"),
    "edge::report_available": ("edge::report_available", "entity::report_available"),
    "global::scenario_shock_flag": ("global::scenario_shock_flag", "global::shock_active"),
}

@dataclass(frozen=True)
class TensorizerState:
    feature_names: tuple[str, ...]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    categorical_vocabs: Mapping[str, Mapping[str, int]]
    metadata_vocabs: Mapping[str, Mapping[str, int]]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | Any) -> TensorizerState:
        def value(name: str) -> Any:
            if isinstance(payload, Mapping):
                return payload[name]
            return getattr(payload, name)

        return cls(
            feature_names=tuple(value("feature_names")),
            numeric_mean=np.asarray(value("numeric_mean"), dtype=np.float32),
            numeric_std=np.asarray(value("numeric_std"), dtype=np.float32),
            categorical_vocabs=value("categorical_vocabs"),
            metadata_vocabs=value("metadata_vocabs"),
        )

@dataclass
class RepresentationBundle:
    encoder: torch.nn.Module
    state_pool_predictor: torch.nn.Module | None
    state_pool_predictor_config: Mapping[str, Any]
    tensorizer_state: TensorizerState
    device: torch.device
    d_model: int
    model_id: str
    revision: str

    @property
    def predictive_hidden_state_available(self) -> bool:
        query = self.state_pool_predictor_config.get("query_vocabulary", {})
        masks = self.state_pool_predictor_config.get("mask_vocabulary", {})
        return self.state_pool_predictor is not None and "hidden_current" in query and any(
            name in masks for name in ("outage", "contiguous_time", "entity_group")
        )

    @property
    def latent_width(self) -> int:
        return (3 + int(self.predictive_hidden_state_available)) * self.d_model

def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)

def representation_from_model_bundle(
    released: Any,
    *,
    model_id: str,
    revision: str,
    device: str,
) -> RepresentationBundle:
    """Adapt the clean package ModelBundle without depending on its internals."""

    core = _field(released, "core", _field(released, "model"))
    if core is None:
        raise TypeError("ModelBundle must expose `core` or `model`")
    tensorizer = _field(released, "tensorizer_config")
    if tensorizer is None:
        tensorizer_object = _field(released, "tensorizer")
        tensorizer = _field(tensorizer_object, "state")
    if tensorizer is None:
        raise TypeError(
            "ModelBundle must expose `tensorizer_config` or `tensorizer.state`"
        )
    raw_config = _field(released, "release_config", _field(released, "config", {}))
    pool_config = (
        raw_config.get("state_pool_predictor", {})
        if isinstance(raw_config, Mapping)
        else {}
    )
    encoder = core.online_encoder
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    pool = _field(released, "state_pool_predictor")
    if pool is not None:
        pool.eval()
        for parameter in pool.parameters():
            parameter.requires_grad_(False)
    return RepresentationBundle(
        encoder=encoder,
        state_pool_predictor=pool,
        state_pool_predictor_config=pool_config,
        tensorizer_state=TensorizerState.from_payload(tensorizer),
        device=torch.device(device),
        d_model=int(core.config.d_model),
        model_id=model_id,
        revision=revision,
    )


def load_state_representation(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "cpu",
) -> RepresentationBundle:
    """Load the immutable release through the package's public API."""

    from supplychain_jepa import load_pretrained

    released = load_pretrained(
        repo_id=model_id, revision=revision, device=device, verify=True
    )
    return representation_from_model_bundle(
        released,
        model_id=model_id,
        revision=revision,
        device=device,
    )


def predictive_hidden_state_latent(
    bundle: RepresentationBundle,
    canonical_state: torch.Tensor,
) -> torch.Tensor:
    """Recover a stockout-shaped hidden operational state in latent space."""

    if not bundle.predictive_hidden_state_available:
        return canonical_state.new_empty((canonical_state.shape[0], 0))
    assert bundle.state_pool_predictor is not None
    query_vocabulary = bundle.state_pool_predictor_config["query_vocabulary"]
    mask_vocabulary = bundle.state_pool_predictor_config["mask_vocabulary"]
    mask_name = next(
        name
        for name in ("outage", "contiguous_time", "entity_group")
        if name in mask_vocabulary
    )
    batch_size = canonical_state.shape[0]
    return bundle.state_pool_predictor(
        canonical_state,
        torch.zeros(
            batch_size,
            dtype=canonical_state.dtype,
            device=canonical_state.device,
        ),
        torch.full(
            (batch_size,),
            int(query_vocabulary["hidden_current"]),
            dtype=torch.long,
            device=canonical_state.device,
        ),
        torch.full(
            (batch_size,),
            int(mask_vocabulary[mask_name]),
            dtype=torch.long,
            device=canonical_state.device,
        ),
    )


def _metadata_id(state: Any, namespace: str, value: str) -> int:
    return int(state.metadata_vocabs[namespace].get(value, 1))


def _metadata_candidate_id(
    state: Any, namespace: str, candidates: tuple[str, ...]
) -> int:
    vocab = state.metadata_vocabs[namespace]
    for candidate in candidates:
        if candidate in vocab:
            return int(vocab[candidate])
    return 1


def _resolve_feature(
    feature_index: Mapping[str, int], logical_name: str
) -> str | None:
    for candidate in _FRESH_FEATURE_CANDIDATES.get(
        logical_name, (logical_name,)
    ):
        if candidate in feature_index:
            return candidate
    return None


def _categorical_id(state: Any, feature: str, value: bool | str) -> int:
    key = (
        f"bool:{str(value).lower()}"
        if isinstance(value, bool)
        else f"str:{value}"
    )
    return int(state.categorical_vocabs.get(feature, {}).get(key, 1))


def _empty_jepa_batch(
    state: Any,
    batch_size: int,
    *,
    device: torch.device,
    operational_kind: str = "edge",
    operational_type: str = "edge_state",
    family_candidates: tuple[str, ...] = ("m5",),
) -> dict[str, torch.Tensor]:
    feature_count = int(state.feature_count)
    values = torch.zeros(
        (batch_size, TOKEN_COUNT, feature_count),
        dtype=torch.float32,
        device=device,
    )
    categorical = torch.zeros(
        (batch_size, TOKEN_COUNT, feature_count),
        dtype=torch.long,
        device=device,
    )
    present = torch.zeros(
        (batch_size, TOKEN_COUNT, feature_count),
        dtype=torch.bool,
        device=device,
    )
    status = torch.zeros_like(categorical)
    age = torch.zeros_like(values)
    edge_positions = torch.arange(0, TOKEN_COUNT, 2, device=device)
    global_positions = edge_positions + 1
    time_index = (
        torch.arange(TOKEN_DAYS, device=device)
        .repeat_interleave(TOKENS_PER_DAY)
        .repeat(batch_size, 1)
    )
    token_kind = torch.zeros(
        (batch_size, TOKEN_COUNT), dtype=torch.long, device=device
    )
    token_type = torch.zeros_like(token_kind)
    token_kind[:, edge_positions] = _metadata_candidate_id(
        state, "token_kind", (operational_kind,)
    )
    token_kind[:, global_positions] = _metadata_id(
        state, "token_kind", "global"
    )
    token_type[:, edge_positions] = _metadata_candidate_id(
        state, "token_type", (operational_type,)
    )
    token_type[:, global_positions] = _metadata_id(
        state, "token_type", "global_state"
    )
    family_id = _metadata_candidate_id(state, "family", family_candidates)
    batch = {
        "context_values": values,
        "context_categorical_ids": categorical,
        "context_feature_present": present,
        "context_status_ids": status,
        "context_age": age,
        "context_time_index": time_index,
        "context_clock": time_index.to(torch.float32),
        "context_family_id": torch.full_like(token_kind, family_id),
        "context_token_kind_id": token_kind,
        "context_type_id": token_type,
        "context_entity_index": torch.full_like(token_kind, 2),
        "context_product_index": torch.full_like(token_kind, 2),
        "context_node_index": torch.full_like(token_kind, 2),
        "context_source_index": torch.full_like(token_kind, 2),
        "context_target_index": torch.full_like(token_kind, 3),
        "context_alignment_index": (
            torch.arange(2, TOKEN_COUNT + 2, device=device)
            .repeat(batch_size, 1)
            .long()
        ),
        "context_padding_mask": torch.zeros(
            (batch_size, TOKEN_COUNT), dtype=torch.bool, device=device
        ),
    }
    return batch


def _set_numeric(
    batch: dict[str, torch.Tensor],
    state: Any,
    feature_index: dict[str, int],
    feature: str,
    positions: torch.Tensor,
    raw: np.ndarray,
    *,
    status: np.ndarray | None = None,
) -> None:
    index = feature_index[feature]
    values = torch.as_tensor(
        np.array(raw, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=batch["context_values"].device,
    )
    normalized = (
        values - float(state.numeric_mean[index])
    ) / float(state.numeric_std[index])
    batch["context_values"][:, positions, index] = normalized
    batch["context_feature_present"][:, positions, index] = True
    if status is None:
        batch["context_status_ids"][:, positions, index] = STATUS_TO_ID[
            "observed"
        ]
    else:
        batch["context_status_ids"][:, positions, index] = torch.as_tensor(
            status,
            dtype=torch.long,
            device=batch["context_values"].device,
        )


def _set_boolean(
    batch: dict[str, torch.Tensor],
    state: Any,
    feature_index: dict[str, int],
    feature: str,
    positions: torch.Tensor,
    raw: np.ndarray,
) -> None:
    index = feature_index[feature]
    false_id = _categorical_id(state, feature, False)
    true_id = _categorical_id(state, feature, True)
    values = np.where(raw, true_id, false_id)
    batch["context_categorical_ids"][:, positions, index] = torch.as_tensor(
        values, dtype=torch.long, device=batch["context_values"].device
    )
    batch["context_feature_present"][:, positions, index] = True
    batch["context_status_ids"][:, positions, index] = STATUS_TO_ID["observed"]


def _fresh_batch(
    data: Any,
    examples: Any,
    state: Any,
    *,
    start: int,
    stop: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    series = examples.series[start:stop]
    target_day = examples.day[start:stop]
    mask = examples.mask[start:stop]
    batch_size = stop - start
    offsets = np.arange(-HISTORY_DAYS, 0, dtype=np.int16)
    history_days = target_day[:, None] + offsets[None, :]
    history_sales = data.sales[series[:, None], history_days]
    history_available = data.stock[series[:, None], history_days] == 0
    observed_history = np.where(history_available, history_sales, 0.0)
    history_total = observed_history.sum(axis=2, dtype=np.float64).astype(
        np.float32
    )
    history_visible_hours = history_available.sum(axis=2).astype(np.float32)
    history_censored = ~history_available.all(axis=2)

    target_sales = data.sales[series, target_day]
    target_visible = ~mask
    target_total = np.sum(
        target_sales * target_visible, axis=1, dtype=np.float64
    ).astype(np.float32)
    target_visible_hours = target_visible.sum(axis=1).astype(np.float32)

    daily_total = np.concatenate((history_total, target_total[:, None]), axis=1)
    visible_hours = np.concatenate(
        (history_visible_hours, target_visible_hours[:, None]), axis=1
    )
    censored = np.concatenate(
        (history_censored, np.ones((batch_size, 1), dtype=bool)), axis=1
    )
    sales_status = np.where(
        censored, STATUS_TO_ID["censored"], STATUS_TO_ID["observed"]
    ).astype(np.int64)

    history_covariates = data.covariates[series[:, None], history_days]
    target_covariates = data.covariates[series, target_day][:, None, :]
    covariates = np.concatenate(
        (history_covariates, target_covariates), axis=1
    )
    holiday = covariates[:, :, benchmark.COVARIATE_COLUMNS.index("holiday_flag")]
    activity = covariates[:, :, benchmark.COVARIATE_COLUMNS.index("activity_flag")]
    discount = covariates[:, :, benchmark.COVARIATE_COLUMNS.index("discount")]
    shock = (holiday > 0.5) | (activity > 0.5) | (discount > 0.0)

    feature_index = {
        name: index for index, name in enumerate(state.feature_names)
    }
    resolved_features = {
        logical_name: _resolve_feature(feature_index, logical_name)
        for logical_name in _FRESH_FEATURE_CANDIDATES
    }
    uses_v2_entity_features = any(
        feature is not None and feature.startswith("entity::")
        for logical_name, feature in resolved_features.items()
        if logical_name.startswith("edge::")
    )
    result = _empty_jepa_batch(
        state,
        batch_size,
        device=device,
        operational_kind="entity" if uses_v2_entity_features else "edge",
        operational_type=(
            "sku_warehouse_state"
            if uses_v2_entity_features
            else "edge_state"
        ),
    )
    edge_positions = torch.arange(0, TOKEN_COUNT, 2, device=device)
    global_positions = edge_positions + 1

    for logical_name in (
        "edge::sales_observed",
        "edge::demand_reported",
        "edge::current_demand_lower_bound",
        "edge::customer_shipments_observed",
    ):
        feature = resolved_features[logical_name]
        if feature is None:
            continue
        _set_numeric(
            result,
            state,
            feature_index,
            feature,
            edge_positions,
            daily_total,
            status=sales_status,
        )
    for logical_name in (
        "global::sales_observed_total",
        "global::demand_reported_total",
        "global::current_demand_lower_bound_total",
        "global::customer_shipments_observed_total",
    ):
        feature = resolved_features[logical_name]
        if feature is None:
            continue
        _set_numeric(
            result,
            state,
            feature_index,
            feature,
            global_positions,
            daily_total,
            status=sales_status,
        )
    reported_events = resolved_features["global::reported_events"]
    if reported_events is not None:
        _set_numeric(
            result,
            state,
            feature_index,
            reported_events,
            global_positions,
            visible_hours,
        )
    day_values = np.broadcast_to(
        np.arange(TOKEN_DAYS, dtype=np.float32), (batch_size, TOKEN_DAYS)
    )
    for logical_name in ("global::clock", "global::day", "global::period"):
        feature = resolved_features[logical_name]
        if feature is None:
            continue
        _set_numeric(
            result,
            state,
            feature_index,
            feature,
            global_positions,
            day_values,
        )
    for logical_name, raw in (
        ("edge::stockout_indicator", censored),
        ("edge::demand_censored_indicator", censored),
        ("edge::report_available", np.ones_like(censored)),
        ("global::scenario_shock_flag", shock),
    ):
        feature = resolved_features[logical_name]
        if feature is None:
            continue
        positions = (
            edge_positions
            if logical_name.startswith("edge::")
            else global_positions
        )
        _set_boolean(
            result,
            state,
            feature_index,
            feature,
            positions,
            raw,
        )
    return result


def encode_examples(
    data: Any,
    examples: Any,
    bundle: RepresentationBundle,
    *,
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    latent_width = bundle.latent_width
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(examples), latent_width):
            return cached
        cache_path.unlink()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.empty((len(examples), latent_width), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            stop = min(start + batch_size, len(examples))
            batch = _fresh_batch(
                data,
                examples,
                bundle.tensorizer_state,
                start=start,
                stop=stop,
                device=bundle.device,
            )
            encoding = bundle.encoder.encode_with_special_tokens(
                batch, "context"
            )
            tokens = encoding.token_latents
            canonical = tokens.mean(dim=1)
            recent = tokens[:, -14:].mean(dim=1)
            endpoint = tokens[:, -2:].mean(dim=1)
            output[start:stop] = (
                torch.cat(
                    (
                        canonical,
                        recent,
                        endpoint,
                        predictive_hidden_state_latent(bundle, canonical),
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            if start == 0 or stop == len(examples) or stop % 10_000 == 0:
                print(
                    json.dumps(
                        {
                            "event": "encoded",
                            "split": examples.name,
                            "rows": stop,
                            "total": len(examples),
                        }
                    ),
                    flush=True,
                )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, output)
    temporary.replace(cache_path)
    return output
