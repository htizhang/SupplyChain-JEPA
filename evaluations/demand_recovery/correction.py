"""JEPA retrieval features and the lightweight demand correction model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import benchmark

if TYPE_CHECKING:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

RANDOM_SEED = 20260728
HISTORY_DAYS = benchmark.HISTORY_DAYS


def interval_targets(data: Any, examples: Any) -> np.ndarray:
    sales = data.sales[examples.series, examples.day]
    return np.sum(sales * examples.mask, axis=1, dtype=np.float64).astype(
        np.float32
    )


def interval_prediction(full_prediction: np.ndarray, examples: Any) -> np.ndarray:
    return np.sum(
        np.asarray(full_prediction, dtype=np.float32) * examples.mask,
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)


def interval_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=np.float64)
    estimate = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, None)
    denominator = max(float(target.sum()), 1e-12)
    return {
        "wape": float(np.abs(estimate - target).sum() / denominator),
        "wpe": float((estimate - target).sum() / denominator),
        "mae": float(np.mean(np.abs(estimate - target))),
        "target_sum": float(target.sum()),
        "prediction_sum": float(estimate.sum()),
        "examples": int(len(target)),
    }


def build_interval_features(
    data: Any,
    examples: Any,
    priors: Any,
    analog_full: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    series = examples.series
    day = examples.day
    mask = examples.mask
    offsets = np.arange(-HISTORY_DAYS, 0, dtype=np.int16)
    history_days = day[:, None] + offsets[None, :]
    history_sales = data.sales[series[:, None], history_days]
    history_available = data.stock[series[:, None], history_days] == 0
    history = np.where(history_available, history_sales, np.nan)
    daily_total = np.nansum(history, axis=2)
    target_sales = data.sales[series, day]
    target_visible = np.where(mask, np.nan, target_sales)
    mask_length = mask.sum(axis=1)
    mask_onset = np.argmax(mask, axis=1)
    mask_offset = 15 - np.argmax(mask[:, ::-1], axis=1)
    visible_sum = np.nansum(target_visible, axis=1)
    visible_count = (~mask).sum(axis=1)
    fallback = benchmark.predict_historical_profile(data, examples, priors)
    analog_total = interval_prediction(analog_full, examples)
    profile_total = interval_prediction(fallback, examples)
    product_prior = np.empty(len(examples), dtype=np.float32)
    category_prior = np.empty(len(examples), dtype=np.float32)
    product_col = benchmark.ID_COLUMNS.index("product_id")
    category_col = benchmark.ID_COLUMNS.index("third_category_id")
    for row, series_index in enumerate(series):
        product = int(data.static[series_index, product_col])
        category = int(data.static[series_index, category_col])
        if product in priors.product_lookup:
            product_values = priors.product_values[
                priors.product_lookup[product]
            ]
        else:
            product_values = fallback[row]
        if category in priors.category_lookup:
            category_values = priors.category_values[
                priors.category_lookup[category]
            ]
        else:
            category_values = priors.global_values
        product_prior[row] = float(product_values[mask[row]].sum())
        category_prior[row] = float(category_values[mask[row]].sum())

    date_int = data.dates.astype("datetime64[D]").astype(np.int64)
    dow = ((date_int[day] + 3) % 7).astype(np.float32)
    target_covariates = data.covariates[series, day]
    static_names = (
        "city_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    )
    columns: list[np.ndarray] = [
        analog_total,
        profile_total,
        mask_length.astype(np.float32),
        mask_onset.astype(np.float32),
        mask_offset.astype(np.float32),
        visible_sum.astype(np.float32),
        visible_count.astype(np.float32),
        (visible_sum / np.maximum(visible_count, 1)).astype(np.float32),
        np.mean(history_available, axis=(1, 2)).astype(np.float32),
        np.nanmean(history, axis=(1, 2)).astype(np.float32),
        np.nanmedian(history, axis=(1, 2)).astype(np.float32),
        daily_total[:, -1].astype(np.float32),
        daily_total[:, -7].astype(np.float32),
        np.mean(daily_total[:, -7:], axis=1).astype(np.float32),
        np.std(daily_total[:, -7:], axis=1).astype(np.float32),
        np.mean(daily_total, axis=1).astype(np.float32),
        np.std(daily_total, axis=1).astype(np.float32),
        product_prior,
        category_prior,
        (priors.global_values[None, :] * mask).sum(axis=1).astype(np.float32),
        priors.series_demand_mean[series].astype(np.float32),
        priors.demand_group[series].astype(np.float32),
        dow,
        np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32),
        np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32),
    ]
    names = [
        "analog_total",
        "profile_total",
        "mask_length",
        "mask_onset",
        "mask_offset",
        "visible_sum",
        "visible_count",
        "visible_mean",
        "history_observed_fraction",
        "history_mean",
        "history_median",
        "daily_total_lag1",
        "daily_total_lag7",
        "daily_total_mean7",
        "daily_total_std7",
        "daily_total_mean28",
        "daily_total_std28",
        "product_prior_total",
        "category_prior_total",
        "global_prior_total",
        "series_demand_mean",
        "demand_group",
        "day_of_week",
        "day_sin",
        "day_cos",
    ]
    for index, name in enumerate(benchmark.COVARIATE_COLUMNS):
        columns.append(target_covariates[:, index].astype(np.float32))
        names.append(f"target_{name}")
    for name in static_names:
        index = benchmark.ID_COLUMNS.index(name)
        columns.append(data.static[series, index].astype(np.float32))
        names.append(name)
    for position, hour in enumerate(benchmark.OPERATIONAL_HOURS):
        columns.append(mask[:, position].astype(np.float32))
        names.append(f"mask_h{int(hour):02d}")
    matrix = np.column_stack(columns).astype(np.float32)
    return matrix, names


@dataclass
class LatentAdapter:
    scaler: StandardScaler
    pca: PCA
    train_projection: np.ndarray


def fit_latent_adapter(latent: np.ndarray, components: int) -> LatentAdapter:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(latent)
    count = min(int(components), scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=count, random_state=RANDOM_SEED)
    projection = pca.fit_transform(scaled).astype(np.float32)
    return LatentAdapter(scaler=scaler, pca=pca, train_projection=projection)


def project_latent(adapter: LatentAdapter, latent: np.ndarray) -> np.ndarray:
    return adapter.pca.transform(adapter.scaler.transform(latent)).astype(
        np.float32
    )


def _weighted_neighbor_features(
    distance: np.ndarray,
    indices: np.ndarray,
    *,
    train_y: np.ndarray,
    train_analog: np.ndarray,
    k_values: tuple[int, ...],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    ratio = (train_y + 0.5) / (train_analog + 0.5)
    normalized_residual = (train_y - train_analog) / (train_analog + 1.0)
    for k in k_values:
        local_distance = distance[:, :k]
        local_indices = indices[:, :k]
        scale = np.maximum(
            np.median(local_distance, axis=1, keepdims=True), 1e-6
        )
        weight = np.exp(
            -(local_distance - local_distance[:, :1]) / scale
        )
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
        neighbor_ratio = np.sum(weight * ratio[local_indices], axis=1)
        neighbor_residual = np.sum(
            weight * normalized_residual[local_indices], axis=1
        )
        neighbor_y = np.sum(weight * train_y[local_indices], axis=1)
        columns.extend(
            [
                neighbor_ratio.astype(np.float32),
                neighbor_residual.astype(np.float32),
                neighbor_y.astype(np.float32),
                local_distance[:, 0].astype(np.float32),
                local_distance.mean(axis=1).astype(np.float32),
                local_distance.std(axis=1).astype(np.float32),
            ]
        )
        names.extend(
            [
                f"jepa_knn_ratio_k{k}",
                f"jepa_knn_normalized_residual_k{k}",
                f"jepa_knn_y_k{k}",
                f"jepa_knn_min_distance_k{k}",
                f"jepa_knn_mean_distance_k{k}",
                f"jepa_knn_std_distance_k{k}",
            ]
        )
    return np.column_stack(columns), names


def build_latent_retrieval_features(
    train_projection: np.ndarray,
    query_projection: np.ndarray,
    *,
    train_y: np.ndarray,
    train_analog: np.ndarray,
    query_is_train: bool,
    threads: int,
) -> tuple[np.ndarray, list[str], NearestNeighbors]:
    from sklearn.neighbors import NearestNeighbors

    k_values = (5, 15, 50)
    maximum = max(k_values) + (1 if query_is_train else 0)
    neighbors = NearestNeighbors(
        n_neighbors=maximum,
        metric="euclidean",
        n_jobs=threads,
    )
    neighbors.fit(train_projection)
    distance, indices = neighbors.kneighbors(query_projection)
    if query_is_train:
        distance = distance[:, 1:]
        indices = indices[:, 1:]
    features, names = _weighted_neighbor_features(
        distance,
        indices,
        train_y=train_y,
        train_analog=train_analog,
        k_values=k_values,
    )
    return features, names, neighbors


def query_latent_retrieval_features(
    neighbors: NearestNeighbors,
    query_projection: np.ndarray,
    *,
    train_y: np.ndarray,
    train_analog: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    k_values = (5, 15, 50)
    distance, indices = neighbors.kneighbors(
        query_projection, n_neighbors=max(k_values)
    )
    return _weighted_neighbor_features(
        distance,
        indices,
        train_y=train_y,
        train_analog=train_analog,
        k_values=k_values,
    )


def _fill_nonfinite(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    reference = np.asarray(train, dtype=np.float32)
    finite = np.isfinite(reference)
    median = np.zeros(reference.shape[1], dtype=np.float32)
    for column in range(reference.shape[1]):
        values = reference[finite[:, column], column]
        median[column] = float(np.median(values)) if len(values) else 0.0

    def clean(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).copy()
        row, column = np.nonzero(~np.isfinite(result))
        if len(row):
            result[row, column] = median[column]
        return result

    return (clean(reference), *(clean(value) for value in others))


def _candidate_configurations() -> list[dict[str, Any]]:
    return [
        {
            "max_depth": 2,
            "min_child_weight": 8,
            "learning_rate": 0.04,
            "n_estimators": 450,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 10.0,
        },
        {
            "max_depth": 3,
            "min_child_weight": 12,
            "learning_rate": 0.035,
            "n_estimators": 550,
            "subsample": 0.85,
            "colsample_bytree": 0.70,
            "reg_lambda": 15.0,
        },
        {
            "max_depth": 4,
            "min_child_weight": 20,
            "learning_rate": 0.025,
            "n_estimators": 650,
            "subsample": 0.80,
            "colsample_bytree": 0.65,
            "reg_lambda": 20.0,
        },
    ]


def _fit_model(
    X: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    *,
    threads: int,
) -> XGBRegressor:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=threads,
        max_bin=256,
        reg_alpha=0.05,
        **config,
    )
    model.fit(X, target, verbose=False)
    return model


def _tune_blend(
    y: np.ndarray,
    analog: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    best: tuple[float, float, float, np.ndarray] | None = None
    for blend in np.linspace(0.0, 1.5, 61):
        raw = np.clip(
            analog + float(blend) * (candidate - analog), 0.0, None
        )
        if float(raw.sum()) <= 1e-12:
            continue
        unbiased = float(y.sum() / raw.sum())
        scales = np.unique(
            np.concatenate(
                (
                    np.asarray([1.0, unbiased]),
                    unbiased * np.linspace(0.92, 1.08, 33),
                )
            )
        )
        for scale in scales:
            prediction = np.clip(raw * float(scale), 0.0, None)
            metrics = interval_metrics(y, prediction)
            score = metrics["wape"]
            tie = abs(metrics["wpe"])
            if best is None or (score, tie) < (best[0], best[1]):
                best = (score, tie, float(blend), prediction)
                best_scale = float(scale)
    if best is None:
        raise RuntimeError("No valid blend candidate")
    return best[3], {
        "validation_wape": best[0],
        "validation_abs_wpe": best[1],
        "blend": best[2],
        "scale": best_scale,
    }


@dataclass
class SelectedModel:
    model: XGBRegressor
    target_kind: str
    configuration: dict[str, Any]
    blend: float
    scale: float
    validation_metrics: dict[str, float]
    validation_prediction: np.ndarray


def train_and_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    analog_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    analog_validation: np.ndarray,
    *,
    threads: int,
) -> tuple[SelectedModel, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    selected: SelectedModel | None = None
    target_payloads = {
        "residual": y_train - analog_train,
        "log_ratio": np.log((y_train + 0.5) / (analog_train + 0.5)),
        "log_target": np.log1p(y_train),
    }
    for target_kind, target in target_payloads.items():
        for index, configuration in enumerate(_candidate_configurations()):
            started = time.perf_counter()
            model = _fit_model(
                X_train, target, configuration, threads=threads
            )
            raw = model.predict(X_validation)
            if target_kind == "residual":
                proposal = np.clip(analog_validation + raw, 0.0, None)
            elif target_kind == "log_ratio":
                proposal = np.clip(
                    (analog_validation + 0.5) * np.exp(raw) - 0.5,
                    0.0,
                    None,
                )
            else:
                proposal = np.clip(np.expm1(raw), 0.0, None)
            tuned_prediction, tuning = _tune_blend(
                y_validation, analog_validation, proposal
            )
            metrics = interval_metrics(y_validation, tuned_prediction)
            record = {
                "target_kind": target_kind,
                "configuration_index": index,
                "configuration": configuration,
                "tuning": tuning,
                "metrics": metrics,
                "fit_seconds": time.perf_counter() - started,
            }
            candidates.append(record)
            print(json.dumps({"event": "candidate", **record}), flush=True)
            if (
                selected is None
                or metrics["wape"] < selected.validation_metrics["wape"]
            ):
                selected = SelectedModel(
                    model=model,
                    target_kind=target_kind,
                    configuration=configuration,
                    blend=float(tuning["blend"]),
                    scale=float(tuning["scale"]),
                    validation_metrics=metrics,
                    validation_prediction=tuned_prediction,
                )
    if selected is None:
        raise RuntimeError("No correction model was selected")
    return selected, candidates


def predict_selected(
    selected: SelectedModel,
    X: np.ndarray,
    analog: np.ndarray,
) -> np.ndarray:
    raw = selected.model.predict(X)
    if selected.target_kind == "residual":
        proposal = np.clip(analog + raw, 0.0, None)
    elif selected.target_kind == "log_ratio":
        proposal = np.clip(
            (analog + 0.5) * np.exp(raw) - 0.5, 0.0, None
        )
    else:
        proposal = np.clip(np.expm1(raw), 0.0, None)
    return np.clip(
        (
            analog
            + selected.blend * (proposal - analog)
        )
        * selected.scale,
        0.0,
        None,
    ).astype(np.float32)
