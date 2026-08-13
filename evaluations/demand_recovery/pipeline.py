"""End-to-end controlled-stockout demand-recovery evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import benchmark, correction, representation
from .contract import verify_dataset


@dataclass(frozen=True)
class EvaluationConfig:
    label_budget_percent: int = 5
    adaptation_pool: int = 60_000
    validation_pool: int = 20_000
    test_cap: int | None = None
    pca_components: int = 64
    encode_batch_size: int = 256
    threads: int = 1
    device: str = "cpu"
    model_id: str = representation.DEFAULT_MODEL_ID
    revision: str = representation.DEFAULT_REVISION

    def validate(self) -> None:
        if not 1 <= self.label_budget_percent <= 100:
            raise ValueError("label_budget_percent must be between 1 and 100")
        if self.adaptation_pool < 1 or self.validation_pool < 1:
            raise ValueError("adaptation_pool and validation_pool must be positive")
        if self.pca_components < 1 or self.encode_batch_size < 1:
            raise ValueError("pca_components and encode_batch_size must be positive")
        if self.threads < 1:
            raise ValueError("threads must be positive")



def _budget_size(pool_size: int, percent: int) -> int:
    return max(1, min(pool_size, round(pool_size * percent / 100)))


def _cache_path(
    cache_dir: Path,
    *,
    revision: str,
    examples: benchmark.ExampleSet,
) -> Path:
    digest = hashlib.sha256()
    for value in (
        examples.series,
        examples.day,
        examples.mask.astype(np.uint8),
        examples.rank,
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return cache_dir / revision[:16] / f"{examples.name}-{digest.hexdigest()[:20]}.npy"


def _prepare_examples(
    data_root: Path, config: EvaluationConfig
) -> tuple[
    benchmark.RetailData,
    benchmark.ExampleSet,
    benchmark.ExampleSet,
    list[benchmark.ExampleSet],
    benchmark.Priors,
    dict[str, Any],
]:
    data = benchmark.load_data(data_root)
    series_split, split_manifest = benchmark.build_store_split(data)
    codes, probabilities, mask_manifest = benchmark.build_mask_bank(
        data, series_split
    )
    adaptation, validation, tests, example_summary = benchmark.build_example_sets(
        data,
        series_split,
        codes,
        probabilities,
        config.adaptation_pool,
        config.validation_pool,
        config.test_cap,
    )
    adaptation = adaptation.subset(
        _budget_size(len(adaptation), config.label_budget_percent)
    )
    priors = benchmark.build_priors(data, series_split)
    protocol = {
        "split": split_manifest,
        "mask_bank": {
            "source_partition": mask_manifest["source_partition"],
            "source_date_max": mask_manifest["source_date_max"],
            "unique_patterns": mask_manifest["unique_patterns"],
            "mean_masked_hours": mask_manifest["mean_masked_hours"],
        },
        "examples": example_summary,
    }
    return data, adaptation, validation, tests, priors, protocol


def _clean_features(
    reference: np.ndarray, *queries: np.ndarray
) -> tuple[np.ndarray, ...]:
    return correction._fill_nonfinite(reference, *queries)


def _split_inputs(
    data: benchmark.RetailData,
    examples: benchmark.ExampleSet,
    priors: benchmark.Priors,
    bundle: representation.RepresentationBundle,
    config: EvaluationConfig,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    analog_full = benchmark.predict_analog_day(data, examples, priors)
    target = correction.interval_targets(data, examples)
    analog = correction.interval_prediction(analog_full, examples)
    latent = representation.encode_examples(
        data,
        examples,
        bundle,
        batch_size=config.encode_batch_size,
        cache_path=_cache_path(
            cache_dir, revision=config.revision, examples=examples
        ),
    )
    base, base_names = correction.build_interval_features(
        data, examples, priors, analog_full
    )
    return target, analog, latent, base, base_names


def _relative_improvement(baseline_wape: float, system_wape: float) -> float:
    return (baseline_wape - system_wape) / baseline_wape


def run_evaluation(
    data_root: Path,
    output: Path,
    *,
    cache_dir: Path,
    artifact_dir: Path | None = None,
    config: EvaluationConfig | None = None,
    model_bundle: representation.RepresentationBundle | None = None,
) -> dict[str, Any]:
    """Run the canonical final evaluation and atomically write deterministic JSON."""

    config = config or EvaluationConfig()
    config.validate()
    dataset_hashes = verify_dataset(data_root)
    data, adaptation, validation, tests, priors, protocol = _prepare_examples(
        data_root.expanduser().resolve(), config
    )
    bundle = model_bundle or representation.load_state_representation(
        model_id=config.model_id,
        revision=config.revision,
        device=config.device,
    )

    train_y, train_analog, train_latent, train_base, base_names = _split_inputs(
        data, adaptation, priors, bundle, config, cache_dir
    )
    val_y, val_analog, val_latent, val_base, val_names = _split_inputs(
        data, validation, priors, bundle, config, cache_dir
    )
    if val_names != base_names:
        raise AssertionError("Adaptation and validation feature schemas differ")

    adapter = correction.fit_latent_adapter(
        train_latent, config.pca_components
    )
    val_projection = correction.project_latent(adapter, val_latent)
    train_retrieval, retrieval_names, neighbors = (
        correction.build_latent_retrieval_features(
            adapter.train_projection,
            adapter.train_projection,
            train_y=train_y,
            train_analog=train_analog,
            query_is_train=True,
            threads=config.threads,
        )
    )
    val_retrieval, val_retrieval_names = (
        correction.query_latent_retrieval_features(
            neighbors,
            val_projection,
            train_y=train_y,
            train_analog=train_analog,
        )
    )
    if val_retrieval_names != retrieval_names:
        raise AssertionError("Adaptation and validation retrieval schemas differ")

    pca_names = [
        f"jepa_pca_{index:03d}"
        for index in range(adapter.train_projection.shape[1])
    ]
    feature_names = [*base_names, *pca_names, *retrieval_names]
    X_train = np.column_stack(
        (train_base, adapter.train_projection, train_retrieval)
    )
    X_val = np.column_stack((val_base, val_projection, val_retrieval))
    X_train, X_val = _clean_features(X_train, X_val)

    selected, candidates = correction.train_and_select(
        X_train,
        train_y,
        train_analog,
        X_val,
        val_y,
        val_analog,
        threads=config.threads,
    )
    # Timings are intentionally excluded so repeated runs serialize identically.
    for candidate in candidates:
        candidate.pop("fit_seconds", None)
    val_prediction = correction.predict_selected(selected, X_val, val_analog)
    validation_result = {
        "baseline": correction.interval_metrics(val_y, val_analog),
        "jepa_system": correction.interval_metrics(val_y, val_prediction),
    }
    validation_result["relative_wape_improvement"] = _relative_improvement(
        validation_result["baseline"]["wape"],
        validation_result["jepa_system"]["wape"],
    )

    test_results: list[dict[str, Any]] = []
    for test in tests:
        test_y, test_analog, test_latent, test_base, test_names = _split_inputs(
            data, test, priors, bundle, config, cache_dir
        )
        if test_names != base_names:
            raise AssertionError(f"Feature schema differs for {test.name}")
        projection = correction.project_latent(adapter, test_latent)
        retrieval, names = correction.query_latent_retrieval_features(
            neighbors,
            projection,
            train_y=train_y,
            train_analog=train_analog,
        )
        if names != retrieval_names:
            raise AssertionError(f"Retrieval schema differs for {test.name}")
        X_test = np.column_stack((test_base, projection, retrieval))
        _, X_test = _clean_features(X_train, X_test)
        prediction = correction.predict_selected(selected, X_test, test_analog)
        baseline_metrics = correction.interval_metrics(test_y, test_analog)
        system_metrics = correction.interval_metrics(test_y, prediction)
        test_results.append(
            {
                "split": test.name,
                "mask_seed": int(test.mask_seed),
                "baseline": baseline_metrics,
                "jepa_system": system_metrics,
                "relative_wape_improvement": _relative_improvement(
                    baseline_metrics["wape"], system_metrics["wape"]
                ),
            }
        )

    target_sum = sum(item["baseline"]["target_sum"] for item in test_results)
    baseline_wape = sum(
        item["baseline"]["wape"] * item["baseline"]["target_sum"]
        for item in test_results
    ) / target_sum
    system_wape = sum(
        item["jepa_system"]["wape"]
        * item["jepa_system"]["target_sum"]
        for item in test_results
    ) / target_sum
    pooled = {
        "baseline_wape": baseline_wape,
        "jepa_system_wape": system_wape,
        "relative_wape_improvement": _relative_improvement(
            baseline_wape, system_wape
        ),
        "winning_test_seeds": sum(
            item["jepa_system"]["wape"] < item["baseline"]["wape"]
            for item in test_results
        ),
        "test_seed_count": len(test_results),
    }

    booster = selected.model.get_booster()
    total_gain = booster.get_score(importance_type="total_gain")
    split_count = booster.get_score(importance_type="weight")

    def is_jepa_feature(raw_name: str) -> bool:
        if raw_name.startswith("f") and raw_name[1:].isdigit():
            return feature_names[int(raw_name[1:])].startswith("jepa_")
        return raw_name.startswith("jepa_")

    all_gain = sum(total_gain.values())
    all_splits = sum(split_count.values())
    result = {
        "schema_version": "supplychain-jepa.demand-recovery.v1",
        "task": "controlled-stockout demand recovery",
        "dataset": {
            "name": "Dingdong-Inc/FreshRetailNet-50K",
            "files": dataset_hashes,
            "external": True,
        },
        "model": {
            "repo_id": config.model_id,
            "revision": config.revision,
            "state_encoder_frozen": True,
            "dynamics_predictor_used": False,
        },
        "configuration": asdict(config),
        "protocol": protocol,
        "method": {
            "baseline": "analog_day_knn",
            "adaptation_examples": len(adaptation),
            "feature_count": len(feature_names),
            "jepa_feature_count": len(pca_names) + len(retrieval_names),
            "selected_target": selected.target_kind,
            "selected_configuration": selected.configuration,
            "blend": selected.blend,
            "scale": selected.scale,
            "jepa_total_gain_share": (
                sum(v for k, v in total_gain.items() if is_jepa_feature(k))
                / all_gain
                if all_gain
                else 0.0
            ),
            "jepa_split_share": (
                sum(v for k, v in split_count.items() if is_jepa_feature(k))
                / all_splits
                if all_splits
                else 0.0
            ),
        },
        "validation": {**validation_result, "candidates": candidates},
        "test": {"per_mask_seed": test_results, "pooled": pooled},
    }

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        selected.model.save_model(artifact_dir / "correction_model.json")
        joblib.dump(
            {
                "scaler": adapter.scaler,
                "pca": adapter.pca,
                "neighbors": neighbors,
                "train_y": train_y,
                "train_analog": train_analog,
                "feature_names": feature_names,
                "selected": {
                    "target_kind": selected.target_kind,
                    "configuration": selected.configuration,
                    "blend": selected.blend,
                    "scale": selected.scale,
                },
            },
            artifact_dir / "retrieval_adapter.joblib",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return result
