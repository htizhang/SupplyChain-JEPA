"""Canonical delayed-reporting SLA-risk evaluation.

The operational baseline and every encoder see the same corrupted Cargo2000
prefixes.  The pretrained State encoder contributes only a low-capacity
transition prototype; matched random encoders retain the architecture,
adapter, label budget, and readout while replacing the pretrained weights.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from supplychain_jepa import DEFAULT_MODEL_ID, DEFAULT_REVISION, ModelBundle, load_pretrained
from supplychain_jepa.model import SequenceEncoder

from .model_adapter import DurationScaler, cache_key, encode_current_state
from .protocol import (
    DELTA_FROM_CHECKPOINT,
    PRIMARY_CHECKPOINT,
    add_targets,
    apply_critical_prefix_delay,
    build_prefix_examples,
    deterministic_rank_fraction,
    feature_columns,
    load_cargo_csv,
    rank_matched_probabilities,
    subset,
)

DATASET_SHA256 = "da40666b12f4a5465a885f3df91d67cd7a1f20946a0ef07e62e9d35b3687984a"
SPLIT_MANIFEST_SHA256 = "47616c4806b101d51624abe0b2e07f8e9ea7cdabea3373d49a5aea9c0ac671de"
MODEL_REVISION = DEFAULT_REVISION
METHOD = "critical-prefix-delayed-reporting"
ALPHA = 0.20
RANDOM_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)
EXPECTED_COUNTS = {
    "source": 1826,
    "target_fit": 47,
    "calibration": 17,
    "development": 230,
    "confirmation": 126,
    "adjacent": 1696,
}
REFERENCE = {
    "development": {
        "baseline": 0.3947828003137619,
        "pretrained": 0.4305829408024346,
        "random_mean": 0.4132930503977935,
    },
    "confirmation": {
        "baseline": 0.3852873947334345,
        "pretrained": 0.41378142555634717,
        "random_mean": 0.39625315905718694,
    },
    "adjacent": {
        "baseline": 0.499222,
        "pretrained": 0.477540,
        "random_mean": 0.479024,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_data_file(root: Path, names: Sequence[str], expected_sha: str) -> Path:
    root = root.expanduser().resolve()
    candidates = [root / name for name in names]
    candidates.extend(root / "sources" / "dataset" / name for name in names)
    candidates.extend(root / "results" / name for name in names)
    for path in candidates:
        if path.is_file():
            actual = sha256_file(path)
            if actual != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {path.name}: {actual} != {expected_sha}")
            return path
    # A data provider may retain its own filename. The content hash, not that
    # filename, is the contract, so inspect only the shallow external tree.
    for path in sorted(root.glob("*/*")):
        if path.is_file() and sha256_file(path) == expected_sha:
            return path
    raise FileNotFoundError(
        f"Could not find any of {list(names)} below {root}; external data is not bundled"
    )


def _verify_external_file(path: Path, expected_sha: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual} != {expected_sha}")
    return path


@dataclass(frozen=True)
class Cohorts:
    source: frozenset[int]
    target_fit: frozenset[int]
    calibration: frozenset[int]
    development: frozenset[int]
    confirmation: frozenset[int]
    adjacent: frozenset[int]

    def evaluation(self, name: str) -> frozenset[int]:
        if name not in {"development", "confirmation", "adjacent"}:
            raise ValueError(f"Unknown evaluation cohort: {name}")
        return getattr(self, name)


def load_cohorts(manifest_path: Path) -> Cohorts:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "budgets",
        "source_process_ids",
        "target_development_process_ids",
        "target_final_test_process_ids",
        "quarantine_process_ids",
    }
    if not required.issubset(manifest):
        raise ValueError("Cargo2000 split manifest is missing required cohorts")
    budget = manifest["budgets"]["64"]
    labeled = frozenset(map(int, budget["target_labeled_process_ids"]))
    cohorts = Cohorts(
        source=frozenset(map(int, manifest["source_process_ids"])),
        target_fit=frozenset(map(int, budget["target_fit_process_ids"])),
        calibration=frozenset(map(int, budget["target_calibration_process_ids"])),
        development=frozenset(map(int, manifest["target_development_process_ids"])) - labeled,
        confirmation=frozenset(map(int, manifest["target_final_test_process_ids"])),
        adjacent=frozenset(map(int, manifest["quarantine_process_ids"])),
    )
    for name, expected in EXPECTED_COUNTS.items():
        actual = len(getattr(cohorts, name))
        if actual != expected:
            raise ValueError(f"{name} cohort has {actual} processes; expected {expected}")
    return cohorts


def _domain_weights(process_ids: pd.Series, target_ids: frozenset[int]) -> np.ndarray:
    target = process_ids.isin(target_ids).to_numpy()
    source_count = int((~target).sum())
    target_count = int(target.sum())
    weights = np.where(target, 0.5 / target_count, 0.5 / source_count)
    return weights / weights.mean()


class PrefixCalibrator:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    @staticmethod
    def _frame(probabilities: np.ndarray, checkpoints: Sequence[str]) -> pd.DataFrame:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return pd.DataFrame(
            {
                "raw_logit": np.log(clipped / (1.0 - clipped)),
                "checkpoint": pd.Series(checkpoints, dtype="string"),
            }
        )

    def fit(
        self, probabilities: np.ndarray, checkpoints: Sequence[str], labels: np.ndarray
    ) -> PrefixCalibrator:
        preprocess = ColumnTransformer(
            [
                ("logit", "passthrough", ["raw_logit"]),
                (
                    "checkpoint",
                    OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                    ["checkpoint"],
                ),
            ],
            sparse_threshold=0.0,
        )
        self.pipeline = Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(C=10.0, solver="lbfgs", max_iter=2000, random_state=20260727),
                ),
            ]
        )
        self.pipeline.fit(self._frame(probabilities, checkpoints), labels)
        return self

    def predict(self, probabilities: np.ndarray, checkpoints: Sequence[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Calibrator has not been fitted")
        return self.pipeline.predict_proba(self._frame(probabilities, checkpoints))[:, 1]


@dataclass
class Baseline:
    pipeline: Pipeline
    calibrator: PrefixCalibrator

    def predict(self, examples: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(examples)[:, 1]
        return self.calibrator.predict(raw, examples["checkpoint"].astype(str).tolist())


def fit_baseline(examples: pd.DataFrame, cohorts: Cohorts) -> Baseline:
    fit = subset(examples, cohorts.source | cohorts.target_fit)
    calibration = subset(examples, cohorts.calibration)
    numeric, categorical = feature_columns(examples)
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                categorical,
            ),
        ],
        sparse_threshold=0.3,
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(C=1.0, solver="liblinear", max_iter=3000, random_state=20260727),
            ),
        ]
    )
    pipeline.fit(
        fit,
        fit["label"].to_numpy(dtype=int),
        model__sample_weight=_domain_weights(fit["process_id"], cohorts.target_fit),
    )
    raw_calibration = pipeline.predict_proba(calibration)[:, 1]
    calibrator = PrefixCalibrator().fit(
        raw_calibration,
        calibration["checkpoint"].astype(str).tolist(),
        calibration["label"].to_numpy(dtype=int),
    )
    return Baseline(pipeline, calibrator)


@dataclass(frozen=True)
class StageDelta:
    process_ids: np.ndarray
    labels: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class DeltaPrototype:
    mean: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    direction: np.ndarray

    def score(self, values: np.ndarray) -> np.ndarray:
        standardized = (values[:, self.keep] - self.mean[self.keep]) / self.scale[self.keep]
        return standardized @ self.direction


def stage_delta(examples: pd.DataFrame, latent: np.ndarray) -> StageDelta:
    if len(examples) != len(latent):
        raise ValueError("Example and latent row counts differ")
    ids: list[int] = []
    labels: list[int] = []
    transitions: list[np.ndarray] = []
    for process_id, process in examples.groupby("process_id", sort=True):
        rows = {str(row.checkpoint): int(index) for index, row in process.iterrows()}
        if DELTA_FROM_CHECKPOINT not in rows or PRIMARY_CHECKPOINT not in rows:
            raise ValueError(f"Process {process_id} lacks required checkpoints")
        process_labels = process["label"].to_numpy(dtype=int)
        if not np.all(process_labels == process_labels[0]):
            raise ValueError(f"Process {process_id} has inconsistent labels")
        ids.append(int(process_id))
        labels.append(int(process_labels[0]))
        transitions.append(latent[rows[PRIMARY_CHECKPOINT]] - latent[rows[DELTA_FROM_CHECKPOINT]])
    return StageDelta(np.asarray(ids), np.asarray(labels), np.asarray(transitions, np.float32))


def fit_prototype(delta: StageDelta, target_fit_ids: frozenset[int]) -> DeltaPrototype:
    mean = delta.values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = delta.values.std(axis=0, dtype=np.float64).astype(np.float32)
    keep = scale > 1e-5
    if not keep.any():
        raise ValueError("All State transition dimensions are constant")
    standardized = (delta.values[:, keep] - mean[keep]) / scale[keep]
    target = np.isin(delta.process_ids, sorted(target_fit_ids))
    positive = target & (delta.labels == 1)
    negative = target & (delta.labels == 0)
    if positive.sum() < 2 or negative.sum() < 2:
        raise ValueError("Target-fit prototype requires both classes")
    direction = standardized[positive].mean(axis=0) - standardized[negative].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8 or not np.isfinite(norm):
        raise ValueError("State transition prototype collapsed")
    return DeltaPrototype(mean, scale, keep, (direction / norm).astype(np.float32))


def blend_ranking(
    baseline: np.ndarray, evidence: np.ndarray, process_ids: np.ndarray, alpha: float = ALPHA
) -> np.ndarray:
    baseline_rank = deterministic_rank_fraction(baseline, process_ids)
    evidence_rank = deterministic_rank_fraction(evidence, process_ids)
    priority = (1.0 - alpha) * baseline_rank + alpha * evidence_rank
    return rank_matched_probabilities(baseline, priority, process_ids)


def _metric(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def aggregate_random_controls(per_seed: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if len(per_seed) != 10:
        raise ValueError("Canonical matched-random aggregation requires exactly 10 seeds")
    values = np.asarray([entry["auprc"] for entry in per_seed], dtype=np.float64)
    return {
        "seeds": [int(entry["seed"]) for entry in per_seed],
        "auprc_mean": float(values.mean()),
        "auprc_sample_sd": float(values.std(ddof=1)),
        "auprc_min": float(values.min()),
        "auprc_max": float(values.max()),
        "per_seed": [dict(entry) for entry in per_seed],
    }


def _stage_rows(examples: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    mask = examples["checkpoint"].astype(str).to_numpy() == PRIMARY_CHECKPOINT
    rows = examples.loc[mask, ["process_id", "label"]].copy()
    rows["baseline"] = probabilities[mask]
    return rows.sort_values("process_id").reset_index(drop=True)


def _encoder_for_seed(bundle: ModelBundle, seed: int) -> SequenceEncoder:
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    encoder = SequenceEncoder(bundle.model.config).to(next(bundle.model.parameters()).device).eval()
    torch.random.set_rng_state(state)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _cache_path(cache_dir: Path | None, label: str, examples: pd.DataFrame) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / label / f"{cache_key(label, examples)}.npy"


def evaluate(
    *,
    data_root: Path,
    split_manifest: Path | None = None,
    model_source: Path | None = None,
    device: str = "cpu",
    batch_size: int = 128,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    verify_model: bool = True,
) -> dict[str, Any]:
    """Run development, confirmation, and adjacent-cohort evaluation."""

    started = time.perf_counter()
    csv_path = _resolve_data_file(data_root, ("c2k_data_comma.csv",), DATASET_SHA256)
    manifest_path = (
        _resolve_data_file(data_root, ("split_manifest.json",), SPLIT_MANIFEST_SHA256)
        if split_manifest is None
        else _verify_external_file(split_manifest, SPLIT_MANIFEST_SHA256)
    )
    raw = add_targets(load_cargo_csv(csv_path))
    cohorts = load_cohorts(manifest_path)
    examples = apply_critical_prefix_delay(build_prefix_examples(raw)).reset_index(drop=True)
    baseline = fit_baseline(examples, cohorts)
    fit_examples = subset(examples, cohorts.source | cohorts.target_fit)
    evaluation_examples = {
        name: subset(examples, cohorts.evaluation(name))
        for name in ("development", "confirmation", "adjacent")
    }
    baseline_probabilities = {
        name: baseline.predict(frame) for name, frame in evaluation_examples.items()
    }

    bundle = load_pretrained(
        source=model_source,
        repo_id=DEFAULT_MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
        verify=verify_model,
        local_files_only=local_files_only,
    )
    if bundle.revision not in {None, MODEL_REVISION}:
        raise ValueError(f"Model revision drifted: {bundle.revision}")
    scaler = DurationScaler.fit(raw, set(cohorts.source))
    pretrained_fit = encode_current_state(
        fit_examples,
        bundle,
        scaler,
        batch_size=batch_size,
        cache_path=_cache_path(cache_dir, "pretrained-fit", fit_examples),
    )
    pretrained_prototype = fit_prototype(stage_delta(fit_examples, pretrained_fit), cohorts.target_fit)

    result_by_cohort: dict[str, Any] = {}
    pretrained_probabilities: dict[str, np.ndarray] = {}
    random_results: dict[str, list[dict[str, float]]] = {
        name: [] for name in evaluation_examples
    }
    for name, frame in evaluation_examples.items():
        latent = encode_current_state(
            frame,
            bundle,
            scaler,
            batch_size=batch_size,
            cache_path=_cache_path(cache_dir, f"pretrained-{name}", frame),
        )
        delta = stage_delta(frame, latent)
        stage = _stage_rows(frame, baseline_probabilities[name])
        if not np.array_equal(stage["process_id"].to_numpy(), delta.process_ids):
            raise AssertionError("Baseline and State encoder process ordering differ")
        pretrained_probabilities[name] = blend_ranking(
            stage["baseline"].to_numpy(), pretrained_prototype.score(delta.values), delta.process_ids
        )

    for seed in RANDOM_SEEDS:
        encoder = _encoder_for_seed(bundle, seed)
        label = f"random-{seed}"
        random_fit = encode_current_state(
            fit_examples,
            bundle,
            scaler,
            encoder=encoder,
            batch_size=batch_size,
            cache_path=_cache_path(cache_dir, f"{label}-fit", fit_examples),
        )
        prototype = fit_prototype(stage_delta(fit_examples, random_fit), cohorts.target_fit)
        for name, frame in evaluation_examples.items():
            latent = encode_current_state(
                frame,
                bundle,
                scaler,
                encoder=encoder,
                batch_size=batch_size,
                cache_path=_cache_path(cache_dir, f"{label}-{name}", frame),
            )
            delta = stage_delta(frame, latent)
            stage = _stage_rows(frame, baseline_probabilities[name])
            probability = blend_ranking(
                stage["baseline"].to_numpy(), prototype.score(delta.values), delta.process_ids
            )
            random_results[name].append(
                {"seed": seed, "auprc": float(average_precision_score(delta.labels, probability))}
            )
        del encoder, random_fit, prototype
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name, frame in evaluation_examples.items():
        stage = _stage_rows(frame, baseline_probabilities[name])
        labels = stage["label"].to_numpy(dtype=int)
        baseline_probability = stage["baseline"].to_numpy()
        random = aggregate_random_controls(random_results[name])
        pretrained = _metric(labels, pretrained_probabilities[name])
        baseline_metric = _metric(labels, baseline_probability)
        result_by_cohort[name] = {
            "processes": len(labels),
            "positives": int(labels.sum()),
            "baseline": baseline_metric,
            "pretrained": pretrained,
            "matched_random": random,
            "relative_auprc_change": {
                "pretrained_vs_baseline": pretrained["auprc"] / baseline_metric["auprc"] - 1.0,
                "pretrained_vs_random_mean": pretrained["auprc"] / random["auprc_mean"] - 1.0,
            },
            "reference_delta": {
                key: (
                    (baseline_metric["auprc"] if key == "baseline" else pretrained["auprc"])
                    if key != "random_mean"
                    else random["auprc_mean"]
                )
                - value
                for key, value in REFERENCE[name].items()
            },
        }

    adjacent = result_by_cohort["adjacent"]
    adjacent_negative = (
        adjacent["pretrained"]["auprc"] < adjacent["baseline"]["auprc"]
        and adjacent["matched_random"]["auprc_mean"] < adjacent["baseline"]["auprc"]
    )
    return {
        "method": METHOD,
        "model": {"repo_id": DEFAULT_MODEL_ID, "revision": MODEL_REVISION},
        "data": {
            "csv_sha256": DATASET_SHA256,
            "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
            "external_files_bundled": False,
        },
        "protocol": {
            "decision_checkpoint": PRIMARY_CHECKPOINT,
            "transition": f"{DELTA_FROM_CHECKPOINT}->{PRIMARY_CHECKPOINT}",
            "reporting_delay": "longest planned incoming leg: RCS then segment-1 DEP/RCF",
            "target_label_budget": 64,
            "rank_blend_alpha": ALPHA,
            "matched_random_seeds": list(RANDOM_SEEDS),
        },
        "cohorts": result_by_cohort,
        "adjacent_boundary_diagnostic": {
            "expected_direction": "negative",
            "observed_negative": adjacent_negative,
            "interpretation": (
                "The fixed State readout is not a general Cargo2000 transfer method; "
                "its positive signal is limited to delayed reporting under joint shift."
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Explicit path to split_manifest.json when it is stored outside --data-root.",
    )
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-model-verification",
        action="store_true",
        help="Skip release-file checksums; intended only for loader diagnostics.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate(
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        model_source=args.model_source,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        verify_model=not args.skip_model_verification,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
