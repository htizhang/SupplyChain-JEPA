from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluations.demand_recovery import benchmark, contract, correction, representation


def _tiny_data(masked_target_value: float) -> benchmark.RetailData:
    days = 40
    hours = 16
    sales = np.zeros((1, days, hours), dtype=np.float32)
    for day in range(days):
        sales[0, day] = np.arange(hours, dtype=np.float32) + (day % 7)
    sales[0, 35, 4:8] = masked_target_value
    stock = np.zeros_like(sales, dtype=np.uint8)
    covariates = np.zeros(
        (1, days, len(benchmark.COVARIATE_COLUMNS)), dtype=np.float32
    )
    static = np.asarray([[1, 10, 20, 30, 40, 50, 60]], dtype=np.int32)
    dates = np.datetime64("2024-01-01") + np.arange(days)
    return benchmark.RetailData(
        sales=sales,
        stock=stock,
        covariates=covariates,
        daily_total=sales.sum(axis=2),
        static=static,
        dates=dates,
        train_day_count=35,
    )


def _tiny_examples() -> benchmark.ExampleSet:
    mask = np.zeros((1, 16), dtype=bool)
    mask[:, 4:8] = True
    return benchmark.ExampleSet(
        name="synthetic",
        series=np.asarray([0], dtype=np.int64),
        day=np.asarray([35], dtype=np.int64),
        mask=mask,
        mask_seed=11,
        rank=np.asarray([0], dtype=np.int32),
    )


def _tiny_priors(data: benchmark.RetailData) -> benchmark.Priors:
    profile = np.nanmedian(data.sales[:, :28], axis=(0, 1)).astype(np.float32)
    return benchmark.Priors(
        product_values=profile[None, :],
        product_lookup={60: 0},
        category_values=profile[None, :],
        category_lookup={50: 0},
        global_values=profile,
        series_demand_mean=np.asarray([float(profile.mean())], dtype=np.float32),
        demand_group=np.asarray([1], dtype=np.int8),
        covariate_mean=np.zeros(len(benchmark.COVARIATE_COLUMNS), dtype=np.float32),
        covariate_std=np.ones(len(benchmark.COVARIATE_COLUMNS), dtype=np.float32),
    )


def test_analog_day_does_not_read_masked_target_values() -> None:
    examples = _tiny_examples()
    low = _tiny_data(masked_target_value=1.0)
    high = _tiny_data(masked_target_value=10_000.0)

    low_prediction = benchmark.predict_analog_day(
        low, examples, _tiny_priors(low), k=7
    )
    high_prediction = benchmark.predict_analog_day(
        high, examples, _tiny_priors(high), k=7
    )

    np.testing.assert_allclose(
        low_prediction[examples.mask], high_prediction[examples.mask]
    )


def test_controlled_masks_are_deterministic_and_nonempty() -> None:
    codes = np.asarray([1, 3, 12], dtype=np.uint32)
    probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    first = benchmark._sample_masks(100, codes, probabilities, seed=29)
    second = benchmark._sample_masks(100, codes, probabilities, seed=29)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (100, 16)
    assert np.all(first.any(axis=1))


def test_interval_metrics_use_only_controlled_mask() -> None:
    examples = _tiny_examples()
    full_prediction = np.zeros((1, 16), dtype=np.float32)
    full_prediction[0, 4:8] = [1.0, 2.0, 3.0, 4.0]
    full_prediction[0, :4] = 10_000.0

    interval = correction.interval_prediction(full_prediction, examples)
    np.testing.assert_allclose(interval, [10.0])
    metrics = correction.interval_metrics(
        np.asarray([12.0], dtype=np.float32), interval
    )
    assert metrics["wape"] == pytest.approx(2.0 / 12.0)
    assert metrics["wpe"] == pytest.approx(-2.0 / 12.0)


def test_blend_selection_can_preserve_the_analog_anchor() -> None:
    target = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    analog = target.copy()
    poor_correction = np.asarray([100.0, 100.0, 100.0], dtype=np.float32)

    prediction, settings = correction._tune_blend(
        target, analog, poor_correction
    )

    np.testing.assert_allclose(prediction, target)
    assert settings["blend"] == 0.0


def test_dataset_hash_contract_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"train.parquet": b"train", "eval.parquet": b"eval"}
    expected = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in files.items()
    }
    monkeypatch.setattr(contract, "DATASET_FILES", expected)
    for name, payload in files.items():
        (tmp_path / name).write_bytes(payload)

    assert contract.verify_dataset(tmp_path) == expected
    (tmp_path / "eval.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        contract.verify_dataset(tmp_path)


def test_reference_result_has_correct_direction_and_all_seed_wins() -> None:
    result = contract.REFERENCE_RESULT
    assert result["baseline_wape"] == pytest.approx(0.41701661785209065)
    assert result["jepa_system_wape"] == pytest.approx(0.3978750022662813)
    assert result["relative_wape_improvement"] == pytest.approx(
        0.04590132566994868
    )
    assert result["jepa_system_wape"] < result["baseline_wape"]
    assert result["winning_test_seeds"] == 3


def test_model_bundle_adapter_prefers_raw_tensorizer_config() -> None:
    class FakeParameter:
        def requires_grad_(self, value: bool) -> None:
            assert value is False

    class FakeModule:
        def eval(self) -> FakeModule:
            return self

        def parameters(self) -> list[FakeParameter]:
            return [FakeParameter()]

    raw = {
        "feature_names": ["entity::sales_observed"],
        "numeric_mean": [1.0],
        "numeric_std": [2.0],
        "categorical_vocabs": {},
        "metadata_vocabs": {
            "family": {"m5": 2},
            "token_kind": {"entity": 2},
            "token_type": {"sku_warehouse_state": 2},
        },
    }
    core = SimpleNamespace(
        online_encoder=FakeModule(), config=SimpleNamespace(d_model=8)
    )
    released = SimpleNamespace(
        model=core,
        tensorizer=object(),  # must not be passed to from_payload
        tensorizer_config=raw,
        state_pool_predictor=None,
        config={"state_pool_predictor": {"enabled": False}},
    )

    bundle = representation.representation_from_model_bundle(
        released, model_id="example/model", revision="a" * 40, device="cpu"
    )

    assert bundle.d_model == 8
    assert bundle.tensorizer_state.feature_names == (
        "entity::sales_observed",
    )
    np.testing.assert_allclose(bundle.tensorizer_state.numeric_mean, [1.0])
