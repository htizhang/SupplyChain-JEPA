"""Canonical FreshRetailNet-50K data construction and analog-day baseline.

The data remain external. This module preserves the controlled-mask protocol and
store-level out-of-distribution split used by the reference evaluation.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_ID = "freshretail-demand-recovery-v1"
# Compatibility salt: changing it changes the canonical store partition.
SPLIT_SEED_SALT = bytes.fromhex("66726e35306b2d7374616765312d7631")
RANDOM_SEED = 20260727
OPERATIONAL_HOURS = np.arange(6, 22, dtype=np.int16)
HISTORY_DAYS = 28
TRAIN_TARGET_START = np.datetime64("2024-04-25")
TRAIN_TARGET_END = np.datetime64("2024-06-11")
VALIDATION_START = np.datetime64("2024-06-12")
VALIDATION_END = np.datetime64("2024-06-25")
TEST_START = np.datetime64("2024-06-26")
TEST_END = np.datetime64("2024-07-02")
TRAIN_MASK_SEED = 101
VALIDATION_MASK_SEED = 29
TEST_MASK_SEEDS = (11, 29, 47)
MAX_TRAIN_DAYS_PER_SERIES = 2
ID_COLUMNS = (
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id",
)
COVARIATE_COLUMNS = (
    "discount", "holiday_flag", "activity_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
)


def _stable_store_order(city_id: int, store_id: int) -> str:
    payload = SPLIT_SEED_SALT + f":{city_id}:{store_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Fast deterministic pseudorandom scores independent of target values."""
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64) + np.uint64(seed) + np.uint64(
            0x9E3779B97F4A7C15
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


@dataclass
class RetailData:
    sales: np.ndarray  # [series, day, 16], float32
    stock: np.ndarray  # [series, day, 16], uint8; 1 means unavailable
    covariates: np.ndarray  # [series, day, 7], float32
    daily_total: np.ndarray  # [series, day], float32; evaluator only
    static: np.ndarray  # [series, 7], int32
    dates: np.ndarray  # [day], datetime64[D]
    train_day_count: int

    @property
    def n_series(self) -> int:
        return int(self.sales.shape[0])


@dataclass
class ExampleSet:
    name: str
    series: np.ndarray
    day: np.ndarray
    mask: np.ndarray
    mask_seed: int
    rank: np.ndarray

    def subset(self, n: int) -> ExampleSet:
        n = min(n, len(self.series))
        return ExampleSet(
            name=self.name,
            series=self.series[:n],
            day=self.day[:n],
            mask=self.mask[:n],
            mask_seed=self.mask_seed,
            rank=self.rank[:n],
        )

    def __len__(self) -> int:
        return len(self.series)


@dataclass
class Priors:
    product_values: np.ndarray
    product_lookup: dict[int, int]
    category_values: np.ndarray
    category_lookup: dict[int, int]
    global_values: np.ndarray
    series_demand_mean: np.ndarray
    demand_group: np.ndarray
    covariate_mean: np.ndarray
    covariate_std: np.ndarray


def _record_batch_column(batch: Any, name: str) -> Any:
    return batch.column(batch.schema.get_field_index(name))


def _load_parquet_split(path: Path, days: int) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    n_rows = parquet.metadata.num_rows
    if n_rows % days:
        raise ValueError(f"{path}: {n_rows} rows is not divisible by {days}")
    n_series = n_rows // days

    sales = np.empty((n_series, days, 16), dtype=np.float32)
    stock = np.empty((n_series, days, 16), dtype=np.uint8)
    covariates = np.empty(
        (n_series, days, len(COVARIATE_COLUMNS)), dtype=np.float32
    )
    daily_total = np.empty((n_series, days), dtype=np.float32)
    ids_flat = np.empty((n_rows, len(ID_COLUMNS)), dtype=np.int32)

    sales_flat = sales.reshape(n_rows, 16)
    stock_flat = stock.reshape(n_rows, 16)
    cov_flat = covariates.reshape(n_rows, len(COVARIATE_COLUMNS))
    total_flat = daily_total.reshape(n_rows)
    columns = [
        *ID_COLUMNS,
        "sale_amount",
        "hours_sale",
        "hours_stock_status",
        *COVARIATE_COLUMNS,
    ]

    offset = 0
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        size = len(batch)
        end = offset + size
        sale_lists = _record_batch_column(batch, "hours_sale")
        stock_lists = _record_batch_column(batch, "hours_stock_status")
        sale_values = np.asarray(
            sale_lists.values.to_numpy(zero_copy_only=False), dtype=np.float32
        ).reshape(size, 24)
        stock_values = np.asarray(
            stock_lists.values.to_numpy(zero_copy_only=False), dtype=np.uint8
        ).reshape(size, 24)
        sales_flat[offset:end] = sale_values[:, 6:22]
        stock_flat[offset:end] = stock_values[:, 6:22]
        total_flat[offset:end] = np.asarray(
            _record_batch_column(batch, "sale_amount").to_numpy(
                zero_copy_only=False
            ),
            dtype=np.float32,
        )
        for index, name in enumerate(ID_COLUMNS):
            ids_flat[offset:end, index] = np.asarray(
                _record_batch_column(batch, name).to_numpy(
                    zero_copy_only=False
                ),
                dtype=np.int32,
            )
        for index, name in enumerate(COVARIATE_COLUMNS):
            cov_flat[offset:end, index] = np.asarray(
                _record_batch_column(batch, name).to_numpy(
                    zero_copy_only=False
                ),
                dtype=np.float32,
            )
        offset = end
    if offset != n_rows:
        raise RuntimeError(f"Read {offset} rows from {path}, expected {n_rows}")

    ids_3d = ids_flat.reshape(n_series, days, len(ID_COLUMNS))
    for column_index, name in enumerate(ID_COLUMNS):
        values = ids_3d[:, :, column_index]
        if np.any(values.min(axis=1) != values.max(axis=1)):
            raise ValueError(f"{name} is not constant within a series in {path}")
    static = ids_3d[:, 0, :].copy()
    del ids_flat, ids_3d

    date_table = pq.read_table(path, columns=["dt"]).slice(0, days)
    dates = np.asarray(
        date_table.column("dt").to_pylist(), dtype="datetime64[D]"
    )
    if len(np.unique(dates)) != days:
        raise ValueError(f"First series in {path} does not contain {days} dates")
    return sales, stock, covariates, daily_total, static, dates


def load_data(root: Path) -> RetailData:
    dataset_dir = root
    train = _load_parquet_split(dataset_dir / "train.parquet", 90)
    evaluation = _load_parquet_split(dataset_dir / "eval.parquet", 7)

    (
        train_sales,
        train_stock,
        train_cov,
        train_total,
        train_static,
        train_dates,
    ) = train
    (
        eval_sales,
        eval_stock,
        eval_cov,
        eval_total,
        eval_static,
        eval_dates,
    ) = evaluation

    if not np.array_equal(train_static[:, [1, 6]], eval_static[:, [1, 6]]):
        train_keys = {
            (int(row[1]), int(row[6])): index
            for index, row in enumerate(train_static)
        }
        order = np.asarray(
            [train_keys[(int(row[1]), int(row[6]))] for row in eval_static],
            dtype=np.int64,
        )
        inverse = np.argsort(order)
        eval_sales = eval_sales[inverse]
        eval_stock = eval_stock[inverse]
        eval_cov = eval_cov[inverse]
        eval_total = eval_total[inverse]
        eval_static = eval_static[inverse]
    if not np.array_equal(train_static, eval_static):
        raise ValueError("Train/eval static entity metadata do not align")

    return RetailData(
        sales=np.concatenate([train_sales, eval_sales], axis=1),
        stock=np.concatenate([train_stock, eval_stock], axis=1),
        covariates=np.concatenate([train_cov, eval_cov], axis=1),
        daily_total=np.concatenate([train_total, eval_total], axis=1),
        static=train_static,
        dates=np.concatenate([train_dates, eval_dates]),
        train_day_count=90,
    )


def build_store_split(data: RetailData) -> tuple[np.ndarray, dict[str, Any]]:
    city_index = ID_COLUMNS.index("city_id")
    store_index = ID_COLUMNS.index("store_id")
    product_index = ID_COLUMNS.index("product_id")
    store_city: dict[int, int] = {}
    store_products: dict[int, set[int]] = {}
    for row in data.static:
        city = int(row[city_index])
        store = int(row[store_index])
        product = int(row[product_index])
        store_city[store] = city
        store_products.setdefault(store, set()).add(product)

    city_stores: dict[int, list[int]] = {}
    for store, city in store_city.items():
        city_stores.setdefault(city, []).append(store)

    assignments: dict[str, list[int]] = {
        "adaptation": [],
        "validation": [],
        "test": [],
    }
    city_assignments: dict[str, dict[str, list[int]]] = {}
    for city, stores in sorted(city_stores.items()):
        ordered = sorted(stores, key=lambda store: _stable_store_order(city, store))
        n = len(ordered)
        n_validation = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        while n - n_validation - n_test < 1:
            n_validation -= 1
        test_stores = ordered[:n_test]
        validation_stores = ordered[n_test : n_test + n_validation]
        adaptation_stores = ordered[n_test + n_validation :]
        assignments["adaptation"].extend(adaptation_stores)
        assignments["validation"].extend(validation_stores)
        assignments["test"].extend(test_stores)
        city_assignments[str(city)] = {
            "adaptation": adaptation_stores,
            "validation": validation_stores,
            "test": test_stores,
        }

    store_to_code = {}
    for code, name in enumerate(("adaptation", "validation", "test")):
        for store in assignments[name]:
            store_to_code[store] = code
    series_split = np.asarray(
        [store_to_code[int(store)] for store in data.static[:, store_index]],
        dtype=np.int8,
    )

    adaptation_products = set()
    for store in assignments["adaptation"]:
        adaptation_products.update(store_products[store])
    statistics = {}
    for name, code in (("adaptation", 0), ("validation", 1), ("test", 2)):
        series = np.flatnonzero(series_split == code)
        products = set(map(int, data.static[series, product_index]))
        statistics[name] = {
            "stores": len(assignments[name]),
            "series": len(series),
            "products": len(products),
            "cities": len(set(map(int, data.static[series, city_index]))),
            "products_seen_in_adaptation": len(products & adaptation_products),
            "products_unseen_in_adaptation": len(products - adaptation_products),
        }

    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "version": 1,
        "unit": "store_id",
        "stratification": "within city_id",
        "seed": hashlib.sha256(SPLIT_SEED_SALT).hexdigest(),
        "algorithm": (
            "Within city, sort stores by SHA256(seed:city_id:store_id); "
            "allocate approximately 70/15/15 percent to adaptation, "
            "validation, and test."
        ),
        "assignments": {
            key: sorted(values) for key, values in assignments.items()
        },
        "city_assignments": city_assignments,
        "statistics": statistics,
    }
    return series_split, manifest


def build_mask_bank(
    data: RetailData, series_split: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_days = np.flatnonzero(data.dates <= TRAIN_TARGET_END)
    source_series = np.flatnonzero(series_split == 0)
    powers = (np.uint32(1) << np.arange(16, dtype=np.uint32))[None, None, :]
    stock = data.stock[source_series][:, source_days]
    codes = np.sum(stock.astype(np.uint32) * powers, axis=2).reshape(-1)
    codes = codes[codes > 0]
    unique, counts = np.unique(codes, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]
    probabilities = probabilities[order]
    lengths = np.asarray(
        [int(int(code).bit_count()) for code in unique], dtype=np.int16
    )
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "version": 1,
        "source_partition": "adaptation stores",
        "source_date_max": str(TRAIN_TARGET_END),
        "operational_hours": OPERATIONAL_HOURS.tolist(),
        "eligible_day_rows": int(len(source_series) * len(source_days)),
        "stockout_day_rows": int(len(codes)),
        "stockout_day_rate": float(
            len(codes) / (len(source_series) * len(source_days))
        ),
        "unique_patterns": int(len(unique)),
        "mean_masked_hours": float(np.average(lengths, weights=counts)),
        "patterns": [
            {
                "code": int(code),
                "pattern_06_to_21": "".join(
                    str((int(code) >> index) & 1) for index in range(16)
                ),
                "count": int(count),
                "probability": float(probability),
                "masked_hours": int(length),
            }
            for code, count, probability, length in zip(
                unique, counts, probabilities, lengths, strict=True
            )
        ],
    }
    return unique.astype(np.uint32), probabilities, manifest


def _sample_masks(
    n: int, codes: np.ndarray, probabilities: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled_codes = rng.choice(codes, size=n, p=probabilities)
    return (
        (sampled_codes[:, None] >> np.arange(16, dtype=np.uint32)) & 1
    ).astype(bool)


def _candidate_examples(
    data: RetailData,
    series_split: np.ndarray,
    split_code: int,
    start: np.datetime64,
    end: np.datetime64,
) -> tuple[np.ndarray, np.ndarray]:
    series = np.flatnonzero(series_split == split_code)
    days = np.flatnonzero((data.dates >= start) & (data.dates <= end))
    clean = data.stock[series][:, days].sum(axis=2) == 0
    local_series, local_day = np.nonzero(clean)
    return series[local_series], days[local_day]


def _ordered_pool(
    series: np.ndarray,
    day: np.ndarray,
    max_examples: int | None,
    max_per_series: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = (
        series.astype(np.uint64) * np.uint64(1_000_003)
        + day.astype(np.uint64)
    )
    order = np.argsort(_splitmix64(keys, seed), kind="stable")
    if max_per_series is None:
        selected = order
    else:
        counts: dict[int, int] = {}
        keep = []
        for index in order:
            entity = int(series[index])
            if counts.get(entity, 0) >= max_per_series:
                continue
            counts[entity] = counts.get(entity, 0) + 1
            keep.append(int(index))
            if max_examples is not None and len(keep) >= max_examples:
                break
        selected = np.asarray(keep, dtype=np.int64)
    if max_examples is not None:
        selected = selected[:max_examples]
    return series[selected], day[selected]


def build_example_sets(
    data: RetailData,
    series_split: np.ndarray,
    codes: np.ndarray,
    probabilities: np.ndarray,
    train_pool: int,
    validation_pool: int,
    test_cap: int | None,
) -> tuple[ExampleSet, ExampleSet, list[ExampleSet], dict[str, Any]]:
    train_series, train_day = _candidate_examples(
        data, series_split, 0, TRAIN_TARGET_START, TRAIN_TARGET_END
    )
    val_series, val_day = _candidate_examples(
        data, series_split, 1, VALIDATION_START, VALIDATION_END
    )
    test_series, test_day = _candidate_examples(
        data, series_split, 2, TEST_START, TEST_END
    )

    eligible_counts = {
        "adaptation": int(len(train_series)),
        "validation": int(len(val_series)),
        "test": int(len(test_series)),
    }
    train_series, train_day = _ordered_pool(
        train_series,
        train_day,
        train_pool,
        MAX_TRAIN_DAYS_PER_SERIES,
        RANDOM_SEED + 1,
    )
    val_series, val_day = _ordered_pool(
        val_series, val_day, validation_pool, None, RANDOM_SEED + 2
    )
    if test_cap:
        test_series, test_day = _ordered_pool(
            test_series, test_day, test_cap, None, RANDOM_SEED + 3
        )

    train = ExampleSet(
        "adaptation",
        train_series,
        train_day,
        _sample_masks(
            len(train_series), codes, probabilities, TRAIN_MASK_SEED
        ),
        TRAIN_MASK_SEED,
        np.arange(len(train_series), dtype=np.int32),
    )
    validation = ExampleSet(
        "validation",
        val_series,
        val_day,
        _sample_masks(
            len(val_series), codes, probabilities, VALIDATION_MASK_SEED
        ),
        VALIDATION_MASK_SEED,
        np.arange(len(val_series), dtype=np.int32),
    )
    test_sets = [
        ExampleSet(
            f"test_seed_{seed}",
            test_series,
            test_day,
            _sample_masks(len(test_series), codes, probabilities, seed),
            seed,
            np.arange(len(test_series), dtype=np.int32),
        )
        for seed in TEST_MASK_SEEDS
    ]
    summary = {
        "eligible_clean_target_days": eligible_counts,
        "selected_examples": {
            "adaptation_pool": len(train),
            "validation": len(validation),
            "test_per_seed": len(test_series),
        },
        "max_adaptation_days_per_series": MAX_TRAIN_DAYS_PER_SERIES,
        "mask_seeds": {
            "adaptation": TRAIN_MASK_SEED,
            "validation": VALIDATION_MASK_SEED,
            "test": list(TEST_MASK_SEEDS),
        },
    }
    return train, validation, test_sets, summary


def build_priors(
    data: RetailData, series_split: np.ndarray
) -> Priors:
    product_col = ID_COLUMNS.index("product_id")
    category_col = ID_COLUMNS.index("third_category_id")
    source_series = np.flatnonzero(series_split == 0)
    source_days = np.flatnonzero(data.dates <= TRAIN_TARGET_END)
    products = sorted(set(map(int, data.static[source_series, product_col])))
    categories = sorted(
        set(map(int, data.static[source_series, category_col]))
    )
    product_lookup = {value: index for index, value in enumerate(products)}
    category_lookup = {value: index for index, value in enumerate(categories)}
    product_sum = np.zeros((len(products), 16), dtype=np.float64)
    product_count = np.zeros((len(products), 16), dtype=np.int64)
    category_sum = np.zeros((len(categories), 16), dtype=np.float64)
    category_count = np.zeros((len(categories), 16), dtype=np.int64)
    global_sum = np.zeros(16, dtype=np.float64)
    global_count = np.zeros(16, dtype=np.int64)

    cov_source = data.covariates[source_series][:, source_days]
    cov_mean = cov_source.reshape(-1, len(COVARIATE_COLUMNS)).mean(axis=0)
    cov_std = cov_source.reshape(-1, len(COVARIATE_COLUMNS)).std(axis=0)
    cov_std = np.where(cov_std > 1e-6, cov_std, 1.0).astype(np.float32)
    del cov_source

    series_sum = np.zeros(data.n_series, dtype=np.float64)
    series_count = np.zeros(data.n_series, dtype=np.int64)
    for start in range(0, data.n_series, 2_000):
        stop = min(start + 2_000, data.n_series)
        sales = data.sales[start:stop, : data.train_day_count]
        stock = data.stock[start:stop, : data.train_day_count]
        observed = np.where(stock == 0, sales, np.nan)
        series_sum[start:stop] = np.nansum(observed, axis=(1, 2))
        series_count[start:stop] = np.sum(~np.isnan(observed), axis=(1, 2))

    for start in range(0, len(source_series), 2_000):
        selected = source_series[start : start + 2_000]
        observed = np.where(
            data.stock[selected][:, source_days] == 0,
            data.sales[selected][:, source_days],
            np.nan,
        )
        sums = np.nansum(observed, axis=1)
        counts = np.sum(~np.isnan(observed), axis=1)
        for row_index, series_index in enumerate(selected):
            product = int(data.static[series_index, product_col])
            category = int(data.static[series_index, category_col])
            pidx = product_lookup[product]
            cidx = category_lookup[category]
            product_sum[pidx] += sums[row_index]
            product_count[pidx] += counts[row_index]
            category_sum[cidx] += sums[row_index]
            category_count[cidx] += counts[row_index]
        global_sum += np.nansum(observed, axis=(0, 1))
        global_count += np.sum(~np.isnan(observed), axis=(0, 1))

    global_values = global_sum / np.maximum(global_count, 1)
    product_values = product_sum / np.maximum(product_count, 1)
    category_values = category_sum / np.maximum(category_count, 1)
    product_values = np.where(
        product_count > 0, product_values, global_values[None, :]
    )
    category_values = np.where(
        category_count > 0, category_values, global_values[None, :]
    )
    series_mean = series_sum / np.maximum(series_count, 1)
    quantiles = np.quantile(series_mean[source_series], [0.33, 0.67])
    demand_group = np.digitize(series_mean, quantiles).astype(np.int8)
    return Priors(
        product_values.astype(np.float32),
        product_lookup,
        category_values.astype(np.float32),
        category_lookup,
        global_values.astype(np.float32),
        series_mean.astype(np.float32),
        demand_group,
        cov_mean.astype(np.float32),
        cov_std,
    )


def _history_arrays(
    data: RetailData, examples: ExampleSet
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(-HISTORY_DAYS, 0, dtype=np.int16)
    days = examples.day[:, None] + offsets[None, :]
    history_sales = data.sales[examples.series[:, None], days]
    history_available = data.stock[examples.series[:, None], days] == 0
    history = np.where(history_available, history_sales, np.nan)
    history_covariates = data.covariates[examples.series[:, None], days]
    return history, history_available, history_covariates


def _fallback_profile(
    data: RetailData, priors: Priors, series: np.ndarray
) -> np.ndarray:
    product_col = ID_COLUMNS.index("product_id")
    category_col = ID_COLUMNS.index("third_category_id")
    result = np.empty((len(series), 16), dtype=np.float32)
    for row, series_index in enumerate(series):
        product = int(data.static[series_index, product_col])
        category = int(data.static[series_index, category_col])
        if product in priors.product_lookup:
            result[row] = priors.product_values[
                priors.product_lookup[product]
            ]
        elif category in priors.category_lookup:
            result[row] = priors.category_values[
                priors.category_lookup[category]
            ]
        else:
            result[row] = priors.global_values
    return result


def predict_historical_profile(
    data: RetailData, examples: ExampleSet, priors: Priors
) -> np.ndarray:
    predictions = np.empty((len(examples), 16), dtype=np.float32)
    for start in range(0, len(examples), 4_000):
        stop = min(start + 4_000, len(examples))
        chunk = ExampleSet(
            examples.name,
            examples.series[start:stop],
            examples.day[start:stop],
            examples.mask[start:stop],
            examples.mask_seed,
            examples.rank[start:stop],
        )
        history, _, _ = _history_arrays(data, chunk)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            profile = np.nanmedian(history, axis=1)
        fallback = _fallback_profile(data, priors, chunk.series)
        predictions[start:stop] = np.where(
            np.isnan(profile), fallback, profile
        )
    return np.clip(predictions, 0, None)


def predict_analog_day(
    data: RetailData, examples: ExampleSet, priors: Priors, k: int = 7
) -> np.ndarray:
    predictions = np.empty((len(examples), 16), dtype=np.float32)
    date_dow = (
        data.dates.astype("datetime64[D]").astype(np.int64) + 3
    ) % 7
    for start in range(0, len(examples), 1_000):
        stop = min(start + 1_000, len(examples))
        chunk = ExampleSet(
            examples.name,
            examples.series[start:stop],
            examples.day[start:stop],
            examples.mask[start:stop],
            examples.mask_seed,
            examples.rank[start:stop],
        )
        history, history_available, history_cov = _history_arrays(data, chunk)
        target_sales = data.sales[chunk.series, chunk.day]
        target_cov = data.covariates[chunk.series, chunk.day]
        visible = ~chunk.mask
        scale = np.maximum(
            np.nanmean(np.abs(history), axis=(1, 2)), 0.02
        ).astype(np.float32)
        common = history_available & visible[:, None, :]
        sales_difference = np.where(
            common,
            np.abs(history - target_sales[:, None, :])
            / scale[:, None, None],
            0.0,
        )
        common_count = common.sum(axis=2)
        sales_distance = sales_difference.sum(axis=2) / np.maximum(
            common_count, 1
        )
        visible_count = np.maximum(visible.sum(axis=1), 1)
        missing_penalty = (
            visible_count[:, None] - common_count
        ) / visible_count[:, None]
        cov_distance = np.mean(
            np.abs(
                (history_cov - target_cov[:, None, :])
                / priors.covariate_std[None, None, :]
            ),
            axis=2,
        )
        history_days = (
            chunk.day[:, None]
            + np.arange(-HISTORY_DAYS, 0, dtype=np.int16)[None, :]
        )
        dow_penalty = (
            date_dow[history_days] != date_dow[chunk.day][:, None]
        ).astype(np.float32)
        distance = (
            sales_distance
            + 0.75 * missing_penalty
            + 0.20 * cov_distance
            + 0.15 * dow_penalty
        )
        nearest = np.argpartition(
            distance, kth=min(k - 1, HISTORY_DAYS - 1), axis=1
        )[:, :k]
        nearest_distance = np.take_along_axis(distance, nearest, axis=1)
        nearest_sales = np.take_along_axis(
            history, nearest[:, :, None], axis=1
        )
        valid = ~np.isnan(nearest_sales)
        weights = np.exp(
            -(nearest_distance - nearest_distance.min(axis=1, keepdims=True))
        )[:, :, None]
        weighted_sum = np.nansum(nearest_sales * weights, axis=1)
        weight_sum = np.sum(weights * valid, axis=1)
        analog = weighted_sum / np.maximum(weight_sum, 1e-8)
        fallback = predict_historical_profile(data, chunk, priors)
        predictions[start:stop] = np.where(
            weight_sum > 0, analog, fallback
        )
    return np.clip(predictions, 0, None)
