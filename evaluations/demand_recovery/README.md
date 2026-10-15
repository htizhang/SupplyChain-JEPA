# Controlled-stockout demand recovery

This evaluation estimates demand hidden by stockout-shaped masks in
FreshRetailNet-50K. It preserves a store-level out-of-distribution split and
uses only 5% of the adaptation pool (3,000 examples) to fit downstream
components.

The system keeps `analog_day_knn` as the operational baseline, freezes the
pretrained SupplyChain-JEPA State encoder, retrieves labeled analogues in its
latent space, and fits a small XGBoost correction model. The Dynamics predictor
is not used.

## Data

Place the external files directly under a local data directory:

```text
<data-root>/train.parquet
<data-root>/eval.parquet
```

The CLI verifies both SHA-256 hashes before reading either file. The parquet
files are never copied into this repository.

## Run

```bash
python -m evaluations.demand_recovery \
  --data-root /path/to/freshretail-parquet \
  --cache-dir /path/to/cache \
  --output /path/to/demand-recovery.json
```

The default model revision is immutable and downloads anonymously from the
public Hugging Face model repository through the client used by `supplychain_jepa`.

## Fresh release result

| Metric | Analog baseline | JEPA system |
|---|---:|---:|
| Pooled test WAPE | 0.4169518 | **0.3991166** |

The relative WAPE reduction is 4.2775%, and the JEPA system wins for all three
controlled-mask test seeds (11, 29, and 47).
