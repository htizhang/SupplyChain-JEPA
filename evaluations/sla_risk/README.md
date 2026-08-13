# Early SLA-risk ranking under delayed reporting

This evaluation asks which shipments should be reviewed when the latest reports
for a critical incoming leg have not arrived. It uses Cargo2000 process traces,
but the external CSV and split manifest are not distributed with this repository.

## Protocol

- The decision point is `in_seg1`.
- The critical incoming leg is selected only from planned durations.
- Its `RCS` report is hidden at `in_rcs`; its `RCS` and first `DEP`/`RCF`
  reports remain hidden at `in_seg1`.
- All aggregate delay features are recomputed after hiding those reports.
- Development, confirmation, and adjacent-shift cohorts are fixed by the
  separately supplied split manifest.

The public-history baseline is a source-and-target logistic model using the
same corrupted observations and 64 target labels. The pretrained system keeps
that baseline fixed and uses a frozen State encoder to represent the change
from `in_rcs` to `in_seg1`. A prototype estimated from the target-fit labels
contributes 20% of the final rank. The baseline probability multiset is
preserved; only shipment ordering changes.

Ten seeded random encoders use the same architecture, tensor adapter, labels,
prototype, and ranking rule. Their deterministic aggregate separates the
effect of pretrained weights from the random nonlinear feature map.

## External inputs

Pass a directory containing the Cargo CSV and split manifest. The evaluator
locates them by content and rejects files whose SHA-256 differs from:

```text
Cargo CSV      da40666b12f4a5465a885f3df91d67cd7a1f20946a0ef07e62e9d35b3687984a
Split manifest 47616c4806b101d51624abe0b2e07f8e9ea7cdabea3373d49a5aea9c0ac671de
```

Run the complete evaluation with the immutable model release:

```bash
python -m evaluations.sla_risk \
  --data-root /path/to/cargo2000 \
  --split-manifest /path/to/split_manifest.json \
  --cache-dir /tmp/supplychain-jepa-sla-risk \
  --output /tmp/sla-risk-results.json
```

`--model-source` may point to an already downloaded release directory. Model
verification remains enabled by default. `--skip-model-verification` exists
only for diagnosing a damaged local cache and must not be used for a reported
result.

## Fresh release result and boundary

The frozen confirmation result is:

| Method | AUPRC |
|---|---:|
| Public-history baseline | 0.386170 |
| Matched random encoders, mean of 10 | 0.387490 |
| Pretrained State encoder | **0.412200** |

Development is also positive (`0.394516` baseline, `0.431458` pretrained).
The adjacent single-shift diagnostic is deliberately retained and is negative:
`0.496580` baseline versus `0.477391` pretrained and `0.478483` matched-random
mean. Therefore this is evidence for the delayed-reporting joint-shift setting,
not a claim of universal improvement on Cargo2000.
