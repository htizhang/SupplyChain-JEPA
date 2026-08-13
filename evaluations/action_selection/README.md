# Action-selection evaluation

This directory contains two deliberately separate surfaces:

1. `protocol.py` aggregates the canonical five-split paired-plan experiment
   from compact run outputs produced with the external simulator corpus.
2. The default module provides a deterministic, self-contained reconstruction
   for API and protocol smoke testing. It is not represented as an exact replay
   of the canonical experiment.

The fixed contract is:

- 16 public-history periods with delayed and missing supplier reports;
- a four-period decision horizon;
- quantity-matched `regular` and `split` replenishment plans;
- common-random-number paired futures;
- no further orders after the decision;
- a frozen pretrained State encoder;
- supervised AdamW adaptation of pretrained Dynamics and a scalar readout;
- learning-rate selection from `{3e-4, 1e-4}` on non-report worlds; and
- five disjoint fit/select/report world splits.

Summarize five canonical protocol outputs:

```bash
python -m evaluations.action_selection.protocol \
  /path/to/split-0.json /path/to/split-1.json /path/to/split-2.json \
  /path/to/split-3.json /path/to/split-4.json \
  --model-revision 493786fda0855b72f61235068c62cc7b2a11fbec \
  --public-history-baseline 9.274 \
  --output /tmp/action-selection-summary.json
```

Run the bounded synthetic smoke evaluation:

```bash
python -m evaluations.action_selection \
  --tiny --worlds 30 --future-replications 3 --epochs 2 \
  --extra-trees-estimators 16 --output-dir /tmp/action-selection-smoke
```

Run the released model through the clean `ModelBundle` API as a smoke check:

```bash
python -m evaluations.action_selection \
  --model-dir /path/to/pinned/model/snapshot \
  --output-dir /tmp/action-selection-release
```

The smoke reconstruction writes `split_00.json` through `split_04.json` and
`summary.json`. Its metrics are not substituted for the canonical external
protocol result.

## Fresh release result

Across five world splits, the canonical protocol obtains mean regret **5.883
± 0.336** for the pretrained system, **6.119 ± 0.746** for the matched-random
diagnostic, and **9.274 ± 2.547** for public-history ExtraTrees. This is a
36.6% complete-system reduction relative to the conventional baseline. The
matched-random reduction is 3.85% with wins on three of five splits and is
reported as a weaker, adaptation-sensitive attribution result.
