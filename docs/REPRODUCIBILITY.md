# Reproducibility

## Immutable model

All reference evaluations pin Hugging Face revision
`493786fda0855b72f61235068c62cc7b2a11fbec`. The loader validates `CHECKSUMS.sha256` before
constructing the model. Do not run a published comparison against a moving branch.

## Environment

The project supports Python 3.11 through 3.13. `uv.lock` freezes the reference dependency graph.
PyTorch does not guarantee bitwise-identical results across every release, platform, or device, so
acceptance uses metric tolerances plus deterministic seeds rather than byte equality for trained
downstream heads.

## Training-chain check

`scripts/run_training_smoke.py` validates generated canonical records and runs one continuous
State-to-Dynamics chain with the small-model override. The corresponding test verifies that the
fitted tensorizer and all State weights are unchanged during frozen Dynamics training while the
Dynamics predictor itself updates. This is a control-flow and inheritance test; full model quality
is evaluated through the pinned release and canonical application protocols.

## First-release gates

- model tensors load strictly and representative outputs match the frozen release;
- demand-recovery WAPE improves by at least 4% and wins under all three fixed masks;
- SLA-risk confirmation AUPRC improves by at least 5%, with the adjacent negative boundary
  evaluated and reported;
- action-selection regret improves by at least 20% over the fixed public-history baseline, while
  the same-architecture matched-random result is retained as a separate attribution diagnostic;
- no external dataset, pretrained weight, fitted binary, credential, or machine-specific path is
  tracked.

If a primary gate fails, the result is marked unresolved and the repository is not promoted as
the clean reference release. Diagnostic controls are never silently promoted to primary gates or
omitted when they weaken the attribution story.
