# Training

## Predictive State pretraining

The State phase uses source-balanced sampling over independent entities, cases, or simulated worlds
rather than raw rows. The released lineage used 37,712 training episodes and 2,035 development
episodes from ten observational and synthetic source families. Union-schema tensorization produced
309 fitted feature identities.

Training jointly covers hidden-current recovery and observational future prediction. The latter is
necessary for event-prefix risk applications; a model trained only to fill individual fields would
not learn process progression.

## Frozen-encoder Dynamics post-training

The released Dynamics corpus contains 4,500 branch records arranged into 2,250 paired
counterfactual groups. Each pair shares future exogenous randomness but changes the candidate
replenishment action. The State encoder is frozen first so that action effects are learned in the
existing representation space rather than rewriting it with a much smaller action corpus.

Limited joint adaptation is an optional downstream procedure, not part of the shared release. When
used, only shallow adapters or upper layers should be updated at a lower learning rate, with State
distillation to reduce forgetting.

## Commands

The clean command surface is:

```bash
python -m supplychain_jepa.train state \
  --config configs/state.json \
  --data /path/to/state_manifest.json \
  --output-dir /path/to/output

python -m supplychain_jepa.train dynamics \
  --config configs/dynamics.json \
  --data /path/to/dynamics_manifest.json \
  --checkpoint /path/to/state_checkpoint.pt \
  --output-dir /path/to/output
```

Training artifacts are intentionally excluded from Git. Promote a checkpoint only after checking
representation rank, matched-versus-shuffled targets, action sensitivity, branch ranking, and
short-rollout stability.
