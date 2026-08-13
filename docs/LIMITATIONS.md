# Limitations

- The released model is a research-scale proof of concept, not a production foundation model.
- Predictive latent state does not enforce inventory balance, capacity, service, or safety
  constraints. A separate optimization or control layer remains necessary.
- Action-conditioned prediction is limited to action semantics and state regimes represented in
  paired training data; it does not identify unsupported interventions.
- The model may compress rare but decision-critical shocks. Risk-sensitive heads, ensembles, and
  scenario methods remain important.
- Downstream results are small-sample and task-specific. The SLA-risk confirmation cohort contains
  only 126 processes with 36 positives, and its adjacent diagnostic cohort is negative.
- In replenishment-plan selection, the complete system strongly beats the fixed public-history
  baseline, but the same-architecture matched-random advantage is smaller and varies across world
  splits. It should not be read as a precise causal estimate of pretraining value.
- Dataset adapters map heterogeneous schemas into a common vocabulary and can omit source-specific
  information. Adapter quality is part of the method.
- Model weights and several datasets are private or governed by source-specific terms, so a fresh
  clone alone is insufficient for full evaluation.
