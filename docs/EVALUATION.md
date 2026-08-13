# Evaluation

## 1. Stockout demand recovery

Normal-availability retail days are subjected to controlled masks shaped like observed stockout
intervals. The label is the true demand inside the hidden interval. The conventional anchor uses
analog-day nearest neighbors. The SupplyChain-JEPA system freezes the State encoder and uses latent
retrieval plus a small correction model trained under a limited adaptation budget.

Primary metric: pooled WAPE across three fixed mask seeds. The clean release rerun obtains
0.416952 for the analog baseline and 0.399117 for the JEPA system, a 4.28% reduction with wins
under all three masks.

## 2. Early SLA-risk ranking under delayed reports

The most time-consuming incoming transport leg is assigned a coherent reporting delay. All methods
receive the same corrupted event prefix, and aggregates are recomputed after hiding events. A
public-history logistic model remains fixed as the baseline. Frozen State-transition evidence is
blended with that score and compared with architecture-matched random encoders.

Primary metric: AUPRC on a method-frozen joint route-and-topology shift cohort. The clean release
rerun obtains 0.386170 for the baseline, 0.387490 for the mean matched random encoder, and 0.412200
for the pretrained encoder. The adjacent single-shift cohort is negative and must be reported:
0.496580 baseline versus 0.477391 pretrained.

## 3. Replenishment-plan selection

At one decision anchor, the controller observes 16 periods of impaired public history and chooses
between quantity-matched regular and split replenishment plans. Both plans are rolled out for four
periods under common random numbers and a fixed no-further-orders continuation. The JEPA system
uses the frozen State encoder, an adapted copy of the pretrained Dynamics predictor, and a scalar
cost readout.

Primary metric: action regret over five disjoint world splits. The clean release rerun obtains
5.883 for the pretrained system, 6.119 for the matched-random diagnostic, and 9.274 for
public-history ExtraTrees. The pretrained system beats matched random by 3.85% on average and on
three of five splits. This weaker attribution result is reported alongside, not substituted for,
the 36.6% complete-system improvement over the fixed conventional baseline.

## Claim boundary

These experiments show that a shared pretrained representation can support three distinct,
partially observed supply-chain applications. They do not show statistical significance in every
cohort, universal dominance over task-specific methods, unrestricted counterfactual validity, or
production readiness.
