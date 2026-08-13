# Architecture

## Operational object

SupplyChain-JEPA does not turn tables into pseudo-images. It represents a supply network as a
sequence of typed entity-time and event tokens. A token can describe a product-location state,
order, shipment, supplier, route milestone, network aggregate, or topology relation. Each token
combines numerical values with feature identity, entity role, observation status, time, source
family, and local topology indices.

The resulting representation is a belief state: a compact summary of operational conditions that
may not be directly observed because records are delayed, censored, missing, or distributed
across systems.

## Shared model chain

The release contains:

- an online context encoder;
- a target encoder trained through exponential moving average and stop-gradient targets;
- a State predictor for hidden-current and observational-future latent targets;
- an action-conditioned Dynamics predictor;
- pooled-state, semantic-concept, and numerical grounding heads.

State and Dynamics are training phases and predictor roles, not two separately pretrained
backbones. Dynamics post-training starts from the State checkpoint and preserves the shared latent
coordinate system.

## State objective

Structured masks remove coherent portions of the operational world: time intervals, entity groups,
network neighborhoods, event suffixes, outages, or future windows. The online encoder receives the
visible context. The target encoder receives the corresponding target view. The State predictor
maps context latents and query metadata to the stop-gradient target latents.

Latent prediction is complemented by variance/covariance regularization and light numerical
grounding so that the representation remains non-collapsed and preserves decision-relevant scale.

## Dynamics objective

The Dynamics predictor receives latent state memory, structured candidate actions, and horizon.
Paired simulator branches vary the action while holding exogenous randomness fixed. Objectives
include future latent prediction, branch contrast, action-shuffle controls, short rollout
consistency, and light flow/cost grounding.

Action conditioning is not a causal guarantee. The predictor is valid only for action semantics and
support represented in its paired training trajectories.

## Release shape

| Component | Parameters |
|---|---:|
| Complete core | 73,629,696 |
| Online State encoder | 27,987,456 |
| Target State encoder | 27,987,456 |
| State predictor | 7,255,680 |
| Dynamics predictor | 10,399,104 |
| Auxiliary State heads | 2,943,285 |

The core uses width 384, 12 attention heads, six encoder layers, three State-predictor layers,
four Dynamics-predictor layers, and a 1,536-dimensional feed-forward block.
