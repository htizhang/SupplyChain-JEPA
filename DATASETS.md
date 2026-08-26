# Dataset and simulator boundary

No dataset is redistributed in this repository. Users must obtain each source from its original
provider and review the current terms. Source-specific converters are included under
`supplychain_jepa.adapters`; [Data adapters](docs/DATA_ADAPTERS.md) documents their required local
inputs and canonical mappings.
The manifests under `data/manifests/` record expected identities and integrity hashes where the
original experiments sealed them.

## Base pretraining

The State corpus combines observational trajectories with mechanism-rich simulation. The
released corpus used M5, BPI Challenge 2019, LaDe, Olist, planned-versus-actual route records,
Stockpyl, SupplyNetPy, MABIM with generated inputs, ISOMORPH, and a project-authored discrete-event
milestone simulator. Observational sources train predictive representations only; they are not
treated as counterfactual action labels.

The Dynamics corpus uses intervention-capable simulators with paired branches sharing the same
exogenous randomness. The released training lineage used Stockpyl and MABIM with generated
inputs. Real observational data is excluded from the action-counterfactual objective.

## Hard downstream separation

The following sources and their generated trajectories are downstream-only and were excluded
from base pretraining:

- FreshRetailNet-50K;
- Cargo2000;
- gym-invmgmt and substantially overlapping environment lineages.

Downstream training partitions may be used for task-specific adapters or heads, but the held-out
evaluation partitions must never influence model or method selection.

## Legal boundary

Dataset terms are separate from the project code. Where source metadata conflicts or is unclear,
the stricter interpretation applies and redistribution is disabled. This document is a technical
inventory, not legal advice.
