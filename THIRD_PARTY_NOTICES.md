# Third-party notices

SupplyChain-JEPA depends on or interoperates with third-party software and datasets. No
third-party source repository or dataset is copied into this repository.

| Project or source | Role | Upstream terms / boundary |
|---|---|---|
| PyTorch | Neural-network runtime | BSD-style license; see the upstream repository |
| Hugging Face Hub and safetensors | Model distribution and serialization | Apache-2.0 |
| Stockpyl | Supply-network simulation adapter | MIT |
| SupplyNetPy | Discrete-event supply-network simulation adapter | MIT |
| MABIM / ReplenishmentEnv | Replenishment simulation adapter | Apache-2.0; bundled datasets are not redistributed |
| ISOMORPH | Synthetic demand and logistics adapter | MIT |
| SimPy | Discrete-event simulation runtime | MIT |
| gym-invmgmt | Downstream evaluation environment | MIT; downstream-only |
| FreshRetailNet-50K | Downstream dataset and baseline schema | Obtain from the original provider; attribution required |
| Cargo2000 | Downstream event-log dataset | Obtain from the original provider; attribution required |

URLs, pinned versions, and dataset-specific restrictions are maintained in `DATASETS.md`,
`docs/REFERENCES.md`, and the machine-readable manifests. Inclusion here does not relicense any
third-party work.
