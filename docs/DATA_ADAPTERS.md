# Data adapters

Every pretraining source has a deterministic adapter in
`supplychain_jepa.adapters`. The adapters preserve source-specific semantics while emitting the
same canonical episode contract consumed by the shared tensorizer and JEPA trainers.

```text
raw dataset or simulator checkout
              |
              v
      source-specific adapter
              |
              v
       canonical episode JSONL
              |
        schema validation
              |
              v
       train-only tensorizer
              |
              v
 topology / entity / event / action tokens
              |
              v
       State or Dynamics JEPA
```

The canonical format is not a requirement that every source expose an identical flat table.
Feature maps remain sparse. The State training split fixes the union of feature identities,
categorical vocabularies, and numerical normalization statistics. Missing source fields remain
absent and receive a false presence mask; adapters must never synthesize measurements merely to
fill another source's schema.

The serialized `schema_version` remains `supplychain-jepa.synthetic-episode.v1` for compatibility
with the released training lineage. Despite that legacy identifier, the contract accepts both
observational and simulated episodes; `objective_eligibility` and provenance carry the relevant
causal distinction.

## Public interface

List the available adapters without importing their optional dependencies:

```bash
uv run python -m supplychain_jepa.adapters list
```

Convert one source:

```bash
uv run --extra adapters python -m supplychain_jepa.adapters convert \
  --source m5 \
  --raw-root /path/to/m5 \
  --output /path/to/corpus/m5.jsonl
```

Build all ten sources from one local configuration:

```bash
cp configs/adapters.example.json /path/to/adapters.json
# Replace every placeholder path with a separately acquired dataset or pinned checkout.
uv run --extra adapters python -m supplychain_jepa.adapters build \
  --config /path/to/adapters.json
```

Raw datasets are not downloaded by these commands. The four external simulator revisions can be
fetched into an isolated directory with:

```bash
uv run python scripts/fetch_simulator_sources.py --root /path/to/simulator-checkouts
```

The fetcher checks out the immutable revisions in `configs/source_pins.json`; adapters reject a
mismatched checkout rather than silently running different simulator code.

## Canonical boundary

Each adapter emits:

- source identity, episode identity, split, and provenance;
- typed topology nodes and edges;
- ordered `steps` with physical time;
- an `observation` view containing only information available to the context encoder;
- a `latent_state` target view;
- observation status for delayed, censored, missing, or unavailable values;
- structured actions and outcomes when the source contains them;
- paired-branch metadata only for controlled simulator interventions.

For observational data, `latent_state` means the finalized factual target view available from the
record, not an invented physical state. Later steps provide future targets through the training
window builder. For simulators, `latent_state` can contain internal state hidden by a separately
generated reporting process.

## Source mappings

### M5

- **Episode:** one bounded item-store daily series window.
- **Topology:** the store is a retail node; the item remains the product identity.
- **Observation/target:** recorded units sold, calendar, hierarchy, events, and SNAP eligibility.
- **Boundary:** M5 contains neither on-hand inventory nor stockout truth. Zeros remain recorded
  sales and are not relabeled as uncensored demand.
- **Objective:** State hidden-current and observational-future prediction only.

### BPI Challenge 2019

- **Episode:** one purchase-order-item process trace ordered by event time.
- **Tokens:** purchasing case, item/vendor/company attributes, and procurement activities.
- **Target:** later factual process states and event suffixes.
- **Boundary:** recorded actions are process observations, not intervention labels.
- **Objective:** State prediction only.

### LaDe

- **Episode:** one courier-day sequence of completed delivery records.
- **Tokens:** order, courier, region, coordinates, acceptance, and completion events.
- **Target:** later delivery milestones, elapsed time, workload progression, and natural
  missingness patterns.
- **Objective:** State prediction only.

### Olist

- **Episode:** one seller's bounded sequence of order-item lifecycles.
- **Tokens:** seller, product, order status, purchase/approval/handoff/delivery timestamps,
  freight, and physical product attributes.
- **Target:** finalized delivery state and delivery-versus-estimate outcome.
- **Boundary:** the dataset has no inventory trajectory or controlled replenishment actions.
- **Objective:** State prediction only.

### Planned-versus-actual routes

- **Episode:** one route in actual stop order.
- **Tokens:** depot, driver, stops, planned and actual sequence indices, time windows, and leg
  distances.
- **Target:** later route suffix, sequence deviation, time-window status, and cumulative execution
  distance.
- **Objective:** State prediction only.

### Stockpyl

- **Episode:** a generated serial inventory network with stochastic demand and lead-time inputs.
- **Tokens:** node-product inventory, backlog, in-transit stock, replenishment edges, demand,
  shipments, costs, and order actions.
- **State view:** observations copy the complete simulator state; State training masks hide parts.
- **Dynamics view:** sibling rollouts start from the same state and replay the same demand stream
  and action prefix, then execute different replenishment quantities.
- **Objective:** State and paired Dynamics prediction.

### SupplyNetPy

- **Episode:** a generated supplier, warehouse, retailer, and customer chain under (s, S) policies.
- **Tokens:** inventory, backorders, shortages, pending orders, demand, fulfillment, and lead times.
- **State view:** deterministic reporting rules withhold part of the internal simulator state.
- **Boundary:** replenishment policy outputs are observational in the released lineage; no paired
  counterfactual claim is made.
- **Objective:** State prediction only.

### MABIM

- **Episode:** a generated multi-warehouse, multi-SKU inventory trajectory.
- **Inputs:** demand, prices, lead times, initial inventory, and policy inputs are generated by the
  adapter; bundled upstream datasets are not read.
- **Tokens:** SKU-warehouse inventory, sales, demand, arrivals, in-transit quantities, backlog,
  costs, and replenishment actions.
- **Dynamics view:** exact low/high replenishment branches share the same generated world,
  historical prefix, and future exogenous stream.
- **Objective:** State and paired Dynamics prediction.

### ISOMORPH

- **Episode:** a generated multi-product supply-network rollout under ISOMORPH's internal policy.
- **Tokens:** facilities, transport edges, product inventory, backlog, in-transit quantities,
  demand, shipments, capacity, and service state.
- **Boundary:** internal policy dispatches remain observations and are not labeled as externally
  controlled interventions.
- **Objective:** State prediction only.

### MilestoneFlow

- **Episode:** a project-authored discrete-event cargo process with queues, consolidation,
  transport, and milestone reports.
- **Tokens:** shipments, operational nodes, event times, report times, queues, service classes,
  process costs, and delivery state.
- **State view:** the physical event stream is the target; the context view receives events only
  after generated reporting delays and omits selected reports.
- **Objective:** State prediction only in the released corpus.

## Dynamics pairing requirement

A Dynamics branch group is valid only when sibling records share simulator family, topology,
decision time, complete pre-action prefix, and a future exogenous stream. Reusing an integer seed
is insufficient if one action changes later random-number consumption. Stockpyl and MABIM adapters
therefore predefine or replay the exogenous inputs and verify that sibling futures diverge only
after the registered action branch.

## Split and leakage rules

- Splits are deterministic hashes of complete cases, worlds, or item-store series, not random
  rows from the same trajectory.
- Tensorizer fitting uses `pretrain_train` only.
- Observational adapters never enable `dynamics_jepa`.
- FreshRetailNet-50K, Cargo2000, gym-invmgmt, and overlapping generated trajectories remain
  excluded from base pretraining.
- Adapter output is validated against the packaged JSON Schema before publication.
