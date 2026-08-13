# Data contract

## Raw episode

Every source adapter emits a canonical episode containing:

- source and episode identity;
- ordered observations or events;
- typed entities and topology relations;
- numerical and categorical feature maps;
- observation-status information;
- optional actions, exogenous variables, costs, and paired-branch metadata.

The raw episode schema and the model tensor-interface schema are separate contracts. Raw adapters
must validate against the former before tensorization. The released Hugging Face bundle provides
the latter together with the fitted vocabulary and normalization statistics.

## Tensorized batch

Tensorization produces context and target token fields, padding masks, numerical presence masks,
categorical IDs, status IDs, time coordinates, entity-role IDs, source-family IDs, and local
topology indices. Unknown fields remain absent; adapters must not invent measurements or refit the
pretrained vocabulary on downstream test data.

`tensorizer.json` is authoritative. `normalization.json` is a deterministic projection for
inspection and interoperability, not an independently fitted object.

## Actions

Structured actions contain supported numerical quantities and categorical action semantics.
Paired branch records must identify a shared initial state and exogenous stream. Merely reusing a
random seed is insufficient if simulator execution consumes randomness differently after the
branch point.
