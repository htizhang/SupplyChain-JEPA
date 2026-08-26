# Training

SupplyChain-JEPA is trained as one continuous weight chain:

```text
canonical trajectories
        │
        ├── Predictive State training
        │     online encoder + State predictor learn
        │     target encoder follows by EMA
        │
        └── State checkpoint
                 │
                 └── Action-conditioned Dynamics training
                       encoders and State predictor stay fixed
                       Dynamics predictor learns paired action effects
```

State and Dynamics are therefore two objectives and training phases, not two unrelated
backbones. The final checkpoint retains the same fitted tensor interface, both State encoders,
the State predictor, and the trained Dynamics predictor.

## 1. Training input

The trainer consumes newline-delimited records satisfying the packaged
[`canonical_episode.schema.json`](../src/supplychain_jepa/schemas/canonical_episode.schema.json).
Each record contains:

- typed supply-network nodes and edges;
- an ordered trajectory of latent states and partial observations;
- observation status such as delayed, censored, missing, or unavailable values;
- optional events and structured actions;
- objective eligibility and train/development split metadata;
- paired-branch metadata when the record can support Dynamics training.

The separation between `observation` and `latent_state` is a hard model boundary. The context
encoder receives topology, observations, and recorded actions. The target encoder receives latent
state. A legally unobserved target is never copied into context.

The loader accepts canonical JSONL directly, a JSON manifest that declares JSONL paths, or a
directory containing such artifacts. All ten raw-to-canonical converters are available under
`supplychain_jepa.adapters` and documented in [Data adapters](DATA_ADAPTERS.md). Third-party raw
data remains outside the repository and under its original terms. The files under
`data/manifests/` document the released training lineage.

### Train-only tensor fitting

Before State optimization, `fit_tensorizer_state` scans only `pretrain_train` records and fixes:

- the union of numerical and categorical feature identities;
- numerical means and standard deviations;
- categorical and metadata vocabularies;
- token families, token types, and local identity indices.

The fitted interface is serialized in the State checkpoint and reused unchanged by Dynamics.
Development or downstream records never refit normalization or vocabularies. The released lineage
produced 309 feature identities.

## 2. Model configuration

[`configs/model.json`](../configs/model.json) defines the released core:

| Setting | Value |
|---|---:|
| Latent width | 384 |
| Attention heads | 12 |
| Encoder layers | 6 |
| Encoder feed-forward width | 1,536 |
| State-predictor layers | 3 |
| Dynamics-predictor layers | 4 |
| Predictor feed-forward width | 1,536 |
| Dropout | 0.05 |
| Local identity encoding | compositional |
| Canonical state pooling | masked mean |
| Action encoding | structured |

The CLI option `--small-model` overrides this with width 32, one encoder layer, one layer per
predictor, feed-forward width 64, and zero dropout. It is intended for interface and control-flow
tests, not as a substitute for the released model.

## 3. Predictive State training

Let \(x_C\) denote the visible operational context and \(x_T\) the hidden-current or future target
view. The online and target encoders produce

\[
z_C = E_\theta(x_C), \qquad
z_T = \operatorname{stopgrad}(E_\xi(x_T)),
\]

and the State predictor uses target-query metadata \(q\), including query role and horizon, to
predict the target representation:

\[
\widehat z_T = P_{\mathrm{state}}(z_C,q).
\]

The target encoder is not optimized by gradients. Its parameters follow the online encoder by an
exponential moving average (EMA).

### Corpus and sampling

The released State lineage contains 37,712 training episodes and 2,035 development episodes from
ten source families. Sampling is hierarchical rather than row-uniform:

1. choose a source family by the configured weight;
2. choose a world or independent case uniformly within that family;
3. choose an episode from that world;
4. construct a deterministic structured query window.

This prevents a source with many nearly duplicate rows from dominating simply because it is wide
or frequently sampled. The configured source weights sum to 100:

| Source family | Weight |
|---|---:|
| M5 | 10 |
| BPI 2019 | 8 |
| LaDe | 8 |
| Olist | 6 |
| Planned-versus-actual routes | 8 |
| Stockpyl | 20 |
| SupplyNetPy | 15 |
| MABIM | 10 |
| ISOMORPH | 10 |
| MilestoneFlow | 5 |

Observational sources support State prediction only. They are not interpreted as evidence about
unobserved interventions.

### Query construction

Every five sampled examples contain three hidden-current queries and two observational-future
queries. Current-state masking removes coherent operational structures rather than isolated random
scalars:

- contiguous time intervals;
- entity groups;
- graph neighborhoods;
- event suffixes;
- reporting or sensor outages.

Future queries use horizons 1, 2, 4, or 8 and remove future actions from context. Their meaning is
policy-marginal observational prediction, not an action-specific counterfactual. Context is capped
at 192 tokens, but the window builder preserves causal global/action evidence, query-relevant
entities, topology neighborhoods, and temporal coverage. It raises an error if these mandatory
tokens cannot fit rather than silently truncating them.

### State loss

The implemented State objective is

\[
\mathcal L_{\mathrm{state}}
= \mathcal L_{\mathrm{token}}
+ \lambda_{\mathrm{pool}}\mathcal L_{\mathrm{pool}}
+ \lambda_{\mathrm{canonical}}\mathcal L_{\mathrm{canonical}}
+ \lambda_{\mathrm{semantic}}\mathcal L_{\mathrm{semantic}}
+ \lambda_{\mathrm{ground}}\mathcal L_{\mathrm{ground}}
+ \mathcal L_{\mathrm{optional\ collapse}}.
\]

The active release configuration is:

| Component | Purpose | Weight |
|---|---|---:|
| Token latent prediction | Smooth-L1 on selected target token latents | 1.00 |
| Pooled target prediction | Preserve the aggregate hidden/future target | 0.25 |
| Canonical-state prediction | Make the exported belief state predictive | 0.20 |
| Semantic concept slots | Preserve cross-schema operational concepts | 0.30 |
| Numerical grounding | Retain scale for inventory, demand, backlog, lead-time, and stockout fields | 0.10 |

The canonical-state term combines absolute Smooth-L1, family-centered mean-squared error, and
family-centered cosine loss with internal weights 0.25, 0.60, and 0.15. The semantic term combines
Smooth-L1 and cosine losses with weights 0.80 and 0.20. Generic identity and raw action/order
concepts are excluded from semantic-slot targets.

The implementation also supports VICReg-style variance/covariance penalties and SIGReg. They are
disabled in the active release configuration (`anti_collapse.method = "none"`, variance and
covariance weights equal zero). Representation mean standard deviation, effective rank, and
leading-principal-component share are still reported throughout training.

### Optimization, EMA, and final alignment

The active optimizer settings are AdamW with learning rate \(5\times10^{-5}\), weight decay 0.05,
betas (0.9, 0.95), gradient clipping at 1.0, 5% warmup, and cosine decay to 10% of the initial
learning rate. EMA momentum increases from 0.996 to 0.9995.

After the main State phase:

1. the configured deployment encoder is selected (`online` in the release configuration);
2. online and target encoders are hard-aligned to that representation space;
3. both encoders are frozen;
4. the lightweight State prediction heads receive 100 final alignment updates at learning rate
   \(10^{-4}\).

This last step removes a train/deploy mismatch: State predictors trained against moving EMA targets
are realigned to the single latent space exported to downstream applications.

## 4. Action-conditioned Dynamics training

Dynamics starts from the completed State checkpoint. A fresh Dynamics predictor is initialized,
while the fitted tensorizer, online encoder, target encoder, and State predictor are restored from
the State stage. The selected State representation is frozen. Only Dynamics-predictor parameters
are passed to the optimizer.

For state memory \(z_t\), action sequence \(a_{t:t+h-1}\), and horizon \(h\), the predictor estimates

\[
\widehat z_{t+h} = P_{\mathrm{dyn}}(z_t,a_{t:t+h-1},h).
\]

### Paired branch contract

Each Dynamics group contains at least two feasible branches. A valid pair must share:

- simulator family and topology;
- decision time;
- the complete pre-action trajectory prefix;
- a common future exogenous stream.

The branches differ in the registered action and then evolve into action-dependent futures. The
released Dynamics lineage contains 4,500 branch records in 2,250 paired groups. Stockpyl and MABIM
receive sampling weights 55 and 45. Real observational records are not used as counterfactual
labels.

The current action representation contains eight fields: quantity, action type, time, actor,
target, product, a presence indicator, and log arrival lag. Quantity receives a signed-log transform
inside the structured action encoder; identity fields are episode-local indices, not portable raw
identifiers. By default the predictor receives actions from the decision time through the requested
horizon.

### Dynamics loss

The active objective is

\[
\begin{aligned}
\mathcal L_{\mathrm{dyn}}
= {}& \mathcal L_{\mathrm{future}}
+ 0.25\mathcal L_{\mathrm{paired\ difference}}
+ 0.10\mathcal L_{\mathrm{control\ margin}} \\
&+ 0.50\mathcal L_{\mathrm{shuffled\ future}}
+ 0.10\mathcal L_{\mathrm{two\ step}}
+ 0.25\mathcal L_{\mathrm{branch\ contrast}}.
\end{aligned}
\]

- **Future latent prediction** matches each registered action to its target future state.
- **Paired difference** predicts the change between sibling action branches, not only their common
  future component.
- **Control margin** makes the registered action fit its future better than no-op and wrong-entity
  controls.
- **Shuffled future** maps a shuffled sibling action to the sibling future it actually represents.
- **Two-step consistency** rolls the predictor twice for horizon-two examples.
- **Branch contrast** distinguishes feasible sibling actions after removing their shared future
  component.

The optimizer is AdamW with learning rate \(2\times10^{-4}\), weight decay 0.05, betas (0.9, 0.95),
gradient clipping at 1.0, two-step gradient accumulation, 5% warmup, and cosine decay to 10% of the
initial learning rate. Training horizons are 1, 2, 4, and 8.

## 5. Full training commands

Install the locked training environment:

```bash
uv sync --all-extras
```

Then train the two stages in sequence. The data arguments must point to already canonicalized
records or a manifest that declares them.

```bash
uv run python -m supplychain_jepa.train state \
  --model-config configs/model.json \
  --config configs/state.json \
  --data /path/to/state-corpus-manifest.json \
  --output-dir /path/to/state-output \
  --max-updates <state-updates> \
  --batch-size <state-micro-batch>

uv run python -m supplychain_jepa.train dynamics \
  --model-config configs/model.json \
  --config configs/dynamics.json \
  --data /path/to/dynamics-corpus-manifest.json \
  --checkpoint /path/to/state-output/state_checkpoint.pt \
  --output-dir /path/to/dynamics-output \
  --max-updates <dynamics-updates> \
  --batch-size <dynamics-micro-batch>
```

`--device auto` uses an available accelerator and otherwise falls back to CPU. Automatic mixed
precision is enabled only on a compatible accelerator; pass `--disable-amp` to turn it off.

State continuation is available through `--init-checkpoint`. Add
`--reuse-parent-tensorizer` only when the new corpus must retain the parent's exact fitted feature
and vocabulary interface.

## 6. Bounded end-to-end smoke recipe

The repository includes a four-record, project-authored canonical corpus generator. It creates two
train and two development branches around one replenishment decision, validates every record
against the packaged JSON Schema, then exercises both training stages on CPU.

```bash
uv run --all-extras python scripts/run_training_smoke.py \
  --work-dir /tmp/supplychain-jepa-smoke \
  --updates 2
```

Expected artifacts are:

```text
<work-dir>/tiny-corpus.jsonl
<work-dir>/state/state_checkpoint.pt
<work-dir>/state/metrics.json
<work-dir>/dynamics/dynamics_checkpoint.pt
<work-dir>/dynamics/metrics.json
<work-dir>/smoke-summary.json
```

The smoke recipe proves the canonical data contract, tensorizer fitting, State-to-Dynamics weight
inheritance, frozen encoder boundary, paired controls, and checkpoint writing. Its tiny losses are
not estimates of the released model's performance.

## 7. Training checkpoint versus model release

Each training command writes one PyTorch checkpoint and one `metrics.json` file. The State
checkpoint contains the full model state, fitted tensorizer, auxiliary State heads, training
configuration, source families, and metrics. The Dynamics checkpoint adds the trained Dynamics
predictor and the State-parent checksum.

Those development checkpoints are not the published model bundle. The pinned model release uses
Safetensors plus explicit model, schema, tensorizer, normalization, release-manifest, and checksum
files. Release packaging must remove machine-local lineage paths while preserving content hashes
and model configuration.

Checkpoint creation is unconditional; promotion is a separate model-selection decision. At
minimum, inspect finite losses, representation geometry, target-shuffle controls, paired branch
ranking, action sensitivity, and short-rollout behavior before selecting a candidate.
