# SupplyChain-JEPA

**A predictive world model for partially observed supply-chain systems.**

Supply-chain decisions are often made from delayed, censored, or incomplete operational
records. Sales can be censored by stockouts, shipment milestones can arrive late, orders in
transit may be unresolved, and the consequences of replenishment actions unfold over several
periods. SupplyChain-JEPA learns a reusable latent operational state from heterogeneous
entity-time and event tokens, then predicts hidden state, future state, and action-conditioned
future state.

This private repository is the clean reference implementation. The pretrained weights are kept
in the private Hugging Face model repository
[`htizhang/SupplyChain-JEPA`](https://huggingface.co/htizhang/SupplyChain-JEPA); datasets and
fitted downstream artifacts are not stored in Git.

## Model

The model follows one continuous training chain:

```text
partial observations ──> context encoder ──> latent operational state
                              │
EMA target encoder ────────────┼──> latent targets
                              │
                              ├──> State predictor: hidden/current/future state
                              └──> Dynamics predictor: future state under candidate actions
```

The released core has **73,629,696 parameters**, a width of 384, six encoder layers,
three State-predictor layers, and four Dynamics-predictor layers. Inputs use a fitted
309-feature schema with numerical values, categorical identities, observation status, time,
entity roles, and local topology indices.

## Install

```bash
uv sync --all-extras
```

Authenticate with Hugging Face through the normal CLI or an environment token, then load the
immutable model revision:

```python
from supplychain_jepa import load_pretrained

bundle = load_pretrained(device="cpu")
model = bundle.model
```

The loader pins revision
`493786fda0855b72f61235068c62cc7b2a11fbec` by default and verifies the release checksums.

## Training

The reference implementation separates two objectives without training two unrelated models:

1. **Predictive State pretraining** learns hidden-state recovery and observational future
   prediction from real and simulated trajectories.
2. **Action-conditioned Dynamics post-training** starts from the same encoder, initially freezes
   it, and learns paired future differences under supported actions.

The detailed [Training guide](docs/TRAINING.md) documents corpus sampling, structured masks,
objective equations and weights, EMA/final alignment, paired action controls, checkpoint contents,
and the distinction between a training checkpoint and the published bundle. The
[Data contract](docs/DATA_CONTRACT.md) defines canonical records, while
[Data adapters](docs/DATA_ADAPTERS.md) documents and exposes converters for all ten real and
simulated pretraining sources.

### Data adapters

The sources are not flattened into one artificial table. Each adapter preserves the source's
entities, topology, timing, observation status, events, and actions, then emits the same canonical
episode format consumed by the shared tensorizer:

```text
raw dataset or pinned simulator -> source adapter -> canonical episode JSONL
    -> schema validation -> train-only tensorizer -> entity / event / action tokens
```

| Source group | Included sources | Training eligibility |
|---|---|---|
| Recorded operational histories | M5, BPI Challenge 2019, LaDe, Olist, planned-versus-actual routes | State JEPA only |
| State-only simulators | SupplyNetPy, ISOMORPH, MilestoneFlow | State JEPA only |
| Intervention-capable simulators | Stockpyl, MABIM | State JEPA and paired Dynamics JEPA |

List the adapters, fetch the four pinned external simulator revisions, or build all ten configured
sources as follows:

```bash
uv run --extra adapters python -m supplychain_jepa.adapters list
uv run python scripts/fetch_simulator_sources.py --root /path/to/simulator-checkouts

cp configs/adapters.example.json /path/to/adapters.json
# Edit the local raw-data and simulator paths in /path/to/adapters.json.
uv run --extra adapters python -m supplychain_jepa.adapters build \
  --config /path/to/adapters.json
```

Raw datasets remain under their original terms and must be acquired separately. See
[Data adapters](docs/DATA_ADAPTERS.md) for required files, field-level mappings, split rules,
missingness semantics, and Dynamics pairing requirements.

A bounded CPU chain generates four canonical records and runs both stages:

```bash
uv run --all-extras python scripts/run_training_smoke.py \
  --work-dir /tmp/supplychain-jepa-smoke \
  --updates 2
```

## Canonical evaluations

| Application | Conventional baseline | SupplyChain-JEPA system | Fresh improvement |
|---|---:|---:|---:|
| Stockout demand recovery | WAPE 0.416952 | 0.399117 | 4.28% lower |
| Early SLA-risk ranking under delayed reports | AUPRC 0.386170 | 0.412200 | 6.74% higher |
| Replenishment-plan selection | Regret 9.274 | 5.883 | 36.6% lower |

For the third application, a same-architecture matched-random diagnostic obtains regret 6.119;
the pretrained system is 3.85% lower and wins three of five world splits. The much stronger
comparison with the fixed public-history baseline therefore supports the complete decision
system, while the smaller and run-sensitive matched-random difference is not presented as a
clean causal estimate of pretraining. These are bounded proof-of-concept results, not evidence
of universal superiority or production readiness. Exact protocols and negative boundary results are documented in
[Evaluation](docs/EVALUATION.md).

## Repository boundary

- Git contains project-authored code, compact configurations, tests, documentation, and
  regenerated summary results.
- Hugging Face contains the pretrained weights and fitted tensor interface.
- External datasets remain under their original terms and must be acquired separately.
- Raw experiments, third-party checkouts, paper full texts, credentials, and machine-specific
  execution metadata are excluded.

No open-source license is granted for this private repository. All rights are reserved pending
an ownership and provenance review. See [Third-party notices](THIRD_PARTY_NOTICES.md) and
[Datasets](DATASETS.md).

## References

The design builds on I-JEPA, V-JEPA 2, T-JEPA, LaT-PFN, and latent world-model research.
See [References](docs/REFERENCES.md) for primary sources.
