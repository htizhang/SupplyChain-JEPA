# Graph Report - SupplyChain-JEPA  (2026-10-15)

> [!NOTE]
> Generated with [graphify](https://github.com/Graphify-Labs/graphify) from this repository only.
> Python code was parsed structurally; every Markdown document and illustration was read for
> concepts and cross-references, with line evidence on each node. Communities were named by hand.
> For the interactive map, download [`graph.html`](graph.html) and open it in a browser; the
> [documentation index](../docs/README.md#knowledge-graph) explains how to read the graph.

## Corpus Check
- 120 files · ~116,338 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1707 nodes · 4679 edges · 69 communities (66 shown, 3 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 531 edges (avg confidence: 0.78)
- Token cost: 289,502,031 input · 1,500,443 output, as reported by the 6 peer documentation lanes for their whole sessions (drafting and extraction; input = 1,574 uncached + 6,974,496 cache-write + 282,525,961 cache-read tokens). Usage for the  lane and the integration pass is not included in this total.

## Community Hubs (Navigation)
- Episode tensorizer, tokens, and masks
- Recorded-data adapters
- Training windows and examples
- Training runners and checkpoints
- Dynamics learning concepts
- Adapter framework, build, and CLI
- Core model and training objectives
- Applications guide and walkthroughs
- Documentation index and source acquisition
- Tiny corpus, schema loading, and adapter tests
- Predictors and token embedding
- Project overview and glossary
- SLA-risk evidence and claim boundaries
- Action-selection model backends
- ISOMORPH simulator adapter
- MilestoneFlow simulator adapter
- MABIM adapter
- State learning: masks, windows, examples
- Public inference API and model bundle
- Reproducibility gates and provenance
- Demand-recovery data and baselines
- Plan-selection application and results
- SLA-risk tensor adapter
- SupplyNetPy simulator adapter
- Demand-recovery representation
- Loss functions and training tests
- Demand-recovery pipeline and CLI
- SLA-risk evaluation and rank blending
- Action-selection evaluation loop
- SLA-risk delayed-report protocol
- References and data sources
- Training smoke chain and CI
- Release loading and verification
- Action-selection world reconstruction
- Stockpyl adapter
- Structured mask sampling
- Demand-recovery correction model
- Seventeen concept classes
- Paired-branch Dynamics corpus
- State pretraining sources
- Action-selection protocol and tests
- Canonical episode JSON Schema
- Training reference
- Model configuration
- Getting started: encoding and plan comparison
- State encoders, predictor, and EMA
- Canonical episode boundary
- Architecture reference
- SLA-risk baseline and calibration
- Tensor interface contract
- Tokens and training windows
- Release verification and parity tests
- State loss terms
- Action-selection CLI and config
- Demand-recovery tests
- Downstream separation and leakage rules
- SLA-risk transition prototype
- Core API tests
- Pooled belief state and canonical head
- Repository boundary check
- Concept mapping code
- M5 walkthrough and censored sales
- Dynamics training settings
- Source-schema gating
- Dataset legal boundary
- Package metadata

## God Nodes (most connected - your core abstractions)
1. `AdapterRequest` - 53 edges
2. `SupplyChainJEPACore` - 44 edges
3. `EpisodeTensors` - 43 edges
4. `stable_digest()` - 38 edges
5. `run_state()` - 38 edges
6. `run_dynamics()` - 34 edges
7. `_device()` - 33 edges
8. `EpisodeTensorizer` - 31 edges
9. `evaluate()` - 30 edges
10. `FittedTensorizerState` - 28 edges

## Surprising Connections (you probably didn't know these)
- `Matched random encoder control` --semantically_similar_to--> `_encoder_for_seed()`  [INFERRED] [semantically similar]
  docs/guide/GETTING_STARTED.md → evaluations/sla_risk/evaluation.py
- `_two_step_rollout_loss()` --implements--> `Two-step consistency term (weight 0.10)`  [INFERRED]
  src/supplychain_jepa/training.py → docs/guide/DYNAMICS_LEARNING.md
- `PlanQualityModel` --implements--> `Adapted Dynamics with a zero-initialized scalar cost head (action selection)`  [INFERRED]
  evaluations/action_selection/evaluate.py → docs/guide/GETTING_STARTED.md
- `_regret()` --implements--> `Optimizer or decision rule`  [EXTRACTED]
  evaluations/action_selection/evaluate.py → docs/guide/DYNAMICS_LEARNING.md
- `check_repository()` --implements--> `Gate: no external dataset, pretrained weight, fitted binary, credential, or machine-specific path tracked`  [INFERRED]
  scripts/check_repository_boundary.py → docs/REPRODUCIBILITY.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **State learning training step** — docs_guide_state_learning_context_encoder, docs_guide_state_learning_query, docs_guide_state_learning_state_predictor, docs_guide_state_learning_target_view, docs_guide_state_learning_target_encoder, docs_guide_state_learning_latent_loss, docs_guide_state_learning_ema_weight_update [EXTRACTED 1.00]
- **Composite State objective terms** — docs_training_token_latent_prediction_loss, docs_training_pooled_target_prediction, docs_training_canonical_state_prediction, docs_training_semantic_concept_slot_prediction, docs_training_numerical_grounding_loss [EXTRACTED 1.00]
- **Joint-embedding predictive architecture lineage** — docs_references_i_jepa_assran_et_al_2023, docs_references_v_jepa_bardes_et_al_2024, docs_references_v_jepa_2_assran_et_al_2025, docs_references_t_jepa_melo_et_al_2024, docs_references_lat_pfn_verdenius_et_al_2024 [EXTRACTED 1.00]
- **Ten source adapters implementing one canonical episode contract** — docs_data_adapters_m5, docs_data_adapters_bpi_challenge_2019, docs_data_adapters_lade, docs_data_adapters_olist, docs_data_adapters_planned_versus_actual_routes, docs_data_adapters_stockpyl, docs_data_adapters_supplynetpy, docs_data_adapters_mabim, docs_data_adapters_isomorph, docs_data_adapters_milestoneflow, docs_data_adapters_canonical_boundary [EXTRACTED 1.00]
- **Leakage-safe path from episode views to training examples** — docs_guide_data_to_episodes_observation_view, docs_guide_data_to_episodes_latent_state_view, docs_guide_data_to_episodes_train_only_tensorizer, docs_guide_data_to_episodes_context_tokens, docs_guide_data_to_episodes_target_tokens, docs_guide_data_to_episodes_episode_local_identifiers, docs_guide_data_to_episodes_window_builder, docs_guide_data_to_episodes_leakage_rules [EXTRACTED 1.00]
- **Paired-branch contract for Dynamics training** — docs_data_contract_paired_branch_records, docs_data_contract_trainer_branch_verification, docs_data_contract_structured_actions, docs_data_adapters_dynamics_pairing_requirement, docs_data_adapters_stockpyl, docs_data_adapters_mabim [INFERRED 0.85]
- **Three application pipelines routed through one frozen release** — docs_guide_using_the_model_routing_diagram, docs_guide_using_the_model_state_encoder, docs_guide_using_the_model_canonical_state_head, docs_guide_using_the_model_dynamics_predictor, docs_guide_using_the_model_demand_recovery_walkthrough, docs_guide_using_the_model_sla_risk_walkthrough, docs_guide_using_the_model_plan_selection_walkthrough [EXTRACTED 1.00]
- **Matched-random controls that separate pretrained weights from architecture and readout** — docs_evaluation_architecture_matched_random_encoders, docs_evaluation_matched_random_diagnostic, evaluations_sla_risk_readme_ten_matched_random_encoders, evaluations_action_selection_readme_matched_random_diagnostic, docs_limitations_matched_random_not_causal, docs_guide_using_the_model_comparison_types [INFERRED 0.85]
- **Readouts built on top of their conventional baseline** — docs_guide_using_the_model_analog_day_baseline, docs_guide_using_the_model_gradient_boosted_correction, docs_guide_using_the_model_fixed_logistic_baseline, docs_guide_using_the_model_rank_blend_reassignment, docs_guide_using_the_model_comparison_types [EXTRACTED 1.00]
- **First-release gates that a clean reference release must pass** — docs_reproducibility_gate_strict_model_load, docs_reproducibility_gate_demand_recovery_wape, docs_reproducibility_gate_sla_risk_auprc, docs_reproducibility_gate_action_selection_regret, docs_reproducibility_gate_repository_boundary [EXTRACTED 1.00]
- **Public CI pipeline that runs without private assets** — github_workflows_ci_checkout, github_workflows_ci_setup_uv, github_workflows_ci_uv_sync_all_extras, github_workflows_ci_ruff_check, github_workflows_ci_pytest_public_selection [EXTRACTED 1.00]
- **Release trust chain: pinned revision, manifest, checksums, verified loading** — provenance_immutable_hugging_face_revision, provenance_release_manifest, docs_reproducibility_checksums_sha256, docs_guide_getting_started_loader_guarantees, scripts_verify_model_release, src_supplychain_jepa_loading_load_pretrained [INFERRED 0.85]
- **Six weighted terms of the Dynamics objective** — docs_guide_dynamics_learning_future_prediction_term, docs_guide_dynamics_learning_paired_difference_comparison, docs_guide_dynamics_learning_control_margin_term, docs_guide_dynamics_learning_shuffled_future_term, docs_guide_dynamics_learning_two_step_consistency_term, docs_guide_dynamics_learning_branch_contrast_term [EXTRACTED 1.00]
- **One-branch Dynamics training flow (Figure 2)** — docs_guide_dynamics_learning_frozen_state_encoder, docs_guide_dynamics_learning_history_summary_z, docs_guide_dynamics_learning_order_plan, docs_guide_dynamics_learning_prediction_horizon, docs_guide_dynamics_learning_dynamics_predictor, docs_guide_dynamics_learning_target_representation, docs_guide_dynamics_learning_smooth_l1_error [EXTRACTED 1.00]
- **Decision-time plan evaluation flow (Figure 5)** — docs_guide_dynamics_learning_frozen_state_encoder, docs_guide_dynamics_learning_dynamics_predictor, docs_guide_dynamics_learning_task_readout, docs_guide_dynamics_learning_decision_rule, docs_guide_dynamics_learning_candidate_plan_evaluation [EXTRACTED 1.00]
- **Records-to-decision chain** — readme_source_adapters, readme_canonical_episodes, readme_state_pretraining, readme_dynamics_post_training, readme_pinned_model_release, readme_task_readout [EXTRACTED 1.00]
- **Three canonical applications and their metrics** — readme_stockout_demand_recovery, readme_sla_risk_ranking, readme_plan_selection, readme_wape, readme_auprc, readme_regret [EXTRACTED 1.00]
- **Conceptual reading route through the four guides** — docs_guide_data_to_episodes, docs_guide_state_learning, docs_guide_dynamics_learning, docs_guide_using_the_model [EXTRACTED 1.00]

## Communities (69 total, 3 thin omitted)

### Community 0 - "Episode tensorizer, tokens, and masks"
Cohesion: 0.07
Nodes (46): _base_features(), _canonical_episode_ids(), _canonical_json(), _CanonicalEpisodeIds, _canonicalize_identifier_features(), _categorical_text(), clone_episode_tensors(), collate_episode_tensors() (+38 more)

### Community 1 - "Recorded-data adapters"
Cohesion: 0.09
Nodes (51): Recorded operational histories (State only), AdapterRequest, Runtime inputs for one source conversion.      ``raw_root`` points to separately, BPI2019Adapter, Any, Adapter for factual BPI Challenge 2019 procurement event histories., Convert recorded purchase-order-item event traces., make_entity() (+43 more)

### Community 2 - "Training windows and examples"
Cohesion: 0.10
Nodes (51): ModelBundle contents, EpisodeTensorizer, Transform canonical episodes with one immutable fitted train state., MaskPlan, One objective mask and the corresponding leakage-safe context view., _build_dynamics_source_agnostic(), _dynamics_units(), Use the canonical builder while admitting any branch-valid source family. (+43 more)

### Community 3 - "Training runners and checkpoints"
Cohesion: 0.11
Nodes (51): Training checkpoint versus release bundle, New source families require source_sampling_weights entries, GradScaler, Module entry point for State and Dynamics training.  Run ``python -m supplychain, _action_sensitivity(), _add_common_arguments(), _amp_context(), _apply_model_overrides() (+43 more)

### Community 4 - "Dynamics learning concepts"
Cohesion: 0.05
Nodes (46): Figure: two order plans from one shared history, Figure: equal errors per plan, different plan effects, Action as a generator-set replenishment instruction, Decision-time candidate plan evaluation, Common-random-numbers design, Optimizer or decision rule, Dynamics predictor (trainable, includes order and horizon encoding), One Dynamics training example (history, plan, horizon, target) (+38 more)

### Community 5 - "Adapter framework, build, and CLI"
Cohesion: 0.08
Nodes (36): AdapterDescriptor, ABC, Any, Public contracts shared by every raw-to-canonical data adapter., Convert one external source into canonical SupplyChain-JEPA episodes., Return deterministic canonical records for ``request``., Lazy registry entry exposed without importing optional dependencies., SourceAdapter (+28 more)

### Community 6 - "Core model and training objectives"
Cohesion: 0.07
Nodes (42): Action controls (no-op, shuffled, wrong entity), Branch contrast term (weight 0.25), Control margin term (no-op and wrong-entity controls, margin 0.05), Shuffled future term (weight 0.50), Linear, One shared encoder chain with State and Dynamics predictors., Freeze migrated raw-feature and family tables that encode source identity., SupplyChainJEPACore (+34 more)

### Community 7 - "Applications guide and walkthroughs"
Cohesion: 0.07
Nodes (42): Figure: one product-store demand-recovery example, Figure: two plans evaluated on the same futures, Figure: one shipment at two reporting checkpoints, Frozen State encoder, adapted Dynamics copy, and scalar cost readout, Analog-day nearest-neighbor anchor, Frozen State encoder with latent retrieval and small correction model, Pooled WAPE across three fixed mask seeds, Stockout demand recovery (application 1) (+34 more)

### Community 8 - "Documentation index and source acquisition"
Cohesion: 0.11
Nodes (30): data/manifests identity and integrity records, Canonical state: masked mean of token latents, 192-token training context budget, Minimal readout recipe: encode once, fit, select, hold out, Observation view versus latent_state target view, OR/OM reading: belief state, imputation, forecasting, transition model, cost function, Latent outputs carry no business units and enforce no constraints, Smoke-corpus worked example: 4-unit versus 14-unit order (+22 more)

### Community 9 - "Tiny corpus, schema loading, and adapter tests"
Cohesion: 0.09
Nodes (38): Schema-shaped builders in supplychain_jepa.adapters.common, Canonical episode record skeleton, Canonical meaning rules: observability, status, absence, reporting, splits, pairing, Common random numbers pairing, Paired branch contract for dynamics_jepa records, Validate canonical JSONL against the schema before training, Deterministic world-level splits (about 90/5/5), Draft202012Validator (+30 more)

### Community 10 - "Predictors and token embedding"
Cohesion: 0.10
Nodes (20): dtype, build_local_target_window(), DynamicsPredictor, FactorizedTokenEmbedding, Any, Tensor, Action-conditioned latent predictor trained over frozen encoder states., Compress heterogeneous action magnitudes before linear projection.          Quan (+12 more)

### Community 11 - "Project overview and glossary"
Cohesion: 0.09
Nodes (39): Overview figure: from partial records to predicted futures, Episode, Glossary of terms used across the guides, Latent representation, Observation and target views, Paired branches, Seventeen semantic concept groups, Task adapter (+31 more)

### Community 12 - "SLA-risk evidence and claim boundaries"
Cohesion: 0.07
Nodes (37): Adjacent single-shift cohort (negative result), Architecture-matched random encoders (SLA risk), AUPRC on the confirmation cohort, Claim boundary of the three evaluations, Coherent reporting delay on the most time-consuming incoming leg, Controlled stockout-shaped masks on normal-availability days, Early SLA-risk ranking under delayed reports (application 2), Method-frozen joint route-and-topology shift cohort (+29 more)

### Community 13 - "Action-selection model backends"
Cohesion: 0.11
Nodes (13): ActionSelectionBackbone, ModelBundleBackbone, ABC, Module, Tensor, Frozen State and adaptable Dynamics backends for action selection., Adapter over the clean, verified :class:`supplychain_jepa.ModelBundle`., Interface that keeps State encoding frozen during downstream fitting. (+5 more)

### Community 14 - "ISOMORPH simulator adapter"
Cohesion: 0.15
Nodes (35): json_safe(), make_action(), make_edge_state(), observation_status(), Any, Convert NumPy and path values to deterministic finite JSON values., stable_digest(), _actions() (+27 more)

### Community 15 - "MilestoneFlow simulator adapter"
Cohesion: 0.13
Nodes (21): make_record_id(), _BranchPolicy, _episode_record(), _force_observation_challenges(), generate_milestone_flow_records(), MilestoneFlowAdapter, _MilestoneWorld, _physical_plan() (+13 more)

### Community 16 - "MABIM adapter"
Cohesion: 0.13
Nodes (33): DatetimeIndex, _actions_for_step(), _checkout_commit(), _checkout_root(), _exogenous_digest(), generate_mabim_records(), _install_gym_compatibility(), _install_pyecharts_compatibility() (+25 more)

### Community 17 - "State learning: masks, windows, examples"
Cohesion: 0.12
Nodes (34): Figure: example 1, the hidden present, Figure: example 2, the near future, Figure: from records to latent summaries, Causal cut, Context budget, Contiguous-time mask, Demand censoring by stockouts, Dynamics predictor (+26 more)

### Community 18 - "Public inference API and model bundle"
Cohesion: 0.12
Nodes (27): Action-conditioned question: what changes if we act differently?, Observational future question: what comes next?, Hidden-current question: what is hidden now?, Shape-explicit inference API, State predictor, ActionComparison, compare_actions(), encode_state() (+19 more)

### Community 19 - "Reproducibility gates and provenance"
Cohesion: 0.09
Nodes (25): Getting the code and environment: clone, uv, optional login, Report baseline and matched-random comparisons side by side, Code on public GitHub, weights on public Hugging Face, Optional Hugging Face login with hf auth login or HF_TOKEN, Matching reference numbers within metric tolerances, Three reproduction tiers, Tier 3: external inputs, First-release gates (+17 more)

### Community 20 - "Demand-recovery data and baselines"
Cohesion: 0.19
Nodes (26): datetime64, build_example_sets(), build_mask_bank(), build_priors(), build_store_split(), _candidate_examples(), ExampleSet, _fallback_profile() (+18 more)

### Community 21 - "Plan-selection application and results"
Cohesion: 0.08
Nodes (25): Action regret over five disjoint world splits, Common random numbers with a no-further-orders continuation, Matched-random attribution diagnostic (plan selection), Public-history ExtraTrees baseline, Quantity-matched regular and split replenishment plans, Replenishment-plan selection (application 3), Adapted Dynamics with a zero-initialized scalar cost head (action selection), Latent retrieval plus gradient-boosted correction (demand recovery) (+17 more)

### Community 22 - "SLA-risk tensor adapter"
Cohesion: 0.12
Nodes (23): cargo_state_batch(), _categorical_id(), DurationScaler, encode_current_state(), _event_rank(), _event_type_name(), _leg_index(), _masked_mean() (+15 more)

### Community 23 - "SupplyNetPy simulator adapter"
Cohesion: 0.14
Nodes (26): ModuleType, _actions(), _build_network(), _checkout_revision(), _delta(), generate_supplynetpy_records(), _load_supplynetpy(), _observation() (+18 more)

### Community 24 - "Demand-recovery representation"
Cohesion: 0.15
Nodes (24): _categorical_id(), _empty_jepa_batch(), encode_examples(), _field(), _fresh_batch(), load_state_representation(), _metadata_candidate_id(), _metadata_id() (+16 more)

### Community 25 - "Loss functions and training tests"
Cohesion: 0.12
Nodes (25): Future prediction term (weight 1.00), Smooth-L1 (Huber-type) prediction error, collapse_diagnostics(), _expanded_mask(), _flatten_valid(), inverse_action_loss(), masked_latent_loss(), paired_difference_loss() (+17 more)

### Community 26 - "Demand-recovery pipeline and CLI"
Cohesion: 0.13
Nodes (22): build_parser(), main(), ArgumentParser, Command-line entry point for demand recovery., file_sha256(), Path, Pinned data and headline-result contracts for demand recovery., Fail closed unless both canonical external parquet files are present. (+14 more)

### Community 27 - "SLA-risk evaluation and rank blending"
Cohesion: 0.14
Nodes (24): aggregate_random_controls(), blend_ranking(), _cache_path(), Cohorts, evaluate(), load_cohorts(), main(), parse_args() (+16 more)

### Community 28 - "Action-selection evaluation loop"
Cohesion: 0.19
Nodes (24): _cost_matrix(), _extra_trees_predictions(), _fit_extra_trees(), _history_matrix(), _model_predictions(), _paired_rows(), PlanQualityModel, _plans() (+16 more)

### Community 29 - "SLA-risk delayed-report protocol"
Cohesion: 0.18
Nodes (24): add_targets(), apply_critical_prefix_delay(), assert_no_hidden_value_leakage(), build_prefix_examples(), critical_incoming_leg(), _event_specs(), EventSpec, feature_columns() (+16 more)

### Community 30 - "References and data sources"
Cohesion: 0.14
Nodes (25): Typed entity-time and event tokens, BPI Challenge 2019 dataset, DINO-WM (Zhou et al.), I-JEPA (Assran et al., 2023), I-JEPA official implementation, ISOMORPH simulator, LaDe dataset, LaT-PFN reference implementation (+17 more)

### Community 31 - "Training smoke chain and CI"
Cohesion: 0.10
Nodes (23): Complete Dynamics objective (six weighted terms), Action-conditioned Dynamics post-training, State predictor (unchanged, not called during Dynamics), CPU suffices for loading, tests, and canonical evaluations; pretraining needs an accelerator, Tier 1: no private assets, Tensorizer and State weights unchanged during frozen Dynamics training, Gate: no external dataset, pretrained weight, fitted binary, credential, or machine-specific path tracked, Small-model override for the smoke chain (+15 more)

### Community 32 - "Release loading and verification"
Cohesion: 0.14
Nodes (21): load_pretrained guarantees: declared files, SHA-256 checks, strict load, parameter check, frozen modules, Offline, local-directory, and explicit-token loading options, Keep verification on; never compare against a moving branch, Verify-then-load workflow, CHECKSUMS.sha256 validated before model construction, main(), Loader verifies release file checksums before use, _json() (+13 more)

### Community 33 - "Action-selection world reconstruction"
Cohesion: 0.16
Nodes (21): Order, PairedFutureCost, Typed contracts for the clean action-selection reconstruction., Configuration for the self-contained Pool-B6-style reconstruction., ReconstructionConfig, WorldSplit, Clean action-selection reconstruction for SupplyChain-JEPA., build_world_splits() (+13 more)

### Community 34 - "Stockpyl adapter"
Cohesion: 0.19
Nodes (23): _build_network(), _checkout_commit(), _executed_quantity(), generate_stockpyl_records(), _load_stockpyl(), _override(), Any, Path (+15 more)

### Community 35 - "Structured mask sampling"
Cohesion: 0.29
Nodes (9): EpisodeTensors, One tensorized episode with separate context and latent target sequences., Any, Random, Tensor, Deterministic structured objective masks for State JEPA., Sample deterministic supply-chain structured masks.      The sampler never mutat, _seed_from() (+1 more)

### Community 36 - "Demand-recovery correction model"
Cohesion: 0.21
Nodes (22): build_interval_features(), build_latent_retrieval_features(), _candidate_configurations(), _fill_nonfinite(), fit_latent_adapter(), _fit_model(), interval_metrics(), interval_prediction() (+14 more)

### Community 37 - "Seventeen concept classes"
Cohesion: 0.11
Nodes (22): action_order concept class, backlog_shortage concept class, calendar_time concept class, capacity_queue concept class, cost_price concept class, Event bookkeeping fields (__event_time, __report_time, __observed), event_progress concept class, fulfillment_service concept class (+14 more)

### Community 38 - "Paired-branch Dynamics corpus"
Cohesion: 0.13
Nodes (21): Dynamics corpus (paired simulator branches), Observational sources are not counterfactual action labels, Dynamics pairing requirement, MABIM source mapping, Stockpyl source mapping, Stockpyl State view: complete observations, masked in training, Paired branch records share initial state and exogenous stream, Structured actions (+13 more)

### Community 39 - "State pretraining sources"
Cohesion: 0.13
Nodes (21): State pretraining corpus, BPI Challenge 2019 source mapping, ISOMORPH source mapping, LaDe source mapping, MilestoneFlow source mapping, Olist source mapping, Planned-versus-actual routes source mapping, SupplyNetPy source mapping (+13 more)

### Community 40 - "Action-selection protocol and tests"
Cohesion: 0.15
Nodes (18): Small deterministic backend used by unit and synthetic smoke tests., TinyBackbone, build_parser(), main(), Any, ArgumentParser, Path, Summarize the canonical paired-plan protocol from compact run outputs. (+10 more)

### Community 41 - "Canonical episode JSON Schema"
Cohesion: 0.25
Nodes (20): action, branch, edge, edgeState, entityState, episode, event, features (+12 more)

### Community 42 - "Training reference"
Cohesion: 0.20
Nodes (16): Training refuses records naming downstream-evaluation sources, Route B: pretrain your own State-to-Dynamics chain, Branch contrast loss, Canonical episode records, Checkpoint promotion review, Composite Dynamics objective, Control margin loss, Dynamics checkpoint (+8 more)

### Community 43 - "Model configuration"
Cohesion: 0.17
Nodes (7): Embedding, ModelConfig, Reject a tensorizer/model capacity mismatch before model execution., Construct a model from a model configuration mapping style mappings.          Te, Set the non-persistent raw-schema residual strength for this encoder., Lightweight role/horizon-conditioned latent target predictor., StatePredictor

### Community 44 - "Getting started: encoding and plan comparison"
Cohesion: 0.12
Nodes (15): 8-field action vector layout, tensorizer.json authoritative; normalization.json a projection, Comparing candidate plans with compare_actions, Cut records at the estimation or decision time, Dynamics post-training on paired Stockpyl and MABIM branches, Dynamics question: how the situation evolves under an action sequence, EMA target encoder, Feature coverage check against the released 309-feature interface (+7 more)

### Community 45 - "State encoders, predictor, and EMA"
Cohesion: 0.21
Nodes (17): AdamW gradient steps, Context encoder, EMA weight update, Encoder summarizes, predictor completes, Final alignment, Query, Simple exponential smoothing, State predictor (+9 more)

### Community 46 - "Canonical episode boundary"
Cohesion: 0.17
Nodes (16): Figure: anatomy of a canonical episode, Adapter CLI: list, convert, build, Canonical boundary (what every adapter emits), latent_state semantics: factual view versus simulator internal state, Pinned simulator revisions, Packaged JSON Schema (authoritative raw-record shape), Raw canonical episode, Raw episode schema and tensor-interface schema are separate contracts (+8 more)

### Community 47 - "Architecture reference"
Cohesion: 0.32
Nodes (13): Action-conditioned Dynamics predictor, Auxiliary State heads, Dynamics objective terms, EMA target encoder, Online context encoder, Paired simulator branches, Released core: 73,629,696 parameters, Shared latent coordinate system (+5 more)

### Community 48 - "SLA-risk baseline and calibration"
Cohesion: 0.26
Nodes (9): Baseline, _domain_weights(), fit_baseline(), _metric(), PrefixCalibrator, DataFrame, ndarray, Series (+1 more)

### Community 49 - "Tensor interface contract"
Cohesion: 0.19
Nodes (13): Canonical episode JSONL, Legacy schema_version identifier, Sparse feature maps; no synthesized measurements, Train-only tensorizer, Released Hugging Face bundle (tensor interface), normalization.json (deterministic projection), Tensorized batch, tensorizer.json (authoritative fitted interface) (+5 more)

### Community 50 - "Tokens and training windows"
Cohesion: 0.23
Nodes (12): Figure: three training questions cut from one episode, 192-token context cap with mandatory tokens, Context tokens, Dynamics question, Future (action-omitted) question, Hidden-current question, Step outcomes (kept, not tokenized), Five structured masks (+4 more)

### Community 51 - "Release verification and parity tests"
Cohesion: 0.21
Nodes (10): Tier 2: downloaded weights, Gate: model tensors load strictly and representative outputs match the frozen release, Pinned model release (revision 493786fd), build_parser(), main(), ArgumentParser, Load and verify the immutable pretrained model release., _released_module() (+2 more)

### Community 52 - "State loss terms"
Cohesion: 0.21
Nodes (12): Concept-slot predictor, Grounding head, Latent loss, Numerical grounding term, Pooled target term, Semantic concept-slot term, Composite State objective, Disabled VICReg and SIGReg penalties (+4 more)

### Community 53 - "Action-selection CLI and config"
Cohesion: 0.20
Nodes (8): Path, EvaluationConfig, Any, Fixed downstream adaptation and evaluation recipe., build_parser(), main(), ArgumentParser, Command-line entry point for the bounded action-selection reconstruction.

### Community 54 - "Demand-recovery tests"
Cohesion: 0.23
Nodes (10): MonkeyPatch, Path, test_analog_day_does_not_read_masked_target_values(), test_blend_selection_can_preserve_the_analog_anchor(), test_controlled_masks_are_deterministic_and_nonempty(), test_dataset_hash_contract_fails_closed(), test_interval_metrics_use_only_controlled_mask(), _tiny_data() (+2 more)

### Community 55 - "Downstream separation and leakage rules"
Cohesion: 0.20
Nodes (11): Cargo2000, FreshRetailNet-50K, gym-invmgmt and overlapping environment lineages, Hard downstream separation, Held-out evaluation partitions, Split and leakage rules, Leakage rules, World-level hash splits (+3 more)

### Community 56 - "SLA-risk transition prototype"
Cohesion: 0.24
Nodes (10): Matched random encoder control, DeltaPrototype, _encoder_for_seed(), fit_prototype(), stage_delta(), StageDelta, Batch-first Transformer encoder over factorized supply-chain tokens., SequenceEncoder (+2 more)

### Community 57 - "Core API tests"
Cohesion: 0.33
Nodes (9): _batch(), _bundle(), MonkeyPatch, Path, Tensor, test_compare_actions_returns_latent_contrasts_not_fake_preferences(), test_hub_snapshot_download_includes_every_checksum_declared_file(), test_release_verification_accepts_content_addressed_snapshot_symlinks() (+1 more)

### Community 58 - "Pooled belief state and canonical head"
Cohesion: 0.29
Nodes (8): Belief state, Belief state in a POMDP, Canonical-state term, Latent representation, Pooled state, Shuffle control diagnostic, State pool predictor, Canonical-state prediction

### Community 59 - "Repository boundary check"
Cohesion: 0.39
Nodes (5): check_repository(), main(), Path, Fail closed when release-forbidden files or environment metadata enter Git., repository_files()

### Community 60 - "Concept mapping code"
Cohesion: 0.38
Nodes (6): feature_concept(), feature_concept_ids(), _has(), Canonical supply-chain concepts shared across heterogeneous source schemas.  The, Return concept IDs aligned exactly with a fitted feature vocabulary., Map one canonical feature name to a stable operational concept.      Rules are o

### Community 61 - "M5 walkthrough and censored sales"
Cohesion: 0.60
Nodes (5): M5 source mapping, demand_sales concept class, M5 (item-store daily sales), M5 walkthrough, Recorded sales are not demand

### Community 62 - "Dynamics training settings"
Cohesion: 0.40
Nodes (5): Bounded training smoke recipe, Continuous State-to-Dynamics weight chain, Dynamics AdamW schedule, Dynamics predictor P_dyn, Structured action representation

## Ambiguous Edges - Review These
- `Disabled VICReg and SIGReg penalties` → `Stable World Models (Galilai Group)`  [AMBIGUOUS]
  docs/TRAINING.md · relation: conceptually_related_to

## Knowledge Gaps
- **46 isolated node(s):** `supplychain-jepa`, `simulator`, `objectiveEligibility`, `Legacy schema_version identifier`, `Tiny corpus generator (examples/training/generate_tiny_corpus.py)` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.


## Integration notes
- 7 semantic fragments were merged with the AST pass: five documentation lanes, the reader-entry lane, and an integration fragment with one node per illustration, file nodes for JSON configurations and results that the AST pass leaves empty, and EXTRACTED edges for explicit links and file paths named in the documents.
- Each repository document is one node for its file, and figure nodes from the lanes were merged into one node per SVG file.
- The episode JSON Schema is kept at definition level: the file node plus 20 `$defs` definitions; 284 field-level schema nodes were folded into them and 89 required-name placeholders were dropped.
- 31 imports of this repository's own packages (`supplychain_jepa.*` in the src layout, and the evaluation packages) were resolved to their modules, and 23 schema `$ref` targets to their definitions.
- 347 AST edges point to standard-library or third-party modules such as `torch` and `pathlib`, which have no node in this repository, and are left out of the graph.
- Files that no document or code edge references appear as single-node communities: `dynamics-smoke.json`, `pretraining_sources.json`, `pyproject.toml`, `state-smoke.json`.
- `graph.json` is written without indentation to stay within the repository's 2 MiB file limit; its content is unchanged.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Disabled VICReg and SIGReg penalties` and `Stable World Models (Galilai Group)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Figure: three training questions cut from one episode` connect `Tokens and training windows` to `Documentation index and source acquisition`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `State JEPA objective` connect `State pretraining sources` to `Tokens and training windows`, `Seventeen concept classes`, `Paired-branch Dynamics corpus`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `Figure: anatomy of a canonical episode` connect `Canonical episode boundary` to `Documentation index and source acquisition`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 122 inferred relationships involving `ValueError` (e.g. with `.__init__()` and `.expected_cost()`) actually correct?**
  _`ValueError` has 122 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `AdapterRequest` (e.g. with `BPI2019Adapter` and `ISOMORPHAdapter`) actually correct?**
  _`AdapterRequest` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `SupplyChainJEPACore` (e.g. with `ModelBundle` and `Released core (73,629,696 parameters)`) actually correct?**
  _`SupplyChainJEPACore` has 5 INFERRED edges - model-reasoned connections that need verification._
