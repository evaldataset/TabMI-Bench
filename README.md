# TabMI-Bench: A Protocol Benchmark for Mechanistic Interpretability of Tabular Foundation Models

Hook-based MI evaluation across **TabPFN v2/v2.5**, **TabICL v2**, **TabDPT**, **iLTM**, and **NAM** (out-of-family holdout), with 8 MI techniques and 4 controlled synthetic probes.

**5 models · 8 MI techniques · 4 function types · 5-seed core comparison · 3-seed holdouts · ED Track submission for NeurIPS 2026.**

## Key Findings

1. **Whole-layer clean activation patching is uninformative on tested ICL TFMs** due to deterministic cascading: replacing layer-L activations with their clean counterparts cascades unchanged through all downstream layers (recovery=1.000 flat). Corruption-based (noising) tracing is the informative alternative.
2. **Three descriptive reference computation profiles** as calibration baselines: staged (TabPFN), distributed (TabICL, TabDPT), preprocessing-dominant (iLTM). 5-seed validated on 3 core models across 4 function types; in-family TabDPT 3-seed holdout and out-of-family NAM 5-seed holdout confirm the categorical separation. Leave-one-function-out (LOFO) preserves the primary-endpoint ratio TabPFN/TabICL >29× in every holdout.
3. **Evidence-coded MI applicability matrix**: 8 techniques × 4 architectures with Supported/Limited/Not-established labels and seed-count superscripts.

## Canonical Sources

Use the following order when documents disagree:

1. `frozen_artifacts/*.json` — bundled aggregated JSONs that drive every numbered table and figure. Used by `make reproduce-paper-frozen` and consumable without GPU/NAS access.
2. `results/rd5_fullscale/aggregated/aggregated_results.json`, `results/phase7/tabdpt_probing/aggregated_3seed.json`, `results/phase7/tabdpt_causal/aggregated_3seed.json`, `results/rd6/tabpfn25_fullscale/aggregated/*.json`, `results/phase8a/nonlinear_probing/lofo_primary_endpoint.json`, `results/phase9_nam_holdout/*.json` — full per-experiment aggregated artifacts.
3. `BENCHMARK_CARD.md` and `croissant.json` — datasheet and machine-readable metadata.

Notes on storage:

- `results/` and `logs/` are symlinks to NAS-backed storage on the build machine. External users should rely on `frozen_artifacts/` and `make reproduce-paper-frozen`.
- All summary numerics trace to JSON files in `frozen_artifacts/` or to the per-experiment aggregates listed above.

## Phase Overview

| Phase | Focus | Methods | Key Discovery |
|-------|-------|---------|---------------|
| **1** | Correlation analysis | Probing, Logit Lens, Attention | Layer 5-8 encodes α, β, a·b in TabPFN |
| **2** | Causal verification | Patching, Ablation, CKA | Layer 5-8 is causally critical |
| **3** | Model manipulation | Steering, SAE | Intentional prediction control + feature decomposition |
| **4** | Cross-architecture | 7 techniques × 3 models | Three distinct computation strategy profiles |
| **5** | Statistical validation | 5-seed full-scale (168 min) | All patterns reproduced across seeds |
| **6** | Strengthening + Expansion | Robust steering, classification, attention | Steering variance 4× reduced; classification ≠ regression ranking |

### Phase 1: Correlation-Based Analysis (3 milestones)

1. **M1 — Reproduction**: 4 experiments from the base paper (coefficient probing, intermediary probing, logit lens, copy mechanism)
2. **RD-4 — Real-world Extension**: Semi-synthetic perturbation sweeps + 5 real-world benchmark datasets
3. **RD-8 — Attention Analysis**: Sample/Feature attention visualization, head specialization, feature interaction, Q-K embedding

### Phase 2: Causal Verification (3 milestones)

4. **RD-1 — Activation Patching**: Causal verification that Layer 5-8 encodings drive predictions
5. **RD-3 — Vector Ablation**: Information removal experiments showing direction-specific degradation
6. **RD-7 — Classification Analysis**: Decision boundary probing + representation geometry (t-SNE, CKA)

### Phase 3: Model Manipulation (2 milestones)

7. **RD-2 — Steering Vectors**: Contrastive activation addition for intentional prediction manipulation
8. **RD-6 — Sparse Autoencoder**: Polysemantic neuron decomposition into monosemantic features

### Phase 4: Cross-Architecture Comparison (1 milestone)

9. **RD-5 — Multi-TFM Comparison**: TabPFN vs TabICL vs iLTM — coefficient probing, intermediary probing, copy mechanism, CKA, activation patching, steering vectors, SAE across three architecturally different tabular foundation models

### Phase 5: Statistical Validation (full-scale reproduction)

10. **Full-Scale Run**: 9 experiments × 5 seeds = 45 runs (44 pass), 168 min total
11. **Deterministic Cascading Discovery**: Standard activation patching is uninformative for ICL-based TFMs → replaced with noising-based causal tracing

### Phase 6: Result Strengthening + Research Expansion (11 milestones)

12. **M23 — Robust Steering**: Multi-pair averaged steering vectors (3 contrastive pairs) — variance reduced ~4× (TabPFN L8 |r|=0.87±0.08)
13. **M24 — Improved SAE**: 16× expansion + JumpReLU — discovered quick-run overfitting (0.70 → 0.24 full-scale)
14. **M25 — Real-World Expansion**: 10 datasets — TabPFN 5 wins, TabICL 4 wins, no single best model
15. **M26 — Classification Probing**: iLTM 100%±0% on breast cancer (L0); classification ranking ≠ regression ranking
16. **M27 — Attention Comparison**: TabPFN entropy=3.99 (selective) vs TabICL entropy=4.57 (uniform)
17. **M28 — TabPFN 2.5**: ✅ v2 vs v2.5 comparison — v2.5 shifts to distributed computation (logit lens R²>0.5 at L2 vs v2's L8)
18. **M29 — Aggregation**: 3-seed full-scale (15/15 pass, 49.9 min), 5 error-bar plots
19. **M30 — TabICL L6 Anomaly Deep Dive**: 3-seed × 5-pair diagnostics — L6 remains highest-variance / lowest-strength layer
20. **M31 — SAE TopK Alternative**: TopK activation integrated — higher sparsity and improved max |r_alpha| for TabPFN/TabICL
21. **M32 — TabICL L5-L6 Zoom**: layer 4-7 high-resolution profiling — L5 emerges as anomaly center in zoomed view
22. **M33 — TopK 3-seed Aggregation**: error-bar aggregation confirms sparsity gains are robust, correlation gains model-dependent

### Phase 7: Paper Strengthening Extensions (16 tasks)

23. **M7A-T1 — Real-World Causal (11 datasets)**: Noising-based causal tracing on 11 OpenML benchmarks — TabPFN L11 (82%), TabICL L0 (100%), iLTM L2 (91%)
24. **M7A-T2 — Multi-Seed Real-World Causal**: 3-seed validation confirms consistency — TabPFN L11 (88%), TabICL L0 (100%), iLTM L2 (97%)
25. **M7A-T3 — L11 Peak Investigation**: Feature count sweep (d=2,8,11) — TabPFN shifts from L5 (d=2) to L11 (d=8+)
26. **M7A-T4 — TabPFN v2.5 Causal**: Real-world causal tracing — v2.5 peaks at L17 (deeper than v2's L11)
27. **M7A-T5 — Real-World Causal Analysis Note**: `research_directions/phase7_realworld_causal.md`
28. **M7B-T1 — SAE Scaling (128×/256×)**: TabPFN capacity-limited (0.26→0.56), TabICL fundamentally polysemantic (flat 0.44→0.46)
29. **M7B-T2 — SAE Scaling 3-Seed**: TabPFN high variance (σ=0.145), TabICL stable (σ=0.041)
30. **M7B-T3 — SAE Diagnostics**: Dead features, L0 sparsity, GPU memory tracked across expansion factors
31. **M7B-T4 — SAE Scaling Analysis Note**: `research_directions/phase7_sae_scaling.md`
32. **M7C-T1 — TabDPT Hooker**: `src/hooks/tabdpt_hooker.py` — 16 layers, 768d, TransformerEncoder
33. **M7C-T2 — TabDPT Probing**: α peak L1 R²=0.998, a·b flat high (>0.99), copy L1 R²>0.95
34. **M7C-T3 — TabDPT Causal + CKA**: L0 peak causal, uniform CKA (mean=0.86) — distributed strategy
35. **M7C-T4 — TabDPT SAE + USAE**: 16× TopK SAE L1 max_r=0.49, L8 max_r=0.40; cross-model with TabPFN/TabICL
36. **M7C-T5 — TabDPT Analysis Note**: `research_directions/phase7_tabdpt.md`
37. **M7D-T1 — Documentation Update**: TabDPT (5th model), SAE scaling, real-world causal milestones documented
38. **M7D-T2 — README/Makefile Update**: Phase 7 documented

## Prerequisites

- **Python** 3.12+
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Model checkpoint**: `tabpfn-v2-regressor.ckpt` in project root

## Environment Setup

```bash
# Create virtual environment
uv venv .venv

# Install dependencies
uv pip install -r requirements.txt

# Verify installation
.venv/bin/python -c "from tabpfn import TabPFNRegressor; print('OK')"
```

Or use the Makefile shortcut:

```bash
make setup
```

## Model Files

Model checkpoints must be placed at the project root:

```
TFMI/tabpfn-v2-regressor.ckpt                  # TabPFN v2 (12L, required)
TFMI/tabpfn-v2.5-regressor-v2.5_default.ckpt   # TabPFN v2.5 (18L, for M28)
```

v2 is loaded by most experiments via `TabPFNRegressor(device="cpu", model_path="tabpfn-v2-regressor.ckpt")`.
v2.5 is used in `rd6_tabpfn25_comparison.py` and requires HuggingFace access to `Prior-Labs/tabpfn_2_5`.

## Quick Start

Run Phase 1 experiments:

```bash
make run_all
```

Run full-scale multi-seed validation:

```bash
# Phase 5: 5-seed × 9 experiments
PYTHONUNBUFFERED=1 .venv/bin/python experiments/run_fullscale.py --include-realworld

# Phase 6: default 5-seed × 5 experiments
PYTHONUNBUFFERED=1 .venv/bin/python experiments/run_fullscale_phase6.py
```

Aggregate results:

```bash
.venv/bin/python experiments/aggregate_results.py
.venv/bin/python experiments/aggregate_phase6.py
```

Run individual experiments (quick-run mode):

```bash
# Phase 1
make run_exp1   # Coefficient Probing
make run_exp2   # Intermediary Probing
make run_exp3   # Answer Probing + Logit Lens
make run_exp4   # Copy Mechanism
make run_rd4a   # Semi-synthetic data
make run_rd4b   # Real-world benchmarks
make run_rd8_viz   # Attention visualization
make run_rd8_copy  # Copy mechanism attention
make run_rd8_head  # Head specialization
make run_rd8_feat  # Feature interaction
make run_rd8_qk    # Q-K embedding

# Phase 4: Multi-model comparison
.venv/bin/python experiments/rd5_coefficient_probing.py
.venv/bin/python experiments/rd5_intermediary_probing.py
.venv/bin/python experiments/rd5_copy_mechanism.py
.venv/bin/python experiments/rd5_cka_comparison.py
.venv/bin/python experiments/rd5_patching_comparison.py
.venv/bin/python experiments/rd5_steering_comparison.py
.venv/bin/python experiments/rd5_sae_comparison.py

# Phase 6: Result strengthening
.venv/bin/python experiments/rd6_robust_steering.py
.venv/bin/python experiments/rd6_improved_sae.py
.venv/bin/python experiments/rd6_realworld_expanded.py
.venv/bin/python experiments/rd6_classification_probing.py
.venv/bin/python experiments/rd6_attention_comparison.py
.venv/bin/python experiments/rd6_tabicl_l6_anomaly.py
.venv/bin/python experiments/rd6_sae_topk_comparison.py
.venv/bin/python experiments/rd6_tabicl_l5l6_zoom.py
.venv/bin/python experiments/aggregate_sae_topk_fullscale.py
```

See all Makefile targets:

```bash
make help
```

## Directory Structure

```
TFMI/
├── README.md                              # This file
├── Makefile                               # Reproducibility targets
├── requirements.txt                       # Python dependencies
├── tabpfn-v2-regressor.ckpt              # TabPFN v2 model checkpoint
│
├── config/                                # Experiment hyperparameters (YAML)
│   ├── exp1_coefficient.yaml             # Phase 1
│   ├── exp2_intermediary.yaml
│   ├── exp3_answer_logitlens.yaml
│   ├── exp4_copy_mechanism.yaml
│   ├── rd4_phase4a.yaml
│   ├── rd4_phase4b.yaml
│   ├── rd8_attention.yaml
│   ├── rd1_activation.yaml               # Phase 2
│   ├── rd3_ablation.yaml
│   ├── rd7_classification.yaml
│   ├── rd2_steering.yaml                 # Phase 3
│   └── rd6_sae.yaml
│
├── src/                                   # Reusable library code
│   ├── __init__.py
│   ├── hooks/
│   │   ├── tabpfn_hooker.py              # TabPFN 12-layer activation extraction
│   │   ├── tabicl_hooker.py              # TabICL 12-block ICL activation extraction
│   │   ├── iltm_hooker.py                # iLTM 3-layer MLP activation extraction
│   │   ├── attention_extractor.py        # Attention weight extraction
│   │   ├── activation_patcher.py         # TabPFN activation patching
│   │   ├── tabicl_patcher.py             # TabICL activation patching
│   │   ├── steering_vector.py            # TabPFN steering vectors
│   │   └── tabicl_steering.py            # TabICL steering vectors
│   ├── probing/
│   │   ├── linear_probe.py               # Linear/MLP probe framework
│   │   └── real_world_targets.py         # Probing targets
│   ├── data/
│   │   ├── synthetic_generator.py        # Synthetic data generation
│   │   ├── real_world_datasets.py        # 16 dataset loaders (10 used in experiments)
│   │   └── classification_generator.py   # Binary classification
│   ├── sae/
│   │   ├── __init__.py
│   │   └── sparse_autoencoder.py         # SAE with ReLU + JumpReLU activation
│   └── visualization/
│       └── plots.py                      # Publication-quality plotting
│
├── experiments/                           # Experiment scripts (entry points)
│   │
│   │  # Phase 1: Base Paper Reproduction
│   ├── exp1_coefficient_probing.py
│   ├── exp2_intermediary_probing.py
│   ├── exp3_answer_probing_logit_lens.py
│   ├── exp4_copy_mechanism.py
│   ├── rd4_phase4a_semisynthetic.py
│   ├── rd4_phase4b_realworld.py
│   ├── rd8_attention_visualization.py
│   ├── rd8_copy_mechanism_attention.py
│   ├── rd8_head_specialization.py
│   ├── rd8_feature_interaction.py
│   ├── rd8_qk_embedding.py
│   │
│   │  # Phase 2: Causal Verification
│   ├── rd1_coefficient_patching.py
│   ├── rd1_layer_sweep.py
│   ├── rd1_intermediary_patching.py
│   ├── rd3_vector_ablation.py
│   ├── rd7_classification_probing.py
│   ├── rd7_representation_geometry.py
│   │
│   │  # Phase 3: Model Manipulation
│   ├── rd2_coefficient_steering.py
│   ├── rd2_layer_steering.py
│   ├── rd2_boundary_steering.py
│   ├── rd6_sae_training.py
│   ├── rd6_feature_analysis.py
│   ├── rd6_feature_ablation.py
│   │
│   │  # Phase 4: Multi-Model Comparison
│   ├── rd5_config.py                     # Shared config (env vars: QUICK_RUN, SEED)
│   ├── rd5_tabicl_smoke_test.py
│   ├── rd5_iltm_smoke_test.py
│   ├── rd5_model_comparison_baseline.py
│   ├── rd5_coefficient_probing.py
│   ├── rd5_intermediary_probing.py
│   ├── rd5_copy_mechanism.py
│   ├── rd5_cka_comparison.py
│   ├── rd5_patching_comparison.py
│   ├── rd5_steering_comparison.py
│   ├── rd5_sae_comparison.py
│   ├── rd5_hypothesis_summary.py
│   │
│   │  # Phase 5: Full-Scale Validation
│   ├── run_fullscale.py           # 5-seed runner (9 experiments)
│   ├── aggregate_results.py               # Result aggregation + plots
│   │
│   │  # Phase 6: Strengthening + Expansion
│   ├── rd6_robust_steering.py            # M23: Multi-pair averaged steering
│   ├── rd6_improved_sae.py               # M24: 16× expansion + JumpReLU
│   ├── rd6_realworld_expanded.py         # M25: 10-dataset expansion
│   ├── rd6_classification_probing.py     # M26: Classification probing
│   ├── rd6_attention_comparison.py       # M27: Cross-model attention
│   ├── rd6_tabpfn25_comparison.py       # M28: TabPFN v2 vs v2.5
│   ├── rd6_tabicl_l6_anomaly.py         # M30: TabICL L6 anomaly deep dive
│   ├── rd6_sae_topk_comparison.py       # M31: TopK SAE comparison
│   ├── rd6_tabicl_l5l6_zoom.py          # M32: TabICL L5-L6 high-resolution
│   ├── aggregate_sae_topk_fullscale.py  # M33: TopK 3-seed aggregation
│   ├── run_fullscale_phase6.py           # default 5-seed runner (5 experiments)
│   └── aggregate_phase6.py              # Result aggregation + plots
│
├── results/                              # Experiment outputs (symlinked to NAS-backed storage)
│   ├── exp1/ .. exp4/                    # Phase 1 results
│   ├── rd4_phase4a/, rd4_phase4b/        # Phase 1 results
│   ├── rd8/                              # Phase 1 results
│   ├── rd1/, rd3/, rd7/                  # Phase 2 results
│   ├── rd2/                              # Phase 3 results
│   ├── rd6/                              # Phase 3 results
│   ├── rd5_fullscale/                    # Phase 5 full-scale (5 seeds)
│   │   ├── seed_{42,123,456,789,1024}/
│   │   └── aggregated/                   # aggregated_results.json + plots
│   ├── rd6_fullscale/                    # Phase 6 full-scale outputs
│   │   ├── seed_*/
│   │   ├── aggregated/                   # aggregated_results.json + plots
│   │   ├── logs/                         # per-experiment logs
│   │   └── run_metadata.json             # archived runner metadata
│   └── rd6/tabpfn25_comparison/       # M28: v2 vs v2.5 (5 plots + results.json)
│
└── research_directions/                  # Research analysis notes (20 files)
    ├── rd0_reproduction_notes.md         # Phase 1: Base paper reproduction
    ├── rd4_phase4a_analysis.md           # Phase 1: Semi-synthetic
    ├── rd4_real_world_extension.md       # Phase 1: Real-world
    ├── rd8_attention_map_analysis.md     # Phase 1: Attention
    ├── phase1_comprehensive_report.md
    ├── rd1_activation_patching.md        # Phase 2: Patching
    ├── rd3_vector_ablation.md            # Phase 2: Ablation
    ├── rd7_classification_analysis.md    # Phase 2: Classification
    ├── phase2_comprehensive_report.md
    ├── rd2_steering_vectors.md           # Phase 3: Steering
    ├── rd6_sparse_autoencoder.md         # Phase 3: SAE
    ├── phase3_comprehensive_report.md
    ├── rd5_architecture_comparison.md    # Phase 4: Architecture
    ├── rd5_multi_tfm_comparison.md       # Phase 4: Multi-TFM
    ├── phase4_comprehensive_report.md
    ├── phase5_fullscale_results.md       # Phase 5: Full-scale
    ├── phase6_results.md                 # Phase 6: Raw results
    ├── phase6_comprehensive_report.md    # Phase 6: Comprehensive report
    └── contribution_assessment.md        # Research contribution assessment
```

## Experiment Descriptions

### Phase 1: Base Paper Reproduction

| Exp | Script | Description | Key Finding |
|-----|--------|-------------|-------------|
| **Exp 1** | `exp1_coefficient_probing.py` | Linear probe for coefficients α, β in z = αx + βy | α, β decodable from Layer 6 (R² > 0.7) |
| **Exp 2** | `exp2_intermediary_probing.py` | Linear probe for intermediary a·b in z = a·b + c | a·b peaks at Layer 5-8, then decays |
| **Exp 3** | `exp3_answer_probing_logit_lens.py` | Linear probe vs logit lens for final answer z | 3-layer gap: probe at Layer 5, logit lens at Layer 8 |
| **Exp 4** | `exp4_copy_mechanism.py` | Recover a, b, c, a·b from answer token activations | All inputs recoverable from Layer 5+ (copy mechanism) |

### Phase 1: Real-world & Attention Extensions

| Exp | Script | Description |
|-----|--------|-------------|
| **Phase 4A** | `rd4_phase4a_semisynthetic.py` | Non-linear functions with noise (σ ∈ {0..2}), missing values (0..30%), high dimensions |
| **Phase 4B** | `rd4_phase4b_realworld.py` | 5 standard datasets: California Housing, Adult Income, Credit Default, Wine Quality, Diabetes 130 |
| **Attention** | `rd8_attention_visualization.py` | Sample/Feature attention heatmaps + entropy curves across 12 layers |
| **Copy Attn** | `rd8_copy_mechanism_attention.py` | Correlation between input similarity and attention-proxy |
| **Head Spec** | `rd8_head_specialization.py` | Pairwise Jensen-Shannon divergence between attention heads per layer |
| **Feature Int** | `rd8_feature_interaction.py` | Tests whether feature attention reflects multiplicative structure |
| **Q-K Embed** | `rd8_qk_embedding.py` | PCA projection of Query/Key vectors into joint 2D space |

### Phase 2: Causal Verification

| Exp | Script | Description | Key Finding |
|-----|--------|-------------|-------------|
| **Patching** | `rd1_coefficient_patching.py` | Activation patching for coefficient direction | Layer 5-8 causally critical |
| **Layer Sweep** | `rd1_layer_sweep.py` | Comprehensive layer-by-layer sensitivity | Noising-based causal tracing |
| **Ablation** | `rd3_vector_ablation.py` | Direction-specific information removal | Targeted degradation in L5-8 |
| **Classification** | `rd7_classification_probing.py` | Decision boundary probing on binary tasks | L5-8 pattern holds for classification |
| **Geometry** | `rd7_representation_geometry.py` | t-SNE visualization + CKA similarity | CKA reveals 3-block structure |

### Phase 3: Model Manipulation

| Exp | Script | Description | Key Finding |
|-----|--------|-------------|-------------|
| **Coeff Steering** | `rd2_coefficient_steering.py` | Extract α-direction, steer at Layer 6, sweep λ | Predictions shift with λ |
| **Layer Steering** | `rd2_layer_steering.py` | Compare steering across layers 0-10 | Layer 6-8 most effective |
| **Boundary Steering** | `rd2_boundary_steering.py` | Steer classification decision boundary | Decision boundary moves |
| **SAE Training** | `rd6_sae_training.py` | Train SAE (4×/8×) on Layer 6 activations | R² > 0.99 reconstruction |
| **SAE Analysis** | `rd6_feature_analysis.py` | Correlate SAE features with α, β | Weak monosemantic features (r ≈ 0.24) |
| **SAE Ablation** | `rd6_feature_ablation.py` | SAE feature ablation comparison | Feature ablation → targeted degradation |

### Phase 4: Multi-Model Comparison

| Exp | Script | Description | Key Finding |
|-----|--------|-------------|-------------|
| **Coeff Probe** | `rd5_coefficient_probing.py` | α, β probing across 3 models | iLTM R²=0.92, TabPFN 0.70, TabICL 0.50 |
| **Intermediary** | `rd5_intermediary_probing.py` | a·b probing across 3 models | TabPFN U-shape, TabICL flat, iLTM decreasing |
| **Copy Mech** | `rd5_copy_mechanism.py` | Input recovery across 3 models | TabICL transparent, TabPFN staged |
| **CKA** | `rd5_cka_comparison.py` | Layer similarity comparison | TabPFN 3-block, TabICL uniform (0.99) |
| **Patching** | `rd5_patching_comparison.py` | Noising-based causal tracing | TabPFN L5 peak, TabICL L0 dominant |
| **Steering** | `rd5_steering_comparison.py` | Steering vector comparison | TabICL L8 r=0.997 |
| **SAE** | `rd5_sae_comparison.py` | SAE feature decomposition | All models: weak α correlation |

### Phase 6: Result Strengthening + Expansion

| Exp | Script | Description | Key Finding |
|-----|--------|-------------|-------------|
| **Robust Steering** | `rd6_robust_steering.py` | Multi-pair averaged steering (3 pairs × 5 layers) | TabPFN L8 \|r\|=0.87±0.08, std reduced ~4× |
| **Improved SAE** | `rd6_improved_sae.py` | 16× expansion + JumpReLU comparison | Full-scale α corr=0.24±0.07 (quick-run 0.70 was overfitting) |
| **Real-World** | `rd6_realworld_expanded.py` | 10 datasets, 3 models | TabPFN 5 wins, TabICL 4 wins |
| **Classification** | `rd6_classification_probing.py` | Cross-model classification probing | iLTM 100%±0% breast cancer; ranking ≠ regression |
| **Attention** | `rd6_attention_comparison.py` | TabPFN vs TabICL attention entropy | TabPFN 3.99 (selective) vs TabICL 4.57 (uniform) |
| **TabPFN 2.5** | `rd6_tabpfn25_comparison.py` | v2 vs v2.5 probing + CKA + logit lens | v2.5 distributed computation, logit lens R²>0.5 at L2 |
| **TabICL L6 Anomaly** | `rd6_tabicl_l6_anomaly.py` | 3-seed layer instability diagnostics | L6 anomaly replicated; L5-L6 unstable segment |
| **TabICL L5-L6 Zoom** | `rd6_tabicl_l5l6_zoom.py` | High-resolution layer-4..7 diagnostics | L5 lowest mean |r|, highest instability |
| **SAE TopK** | `rd6_sae_topk_comparison.py` | ReLU/JumpReLU/TopK comparative SAE | TopK strongly boosts sparsity |
| **SAE TopK 3-seed** | `aggregate_sae_topk_fullscale.py` | Multi-seed error-bar aggregation | TopK gains are sparsity-dominant, model-dependent on correlation |

## Results Summary

### Three Computation Strategies (Confirmed by 8 Methods)

| Aspect | TabPFN v2 | TabPFN v2.5 | TabICL | TabDPT | iLTM |
|--------|-----------|-------------|--------|--------|------|
| **Strategy** | Localized (L5-8) | **Semi-distributed (L3-14)** | Distributed (L0-11) | **Distributed (L0-15)** | Preprocessing-dominant |
| **Probing Profile** | U-shape (dip L2-3, peak L5-8) | Peak at L11 | Flat high (all layers R²>0.93) | Flat high (all layers R²>0.98) | L0 peak, rapid decay |
| **CKA** | 3-block structure | **Single massive block** | Uniform (CKA>0.98) | Uniform (CKA=0.86) | N/A (3 layers) |
| **Logit Lens** | Jump at L8 | **Gradual from L2** | — | — | — |
| **Steering Best** | L8 (\|r\|=0.87±0.08) | — | L11 (\|r\|=1.00±0.00) | — | N/A |
| **Attention** | Variable entropy (avg 3.99) | 3 heads, 64 d_k | Uniform high (avg 4.57) | — | N/A |
| **Classification** | Progressive (L3-4) | — | Immediate but unstable | — | Immediate and stable |
| **SAE α corr** | 0.24±0.07 | — | 0.25±0.05 | 0.49 (16× L1) | 0.31±0.05 |
| **Real-World Causal** | L11 (88%) | L17 (3 ds) | L0 (100%) | L0 (100%) | L2 (97%) |
| **Thinking Tokens** | — | **64 tokens** | — | — | — |

### Full-Scale Validation

| Phase | Seeds | Experiments | Pass Rate | Total Time |
|-------|-------|-------------|-----------|------------|
| 5 | 5 (42, 123, 456, 789, 1024) | 9 | 44/45 (97.8%) | 168 min |
| 6 | 3 (42, 123, 456) | 5 + M28 | 15/15 + M28 ✅ | 49.9 + 2.7 min |
| 7 | 3 (42, 123, 456) | 4 milestones | 16/16 ✅ | ~120 min |

## Global Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `random_seed` | 42 | Default reproducibility seed |
| `device` | cpu | PyTorch device |
| `model_path` | `tabpfn-v2-regressor.ckpt` | TabPFN v2 checkpoint |
| `QUICK_RUN` | True (env) | Fast testing mode; set `QUICK_RUN=0` for full-scale |
| `SEED` | 42 (env) | Overridable seed for multi-seed runs |

## Configuration Files

| Config | Experiment |
|--------|------------|
| `config/exp1_coefficient.yaml` | Exp 1: Coefficient Probing |
| `config/exp2_intermediary.yaml` | Exp 2: Intermediary Probing |
| `config/exp3_answer_logitlens.yaml` | Exp 3: Answer Probing + Logit Lens |
| `config/exp4_copy_mechanism.yaml` | Exp 4: Copy Mechanism |
| `config/rd4_phase4a.yaml` | RD-4 Phase 4A: Semi-synthetic |
| `config/rd4_phase4b.yaml` | RD-4 Phase 4B: Real-world |
| `config/rd8_attention.yaml` | RD-8: All 5 attention experiments |
| `config/rd1_activation.yaml` | RD-1: Activation Patching |
| `config/rd3_ablation.yaml` | RD-3: Vector Ablation |
| `config/rd7_classification.yaml` | RD-7: Classification Analysis |
| `config/rd2_steering.yaml` | RD-2: Steering Vectors |
| `config/rd6_sae.yaml` | RD-6: Sparse Autoencoder |

## Research Notes

Detailed analysis for each phase in `research_directions/`:

### Phase 1: Correlation Analysis
- [`rd0_reproduction_notes.md`](research_directions/rd0_reproduction_notes.md) — Base paper reproduction
- [`rd4_phase4a_analysis.md`](research_directions/rd4_phase4a_analysis.md) — Semi-synthetic robustness
- [`rd4_real_world_extension.md`](research_directions/rd4_real_world_extension.md) — Real-world extension
- [`rd8_attention_map_analysis.md`](research_directions/rd8_attention_map_analysis.md) — Attention patterns
- [`phase1_comprehensive_report.md`](research_directions/phase1_comprehensive_report.md) — **Phase 1 comprehensive report**

### Phase 2: Causal Verification
- [`rd1_activation_patching.md`](research_directions/rd1_activation_patching.md) — Activation patching
- [`rd3_vector_ablation.md`](research_directions/rd3_vector_ablation.md) — Vector ablation
- [`rd7_classification_analysis.md`](research_directions/rd7_classification_analysis.md) — Classification analysis
- [`phase2_comprehensive_report.md`](research_directions/phase2_comprehensive_report.md) — **Phase 2 comprehensive report**

### Phase 3: Model Manipulation
- [`rd2_steering_vectors.md`](research_directions/rd2_steering_vectors.md) — Steering vectors
- [`rd6_sparse_autoencoder.md`](research_directions/rd6_sparse_autoencoder.md) — SAE feature decomposition
- [`phase3_comprehensive_report.md`](research_directions/phase3_comprehensive_report.md) — **Phase 3 comprehensive report**

### Phase 4: Cross-Architecture Comparison
- [`rd5_architecture_comparison.md`](research_directions/rd5_architecture_comparison.md) — Architecture comparison
- [`rd5_multi_tfm_comparison.md`](research_directions/rd5_multi_tfm_comparison.md) — Multi-TFM comparison
- [`phase4_comprehensive_report.md`](research_directions/phase4_comprehensive_report.md) — **Phase 4 comprehensive report**

### Phase 5: Statistical Validation
- [`phase5_fullscale_results.md`](research_directions/phase5_fullscale_results.md) — **Phase 5 full-scale results**

### Phase 6: Strengthening + Expansion
- [`phase6_results.md`](research_directions/phase6_results.md) — Phase 6 raw results (392 lines)
- [`phase6_comprehensive_report.md`](research_directions/phase6_comprehensive_report.md) — **Phase 6 comprehensive report**
- [`phase6_option23_synthesis.md`](research_directions/phase6_option23_synthesis.md) — **Option 2+3 synthesis report**

### Research Assessment
- [`contribution_assessment.md`](research_directions/contribution_assessment.md) — 7 "firsts" contribution assessment

## References

### Core

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| Gupta et al. "TabPFN Through The Looking Glass" (2026) | [2601.08181](https://arxiv.org/abs/2601.08181) | Base paper — reproduction target |
| Hollmann et al. "Accurate Predictions on Small Data" (2025) | [2501.02013](https://arxiv.org/abs/2501.02013) | TabPFN v2 architecture |
| Qu et al. "TabICL" (2025) | [2502.05564](https://arxiv.org/abs/2502.05564) | TabICL architecture |
| Ye et al. "iLTM" (2025) | [2511.15941](https://arxiv.org/abs/2511.15941) | iLTM architecture |

### Mechanistic Interpretability

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| Alain & Bengio "Understanding intermediate layers" (2016) | [1610.01644](https://arxiv.org/abs/1610.01644) | Linear probing methodology |
| nostalgebraist "Interpreting GPT: Logit Lens" (2020) | [LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | Logit lens technique |
| Olsson et al. "In-context Learning and Induction Heads" (2022) | [Transformer Circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Copy mechanism |
| Heimersheim & Nanda "How to use activation patching" (2024) | [2404.15255](https://arxiv.org/abs/2404.15255) | RD-1 methodology |
| Zhang & Nanda "Best Practices of Activation Patching" (2023) | [2309.16042](https://arxiv.org/abs/2309.16042) | RD-1 best practices |
| Elhage et al. "Toy Models of Superposition" (2022) | [Transformer Circuits](https://transformer-circuits.pub/2022/toy_model/index.html) | RD-3 theory |
| Turner et al. "Activation Addition" (2023) | — | RD-2 steering methodology |
| Bricken et al. "Towards Monosemanticity" (2023) | [Transformer Circuits](https://transformer-circuits.pub/2023/monosemantic-features) | RD-6 SAE methodology |
| Chanin et al. "A is for Absorption" (2024) | [2409.14507](https://arxiv.org/abs/2409.14507) | SAE feature absorption |

### Representation Analysis

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| Kornblith et al. "Similarity of Neural Network Representations" (2019) | [1905.00414](https://arxiv.org/abs/1905.00414) | CKA original paper |
| Wiliński et al. "Representations in Time Series FMs" (2025) | [2501.01365](https://arxiv.org/abs/2501.01365) | CKA methodology |
| Sirikova & Chan "Architecture Determines Representation Similarity" (2026) | [2601.17093](https://arxiv.org/abs/2601.17093) | Cross-architecture comparison |
| Koleva et al. "Attention in Tabular LMs" (2023) | [2302.14278](https://arxiv.org/abs/2302.14278) | Tabular attention analysis |
| Yeh et al. "AttentionViz" (2023) | [2305.03210](https://arxiv.org/abs/2305.03210) | Q-K embedding methodology |

### Benchmarks

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| TabArena (2025) | [2506.16791](https://arxiv.org/abs/2506.16791) | Living benchmark |
| Dragoi et al. "Fourier Features in TabPFN" (2026) | [2602.23182](https://arxiv.org/abs/2602.23182) | Complementary analysis |

## License

Research project — not for redistribution.
