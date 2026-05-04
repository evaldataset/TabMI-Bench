# Makefile — TFMI Phase 1–7: Tabular Foundation Model Mechanistic Interpretability
# Reproducibility package for Gupta et al. (2026) reproduction + extensions
#
# Usage:
#   make help          Show available targets
#   make setup         Create venv and install dependencies
#   make run_all       Run all Phase 1 experiments (11 total)
#   make run_all_phase7 Run all Phase 7 extensions
#   make clean         Remove results/ directory

PYTHON = .venv/bin/python
UV     = uv

# ==============================================================================
# Phony targets
# ==============================================================================

.PHONY: help setup clean \
        run_all run_all_phase2 run_all_phase3 run_all_phase4 run_all_phase7 \
        run_reproduction run_rd4 run_rd8 \
        run_exp1 run_exp2 run_exp3 run_exp4 \
        run_rd4a run_rd4b \
        run_rd8_viz run_rd8_copy run_rd8_head run_rd8_feat run_rd8_qk \
        run_rd1 run_rd1_coeff run_rd1_sweep run_rd1_intermediary \
        run_rd3 run_rd7 run_rd7_probing run_rd7_geometry \
        run_rd2 run_rd2_coeff run_rd2_layer run_rd2_boundary \
        run_rd6 run_rd6_train run_rd6_features run_rd6_ablation \
        run_rd5 run_rd5_coeff run_rd5_intermed run_rd5_copy run_rd5_cka \
        run_rd5_patching run_rd5_steering run_rd5_sae run_rd5_summary \
        run_phase7_realworld run_phase7_sae run_phase7_tabdpt

.DEFAULT_GOAL := help

# ==============================================================================
# Help
# ==============================================================================

help:
	@echo ""
	@echo "TFMI Phase 1, 2 & 3 — Reproducibility Makefile"
	@echo "============================================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup            Create .venv and install requirements"
	@echo ""
	@echo "Run all:"
	@echo "  make run_all          Run all Phase 1 experiments (11 total)"
	@echo "  make run_all_phase2   Run all Phase 2 experiments (6 total)"
	@echo "  make run_all_phase3   Run all Phase 3 experiments (6 total)"
	@echo ""
	@echo "Phase 1 experiment groups:"
	@echo "  make run_reproduction Run exp1..exp4 (base paper reproduction)"
	@echo "  make run_rd4          Run rd4_phase4a + rd4_phase4b"
	@echo "  make run_rd8          Run all 5 RD-8 attention experiments"
	@echo ""
	@echo "Individual experiments (M1 — Reproduction):"
	@echo "  make run_exp1         Exp 1: Coefficient Probing"
	@echo "  make run_exp2         Exp 2: Intermediary Probing"
	@echo "  make run_exp3         Exp 3: Answer Probing + Logit Lens"
	@echo "  make run_exp4         Exp 4: Copy Mechanism"
	@echo ""
	@echo "Individual experiments (RD-4 — Real-world Extension):"
	@echo "  make run_rd4a         RD-4 Phase 4A: Semi-synthetic data"
	@echo "  make run_rd4b         RD-4 Phase 4B: Real-world benchmarks"
	@echo ""
	@echo "Individual experiments (RD-8 — Attention Analysis):"
	@echo "  make run_rd8_viz      RD-8: Attention heatmap visualization"
	@echo "  make run_rd8_copy     RD-8: Copy mechanism attention"
	@echo "  make run_rd8_head     RD-8: Head specialization (JSD)"
	@echo "  make run_rd8_feat     RD-8: Feature interaction structure"
	@echo "  make run_rd8_qk       RD-8: Q-K joint embedding (PCA)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Remove results/ directory"
	@echo ""
	@echo "Phase 2 experiment groups:"
	@echo "  make run_rd1          RD-1: All 3 Activation Patching experiments"
	@echo "  make run_rd3          RD-3: Vector Ablation"
	@echo "  make run_rd7          RD-7: All 2 Classification Analysis experiments"
	@echo ""
	@echo "Individual experiments (RD-1 — Activation Patching):"
	@echo "  make run_rd1_coeff         RD-1: Coefficient patching (Layer 6 causality)"
	@echo "  make run_rd1_sweep         RD-1: Full layer sweep (a/b/c corruption)"
	@echo "  make run_rd1_intermediary  RD-1: Intermediary a·b patching"
	@echo ""
	@echo "Individual experiments (RD-3 — Vector Ablation):"
	@echo "  make run_rd3               RD-3: Direction extraction + ablation"
	@echo ""
	@echo "Individual experiments (RD-7 — Classification Analysis):"
	@echo "  make run_rd7_probing       RD-7: Decision boundary probing"
	@echo "  make run_rd7_geometry      RD-7: t-SNE + CKA representation geometry"
	@echo ""
	@echo "Phase 3 experiment groups:"
	@echo "  make run_rd2          RD-2: All 3 Steering Vector experiments"
	@echo "  make run_rd6          RD-6: All 3 SAE experiments"
	@echo ""
	@echo "Individual experiments (RD-2 — Steering Vectors):"
	@echo "  make run_rd2_coeff         RD-2: Coefficient steering (Layer 6)"
	@echo "  make run_rd2_layer         RD-2: Layer-by-layer steering comparison"
	@echo "  make run_rd2_boundary      RD-2: Classification boundary steering"
	@echo ""
	@echo "Individual experiments (RD-6 — Sparse Autoencoder):"
	@echo "  make run_rd6_train         RD-6: SAE training + basic analysis"
	@echo "  make run_rd6_features      RD-6: SAE feature correlation analysis"
	@echo "  make run_rd6_ablation      RD-6: SAE feature ablation vs RD-3"
	@echo ""

# ==============================================================================
# Setup
# ==============================================================================

setup:
	@echo "[setup] Creating virtual environment with uv..."
	$(UV) venv .venv
	@echo "[setup] Installing dependencies..."
	$(UV) pip install -r requirements.txt
	@echo "[setup] Done. Activate with: source .venv/bin/activate"

# ==============================================================================
# Milestone 1: Base Paper Reproduction (exp1..exp4)
# ==============================================================================

run_exp1:
	@echo "[exp1] Coefficient Probing..."
	$(PYTHON) experiments/exp1_coefficient_probing.py
	@echo "[exp1] Done. Results: results/exp1/"

run_exp2:
	@echo "[exp2] Intermediary Probing..."
	$(PYTHON) experiments/exp2_intermediary_probing.py
	@echo "[exp2] Done. Results: results/exp2/"

run_exp3:
	@echo "[exp3] Answer Probing + Logit Lens..."
	$(PYTHON) experiments/exp3_answer_probing_logit_lens.py
	@echo "[exp3] Done. Results: results/exp3/"

run_exp4:
	@echo "[exp4] Copy Mechanism..."
	$(PYTHON) experiments/exp4_copy_mechanism.py
	@echo "[exp4] Done. Results: results/exp4/"

run_reproduction: run_exp1 run_exp2 run_exp3 run_exp4
	@echo "[reproduction] All 4 base paper experiments completed."

# ==============================================================================
# Milestone 2: RD-4 — Real-world Data Extension
# ==============================================================================

run_rd4a:
	@echo "[rd4a] Phase 4A: Semi-synthetic data..."
	$(PYTHON) experiments/rd4_phase4a_semisynthetic.py
	@echo "[rd4a] Done. Results: results/rd4_phase4a/"

run_rd4b:
	@echo "[rd4b] Phase 4B: Real-world benchmarks..."
	$(PYTHON) experiments/rd4_phase4b_realworld.py
	@echo "[rd4b] Done. Results: results/rd4_phase4b/"

run_rd4: run_rd4a run_rd4b
	@echo "[rd4] All RD-4 experiments completed."

# ==============================================================================
# Milestone 3: RD-8 — Attention Map Analysis
# ==============================================================================

run_rd8_viz:
	@echo "[rd8-viz] Attention heatmap visualization..."
	$(PYTHON) experiments/rd8_attention_visualization.py
	@echo "[rd8-viz] Done. Results: results/rd8/attention_heatmaps/"

run_rd8_copy:
	@echo "[rd8-copy] Copy mechanism attention..."
	$(PYTHON) experiments/rd8_copy_mechanism_attention.py
	@echo "[rd8-copy] Done. Results: results/rd8/copy_mechanism_attention/"

run_rd8_head:
	@echo "[rd8-head] Head specialization (JSD)..."
	$(PYTHON) experiments/rd8_head_specialization.py
	@echo "[rd8-head] Done. Results: results/rd8/head_analysis/"

run_rd8_feat:
	@echo "[rd8-feat] Feature interaction structure..."
	$(PYTHON) experiments/rd8_feature_interaction.py
	@echo "[rd8-feat] Done. Results: results/rd8/feature_interaction/"

run_rd8_qk:
	@echo "[rd8-qk] Q-K joint embedding (PCA)..."
	$(PYTHON) experiments/rd8_qk_embedding.py
	@echo "[rd8-qk] Done. Results: results/rd8/qk_embeddings/"

run_rd8: run_rd8_viz run_rd8_copy run_rd8_head run_rd8_feat run_rd8_qk
	@echo "[rd8] All 5 RD-8 experiments completed."

# ==============================================================================
# Run all
# ==============================================================================

run_all: run_reproduction run_rd4 run_rd8
	@echo ""
	@echo "========================================"
	@echo " All 11 Phase 1 experiments completed."
	@echo " Results saved to: results/"
	@echo "========================================"

# ==============================================================================
# Phase 2: RD-1 — Activation Patching
# ==============================================================================

run_rd1_coeff:
	@echo "[rd1-coeff] Coefficient Patching (Layer 6 causality)..."
	$(PYTHON) experiments/rd1_coefficient_patching.py
	@echo "[rd1-coeff] Done. Results: results/rd1/coefficient_patching/"

run_rd1_sweep:
	@echo "[rd1-sweep] Full Layer Sweep (a/b/c corruption)..."
	$(PYTHON) experiments/rd1_layer_sweep.py
	@echo "[rd1-sweep] Done. Results: results/rd1/layer_sweep/"

run_rd1_intermediary:
	@echo "[rd1-intermediary] Intermediary a·b Patching..."
	$(PYTHON) experiments/rd1_intermediary_patching.py
	@echo "[rd1-intermediary] Done. Results: results/rd1/intermediary_patching/"

run_rd1: run_rd1_coeff run_rd1_sweep run_rd1_intermediary
	@echo "[rd1] All 3 RD-1 Activation Patching experiments completed."

# ==============================================================================
# Phase 2: RD-3 — Vector Ablation
# ==============================================================================

run_rd3:
	@echo "[rd3] Vector Ablation (direction extraction + ablation)..."
	$(PYTHON) experiments/rd3_vector_ablation.py
	@echo "[rd3] Done. Results: results/rd3/"

# ==============================================================================
# Phase 2: RD-7 — Classification Task Analysis
# ==============================================================================

run_rd7_probing:
	@echo "[rd7-probing] Classification Decision Boundary Probing..."
	$(PYTHON) experiments/rd7_classification_probing.py
	@echo "[rd7-probing] Done. Results: results/rd7/probing/"

run_rd7_geometry:
	@echo "[rd7-geometry] Representation Geometry (t-SNE + CKA)..."
	$(PYTHON) experiments/rd7_representation_geometry.py
	@echo "[rd7-geometry] Done. Results: results/rd7/geometry/"

run_rd7: run_rd7_probing run_rd7_geometry
	@echo "[rd7] All 2 RD-7 Classification Analysis experiments completed."

# ==============================================================================
# Run all Phase 2
# ==============================================================================

run_all_phase2: run_rd1 run_rd3 run_rd7
	@echo ""
	@echo "============================================"
	@echo " All 6 Phase 2 experiments completed."
	@echo " Results saved to: results/rd1/ results/rd3/ results/rd7/"
	@echo "============================================"
# ==============================================================================
# Clean
# ==============================================================================

clean:
	@echo "[clean] Cleaning generated artifacts..."
	@if [ -L results ]; then \
		echo "[clean] WARNING: results/ is a symlink — skipping to protect external data"; \
	else \
		echo "[clean] Removing results/ directory..."; \
		rm -rf results/; \
	fi
	@echo "[clean] Done."

# ==============================================================================
# Phase 3: RD-2 — Steering Vectors
# ==============================================================================

run_rd2_coeff:
	@echo "[rd2-coeff] Coefficient Steering (Layer 6)..."
	$(PYTHON) experiments/rd2_coefficient_steering.py
	@echo "[rd2-coeff] Done. Results: results/rd2/coefficient_steering/"

run_rd2_layer:
	@echo "[rd2-layer] Layer-by-Layer Steering Comparison..."
	$(PYTHON) experiments/rd2_layer_steering.py
	@echo "[rd2-layer] Done. Results: results/rd2/layer_steering/"

run_rd2_boundary:
	@echo "[rd2-boundary] Classification Boundary Steering..."
	$(PYTHON) experiments/rd2_boundary_steering.py
	@echo "[rd2-boundary] Done. Results: results/rd2/boundary_steering/"

run_rd2: run_rd2_coeff run_rd2_layer run_rd2_boundary
	@echo "[rd2] All 3 RD-2 Steering Vector experiments completed."

# ==============================================================================
# Phase 3: RD-6 — Sparse Autoencoder
# ==============================================================================

run_rd6_train:
	@echo "[rd6-train] SAE Training + Basic Analysis..."
	$(PYTHON) experiments/rd6_sae_training.py
	@echo "[rd6-train] Done. Results: results/rd6/training/"

run_rd6_features:
	@echo "[rd6-features] SAE Feature Correlation Analysis..."
	$(PYTHON) experiments/rd6_feature_analysis.py
	@echo "[rd6-features] Done. Results: results/rd6/features/"

run_rd6_ablation:
	@echo "[rd6-ablation] SAE Feature Ablation vs RD-3..."
	$(PYTHON) experiments/rd6_feature_ablation.py
	@echo "[rd6-ablation] Done. Results: results/rd6/ablation/"

run_rd6: run_rd6_train run_rd6_features run_rd6_ablation
	@echo "[rd6] All 3 RD-6 SAE experiments completed."

# ==============================================================================
# Run all Phase 3
# ==============================================================================

run_all_phase3: run_rd2 run_rd6
	@echo ""
	@echo "============================================"
	@echo " All 6 Phase 3 experiments completed."
	@echo " Results saved to: results/rd2/ results/rd6/"
	@echo "============================================"

# ==============================================================================
# Phase 4: RD-5 Multi-TFM Comparative Analysis
# ==============================================================================

run_rd5_coeff:
	@echo "[rd5] Running coefficient probing comparison..."
	$(PYTHON) experiments/rd5_coefficient_probing.py

run_rd5_intermed:
	@echo "[rd5] Running intermediary probing comparison..."
	$(PYTHON) experiments/rd5_intermediary_probing.py

run_rd5_copy:
	@echo "[rd5] Running copy mechanism comparison..."
	$(PYTHON) experiments/rd5_copy_mechanism.py

run_rd5_cka:
	@echo "[rd5] Running CKA comparison..."
	$(PYTHON) experiments/rd5_cka_comparison.py

run_rd5_patching:
	@echo "[rd5] Running patching comparison..."
	$(PYTHON) experiments/rd5_patching_comparison.py

run_rd5_steering:
	@echo "[rd5] Running steering comparison..."
	$(PYTHON) experiments/rd5_steering_comparison.py

run_rd5_sae:
	@echo "[rd5] Running SAE comparison..."
	$(PYTHON) experiments/rd5_sae_comparison.py

run_rd5_summary:
	@echo "[rd5] Running hypothesis summary..."
	$(PYTHON) experiments/rd5_hypothesis_summary.py

run_rd5: run_rd5_coeff run_rd5_intermed run_rd5_copy run_rd5_cka run_rd5_patching run_rd5_steering run_rd5_sae run_rd5_summary
	@echo "[rd5] All 8 RD-5 experiments completed."

# ==============================================================================
# Run all Phase 4
# ==============================================================================

run_all_phase4: run_rd5
	@echo ""
	@echo "============================================"
	@echo " All 8 Phase 4 experiments completed."
	@echo " Results saved to: results/rd5/"
	@echo "============================================"

# ==============================================================================
# Phase 5: Full-scale + Multi-seed + Real-world
# ==============================================================================

run_rd5_realworld_probing:
	@echo "[rd5] Running real-world probing..."
	$(PYTHON) experiments/rd5_realworld_probing.py

run_rd5_realworld_cka:
	@echo "[rd5] Running real-world CKA..."
	$(PYTHON) experiments/rd5_realworld_cka.py

run_fullscale_quick:
	@echo "[rd5] Quick smoke test (1 seed)..."
	$(PYTHON) experiments/run_fullscale.py --quick --seeds 42

run_fullscale:
	@echo "[rd5] Full-scale experiments (5 seeds)..."
	$(PYTHON) experiments/run_fullscale.py

aggregate:
	@echo "[rd5] Aggregating multi-seed results..."
	$(PYTHON) experiments/aggregate_results.py

run_all_phase5: run_fullscale aggregate
	@echo "Phase 5 complete. Results: results/rd5_fullscale/"

# ==============================================================================
# Phase 6: Result Strengthening + Expansion
# ==============================================================================

run_rd6_robust_steering:
	@echo "[rd6] Robust multi-pair steering..."
	$(PYTHON) experiments/rd6_robust_steering.py

run_rd6_improved_sae:
	@echo "[rd6] Improved SAE (16x + JumpReLU)..."
	$(PYTHON) experiments/rd6_improved_sae.py

run_rd6_realworld:
	@echo "[rd6] Real-world expanded (up to 16 candidate datasets)..."
	$(PYTHON) experiments/rd6_realworld_expanded.py

run_rd6_classification:
	@echo "[rd6] Classification probing..."
	$(PYTHON) experiments/rd6_classification_probing.py

run_rd6_attention:
	@echo "[rd6] Cross-model attention comparison..."
	$(PYTHON) experiments/rd6_attention_comparison.py

run_rd6_tabpfn25:
	@echo "[rd6] TabPFN v2 vs v2.5 comparison..."
	$(PYTHON) experiments/rd6_tabpfn25_comparison.py

run_rd6_topk:
	@echo "[rd6] SAE TopK comparison..."
	$(PYTHON) experiments/rd6_sae_topk_comparison.py

run_all_phase6: run_rd6_robust_steering run_rd6_improved_sae run_rd6_realworld run_rd6_classification run_rd6_attention run_rd6_tabpfn25 run_rd6_topk
	@echo "Phase 6 complete. Results: results/rd6_fullscale/"

# ==============================================================================
# Aggregation & Analysis
# ==============================================================================

aggregate_phase5:
	@echo "[agg] Aggregating Phase 5 results..."
	$(PYTHON) experiments/aggregate_results.py

aggregate_phase6:
	@echo "[agg] Aggregating Phase 6 results..."
	$(PYTHON) experiments/aggregate_phase6.py

aggregate_sae_topk:
	@echo "[agg] Aggregating SAE TopK results..."
	$(PYTHON) experiments/aggregate_sae_topk_fullscale.py

stats:
	@echo "[stats] Running statistical analysis..."
	$(PYTHON) experiments/statistical_analysis.py

figures:
	@echo "[fig] Generating paper figures..."
	$(PYTHON) experiments/generate_paper_figures.py
	$(PYTHON) experiments/generate_appendix_figures.py

aggregate_all: aggregate_phase5 aggregate_phase6 aggregate_sae_topk stats figures
	@echo "All aggregation and figure generation complete."

# ==============================================================================
# Full Pipeline
# ==============================================================================

all: run_all run_all_phase2 run_all_phase3 run_all_phase4 run_all_phase5 run_all_phase6 run_all_phase7 aggregate_all
	@echo ""
	@echo "========================================"
	@echo " ALL experiments + aggregation complete."
	@echo "========================================"

# ==============================================================================
# Phase 7: Paper Strengthening Extensions
# ==============================================================================

run_phase7_realworld:
	@echo "[p7] Real-world causal tracing (11 datasets, seed 42)..."
	SEED=42 $(PYTHON) experiments/phase7_realworld_causal.py

run_phase7_sae:
	@echo "[p7] SAE scaling (4x–256x, 3 seeds)..."
	$(PYTHON) experiments/phase7_sae_scaling.py

run_phase7_tabdpt:
	@echo "[p7] TabDPT probing + causal + SAE..."
	$(PYTHON) experiments/phase7_tabdpt_probing.py
	$(PYTHON) experiments/phase7_tabdpt_causal.py
	$(PYTHON) experiments/phase7_tabdpt_sae.py

run_all_phase7: run_phase7_realworld run_phase7_sae run_phase7_tabdpt
	@echo "Phase 7 complete. Results: results/phase7/"

# ==============================================================================
# Phase 8: NeurIPS Strengthening
# ==============================================================================

run_phase8a_probing:
	@echo "[p8a] Non-linear function invariance — probing..."
	QUICK_RUN=0 $(PYTHON) experiments/phase8a_nonlinear_probing.py

run_phase8a_causal:
	@echo "[p8a] Non-linear function invariance — causal tracing..."
	QUICK_RUN=0 $(PYTHON) experiments/phase8a_nonlinear_causal.py

run_phase8b_steering:
	@echo "[p8b] Real-world steering validation..."
	QUICK_RUN=0 $(PYTHON) experiments/phase8b_realworld_steering.py

run_phase8d_complexity:
	@echo "[p8d] Feature complexity sweep..."
	QUICK_RUN=0 $(PYTHON) experiments/phase8d_feature_complexity.py

run_phase8c_phase6_extra:
	@echo "[p8c] Phase 6 additional seeds (789, 1024)..."
	$(PYTHON) experiments/run_fullscale_phase6.py --seeds 789 1024

run_phase8c_scaling:
	@echo "[p8c] N=10K scaling (5 seeds)..."
	DEVICE=cuda:1 $(PYTHON) experiments/neurips_c3_scaling_multiseed.py

run_phase8c_stats:
	@echo "[p8c] Statistical analysis with BH-FDR..."
	PYTHONPATH=. $(PYTHON) experiments/statistical_analysis.py

run_all_phase8: run_phase8a_probing run_phase8a_causal run_phase8b_steering run_phase8d_complexity run_phase8c_stats
	@echo "Phase 8 complete. Results: results/phase8a/ results/phase8b/ results/phase8d/"

# ==============================================================================
# Paper Reproduction (from frozen JSON artifacts; no GPU required)
# ==============================================================================

# reproduce-paper: regenerate paper-facing tables and figures.
# Default target uses NAS-backed `results/` (full live state).
# For external reviewer reproduction without NAS, use: make reproduce-paper-frozen
reproduce-paper: aggregate_phase5 aggregate_phase6 aggregate_sae_topk stats figures
	@echo "[reproduce] Aggregating TabDPT 3-seed in-family holdout..."
	@$(PYTHON) -c "import json, glob, numpy as np; ps=sorted(glob.glob('results/phase7/tabdpt_probing/tabdpt_probing_seed*.json')); P=np.stack([np.array(json.load(open(p))['intermediary']['ab_r2_by_layer']) for p in ps]); m=P.mean(0); out={'n_seeds':len(ps),'mean_profile':m.tolist(),'std_profile':P.std(0).tolist(),'peak_r2_mean':float(P.max(1).mean()),'peak_r2_std':float(P.max(1).std()),'sigma2_profile_mean_profile':float(m.var()),'min_r2_mean_profile':float(m.min())}; json.dump(out, open('results/phase7/tabdpt_probing/aggregated_3seed.json','w'), indent=2)"
	@echo "[reproduce] All paper-facing artifacts regenerated."
	@echo "[reproduce] To rebuild PDF: cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"

# reproduce-paper-frozen: verify paper-facing summary numerics WITHOUT requiring
# the NAS-backed `results/` symlink. Reads only from `frozen_artifacts/*.json`
# (bundled in the supplementary zip) and prints the canonical values that drive
# Tables 1, 7, the LOFO appendix, the TabDPT holdouts, the NAM holdout, and v2.5.
reproduce-paper-frozen:
	@$(PYTHON) scripts/verify_frozen_artifacts.py
