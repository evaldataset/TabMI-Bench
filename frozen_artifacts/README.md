# Frozen JSON Artifacts for `make reproduce-paper`

This directory contains the canonical aggregated JSON files that the paper's
tables and figures are computed from. Bundling these allows reviewers to
regenerate paper-facing artifacts without re-running 40 GPU-hours of experiments.

| File | Source | Used by |
|---|---|---|
| `rd5_fullscale_aggregated.json` | `results/rd5_fullscale/aggregated/aggregated_results.json` | Table 1 (3 strategies), Fig 1 |
| `tabdpt_probing_3seed.json` | `results/phase7/tabdpt_probing/aggregated_3seed.json` | Table 1 TabDPT row, Fig 1 |
| `tabdpt_causal_3seed.json` | `results/phase7/tabdpt_causal/aggregated_3seed.json` | §4.4 TabDPT causal claim |
| `nam_holdout.json` | `results/phase9_nam_holdout/nam_holdout_results.json` | §4.3 NAM holdout |
| `lofo_primary_endpoint.json` | `results/phase8a/nonlinear_probing/lofo_primary_endpoint.json` | Appendix LOFO table |
| `c1c2_baselines.json` | `results/neurips/c1c2_baselines.json` | §4.5 shuffled-label control |
| `scale_10k_multiseed.json` | `results/scale_10k_multiseed/results.json` | Appendix G N=10K scaling |
| `tabpfn25_fullscale_aggregated.json` | `results/rd6/tabpfn25_fullscale/aggregated/*.json` | Table 7 (v2 vs v2.5) |

To regenerate paper figures/tables from these artifacts without GPU:
```bash
make reproduce-paper
```
