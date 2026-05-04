# TabMI-Bench — Benchmark Datasheet

Following Gebru et al. (2021) "Datasheets for Datasets" and Mitchell et al. (2019) "Model Cards" templates.

---

## 1. Motivation

- **Purpose**: TabMI-Bench provides a standardized protocol for evaluating mechanistic interpretability (MI) methods on Tabular Foundation Models (TFMs). It fills the gap between mature LLM MI evaluation (e.g., MIB) and the growing but uninterpreted TFM ecosystem.
- **Creators**: Anonymous (double-blind submission). Will be de-anonymized upon acceptance.
- **Funding**: [To be disclosed upon acceptance]
- **Date of creation**: January–April 2026

## 2. Composition

### Benchmark Components

| Component | Description | Files | Format |
|-----------|-------------|-------|--------|
| Hook infrastructure | Activation extraction for 5 TFMs | `src/hooks/*.py` (10 files) | Python |
| Synthetic data generators | 4 function types (bilinear, sinusoidal, polynomial, mixed) | `src/data/synthetic_generator.py` | Python |
| Real-world dataset loaders | 11 causal tracing + 4 steering datasets | `src/data/real_world_datasets.py` | Python (scikit-learn, OpenML) |
| Evaluation scripts | Experiment scripts for all MI techniques | `experiments/*.py` | Python |
| Reference baselines | Multi-seed results for 5 models | `results/` | JSON |
| Negative controls | Random-target, shuffled-label, random-vector | `experiments/neurips_c1c2_baselines_*.py` | Python |
| Benchmark card | This document | `BENCHMARK_CARD.md` | Markdown |
| Croissant metadata | Machine-readable metadata | `croissant.json` | JSON-LD |
| Example: add new model | Worked extensibility example | `examples/add_new_model.py` | Python |

### Instance Counts

- **Synthetic probes**: 4 function families × configurable N (default N_train=100, N_test=50)
- **Real-world datasets**: 11 for causal tracing, 4 for steering (all public, from scikit-learn/OpenML)
- **Models**: 5 TFMs (TabPFN v2, TabPFN v2.5, TabICL v2, TabDPT, iLTM)
- **MI techniques**: 8 (linear probing, logit lens, activation patching, vector ablation, steering vectors, SAE, CKA, attention analysis)
- **Reference seeds**: 5 (core probing on TabPFN/TabICL/iLTM), 3 (TabDPT in-family holdout, N=10K probing scale, real-world causal), 5 (NAM out-of-family holdout, real-world steering, function invariance), 10 (robust steering, SAE expansion)

### Data Types

- Synthetic: continuous numeric features only (no categorical, no missing values)
- Real-world: mixed (numeric + categorical with encoding; no explicit missing value handling in synthetic core)

### Sensitive Information

- No personally identifiable information (PII) in synthetic data
- Real-world datasets inherit properties of their public sources:
  - Adult Income (OpenML 1590): contains age, sex, race attributes
  - California Housing: geographic data reflecting historical patterns
- The benchmark evaluates MI methods, not model fairness

### Instances Not Included

- Concrete dataset excluded due to OpenML cache failure
- Closed-source TFMs excluded (no checkpoint access)
- Non-ICL tabular methods (XGBoost, LightGBM) excluded by design

## 3. Collection Process

- **Synthetic data**: Generated deterministically from mathematical functions with fixed random seeds. No human data collection.
- **Real-world data**: Loaded from scikit-learn built-in datasets and OpenML public repositories at runtime. No new data collection.
- **Model activations**: Extracted via forward hooks registered on PyTorch modules during inference. No model training performed.
- **Timeframe**: Experiments conducted January–March 2026

## 4. Preprocessing

- **Synthetic**: Features generated from known distributions; targets computed from closed-form functions with Gaussian noise.
- **Real-world**: StandardScaler applied to features (zero mean, unit variance). Train/test split with fixed random seed.
- **Activations**: Raw layer outputs stored as float64 numpy arrays. No dimensionality reduction applied before probing.

## 5. Uses

### Intended Uses

1. **Evaluating new MI techniques**: Run the 4-step protocol, compare results against reference baselines
2. **Profiling new TFMs**: Implement hook interface (~150 lines), run existing scripts
3. **Educational**: Understanding how different TFM architectures process tabular data internally
4. **Reproducibility**: Independent verification of the reported computation profiles

### Not Intended For

- **Deployment certification**: Benchmark outputs are diagnostic measurements, NOT safety/fairness/compliance certifications
- **Model ranking**: TabMI-Bench is a protocol benchmark, not a scored leaderboard
- **Prediction accuracy comparison**: Use TabZilla, TabReD, or OpenML-CC18 for predictive performance
- **Security auditing**: The benchmark does not test adversarial robustness or data poisoning

### Potential Misuse

- Over-interpreting "staged" or "distributed" labels as complete model understanding
- Using steering controllability results as evidence of safe/aligned controllability
- Treating benchmark pass as regulatory compliance evidence

## 6. Distribution

- **License**: MIT (see `LICENSE` file)
- **Hosting**:
  - Code and scripts: GitHub (anonymous during review, public upon acceptance)
  - Archived artifacts: Zenodo (DOI-minted for long-term persistence)
  - ML-ecosystem discovery: HuggingFace Datasets (Croissant-annotated)
- **Access**: Open access. No registration required for code or synthetic data. Real-world datasets require internet for OpenML downloads.
- **Export controls**: None. No dual-use concerns.

## 7. Maintenance

- **Maintainers**: Paper authors (contact: [upon acceptance])
- **Update policy**: Version-pinned snapshots for reproducibility. New model hooks and reference baselines added as TFMs are released.
- **Versioning**: Semantic versioning (major.minor.patch). Breaking changes increment major version.
- **Community contributions**: Pull requests welcomed for new model hook implementations
- **Deprecation**: Individual model hooks may be deprecated when models are superseded, but archived results are preserved

## 8. Known Limitations

| Limitation | Scope | Mitigation |
|-----------|-------|------------|
| Small scale (N=100) | Core experiments | N=10K scale check preserves patterns |
| Regression only (synthetic) | Core probes | Classification probing in appendix |
| 5 models, 3 families | Taxonomy scope | Holdout validation on TabDPT (3-seed) and NAM (5-seed); extensible hook design |
| Post-hoc taxonomy | Scientific claim | Profile variance metric + ±50% threshold robustness + leave-one-function-out (LOFO) verified primary-endpoint ordering with ratio >29× in every holdout |
| N=10K steering sub-check is single-seed | Evidence tier | Probing component uses 3 seeds; steering sub-check labeled supporting |
| No categorical synthetic features | Probe design | Real-world datasets include categorical |
| Probing collapses at d≥8 | Method limitation | Causal tracing remains informative |

## 9. Ethical Considerations

- The benchmark studies existing open-source models and does not create new capabilities
- Real-world datasets are publicly available and widely used in ML research
- Steering demonstrations should NOT be interpreted as evidence of safe model controllability
- Users should not rely on benchmark outputs for regulatory compliance without additional validation

## 10. Citation

```bibtex
@inproceedings{anonymous2026tabmibench,
  title={TabMI-Bench: Evaluating Mechanistic Interpretability Methods Across Tabular Foundation Model Architectures},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
