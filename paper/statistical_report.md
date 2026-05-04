# Statistical Analysis Report — TFMI

## Summary
- Total statistical tests: 5
- Significant after Bonferroni correction (alpha=0.01000): 4
- Significant after BH-FDR correction (alpha=0.05): 4
- Large effect sizes (|effect| > 0.8): 4

## 1. Cross-Model Computation Strategy Differences

### 1.1 Profile variance (σ²_profile) [PRIMARY]
- Claim: TabPFN has concentrated computation (high variance) vs TabICL distributed (low variance)
- TabPFN (Lall): σ²_profile=0.033±0.005, n=5
- TabICL (Lall): σ²_profile=0.000±0.000, n=5
- 95% t-CI (TabPFN): [0.027, 0.039] | Bootstrap CI: [0.030, 0.037]
- 95% t-CI (TabICL): [0.000, 0.000] | Bootstrap CI: [0.000, 0.000]
- Welch t-test: t=15.477, p=0.000102
- Permutation test (exact): p=0.007937
- Cohen's d: 9.79 (large)
- Interpretation: Non-overlapping CIs support the claim.

### 1.2 Intermediary R² peak layer [sensitivity]
- Claim: TabPFN and TabICL use different computation strategies
- TabPFN (L9): R²=0.984±0.002, n=5
- TabICL (L1): R²=0.998±0.001, n=5
- 95% t-CI (TabPFN): [0.981, 0.988] | Bootstrap CI: [0.982, 0.986]
- 95% t-CI (TabICL): [0.997, 0.999] | Bootstrap CI: [0.998, 0.999]
- Welch t-test: t=-11.958, p=0.000133
- Permutation test (exact): p=0.007937
- Cohen's d: -7.56 (large)
- Interpretation: Non-overlapping CIs support the claim.

### 1.3 Patching sensitivity [PRIMARY]
- Claim: Causal computation locus differs between models
- TabPFN (L5): sensitivity=0.593±0.081, n=5
- TabICL (L0): sensitivity=0.999±0.003, n=5
- 95% t-CI (TabPFN): [0.492, 0.694] | Bootstrap CI: [0.532, 0.657]
- 95% t-CI (TabICL): [0.994, 1.003] | Bootstrap CI: [0.996, 1.000]
- Welch t-test: t=-11.130, p=0.000364
- Permutation test (exact): p=0.007937
- Cohen's d: -7.04 (large)
- Interpretation: Non-overlapping CIs support the claim.

### 1.4 Steering effectiveness [secondary]
- Claim: Robust steering strength peaks at different layers by architecture
- TabPFN (Lbest): |r|=0.971±0.079, n=10
- TabICL (Lbest): |r|=1.000±0.000, n=10
- 95% t-CI (TabPFN): [0.915, 1.027] | Bootstrap CI: [0.920, 0.998]
- 95% t-CI (TabICL): [1.000, 1.000] | Bootstrap CI: [1.000, 1.000]
- Welch t-test: t=-1.163, p=0.274563
- Permutation test (exact): p=0.000210
- Cohen's d: -0.52 (medium)
- Interpretation: CI overlap exists; interpret with caution.

## 2. SAE Paired Comparisons (Wilcoxon Signed-Rank)

### TopK vs ReLU — Sparsity (tabpfn)
- ReLU: 0.706±0.006 | TopK: 0.938±0.000 (n=5)
- Wilcoxon two-sided: W=0.000, p=0.062500
- Paired permutation test: p=0.062500
- Effect size r: 0.83
- Mean paired diff (TopK-ReLU): 0.231
- 95% bootstrap CI of paired diff: [0.226, 0.236]

### TopK vs ReLU — Sparsity (tabicl)
- ReLU: 0.867±0.005 | TopK: 0.945±0.001 (n=5)
- Wilcoxon two-sided: W=0.000, p=0.062500
- Paired permutation test: p=0.062500
- Effect size r: 0.83
- Mean paired diff (TopK-ReLU): 0.078
- 95% bootstrap CI of paired diff: [0.074, 0.082]

### TopK vs ReLU — Sparsity (iltm)
- ReLU: 0.936±0.005 | TopK: 0.972±0.002 (n=5)
- Wilcoxon two-sided: W=0.000, p=0.062500
- Paired permutation test: p=0.062500
- Effect size r: 0.83
- Mean paired diff (TopK-ReLU): 0.036
- 95% bootstrap CI of paired diff: [0.033, 0.039]

## 3. CKA Data Completeness

- CKA experiment coverage: 5/5 seeds (missing: none)
- All other key experiments use 5 seeds (Phase 5) or 3 seeds (Phase 6)
- Representative CKA summary metric (mean adjacent-layer CKA, TabPFN): 0.697±0.007, n=5
- 95% t-CI (TabPFN adjacent CKA): [0.689, 0.705]
- Representative CKA summary metric (mean adjacent-layer CKA, TabICL): 0.990±0.000, n=5
- 95% t-CI (TabICL adjacent CKA): [0.989, 0.990]
- Limitation: n=4 reduces precision versus the 5-seed Phase 5 setup.

## 4. Summary Table
| Comparison | n | Test | p-value | p (Bonf.) | p (BH-FDR) | Effect size | Bonf.? | BH? |
|------------|---|------|---------|-----------|-----------|-------------|--------|-----|
| Profile variance (σ²_profile) [PRIMARY] | 5 vs 5 | Exact permutation | 0.007937 | 0.039683 | 0.009921 | 9.79 (large) | Yes | Yes |
| Intermediary R² peak layer [sensitivity] | 5 vs 5 | Exact permutation | 0.007937 | 0.039683 | 0.009921 | -7.56 (large) | Yes | Yes |
| Patching sensitivity [PRIMARY] | 5 vs 5 | Exact permutation | 0.007937 | 0.039683 | 0.009921 | -7.04 (large) | Yes | Yes |
| Steering effectiveness [secondary] | 10 vs 10 | Exact permutation | 0.000210 | 0.001050 | 0.001050 | -0.52 (medium) | Yes | Yes |
| SAE TopK vs ReLU (representative: tabpfn) | 5 paired | Wilcoxon signed-rank | 0.062500 | 0.312500 | 0.062500 | 0.83 (large) | No | No |
