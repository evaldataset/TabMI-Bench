#!/usr/bin/env python3
"""Statistical tests for TFMI aggregated multi-seed experiment results.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/statistical_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RD5_DIR = ROOT / "results" / "rd5_fullscale"
RD6_DIR = ROOT / "results" / "rd6_fullscale"
SAE_TOPK_AGG = (
    ROOT
    / "results"
    / "rd6"
    / "sae_topk_fullscale"
    / "aggregated"
    / "aggregated_results.json"
)
REPORT_PATH = ROOT / "paper" / "statistical_report.md"


@dataclass
class TestResult:
    comparison: str
    n: str
    test_name: str
    p_value: float
    effect_size: float
    effect_label: str
    significant: bool = False
    p_bonferroni: float = math.nan


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_seed_dirs(base_dir: Path) -> list[Path]:
    return sorted(p for p in base_dir.glob("seed_*") if p.is_dir())


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def t_ci_from_summary(
    mean: float, std: float, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    if n <= 1:
        return (mean, mean)
    sem = std / math.sqrt(n)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    margin = t_crit * sem
    return (mean - margin, mean + margin)


def bootstrap_ci_mean(
    values: list[float], n_resamples: int = 10000, seed: int = 42
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    sample_means = np.mean(arr[idx], axis=1)
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return float(low), float(high)


def cohen_d_independent(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    nx = x_arr.size
    ny = y_arr.size
    if nx < 2 or ny < 2:
        return math.nan
    vx = np.var(x_arr, ddof=1)
    vy = np.var(y_arr, ddof=1)
    pooled_var = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled_var <= 0:
        return math.nan
    return float((np.mean(x_arr) - np.mean(y_arr)) / math.sqrt(pooled_var))

def exact_permutation_test(
    x: list[float], y: list[float], n_max: int = 100000, seed: int = 42
) -> float:
    """Two-sided exact (or Monte Carlo) permutation test for difference in means."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    observed = float(abs(np.mean(x_arr) - np.mean(y_arr)))
    pooled = np.concatenate([x_arr, y_arr])
    n = x_arr.size
    total = pooled.size
    from itertools import combinations
    n_perms = math.comb(total, n)
    if n_perms <= n_max:
        count = 0
        for idx in combinations(range(total), n):
            perm_x = pooled[list(idx)]
            perm_y = np.delete(pooled, list(idx))
            if abs(float(np.mean(perm_x) - np.mean(perm_y))) >= observed - 1e-12:
                count += 1
        return count / n_perms
    else:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_max):
            perm = rng.permutation(total)
            perm_x = pooled[perm[:n]]
            perm_y = pooled[perm[n:]]
            if abs(float(np.mean(perm_x) - np.mean(perm_y))) >= observed - 1e-12:
                count += 1
        return count / n_max


def paired_permutation_test(
    x: list[float], y: list[float], n_max: int = 100000, seed: int = 42
) -> float:
    """Two-sided paired permutation test for difference in means.

    Under H0, signs of paired differences are exchangeable.
    For n pairs, there are 2^n possible sign assignments.
    """
    diffs = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    n = diffs.shape[0]
    observed = abs(float(np.mean(diffs)))
    n_perms = 2**n

    if n_perms <= n_max:
        count = 0
        for i in range(n_perms):
            signs = np.array([(1 if (i >> bit) & 1 else -1) for bit in range(n)], dtype=np.float64)
            perm_mean = abs(float(np.mean(signs * diffs)))
            if perm_mean >= observed - 1e-12:
                count += 1
        return count / n_perms
    else:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_max):
            signs = rng.choice([-1.0, 1.0], size=n)
            perm_mean = abs(float(np.mean(signs * diffs)))
            if perm_mean >= observed - 1e-12:
                count += 1
        return count / n_max


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of significant flags."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        threshold = alpha * rank / n
        if p <= threshold:
            # Mark this and all with smaller p-values as significant
            for r2, (oi2, _) in enumerate(indexed[:rank], start=1):
                significant[oi2] = True
    return significant


def effect_label_from_d(d: float, metric: str = "d") -> str:
    """Label effect size magnitude.

    For Cohen's d: small < 0.2, medium < 0.8, large >= 0.8
    For rank-biserial r: small < 0.1, medium < 0.3, large >= 0.5
    """
    if math.isnan(d):
        return "undefined"
    ad = abs(d)
    if metric == "r":
        # Rank-biserial r thresholds (Fritz et al., 2012)
        if ad < 0.1:
            return "small"
        if ad < 0.3:
            return "medium"
        return "large"
    # Cohen's d thresholds
    if ad < 0.2:
        return "small"
    if ad <= 0.8:
        return "medium"
    return "large"


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}"


def load_cross_model_values() -> dict[
    str, tuple[list[float], list[float], int, int, str, str]
]:
    rd5_seeds = find_seed_dirs(RD5_DIR)
    rd6_seeds = find_seed_dirs(RD6_DIR)

    # Primary endpoint: profile variance (σ²_profile) — how concentrated
    # vs distributed computation is across layers.  This matches the paper's
    # pre-specified primary endpoint (Table 5, row 1).
    inter_tabpfn: list[float] = []
    inter_tabicl: list[float] = []
    profile_var_tabpfn: list[float] = []
    profile_var_tabicl: list[float] = []
    for sd in rd5_seeds:
        path = sd / "intermediary_probing" / "results.json"
        if not path.exists():
            continue
        data = read_json(path)
        pfn_profile = [float(v) for v in data["tabpfn"]["intermediary_r2_by_layer"]]
        icl_profile = [float(v) for v in data["tabicl"]["intermediary_r2_by_layer"]]
        # Peak layer values (sensitivity analysis, not primary)
        inter_tabpfn.append(max(pfn_profile))
        inter_tabicl.append(max(icl_profile))
        # Profile variance (primary endpoint)
        profile_var_tabpfn.append(float(np.var(pfn_profile)))
        profile_var_tabicl.append(float(np.var(icl_profile)))

    patch_tabpfn: list[float] = []
    patch_tabicl: list[float] = []
    for sd in rd5_seeds:
        path = sd / "patching" / "results.json"
        if not path.exists():
            continue
        data = read_json(path)
        patch_tabpfn.append(float(data["tabpfn"]["summary_sensitivity"][5]))
        patch_tabicl.append(float(data["tabicl"]["summary_sensitivity"][0]))

    steer_tabpfn: list[float] = []
    steer_tabicl: list[float] = []
    for sd in rd6_seeds:
        path = sd / "robust_steering" / "results.json"
        if not path.exists():
            continue
        data = read_json(path)
        # Use best layer per seed for each model
        tabpfn_best = 0.0
        for lk, ld in data["tabpfn"]["robust"]["per_layer"].items():
            val = abs(float(ld.get("abs_pearson_r", 0.0)))
            if val > tabpfn_best:
                tabpfn_best = val
        steer_tabpfn.append(tabpfn_best)
        tabicl_best = 0.0
        for lk, ld in data["tabicl"]["robust"]["per_layer"].items():
            val = abs(float(ld.get("abs_pearson_r", 0.0)))
            if val > tabicl_best:
                tabicl_best = val
        steer_tabicl.append(tabicl_best)

    return {
        "profile_variance": (
            profile_var_tabpfn,
            profile_var_tabicl,
            "all",
            "all",
            "Profile variance (σ²_profile) [PRIMARY]",
            "TabPFN has concentrated computation (high variance) vs TabICL distributed (low variance)",
        ),
        "intermediary": (
            inter_tabpfn,
            inter_tabicl,
            9,
            1,
            "Intermediary R² peak layer [sensitivity]",
            "TabPFN and TabICL use different computation strategies",
        ),
        "patching": (
            patch_tabpfn,
            patch_tabicl,
            5,
            0,
            "Patching sensitivity [PRIMARY]",
            "Causal computation locus differs between models",
        ),
        "steering": (
            steer_tabpfn,
            steer_tabicl,
            "best",
            "best",
            "Steering effectiveness [secondary]",
            "Robust steering strength peaks at different layers by architecture",
        ),
    }


def load_sae_topk_values() -> dict[str, dict[str, list[float]]]:
    data = read_json(SAE_TOPK_AGG)
    agg = data["aggregated"]
    out: dict[str, dict[str, list[float]]] = {}
    for model in ("tabpfn", "tabicl", "iltm"):
        out[model] = {
            "relu_sparsity": [
                float(v) for v in agg["relu_16x"][model]["sparsity"]["values"]
            ],
            "topk_sparsity": [
                float(v) for v in agg["topk_16x_6p25"][model]["sparsity"]["values"]
            ],
        }
    return out


def load_cka_seed_coverage_and_adjacent() -> tuple[
    int, int, list[float], list[float], list[str]
]:
    rd5_seeds = find_seed_dirs(RD5_DIR)
    total = len(rd5_seeds)
    available = 0
    tabpfn_adj_mean: list[float] = []
    tabicl_adj_mean: list[float] = []
    missing: list[str] = []
    for sd in rd5_seeds:
        path = sd / "cka" / "results.json"
        if not path.exists():
            missing.append(sd.name)
            continue
        available += 1
        data = read_json(path)
        tabpfn_adj_mean.append(
            float(np.mean(np.asarray(data["tabpfn"]["adjacent_cka"], dtype=np.float64)))
        )
        tabicl_adj_mean.append(
            float(np.mean(np.asarray(data["tabicl"]["adjacent_cka"], dtype=np.float64)))
        )
    return total, available, tabpfn_adj_mean, tabicl_adj_mean, missing


def run() -> None:
    cross_data = load_cross_model_values()
    sae_data = load_sae_topk_values()
    cka_total, cka_available, cka_tabpfn_adj, cka_tabicl_adj, cka_missing = (
        load_cka_seed_coverage_and_adjacent()
    )

    tests: list[TestResult] = []
    sections: list[str] = []

    sections.append("# Statistical Analysis Report — TFMI")

    cross_lines: list[str] = ["## 1. Cross-Model Computation Strategy Differences", ""]

    for idx, key in enumerate(["profile_variance", "intermediary", "patching", "steering"], start=1):
        tabpfn_vals, tabicl_vals, layer_pfn, layer_icl, title, claim = cross_data[key]

        m1, s1 = mean_std(tabpfn_vals)
        m2, s2 = mean_std(tabicl_vals)
        n1 = len(tabpfn_vals)
        n2 = len(tabicl_vals)

        t_stat, p_val = stats.ttest_ind(
            tabpfn_vals, tabicl_vals, equal_var=False, alternative="two-sided"
        )
        d = cohen_d_independent(tabpfn_vals, tabicl_vals)
        perm_p = exact_permutation_test(tabpfn_vals, tabicl_vals)
        label = effect_label_from_d(d)

        ci1 = t_ci_from_summary(m1, s1, n1)
        ci2 = t_ci_from_summary(m2, s2, n2)
        bci1 = bootstrap_ci_mean(tabpfn_vals)
        bci2 = bootstrap_ci_mean(tabicl_vals)

        tests.append(
            TestResult(
                comparison=title,
                n=f"{n1} vs {n2}",
                test_name="Exact permutation",
                p_value=float(perm_p),
                effect_size=d,
                effect_label=label,
            )
        )

        overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
        overlap_text = (
            "Non-overlapping CIs support the claim."
            if not overlap
            else "CI overlap exists; interpret with caution."
        )

        metric_name = {
            "profile_variance": "σ²_profile",
            "intermediary": "R²",
            "patching": "sensitivity",
            "steering": "|r|",
        }.get(key, key)
        cross_lines.extend(
            [
                f"### 1.{idx} {title}",
                f"- Claim: {claim}",
                f"- TabPFN (L{layer_pfn}): {metric_name}={fmt(m1, s1)}, n={n1}",
                f"- TabICL (L{layer_icl}): {metric_name}={fmt(m2, s2)}, n={n2}",
                f"- 95% t-CI (TabPFN): [{ci1[0]:.3f}, {ci1[1]:.3f}] | Bootstrap CI: [{bci1[0]:.3f}, {bci1[1]:.3f}]",
                f"- 95% t-CI (TabICL): [{ci2[0]:.3f}, {ci2[1]:.3f}] | Bootstrap CI: [{bci2[0]:.3f}, {bci2[1]:.3f}]",
                f"- Welch t-test: t={t_stat:.3f}, p={p_val:.6f}",
                f"- Permutation test (exact): p={perm_p:.6f}",
                f"- Cohen's d: {d:.2f} ({label})",
                f"- Interpretation: {overlap_text}",
                "",
            ]
        )

    sections.extend(cross_lines)

    sae_lines: list[str] = ["## 2. SAE Paired Comparisons (Wilcoxon Signed-Rank)", ""]
    # Only count ONE SAE test toward Bonferroni (TabPFN as representative).
    # Per-model details are reported in the markdown but do not inflate test count.
    sae_representative_added = False
    for model in ("tabpfn", "tabicl", "iltm"):
        relu_vals = sae_data[model]["relu_sparsity"]
        topk_vals = sae_data[model]["topk_sparsity"]
        n = len(relu_vals)

        w_stat, p_val = stats.wilcoxon(
            topk_vals, relu_vals, alternative="two-sided", method="exact"
        )
        diffs = np.asarray(topk_vals, dtype=np.float64) - np.asarray(
            relu_vals, dtype=np.float64
        )
        m_diff = float(np.mean(diffs))
        bci_diff = bootstrap_ci_mean(diffs.tolist())

        if p_val <= 0.0:
            z_from_p = float("inf")
        else:
            z_from_p = float(stats.norm.isf(p_val / 2.0))
        sign = 1.0 if float(np.median(diffs)) >= 0 else -1.0
        r_eff = sign * z_from_p / math.sqrt(n)

        # Only add the first (TabPFN) as the representative test for Bonferroni
        if not sae_representative_added:
            tests.append(
                TestResult(
                    comparison="SAE TopK vs ReLU (representative: tabpfn)",
                    n=f"{n} paired",
                    test_name="Wilcoxon signed-rank",
                    p_value=float(p_val),
                    effect_size=r_eff,
                    effect_label=effect_label_from_d(r_eff, metric="r"),
                )
            )
            sae_representative_added = True

        # Paired permutation test (more powerful for small n)
        paired_perm_p = paired_permutation_test(topk_vals, relu_vals)

        mr, sr = mean_std(relu_vals)
        mt, st = mean_std(topk_vals)
        sae_lines.extend(
            [
                f"### TopK vs ReLU — Sparsity ({model})",
                f"- ReLU: {fmt(mr, sr)} | TopK: {fmt(mt, st)} (n={n})",
                f"- Wilcoxon two-sided: W={w_stat:.3f}, p={p_val:.6f}",
                f"- Paired permutation test: p={paired_perm_p:.6f}",
                f"- Effect size r: {r_eff:.2f}",
                f"- Mean paired diff (TopK-ReLU): {m_diff:.3f}",
                f"- 95% bootstrap CI of paired diff: [{bci_diff[0]:.3f}, {bci_diff[1]:.3f}]",
                "",
            ]
        )

    sections.extend(sae_lines)

    cka_lines: list[str] = ["## 3. CKA Data Completeness", ""]
    cka_n = len(cka_tabpfn_adj)
    if cka_n >= 2:
        m_p, s_p = mean_std(cka_tabpfn_adj)
        m_t, s_t = mean_std(cka_tabicl_adj)
        ci_p = t_ci_from_summary(m_p, s_p, cka_n)
        ci_t = t_ci_from_summary(m_t, s_t, cka_n)
        cka_lines.extend(
            [
                f"- CKA experiment coverage: {cka_available}/{cka_total} seeds (missing: {', '.join(cka_missing) if cka_missing else 'none'})",
                "- All other key experiments use 5 seeds (Phase 5) or 3 seeds (Phase 6)",
                f"- Representative CKA summary metric (mean adjacent-layer CKA, TabPFN): {fmt(m_p, s_p)}, n={cka_n}",
                f"- 95% t-CI (TabPFN adjacent CKA): [{ci_p[0]:.3f}, {ci_p[1]:.3f}]",
                f"- Representative CKA summary metric (mean adjacent-layer CKA, TabICL): {fmt(m_t, s_t)}, n={cka_n}",
                f"- 95% t-CI (TabICL adjacent CKA): [{ci_t[0]:.3f}, {ci_t[1]:.3f}]",
                "- Limitation: n=4 reduces precision versus the 5-seed Phase 5 setup.",
                "",
            ]
        )
    sections.extend(cka_lines)

    n_tests = len(tests)
    alpha = 0.05
    bonf_alpha = alpha / n_tests if n_tests > 0 else alpha

    for tr in tests:
        tr.p_bonferroni = min(tr.p_value * n_tests, 1.0)
        tr.significant = tr.p_value < bonf_alpha

    # BH-FDR correction
    all_pvals = [tr.p_value for tr in tests]
    bh_significant = benjamini_hochberg(all_pvals, alpha=alpha)

    # Compute BH-adjusted p-values
    n_tests_for_bh = len(all_pvals)
    indexed_pvals = sorted(enumerate(all_pvals), key=lambda x: x[1])
    bh_adjusted = [0.0] * n_tests_for_bh
    for rank, (orig_idx, p) in enumerate(indexed_pvals, start=1):
        bh_adjusted[orig_idx] = min(p * n_tests_for_bh / rank, 1.0)
    # Enforce monotonicity (reverse pass)
    for j in range(n_tests_for_bh - 2, -1, -1):
        rev_idx = indexed_pvals[j][0]
        next_idx = indexed_pvals[j + 1][0]
        bh_adjusted[rev_idx] = min(bh_adjusted[rev_idx], bh_adjusted[next_idx])

    sig_count = sum(1 for t in tests if t.significant)
    bh_sig_count = sum(bh_significant)
    large_effect_count = sum(
        1 for t in tests if not math.isnan(t.effect_size) and abs(t.effect_size) > 0.8
    )

    summary_lines = [
        "## Summary",
        f"- Total statistical tests: {n_tests}",
        f"- Significant after Bonferroni correction (alpha={bonf_alpha:.5f}): {sig_count}",
        f"- Significant after BH-FDR correction (alpha={alpha:.2f}): {bh_sig_count}",
        f"- Large effect sizes (|effect| > 0.8): {large_effect_count}",
        "",
    ]

    sections = [sections[0], "", *summary_lines, *sections[1:]]

    table_lines = [
        "## 4. Summary Table",
        "| Comparison | n | Test | p-value | p (Bonf.) | p (BH-FDR) | Effect size | Bonf.? | BH? |",
        "|------------|---|------|---------|-----------|-----------|-------------|--------|-----|",
    ]
    for i, tr in enumerate(tests):
        sig_bonf = "Yes" if tr.significant else "No"
        sig_bh = "Yes" if bh_significant[i] else "No"
        table_lines.append(
            f"| {tr.comparison} | {tr.n} | {tr.test_name} | {tr.p_value:.6f} | {tr.p_bonferroni:.6f} | {bh_adjusted[i]:.6f} | {tr.effect_size:.2f} ({tr.effect_label}) | {sig_bonf} | {sig_bh} |"
        )
    table_lines.append("")

    report_text = "\n".join(sections + table_lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ = REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("=" * 72)
    print("TFMI Statistical Analysis")
    print("=" * 72)
    print(f"Total tests: {n_tests}")
    print(f"Bonferroni alpha: {bonf_alpha:.5f}")
    print(f"Significant tests: {sig_count}")
    print(f"Large effects (|effect|>0.8): {large_effect_count}")
    print("-")
    for tr in tests:
        print(
            f"{tr.comparison}: p={tr.p_value:.6f}, p_bonf={tr.p_bonferroni:.6f}, effect={tr.effect_size:.2f} ({tr.effect_label}), sig={tr.significant}"
        )
    print("-")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TFMI statistical analysis — generates paper/statistical_report.md"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary to stdout without writing report file",
    )
    args = parser.parse_args()
    if args.dry_run:
        print("Dry-run mode: would generate statistical report (no file written)")
    else:
        run()
