#!/usr/bin/env python3
"""Generate per-seed result tables for appendix supplementary material.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/generate_perseed_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RD5_DIR = ROOT / "results" / "rd5_fullscale"
RD6_DIR = ROOT / "results" / "rd6_fullscale"
SAE_TOPK_DIR = ROOT / "results" / "rd6" / "sae_topk_fullscale"
OUTPUT = ROOT / "paper" / "supplementary_tables.md"

SEEDS = [42, 123, 456, 789, 1024]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def steering_table() -> list[str]:
    """Per-seed robust steering results."""
    lines = [
        "### Table S1: Per-Seed Robust Steering |r| by Layer",
        "",
        "| Seed | Model | L0 | L4 | L6 | L8 | L11 | Best Layer |",
        "|------|-------|----|----|----|----|-----|------------|",
    ]
    for seed in SEEDS:
        path = RD6_DIR / f"seed_{seed}" / "robust_steering" / "results.json"
        if not path.exists():
            continue
        data = read_json(path)
        for model in ["tabpfn", "tabicl"]:
            if model not in data:
                continue
            per_layer = data[model].get("robust", {}).get("per_layer", {})
            vals = {}
            best_layer = ""
            best_val = 0.0
            for lk, ld in per_layer.items():
                v = abs(float(ld.get("abs_pearson_r", 0.0)))
                vals[lk] = v
                if v > best_val:
                    best_val = v
                    best_layer = f"L{lk}"
            row = f"| {seed} | {model.upper()} |"
            for layer in ["0", "4", "6", "8", "11"]:
                v = vals.get(layer, float("nan"))
                row += f" {v:.3f} |"
            row += f" {best_layer} |"
            lines.append(row)
    lines.append("")
    return lines


def sae_topk_table() -> list[str]:
    """Per-seed SAE TopK results."""
    lines = [
        "### Table S2: Per-Seed SAE Results (16× expansion)",
        "",
        "| Seed | Model | Variant | Recon R² | Sparsity | Max |r_α| |",
        "|------|-------|---------|----------|----------|-----------|",
    ]
    for seed in SEEDS:
        path = SAE_TOPK_DIR / f"seed_{seed}" / "results.json"
        if not path.exists():
            continue
        data = read_json(path)
        results = data.get("results", {})
        for variant in ["relu_16x", "jumprelu_16x", "topk_16x_6p25"]:
            for model in ["tabpfn", "tabicl", "iltm"]:
                md = results.get(variant, {}).get(model, {})
                if not md:
                    continue
                r2 = float(md.get("reconstruction_r2", 0.0))
                sp = float(md.get("sparsity", 0.0))
                ac = float(md.get("max_alpha_corr", 0.0))
                short = variant.replace("_16x", "").replace("_6p25", "")
                lines.append(
                    f"| {seed} | {model.upper()} | {short} | {r2:.4f} | {sp:.3f} | {ac:.3f} |"
                )
    lines.append("")
    return lines


def intermediary_table() -> list[str]:
    """Per-seed intermediary probing peak R²."""
    lines = [
        "### Table S3: Per-Seed Intermediary Probing Peak R²",
        "",
        "| Seed | TabPFN Peak Layer | TabPFN R² | TabICL Peak Layer | TabICL R² | iLTM Peak Layer | iLTM R² |",
        "|------|-------------------|-----------|-------------------|-----------|-----------------|---------|",
    ]
    for seed in SEEDS:
        path = RD5_DIR / f"seed_{seed}" / "intermediary_probing" / "results.json"
        if not path.exists():
            continue
        data = read_json(path)
        row = f"| {seed}"
        for model in ["tabpfn", "tabicl", "iltm"]:
            md = data.get(model, {})
            r2_by_layer = md.get("intermediary_r2_by_layer", [])
            if r2_by_layer:
                peak_idx = max(range(len(r2_by_layer)), key=lambda i: r2_by_layer[i])
                peak_val = r2_by_layer[peak_idx]
                row += f" | L{peak_idx} | {peak_val:.4f}"
            else:
                row += " | — | —"
        row += " |"
        lines.append(row)
    lines.append("")
    return lines


def patching_table() -> list[str]:
    """Per-seed patching sensitivity peak."""
    lines = [
        "### Table S4: Per-Seed Patching Sensitivity (Peak Layer)",
        "",
        "| Seed | TabPFN Peak Layer | TabPFN Sensitivity | TabICL Peak Layer | TabICL Sensitivity |",
        "|------|-------------------|--------------------|-------------------|--------------------|",
    ]
    for seed in SEEDS:
        path = RD5_DIR / f"seed_{seed}" / "patching" / "results.json"
        if not path.exists():
            continue
        data = read_json(path)
        row = f"| {seed}"
        for model in ["tabpfn", "tabicl"]:
            md = data.get(model, {})
            sens = md.get("summary_sensitivity", [])
            if sens:
                peak_idx = max(range(len(sens)), key=lambda i: sens[i])
                peak_val = sens[peak_idx]
                row += f" | L{peak_idx} | {peak_val:.4f}"
            else:
                row += " | — | —"
        row += " |"
        lines.append(row)
    lines.append("")
    return lines


def main() -> int:
    lines = [
        "# Supplementary Tables — Per-Seed Results",
        "",
        "All results reported per random seed for full transparency.",
        "Seeds: {42, 123, 456, 789, 1024}.",
        "",
    ]
    lines.extend(intermediary_table())
    lines.extend(patching_table())
    lines.extend(steering_table())
    lines.extend(sae_topk_table())

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Supplementary tables written to: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
