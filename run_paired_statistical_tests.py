from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# User settings
# ============================================================

# Put your DETAIL csv path here, not the summary csv.
# Example:
# INPUT_DETAIL_CSV = Path("Results/multiseed_aggregation_ablation_detail_20260516_144841.csv")
INPUT_DETAIL_CSV = Path("Results/multiseed_aggregation_ablation_detail_20260516_144841.csv")

RESULTS_DIR = Path("Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "paired_statistical_tests_plain_vs_proposed.csv"
OUTPUT_LATEX = RESULTS_DIR / "paired_statistical_tests_plain_vs_proposed_table.tex"

BASELINE_MODEL = "PlainSSM"
PROPOSED_MODEL = "Proposed_Unnormalized_Base"

METRICS = ["accuracy", "macro_f1"]


# ============================================================
# Helpers
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """
    Returns paired Wilcoxon signed-rank p-value for y - x.

    With only five seeds per dataset, this test has low power.
    The output should be interpreted as supporting evidence, not definitive inference.
    """
    diff = y - x
    diff = diff[np.isfinite(diff)]

    if len(diff) < 2:
        return np.nan

    # Wilcoxon cannot test if all differences are exactly zero.
    if np.allclose(diff, 0):
        return 1.0

    if not SCIPY_AVAILABLE:
        return np.nan

    try:
        # two-sided test; exact is used automatically when appropriate.
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def fmt_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    return f"{p:.4f}{stars(p)}"


# ============================================================
# Main analysis
# ============================================================

def main() -> None:
    if not INPUT_DETAIL_CSV.exists():
        raise FileNotFoundError(
            f"Could not find detail CSV: {INPUT_DETAIL_CSV}\n"
            "Please set INPUT_DETAIL_CSV to your multiseed_aggregation_ablation_detail_*.csv file."
        )

    df = pd.read_csv(INPUT_DETAIL_CSV)

    required_cols = {"dataset", "model", "seed", "status", "accuracy", "macro_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in detail CSV: {missing}")

    df = df[df["status"] == "success"].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    rows: List[Dict] = []

    datasets = sorted(df["dataset"].dropna().unique().tolist())

    # Per-dataset paired tests across the five seeds.
    for dataset in datasets:
        d = df[df["dataset"] == dataset].copy()

        for metric in METRICS:
            pivot = d.pivot_table(
                index="seed",
                columns="model",
                values=metric,
                aggfunc="first",
            )

            if BASELINE_MODEL not in pivot.columns or PROPOSED_MODEL not in pivot.columns:
                continue

            pair = pivot[[BASELINE_MODEL, PROPOSED_MODEL]].dropna()
            base = pair[BASELINE_MODEL].to_numpy(dtype=float)
            prop = pair[PROPOSED_MODEL].to_numpy(dtype=float)
            diff = prop - base

            p_value = paired_wilcoxon(base, prop)

            rows.append({
                "scope": "per_dataset",
                "dataset": dataset,
                "metric": metric,
                "n_pairs": len(pair),
                "baseline_mean": np.mean(base) if len(base) else np.nan,
                "baseline_std": np.std(base, ddof=1) if len(base) > 1 else 0.0,
                "proposed_mean": np.mean(prop) if len(prop) else np.nan,
                "proposed_std": np.std(prop, ddof=1) if len(prop) > 1 else 0.0,
                "mean_difference": np.mean(diff) if len(diff) else np.nan,
                "median_difference": np.median(diff) if len(diff) else np.nan,
                "wilcoxon_p_value": p_value,
            })

    # Pooled paired test across dataset-seed pairs.
    # This is useful as a global supporting analysis, but it should be described cautiously
    # because observations from the same dataset are not fully independent.
    for metric in METRICS:
        pivot = df.pivot_table(
            index=["dataset", "seed"],
            columns="model",
            values=metric,
            aggfunc="first",
        )

        if BASELINE_MODEL not in pivot.columns or PROPOSED_MODEL not in pivot.columns:
            continue

        pair = pivot[[BASELINE_MODEL, PROPOSED_MODEL]].dropna()
        base = pair[BASELINE_MODEL].to_numpy(dtype=float)
        prop = pair[PROPOSED_MODEL].to_numpy(dtype=float)
        diff = prop - base

        p_value = paired_wilcoxon(base, prop)

        rows.append({
            "scope": "pooled_dataset_seed_pairs",
            "dataset": "All dataset--seed pairs",
            "metric": metric,
            "n_pairs": len(pair),
            "baseline_mean": np.mean(base) if len(base) else np.nan,
            "baseline_std": np.std(base, ddof=1) if len(base) > 1 else 0.0,
            "proposed_mean": np.mean(prop) if len(prop) else np.nan,
            "proposed_std": np.std(prop, ddof=1) if len(prop) > 1 else 0.0,
            "mean_difference": np.mean(diff) if len(diff) else np.nan,
            "median_difference": np.median(diff) if len(diff) else np.nan,
            "wilcoxon_p_value": p_value,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    # Create compact LaTeX table for per-dataset results only.
    per = out[out["scope"] == "per_dataset"].copy()

    metric_label = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
    }

    with open(OUTPUT_LATEX, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Paired Wilcoxon signed-rank tests comparing PlainSSM and Proposed Base across five random seeds. The tests are used as supporting evidence because each dataset has only five paired runs.}" + "\n")
        f.write(r"\label{tab:paired_tests}" + "\n")
        f.write(r"\resizebox{\textwidth}{!}{" + "\n")
        f.write(r"\begin{tabular}{llccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Dataset & Metric & PlainSSM & Proposed Base & Mean diff. & Median diff. & \(p\)-value \\" + "\n")
        f.write(r"\midrule" + "\n")

        for _, r in per.iterrows():
            f.write(
                f"{r['dataset']} & "
                f"{metric_label.get(r['metric'], r['metric'])} & "
                f"{fmt_mean_std(r['baseline_mean'], r['baseline_std'])} & "
                f"{fmt_mean_std(r['proposed_mean'], r['proposed_std'])} & "
                f"{r['mean_difference']:.3f} & "
                f"{r['median_difference']:.3f} & "
                f"{fmt_p(r['wilcoxon_p_value'])} \\\\\n"
            )

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")

    print("Statistical testing completed.")
    print("Input detail CSV:", INPUT_DETAIL_CSV)
    print("Output CSV:", OUTPUT_CSV)
    print("Output LaTeX:", OUTPUT_LATEX)

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy was not available, so Wilcoxon p-values were not computed.")


if __name__ == "__main__":
    main()
