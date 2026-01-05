#!/usr/bin/env python3
"""
analyze_optpixel_scan.py

Reusable analysis pipeline for any scan CSV.
- Input: path to a CSV (e.g., results/csvs/optpixel_scan.csv)
- Output: plots saved under results/plots/<csv_stem>_results/ by default

By default, this script DOES NOT attempt to load per-geometry pT-curve files.
If you later generate curve files, run with:
  --with-curves --curves-dir results/curves
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Helpers
# =========================================================
def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_radii(radii_str: Any) -> List[float]:
    s = str(radii_str)
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return [float(x) for x in nums]


def safe_min_spacing(radii: List[float]) -> float:
    if len(radii) < 2:
        return np.nan
    return float(np.min(np.diff(radii)))


def safe_avg_spacing(radii: List[float]) -> float:
    if len(radii) < 2:
        return np.nan
    return float(np.mean(np.diff(radii)))


def calculate_cost_from_radii(
    radii_cm: List[float],
    half_length_cm: float = 25.0,
    cost_per_cm2: float = 200.0,
    fixed_per_layer: float = 475000.0,
) -> float:
    length_cm = 2.0 * half_length_cm
    total_cost_chf = 0.0
    for r in radii_cm:
        area_cm2 = 2.0 * np.pi * r * length_cm
        total_cost_chf += (area_cm2 * cost_per_cm2) + fixed_per_layer
    return total_cost_chf / 1e6  # MCHF


def compute_pareto_front_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    best_y = np.inf
    front = []
    for idx in order:
        if y[idx] < best_y:
            front.append(idx)
            best_y = y[idx]
    return np.array(front, dtype=int)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# =========================================================
# Main analysis function
# =========================================================
def analyze_scan_csv(
    csv_path: str | Path,
    plots_root: Optional[str | Path] = None,
    dpi: int = 150,
    drop_outliers_quantile: float = 0.98,
    with_curves: bool = False,
    curves_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Infer "results_dir" and default plots root:
    if csv_path.parent.name.lower() == "csvs" and csv_path.parent.parent.name.lower() == "results":
        results_dir = csv_path.parent.parent
    else:
        results_dir = csv_path.parent

    run_name = csv_path.stem
    plots_root_path = (results_dir / "plots") if plots_root is None else Path(plots_root).expanduser().resolve()
    plots_dir = plots_root_path / f"{run_name}_results"
    ensure_dir(plots_dir)

    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows")

    # Identify columns robustly
    col_radii = first_existing_col(df, ["radii_cm", "radii", "radii_str"])
    col_r1    = first_existing_col(df, ["r1_cm", "r1"])
    col_N     = first_existing_col(df, ["N", "n_layers", "nlayers"])
    col_d0    = first_existing_col(df, ["d0_1GeV_um", "d0_um", "d0"])
    col_pt    = first_existing_col(df, ["pt_100GeV_percent", "pt_pct", "pt_percent", "pt"])
    col_geom  = first_existing_col(df, ["geom_file", "geomPath", "geometry_file", "geom"])

    missing = [("radii", col_radii), ("d0", col_d0), ("pt", col_pt)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise ValueError(
            f"Missing required columns for: {missing}. "
            "CSV must contain radii + d0 + pt resolution."
        )

    # Parse radii list
    df["_radii_list"] = df[col_radii].apply(parse_radii)

    # Derive r1 if missing
    if col_r1 is None:
        df["r1_cm"] = df["_radii_list"].apply(lambda r: r[0] if len(r) else np.nan)
        col_r1 = "r1_cm"

    # Derive N if missing
    if col_N is None:
        df["N"] = df["_radii_list"].apply(len)
        col_N = "N"

    # Derived fields
    df["cost_MCHF"] = df["_radii_list"].apply(calculate_cost_from_radii)
    df["min_spacing_cm"] = df["_radii_list"].apply(safe_min_spacing)
    df["avg_spacing_cm"] = df["_radii_list"].apply(safe_avg_spacing)

    print(f"[INFO] Avg cost: {df['cost_MCHF'].mean():.2f} MCHF")

    # =========================================================
    # Task 5.1: Best-achievable performance vs r1
    # =========================================================
    print("[INFO] Plot: Task 5.1 (best vs r1)")
    df["r1_round"] = df[col_r1].round(2)
    best_by_r1 = (
        df.groupby("r1_round", as_index=False)
          .agg(best_d0=(col_d0, "min"), best_pt=(col_pt, "min"))
          .sort_values("r1_round")
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    (line1,) = ax1.plot(best_by_r1["r1_round"], best_by_r1["best_d0"], marker="o", linewidth=2,
                        label=r"$\sigma(d_0)$ @ 1 GeV")
    ax1.set_xlabel(r"First layer radius $r_1$ [cm]")
    ax1.set_ylabel(r"$\sigma(d_0)$ [$\mu$m]")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    (line2,) = ax2.plot(best_by_r1["r1_round"], best_by_r1["best_pt"], marker="s", linewidth=2,
                        label=r"$\sigma(p_T)/p_T$ @ 100 GeV")
    ax2.set_ylabel(r"$\sigma(p_T)/p_T$ [%]")

    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="upper center")
    plt.title("Task 5.1: Impact of First Layer Radius ($r_1$)")
    savefig(plots_dir / "task5_1_r1_study.png", dpi=dpi)

    # =========================================================
    # Task 5.2: Momentum resolution vs N (boxplot)
    # =========================================================
    print("[INFO] Plot: Task 5.2 (pt vs N)")
    Ns = sorted(df[col_N].dropna().unique())
    pt_data = [df.loc[df[col_N] == n, col_pt].dropna().values for n in Ns]

    plt.figure(figsize=(10, 6))
    plt.boxplot(pt_data, tick_labels=[f"N={int(n)}" for n in Ns], patch_artist=True)
    plt.xlabel("Number of layers")
    plt.ylabel(r"$\sigma(p_T)/p_T$ @ 100 GeV [%]")
    plt.title("Task 5.2: Momentum Resolution vs. Number of Layers")
    plt.grid(True, axis="y", alpha=0.4)
    savefig(plots_dir / "task5_2_layers_pt.png", dpi=dpi)

    # =========================================================
    # Task 5.3: Trade-off cloud colored by cost
    # =========================================================
    print("[INFO] Plot: Task 5.3 (tradeoff cloud)")
    d0_limit = df[col_d0].quantile(drop_outliers_quantile)
    pt_limit = df[col_pt].quantile(drop_outliers_quantile)
    df_cloud = df[(df[col_d0] < d0_limit) & (df[col_pt] < pt_limit)].copy()

    plt.figure(figsize=(11, 8))
    sc = plt.scatter(df_cloud[col_d0], df_cloud[col_pt], c=df_cloud["cost_MCHF"], s=20, alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Estimated Cost [MCHF]")

    plt.xlabel(r"$\sigma(d_0)$ @ 1 GeV [$\mu$m] (lower is better)")
    plt.ylabel(r"$\sigma(p_T)/p_T$ @ 100 GeV [%] (lower is better)")
    plt.title("Task 5.3: Resolution Trade-off (colored by cost)")
    plt.grid(True, alpha=0.3)
    savefig(plots_dir / "task5_3_tradeoff.png", dpi=dpi)

    # =========================================================
    # Task 6.1 selections: best_d0, best_pt, best compromise
    # =========================================================
    best_d0_row = df.loc[df[col_d0].idxmin()]
    best_pt_row = df.loc[df[col_pt].idxmin()]

    d0_min = float(df[col_d0].min())
    pt_min = float(df[col_pt].min())
    df["_compromise_score"] = (df[col_d0] / d0_min) + (df[col_pt] / pt_min)
    best_comp_row = df.loc[df["_compromise_score"].idxmin()]

    # =========================================================
    # Task 6.1-2: Heatmap best d0 vs (r1, avg spacing) [derived]
    # =========================================================
    print("[INFO] Plot: Task 6.1-2 (heatmap d0 vs r1 & avg spacing)")
    heat = df.dropna(subset=[col_r1, "avg_spacing_cm", col_d0]).copy()
    heat["r1_bin"] = heat[col_r1].round(2)
    heat["spacing_bin"] = heat["avg_spacing_cm"].round(2)

    pivot = (
        heat.pivot_table(index="spacing_bin", columns="r1_bin", values=col_d0, aggfunc="min")
        .sort_index()
        .sort_index(axis=1)
    )

    plt.figure(figsize=(11, 7))
    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            float(pivot.columns.min()), float(pivot.columns.max()),
            float(pivot.index.min()), float(pivot.index.max()),
        ],
    )
    plt.colorbar(label=r"Best $\sigma(d_0)$ @ 1 GeV [$\mu$m]")
    plt.xlabel(r"$r_1$ [cm]")
    plt.ylabel("Average layer spacing [cm] (derived)")
    plt.title(r"Task 6.1-2: Best $d_0$ vs $(r_1,\ \mathrm{avg\ spacing})$")
    savefig(plots_dir / "task6_2_heatmap_d0_r1_spacing.png", dpi=dpi)

    # =========================================================
    # Task 6.1-3: Pareto front (computed on FULL data)
    # =========================================================
    print("[INFO] Plot: Task 6.1-3 (Pareto front)")
    pf_all = df.dropna(subset=[col_d0, col_pt]).copy()
    x_all = pf_all[col_d0].to_numpy()
    y_all = pf_all[col_pt].to_numpy()
    front_idx = compute_pareto_front_indices(x_all, y_all)

    plt.figure(figsize=(11, 8))
    plt.scatter(df_cloud[col_d0], df_cloud[col_pt], s=18, alpha=0.35, edgecolors="none",
                label=f"Cloud (<= q{drop_outliers_quantile:.2f})")

    fx = x_all[front_idx]
    fy = y_all[front_idx]
    order = np.argsort(fx)
    plt.plot(fx[order], fy[order], linewidth=2.5, label="Pareto front (full data)")

    plt.xlabel(r"$\sigma(d_0)$ @ 1 GeV [$\mu$m] (lower is better)")
    plt.ylabel(r"$\sigma(p_T)/p_T$ @ 100 GeV [%] (lower is better)")
    plt.title("Task 6.1-3: Pareto front (d0 vs pT trade-off)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(plots_dir / "task6_3_pareto_front.png", dpi=dpi)

    # =========================================================
    # Task 6.1-4: Cost vs performance
    # =========================================================
    print("[INFO] Plot: Task 6.1-4 (cost vs improvement)")
    base = df.loc[df["cost_MCHF"].idxmin()]
    base_cost = float(base["cost_MCHF"])
    base_d0 = float(base[col_d0])
    base_pt = float(base[col_pt])

    tmp = df.dropna(subset=["cost_MCHF", col_d0, col_pt]).copy()
    tmp["rel_improve_d0"] = (base_d0 - tmp[col_d0]) / base_d0
    tmp["rel_improve_pt"] = (base_pt - tmp[col_pt]) / base_pt
    tmp["combined_improve"] = 0.5 * (tmp["rel_improve_d0"] + tmp["rel_improve_pt"])

    tmp["delta_cost"] = tmp["cost_MCHF"] - base_cost
    tmp["improve_per_MCHF"] = np.where(
        tmp["delta_cost"] > 1e-9,
        tmp["combined_improve"] / tmp["delta_cost"],
        np.nan,
    )

    plt.figure(figsize=(11, 7))
    plt.scatter(tmp["cost_MCHF"], tmp["combined_improve"], s=18, alpha=0.45, edgecolors="none")
    plt.axvline(base_cost, linestyle="--", linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Estimated cost [MCHF]")
    plt.ylabel("Combined relative improvement vs cheapest baseline")
    plt.title("Task 6.1-4: Cost vs performance (relative improvement)")
    plt.grid(True, alpha=0.3)
    savefig(plots_dir / "task6_4_cost_vs_improvement.png", dpi=dpi)

    plt.figure(figsize=(11, 7))
    tmp2 = tmp.dropna(subset=["improve_per_MCHF"])
    plt.scatter(tmp2["cost_MCHF"], tmp2["improve_per_MCHF"], s=18, alpha=0.45, edgecolors="none")
    plt.axvline(base_cost, linestyle="--", linewidth=1)
    plt.xlabel("Estimated cost [MCHF]")
    plt.ylabel("Improvement per MCHF (1/MCHF)")
    plt.title("Task 6.1-4: Resolution improvement per CHF (per MCHF)")
    plt.grid(True, alpha=0.3)
    savefig(plots_dir / "task6_4_improvement_per_MCHF.png", dpi=dpi)

    # =========================================================
    # Print summary
    # =========================================================
    def pretty_row(label: str, row: pd.Series) -> str:
        geom = row[col_geom] if (col_geom and col_geom in row.index) else "<no geom_file col>"
        radii = row[col_radii]
        return (
            f"\n--- {label} ---\n"
            f"Geom:  {geom}\n"
            f"Radii: {radii}\n"
            f"d0:    {float(row[col_d0]):.4f} um\n"
            f"pT:    {float(row[col_pt]):.4f} %\n"
            f"Cost:  {float(row['cost_MCHF']):.2f} MCHF\n"
        )

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"[INFO] Plots saved in: {plots_dir}")
    print(pretty_row("Best d0 configuration (low-pT)", best_d0_row))
    print(pretty_row("Best pT configuration (high-pT)", best_pt_row))
    print(pretty_row("Best compromise configuration", best_comp_row))

    return {
        "csv_path": str(csv_path),
        "results_dir": str(results_dir),
        "plots_dir": str(plots_dir),
        "best_d0_row": best_d0_row.to_dict(),
        "best_pt_row": best_pt_row.to_dict(),
        "best_compromise_row": best_comp_row.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an OptPixel scan CSV and generate plots.")
    parser.add_argument("csv", help="Path to scan CSV (e.g., results/csvs/optpixel_scan.csv)")
    parser.add_argument("--plots-root", default=None, help="Root directory for plots (default: inferred results/plots)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output figures")
    parser.add_argument("--outlier-q", type=float, default=0.98, help="Quantile cutoff for cloud plots (default 0.98)")
    # Curves disabled by default; keep flags for later
    parser.add_argument("--with-curves", action="store_true", help="Enable curve-file plots (Task 6.1-1)")
    parser.add_argument("--curves-dir", default=None, help="Directory containing <geom_stem>_results.txt curve files")
    args = parser.parse_args()

    analyze_scan_csv(
        csv_path=args.csv,
        plots_root=args.plots_root,
        dpi=args.dpi,
        drop_outliers_quantile=args.outlier_q,
        with_curves=args.with_curves,
        curves_dir=args.curves_dir,
    )


if __name__ == "__main__":
    main()
