import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 0. OUTPUT DIRECTORY (SAVE EVERYTHING TO results/)
# =========================================================
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)

def outpath(fname: str) -> str:
    return os.path.join(OUTDIR, fname)

# =========================================================
# 0. HELPERS (robust parsing + Pareto + curve loading)
# =========================================================
def first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_radii(radii_str):
    """
    Robustly parse radii from strings like:
      "1.0;2.0;3.0"
      "1.0, 2.0, 3.0"
      "[1.0, 2.0, 3.0]"
    Returns list[float].
    """
    s = str(radii_str)
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return [float(x) for x in nums]

def safe_min_spacing(radii):
    if len(radii) < 2:
        return np.nan
    return float(np.min(np.diff(radii)))

def safe_avg_spacing(radii):
    if len(radii) < 2:
        return np.nan
    return float(np.mean(np.diff(radii)))

def calculate_cost_from_radii(
    radii,
    half_length_cm=25.0,
    cost_per_cm2=200.0,
    fixed_per_layer=475000.0
):
    """
    Cost model:
      Cost = (Surface Area * 200 CHF/cm2) + (N_layers * 475,000 CHF)
    where surface area sums cylindrical barrel areas for each layer:
      A_layer = 2*pi*r*L, L = 2*half_length
    Returns MCHF.
    """
    length_cm = 2 * half_length_cm
    total_cost_chf = 0.0
    for r in radii:
        area = 2 * np.pi * r * length_cm
        total_cost_chf += (area * cost_per_cm2) + fixed_per_layer
    return total_cost_chf / 1e6  # MCHF

def find_curve_file_for_row(row: pd.Series):
    """
    Try to locate per-geometry curve file produced by EvaluateGeometry.C:
      results/<geomName>_results.txt
    We infer geomName from common columns (geom_file/geomPath/etc).
    """
    # direct columns if present
    for col in ["results_file", "result_file", "curve_file"]:
        if col in row.index and pd.notna(row[col]):
            p = str(row[col])
            if os.path.exists(p):
                return p

    geom_col = None
    for c in ["geom_file", "geomPath", "geometry_file", "geom"]:
        if c in row.index:
            geom_col = c
            break

    if geom_col and pd.notna(row[geom_col]):
        geom_path = str(row[geom_col])
        base = os.path.basename(geom_path)
        name = os.path.splitext(base)[0]
        candidates = [
            os.path.join(OUTDIR, f"{name}_results.txt"),
            os.path.join(OUTDIR, f"{base}_results.txt"),
            os.path.join(OUTDIR, f"{name}.txt"),
        ]
        for f in candidates:
            if os.path.exists(f):
                return f

    return None

def read_curve_file(curve_path: str) -> pd.DataFrame:
    """
    Parse curve files like:
      # pT[GeV] d0[um] pt[%]
      0.5  123.4  56.7
    """
    dfc = pd.read_csv(
        curve_path,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=["pt_GeV", "d0_um", "pt_pct"]
    ).dropna()
    return dfc

def compute_pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Minimize x and y. Returns indices of Pareto-optimal points (lower envelope)
    after sorting by x.
    """
    order = np.argsort(x)
    best_y = np.inf
    front_idx = []
    for idx in order:
        if y[idx] < best_y:
            front_idx.append(idx)
            best_y = y[idx]
    return np.array(front_idx, dtype=int)

# =========================================================
# 1. LOAD DATA
# =========================================================
print("Loading results...")
csv_path = os.path.join(OUTDIR, "optpixel_scan.csv")
if not os.path.exists(csv_path):
    # allow original relative path too
    csv_path = "results/optpixel_scan.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} geometries from {csv_path}.")
except FileNotFoundError:
    print("Error: 'results/optpixel_scan.csv' not found. Run the scan first!")
    raise SystemExit(1)

# Identify columns robustly (supports older naming)
col_radii = first_existing_col(df, ["radii_cm", "radii", "radii_str"])
col_r1    = first_existing_col(df, ["r1_cm", "r1"])
col_N     = first_existing_col(df, ["N", "n_layers", "nlayers"])
col_d0    = first_existing_col(df, ["d0_1GeV_um", "d0_um", "d0"])
col_pt    = first_existing_col(df, ["pt_100GeV_percent", "pt_pct", "pt_percent", "pt"])
col_geom  = first_existing_col(df, ["geom_file", "geomPath", "geometry_file", "geom"])

missing = [("radii", col_radii), ("d0", col_d0), ("pt", col_pt)]
missing = [name for name, col in missing if col is None]
if missing:
    print(f"ERROR: Missing required columns for: {missing}")
    print("Your CSV must contain at least radii + d0 + pt resolution columns.")
    raise SystemExit(1)

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

# =========================================================
# 2. COST + SPACING FIELDS
# =========================================================
df["cost_MCHF"] = df["_radii_list"].apply(calculate_cost_from_radii)
df["min_spacing_cm"] = df["_radii_list"].apply(safe_min_spacing)
df["avg_spacing_cm"] = df["_radii_list"].apply(safe_avg_spacing)

print(f"Cost calculation complete. Average cost: {df['cost_MCHF'].mean():.2f} MCHF")

# =========================================================
# 3. TASK 5.1: Best-achievable performance vs r1 (dual axis)
# =========================================================
print("Generating Plot 5.1...")
df["r1_round"] = df[col_r1].round(2)

best_by_r1 = (
    df.groupby("r1_round", as_index=False)
      .agg(
          best_d0=(col_d0, "min"),
          best_pt=(col_pt, "min")
      )
      .sort_values("r1_round")
)

fig, ax1 = plt.subplots(figsize=(10, 6))
line1, = ax1.plot(
    best_by_r1["r1_round"], best_by_r1["best_d0"],
    marker="o", linewidth=2, color="tab:blue",
    label=r"$\sigma(d_0)$ at 1 GeV"
)
ax1.set_xlabel(r"First layer radius $r_1$ [cm]", fontsize=12)
ax1.set_ylabel(r"$\sigma(d_0)$ [$\mu$m]", color="tab:blue", fontsize=12)
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
line2, = ax2.plot(
    best_by_r1["r1_round"], best_by_r1["best_pt"],
    marker="s", linewidth=2, color="tab:red",
    label=r"$\sigma(p_T)/p_T$ at 100 GeV"
)
ax2.set_ylabel(r"$\sigma(p_T)/p_T$ [%]", color="tab:red", fontsize=12)
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="upper center")
plt.title("Task 5.1: Impact of First Layer Radius ($r_1$)", fontsize=14)
plt.tight_layout()
plt.savefig(outpath("task5_1_r1_study.png"), dpi=150)
plt.close()

# =========================================================
# 4. TASK 5.2: Momentum resolution vs N (boxplot)
# =========================================================
print("Generating Plot 5.2...")
Ns = sorted(df[col_N].dropna().unique())
pt_data = [df.loc[df[col_N] == n, col_pt].dropna().values for n in Ns]

plt.figure(figsize=(10, 6))
plt.boxplot(pt_data, labels=[f"N={n}" for n in Ns], patch_artist=True)
plt.xlabel("Number of layers", fontsize=12)
plt.ylabel(r"$\sigma(p_T)/p_T$ at 100 GeV [%]", fontsize=12)
plt.title("Task 5.2: Momentum Resolution vs. Number of Layers", fontsize=14)
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(outpath("task5_2_layers_pt.png"), dpi=150)
plt.close()

# =========================================================
# 5. TASK 5.3: Trade-off cloud colored by cost
# =========================================================
print("Generating Plot 5.3...")
plt.figure(figsize=(11, 8))

d0_limit = df[col_d0].quantile(0.98)
pt_limit = df[col_pt].quantile(0.98)
df_clean = df[(df[col_d0] < d0_limit) & (df[col_pt] < pt_limit)].copy()

sc = plt.scatter(
    df_clean[col_d0], df_clean[col_pt],
    c=df_clean["cost_MCHF"],
    cmap="viridis",
    s=20,
    alpha=0.7,
    edgecolors="none"
)
cbar = plt.colorbar(sc)
cbar.set_label("Estimated Cost [MCHF]", fontsize=12)

plt.xlabel(r"$\sigma(d_0)$ at 1 GeV [$\mu$m] $\rightarrow$ (Lower is Better)", fontsize=12)
plt.ylabel(r"$\sigma(p_T)/p_T$ at 100 GeV [%] $\rightarrow$ (Lower is Better)", fontsize=12)
plt.title("Task 5.3: Resolution Trade-off (Cloud, colored by cost)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outpath("task5_3_tradeoff.png"), dpi=150)
plt.close()

# =========================================================
# 6. TASK 6.1 PLOTS
#    1) Resolution vs pT curves for best configs
#    2) Heatmap: d0 vs (r1, spacing)
#    3) Pareto front plot
#    4) Cost vs performance: improvement per CHF
# =========================================================
print("Generating Task 6.1 plots...")

# ---- Best configurations: best d0, best pt, best compromise
best_d0_row = df.loc[df[col_d0].idxmin()]
best_pt_row = df.loc[df[col_pt].idxmin()]

d0_min = float(df[col_d0].min())
pt_min = float(df[col_pt].min())
df["_compromise_score"] = (df[col_d0] / d0_min) + (df[col_pt] / pt_min)
best_comp_row = df.loc[df["_compromise_score"].idxmin()]

candidates = [
    ("best_d0", best_d0_row),
    ("best_pt", best_pt_row),
    ("best_compromise", best_comp_row),
]

# ---------------------------------------------------------
# 6.1-1: Resolution vs pT curves (requires per-geometry *_results.txt)
# ---------------------------------------------------------
curves = []
for label, row in candidates:
    curve_file = find_curve_file_for_row(row)
    if curve_file is None:
        print(f"  [WARN] No pT-curve file found for {label}. Skipping curve.")
        continue
    try:
        cdf = read_curve_file(curve_file)
        cdf["label"] = label
        curves.append(cdf)
        print(f"  Found curve for {label}: {curve_file}")
    except Exception as e:
        print(f"  [WARN] Failed to read curve for {label} from {curve_file}: {e}")

if curves:
    curves_df = pd.concat(curves, ignore_index=True)

    # d0 vs pT
    plt.figure(figsize=(10, 6))
    for label in curves_df["label"].unique():
        sub = curves_df[curves_df["label"] == label].sort_values("pt_GeV")
        plt.plot(sub["pt_GeV"], sub["d0_um"], marker="o", linewidth=2, label=label)
    plt.xscale("log")
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel(r"$\sigma(d_0)$ [$\mu$m]")
    plt.title("Task 6.1-1: Impact parameter resolution vs $p_T$ (best configs)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath("task6_1_resolution_vs_pt_d0.png"), dpi=150)
    plt.close()

    # pt resolution vs pT
    plt.figure(figsize=(10, 6))
    for label in curves_df["label"].unique():
        sub = curves_df[curves_df["label"] == label].sort_values("pt_GeV")
        plt.plot(sub["pt_GeV"], sub["pt_pct"], marker="s", linewidth=2, label=label)
    plt.xscale("log")
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel(r"$\sigma(p_T)/p_T$ [%]")
    plt.title("Task 6.1-1: Momentum resolution vs $p_T$ (best configs)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath("task6_1_resolution_vs_pt_pt.png"), dpi=150)
    plt.close()
else:
    print("  [INFO] No curve files found, so resolution-vs-pT plots were not produced.")
    print("         Ensure results/<geomName>_results.txt exists for best geometries.")

# ---------------------------------------------------------
# 6.1-2: Heatmap: best d0 vs (r1, spacing)
# Use avg_spacing_cm as 'spacing' (robust even for random scans).
# ---------------------------------------------------------
heat = df.dropna(subset=[col_r1, "avg_spacing_cm", col_d0]).copy()
heat["r1_bin"] = heat[col_r1].round(2)
heat["spacing_bin"] = heat["avg_spacing_cm"].round(2)

pivot = heat.pivot_table(
    index="spacing_bin",
    columns="r1_bin",
    values=col_d0,
    aggfunc="min"
).sort_index().sort_index(axis=1)

plt.figure(figsize=(11, 7))
plt.imshow(
    pivot.values,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[
        float(pivot.columns.min()), float(pivot.columns.max()),
        float(pivot.index.min()), float(pivot.index.max())
    ],
)
plt.colorbar(label=r"Best $\sigma(d_0)$ at 1 GeV [$\mu$m]")
plt.xlabel(r"$r_1$ [cm]")
plt.ylabel(r"Average layer spacing [cm]")
plt.title(r"Task 6.1-2: Best $d_0$ vs $(r_1,\ \mathrm{spacing})$ heatmap")
plt.tight_layout()
plt.savefig(outpath("task6_2_heatmap_d0_r1_spacing.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# 6.1-3: Pareto front (cloud + extracted Pareto front overlay)
# ---------------------------------------------------------
pf = df_clean.dropna(subset=[col_d0, col_pt]).copy()
x = pf[col_d0].to_numpy()
y = pf[col_pt].to_numpy()
front_idx = compute_pareto_front(x, y)

plt.figure(figsize=(11, 8))
plt.scatter(x, y, s=18, alpha=0.35, edgecolors="none", label="All configs (filtered)")

fx = x[front_idx]
fy = y[front_idx]
order = np.argsort(fx)
plt.plot(fx[order], fy[order], linewidth=2.5, label="Pareto front")

plt.xlabel(r"$\sigma(d_0)$ at 1 GeV [$\mu$m] (lower is better)")
plt.ylabel(r"$\sigma(p_T)/p_T$ at 100 GeV [%] (lower is better)")
plt.title("Task 6.1-3: Pareto front (d0 vs pT trade-off)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(outpath("task6_3_pareto_front.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# 6.1-4: Cost vs performance (improvement per CHF)
# Baseline = cheapest configuration (min cost).
# combined_improve = average relative improvement in (d0, pt).
# improve_per_MCHF = combined_improve / (cost - baseline_cost).
# ---------------------------------------------------------
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
    np.nan
)

# Cost vs combined improvement
plt.figure(figsize=(11, 7))
plt.scatter(tmp["cost_MCHF"], tmp["combined_improve"], s=18, alpha=0.45, edgecolors="none")
plt.axvline(base_cost, linestyle="--", linewidth=1)
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Estimated cost [MCHF]")
plt.ylabel("Combined relative improvement vs cheapest baseline")
plt.title("Task 6.1-4: Cost vs performance (relative improvement)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outpath("task6_4_cost_vs_improvement.png"), dpi=150)
plt.close()

# Cost vs improvement per MCHF
plt.figure(figsize=(11, 7))
tmp2 = tmp.dropna(subset=["improve_per_MCHF"])
plt.scatter(tmp2["cost_MCHF"], tmp2["improve_per_MCHF"], s=18, alpha=0.45, edgecolors="none")
plt.axvline(base_cost, linestyle="--", linewidth=1)
plt.xlabel("Estimated cost [MCHF]")
plt.ylabel("Improvement per MCHF (1/MCHF)")
plt.title("Task 6.1-4: Resolution improvement per CHF (normalized per MCHF)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outpath("task6_4_improvement_per_MCHF.png"), dpi=150)
plt.close()

# =========================================================
# 7. REPORT OUTPUT: Best Candidates
# =========================================================
print("\n=== ANALYSIS COMPLETE ===")
print(f"All plots saved under: {OUTDIR}/")
print("Task 5 plots:")
print(f"  - {outpath('task5_1_r1_study.png')}")
print(f"  - {outpath('task5_2_layers_pt.png')}")
print(f"  - {outpath('task5_3_tradeoff.png')}")
print("Task 6.1 plots:")
print(f"  - {outpath('task6_2_heatmap_d0_r1_spacing.png')}")
print(f"  - {outpath('task6_3_pareto_front.png')}")
print(f"  - {outpath('task6_4_cost_vs_improvement.png')}")
print(f"  - {outpath('task6_4_improvement_per_MCHF.png')}")
print("Resolution-vs-pT curves (only if curve files exist):")
print(f"  - {outpath('task6_1_resolution_vs_pt_d0.png')}")
print(f"  - {outpath('task6_1_resolution_vs_pt_pt.png')}")

def pretty_row(label, row):
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

print(pretty_row("Best d0 configuration (low-pT physics)", best_d0_row))
print(pretty_row("Best pT configuration (high-pT physics)", best_pt_row))
print(pretty_row("Best compromise configuration", best_comp_row))
