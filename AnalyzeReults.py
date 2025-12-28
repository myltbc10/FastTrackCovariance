import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. SETUP AND DATA LOADING
# ---------------------------------------------------------
# Load the data
df = pd.read_csv("results/optpixel_scan.csv")

# Constants for Cost Calculation (from PDF Task 4.1)
# Cost = Silicon Area * 200 CHF/cm2 + Layers * 475,000 CHF
HALF_LENGTH_CM = 25.0
COST_PER_CM2 = 200.0
FIXED_PER_LAYER = 475000.0

def calculate_cost(radii_str):
    # Parse "1.0;2.0;3.0" into a list of floats
    radii = [float(x) for x in radii_str.split(';')]
    
    total_cost = 0.0
    for r in radii:
        # Surface area of a cylinder = 2*pi*r * length (length = 2 * half_length)
        area = 2 * np.pi * r * (2 * HALF_LENGTH_CM)
        total_cost += area * COST_PER_CM2 + FIXED_PER_LAYER
        
    return total_cost / 1e6  # Return in MCHF (Millions)

# Apply cost calculation to every row
df['cost_MCHF'] = df['radii_cm'].apply(calculate_cost)

print(f"Loaded {len(df)} geometries.")
print(df.head())

# ---------------------------------------------------------
# 2. PLOT 1: First Layer Position Study (Task 5.1)
# ---------------------------------------------------------
# We want to see how d0 depends on r1.
# Since we have many geometries for each r1, we plot the "Best" (lowest d0) 
# for each r1 value to see the theoretical limit.

plt.figure(figsize=(10, 6))

# Group by r1 and find the minimum d0 for that specific radius
# We round r1 to 1 decimal place to group them properly
df['r1_rounded'] = df['r1_cm'].round(1)
best_d0_per_r1 = df.groupby('r1_rounded')['d0_1GeV_um'].min()

plt.plot(best_d0_per_r1.index, best_d0_per_r1.values, 'o-', linewidth=2, color='blue', label='Best Achievable $d_0$')

# Optional: Scatter all points to show the spread
plt.scatter(df['r1_cm'], df['d0_1GeV_um'], alpha=0.1, color='gray', s=5, label='All Geometries')

plt.title(r'Impact Parameter Resolution vs. First Layer Radius ($r_1$)', fontsize=14)
plt.xlabel(r'First Layer Radius $r_1$ [cm]', fontsize=12)
plt.ylabel(r'$\sigma(d_0)$ at 1 GeV [$\mu m$]', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("plot_5_1_r1_study.png")
print("Saved plot_5_1_r1_study.png")

# ---------------------------------------------------------
# 3. PLOT 2: Number of Layers Study (Task 5.2)
# ---------------------------------------------------------
# Compare performance distributions for N=3, 4, 5, 6
# We look at pT resolution since adding layers helps pattern recognition and pT more than d0

plt.figure(figsize=(10, 6))

data_to_plot = []
labels = []

for n_layers in sorted(df['N'].unique()):
    subset = df[df['N'] == n_layers]['pt_100GeV_percent']
    data_to_plot.append(subset)
    labels.append(f"N={n_layers}")

plt.boxplot(data_to_plot, labels=labels, patch_artist=True)

plt.title(r'Momentum Resolution Distribution by Number of Layers', fontsize=14)
plt.xlabel('Number of Layers', fontsize=12)
plt.ylabel(r'$\sigma(p_T)/p_T$ at 100 GeV [%]', fontsize=12)
plt.grid(True, axis='y', ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig("plot_5_2_layers_study.png")
print("Saved plot_5_2_layers_study.png")

# ---------------------------------------------------------
# 4. PLOT 3: Trade-off Analysis / Pareto Front (Task 5.3)
# ---------------------------------------------------------
# Scatter plot of d0 vs pT, colored by Cost
# The "Pareto Front" is the bottom-left boundary of this cloud.

plt.figure(figsize=(11, 8))

sc = plt.scatter(df['d0_1GeV_um'], df['pt_100GeV_percent'], 
                 c=df['cost_MCHF'], cmap='viridis', 
                 alpha=0.6, s=15, edgecolors='none')

cbar = plt.colorbar(sc)
cbar.set_label('Estimated Cost [MCHF]', fontsize=12)

plt.title('Pareto Front: Resolution Trade-off', fontsize=16)
plt.xlabel(r'$\sigma(d_0)$ at 1 GeV [$\mu m$] (Lower is Better)', fontsize=14)
plt.ylabel(r'$\sigma(p_T)/p_T$ at 100 GeV [%] (Lower is Better)', fontsize=14)

# Set limits to focus on the interesting region (ignore very bad geometries)
# You might need to adjust these based on your specific data range
plt.xlim(0, df['d0_1GeV_um'].quantile(0.95)) 
plt.ylim(0, df['pt_100GeV_percent'].quantile(0.95))

plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig("plot_5_3_pareto_front.png")
print("Saved plot_5_3_pareto_front.png")

# ---------------------------------------------------------
# 5. BONUS: Identify Best Candidates for Report
# ---------------------------------------------------------
# Find the specific row with the absolute best d0
best_d0_idx = df['d0_1GeV_um'].idxmin()
print("\n--- Best d0 Configuration ---")
print(df.loc[best_d0_idx])

# Find the specific row with the absolute best pT
best_pt_idx = df['pt_100GeV_percent'].idxmin()
print("\n--- Best pT Configuration ---")
print(df.loc[best_pt_idx])