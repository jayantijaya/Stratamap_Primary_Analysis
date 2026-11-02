import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------
# Step 1: Read normalized H5AD file
# --------------------------------------------------

input_file = ("/home/users/jayanthi/stratamap_hippocampus_normalized_protein.h5ad")

adata_protein_filtered = sc.read_h5ad(input_file)

print("Dataset:", adata_protein_filtered)
print("Cell bins:", adata_protein_filtered.n_obs)
print("Genes:", adata_protein_filtered.n_vars)

# --------------------------------------------------
# Step 2: Select gene
# --------------------------------------------------

gene = "PROX1"

if gene not in adata_protein_filtered.var_names:
    raise ValueError(f"{gene} not found in the dataset.")

# --------------------------------------------------
# Step 3: Extract spatial coordinates
# --------------------------------------------------

if "spatial" not in adata_protein_filtered.obsm:
    raise ValueError("Spatial coordinates not found in adata.obsm['spatial'].")

coords = adata_protein_filtered.obsm["spatial"]

# --------------------------------------------------
# Step 4: Extract normalized expression
# --------------------------------------------------

expr = adata_protein_filtered[:, gene].X

if hasattr(expr, "toarray"):
    expr = expr.toarray().ravel()
else:
    expr = np.asarray(expr).ravel()

print("Minimum expression:", expr.min())
print("Maximum expression:", expr.max())

# --------------------------------------------------
# Step 5: Threshold-based spatial expression plot
# --------------------------------------------------

threshold = 2
mask = expr > threshold

print("Bins expressing gene:", mask.sum())
print(
    "Percentage of bins:",
    round(mask.mean() * 100, 2),
    "%"
)

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(
    coords[mask, 0],
    coords[mask, 1],
    color="green",
    s=3,
    alpha=0.8,
    linewidths=0,
    rasterized=True
)

ax.set_xlabel("X coordinate", fontsize=12)
ax.set_ylabel("Y coordinate", fontsize=12)

ax.set_title(
    f"Spatial Expression of {gene} (Normalized Expression > {threshold})",
    fontsize=14
)

ax.invert_yaxis()
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

# Save threshold plot
threshold_output = (f"/home/users/jayanthi/hippocampus_{gene}_spatial_expression.png")

fig.savefig(
    threshold_output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Threshold plot saved:", threshold_output)

# --------------------------------------------------
# Step 6: Continuous spatial gradient plot
# --------------------------------------------------

# White → light green → medium green → dark green
green_cmap = LinearSegmentedColormap.from_list(
    "white_darkgreen",
    ["white", "#C7E9C0", "#41AB5D", "#006400"]
)

# Gradient starts at expression > 1
gradient_min = 1

# Use the maximum observed expression as the upper limit
gradient_max = expr.max()

fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=expr,
    cmap=green_cmap,
    vmin=gradient_min,
    vmax=gradient_max,
    s=3,
    alpha=0.9,
    linewidths=0,
    rasterized=True
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Normalized log1p expression", fontsize=11)

ax.set_title(
    f"{gene} - Spatial Expression Gradient (>1)",
    fontsize=14
)

ax.set_xlabel("X coordinate", fontsize=12)
ax.set_ylabel("Y coordinate", fontsize=12)

ax.invert_yaxis()
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

# Save gradient plot
gradient_output = (
    f"/home/users/jayanthi/"
    f"hippocampus_{gene}_gradient.png"
)

fig.savefig(
    gradient_output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Gradient plot saved:", gradient_output)

print("\nBoth spatial expression plots generated successfully.")

