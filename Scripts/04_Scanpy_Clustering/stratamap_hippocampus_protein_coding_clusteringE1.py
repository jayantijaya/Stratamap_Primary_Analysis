# =======================================================================
# Protein-Coding Gene-Based Clustering of StrataMap Hippocampal Cell Bins
# Includes:
# 1. Load protein-coding dataset after QC and upper-count filtering
# 2. Normalize and log-transform gene expression
# 3. Identify highly variable genes (HVGs)
# 4. Scale expression and perform PCA
# 5. Construct nearest-neighbor graph
# 6. UMAP visualization
# 7. Leiden clustering
# 8. Spatial cluster visualization
# 9. Identify marker genes for all clusters
# 10. Export marker genes and cluster assignments to CSV
# =======================================================================

# -----------------------------------------------------------------------
# Step 1: Import libraries
# -----------------------------------------------------------------------
import os
import time
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Step 2: Define input and output paths
# -----------------------------------------------------------------------
input_file = (
    "/home/users/jayanthi/"
    "adata_filtered_upper_cutoff.h5ad"
)

normalized_file = (
    "/home/users/jayanthi/"
    "stratamap_hippocampus_normalized_protein.h5ad"
)

clustered_file = (
    "/home/users/jayanthi/"
    "stratamap_hippocampus_pca_umap_clusters_protein.h5ad"
)

marker_file = (
    "/home/users/jayanthi/"
    "stratamap_hippocampus_protein_all_marker_genes.csv"
)

cluster_assignment_file = (
    "/home/users/jayanthi/"
    "stratamap_hippocampus_protein_cluster_assignments.csv"
)

cluster_summary_file = (
    "/home/users/jayanthi/"
    "stratamap_hippocampus_protein_cluster_summary.csv"
)

# -----------------------------------------------------------------------
# Step 3: Load protein-coding dataset after QC and upper-count filtering
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 1: Loading protein-coding dataset")
print("=" * 70)

adata_protein_upper_cutoff = sc.read_h5ad(input_file)

print("Dataset shape:", adata_protein_upper_cutoff.shape)
print(f"Number of cell bins: {adata_protein_upper_cutoff.n_obs:,}")
print(f"Number of genes: {adata_protein_upper_cutoff.n_vars:,}")

# -----------------------------------------------------------------------
# Step 4: Normalize gene expression
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Normalization and log transformation")
print("=" * 70)

# Create a separate object for normalization
adata_norm = adata_protein_upper_cutoff.copy()

# Preserve original counts
adata_norm.layers["counts"] = adata_norm.X.copy()

# Normalize each cell bin to 10,000 counts
sc.pp.normalize_total(
    adata_norm,
    target_sum=1e4
)

# Log-transform normalized expression
sc.pp.log1p(adata_norm)

# Preserve normalized, log-transformed expression
adata_norm.raw = adata_norm.copy()

# Save normalized dataset
adata_norm.write_h5ad(
    normalized_file,
    compression="gzip"
)

print("Normalization completed.")
print(f"Cell bins: {adata_norm.n_obs:,}")
print(f"Genes: {adata_norm.n_vars:,}")
print("Saved normalized dataset:", normalized_file)

# -----------------------------------------------------------------------
# Step 5: Identify highly variable genes
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Highly variable gene selection")
print("=" * 70)

sc.pp.highly_variable_genes(
    adata_norm,
    n_top_genes=3000,
    flavor="seurat"
)

n_hvg = int(adata_norm.var["highly_variable"].sum())

print("Number of highly variable genes:", n_hvg)

# Create a copy containing only highly variable genes
adata_hvg = adata_norm[
    :, adata_norm.var["highly_variable"]
].copy()

print("HVG dataset shape:", adata_hvg.shape)

# -----------------------------------------------------------------------
# Step 6: Scale expression
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Scaling expression")
print("=" * 70)

sc.pp.scale(
    adata_hvg,
    zero_center=True,
    max_value=10
)

print("Expression scaling completed.")

# NOTE:
# Scaling approximately 586,000 cell bins across 3,000 genes can use
# substantial RAM because zero_center=True may densify the matrix.
# Monitor available memory before running this step.

# -----------------------------------------------------------------------
# Step 7: PCA
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Principal component analysis")
print("=" * 70)

sc.tl.pca(
    adata_hvg,
    n_comps=50,
    svd_solver="arpack",
    random_state=42
)

print("PCA completed.")
print("PCA shape:", adata_hvg.obsm["X_pca"].shape)

# Plot PCA variance explained
sc.pl.pca_variance_ratio(
    adata_hvg,
    n_pcs=50,
    log=True,
    show=True
)

# -----------------------------------------------------------------------
# Step 8: Construct nearest-neighbor graph
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Constructing nearest-neighbor graph")
print("=" * 70)

sc.pp.neighbors(
    adata_hvg,
    n_neighbors=15,
    n_pcs=30,
    random_state=42
)

print("Nearest-neighbor graph constructed.")

# -----------------------------------------------------------------------
# Step 9: UMAP
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: UMAP")
print("=" * 70)

start = time.time()

sc.tl.umap(
    adata_hvg,
    random_state=42
)

print(
    f"UMAP completed in "
    f"{(time.time() - start) / 60:.2f} minutes"
)

print("UMAP shape:", adata_hvg.obsm["X_umap"].shape)

# -----------------------------------------------------------------------
# Step 10: Leiden clustering
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Leiden clustering")
print("=" * 70)

start = time.time()

sc.tl.leiden(
    adata_hvg,
    resolution=1.0,
    key_added="leiden_1.0",
    random_state=42,
    flavor="igraph",
    n_iterations=1,
    directed=False
)

print(
    f"Leiden clustering completed in "
    f"{(time.time() - start) / 60:.2f} minutes"
)

n_clusters = adata_hvg.obs["leiden_1.0"].nunique()

print("Number of clusters:", n_clusters)

print("\nCluster sizes:")
print(
    adata_hvg.obs["leiden_1.0"]
    .value_counts()
    .sort_index()
)

# -----------------------------------------------------------------------
# Step 11: Visualize UMAP clusters
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: UMAP visualization")
print("=" * 70)

sc.pl.umap(
    adata_hvg,
    color="leiden_1.0",
    legend_loc="on data",
    title="StrataMap Hippocampus — Leiden Clustering",
    frameon=True
)

# -----------------------------------------------------------------------
# Step 12: Save PCA, UMAP, and clustering results
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: Saving clustering results")
print("=" * 70)

adata_hvg.write_h5ad(
    clustered_file,
    compression="gzip"
)

print("Clustered AnnData saved to:", clustered_file)

# -----------------------------------------------------------------------
# Step 13: Prepare consistent cluster colors
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 11: Preparing spatial cluster colors")
print("=" * 70)

adata_hvg.obs["leiden_1.0"] = (
    adata_hvg.obs["leiden_1.0"].astype("category")
)

# Generate Scanpy's default cluster colors if absent
if "leiden_1.0_colors" not in adata_hvg.uns:
    sc.pl.embedding(
        adata_hvg,
        basis="spatial",
        color="leiden_1.0",
        show=False
    )
    plt.close("all")

clusters = adata_hvg.obs["leiden_1.0"].cat.categories

cluster_colors = dict(
    zip(
        clusters,
        adata_hvg.uns["leiden_1.0_colors"]
    )
)

print("Cluster colors:")
print(cluster_colors)

# -----------------------------------------------------------------------
# Step 14: Spatial visualization of Leiden clusters
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 12: Spatial cluster visualization")
print("=" * 70)

if "spatial" not in adata_hvg.obsm:
    raise ValueError(
        "Spatial coordinates are missing from adata_hvg.obsm['spatial']."
    )

coords = adata_hvg.obsm["spatial"]
labels = adata_hvg.obs["leiden_1.0"]

fig, ax = plt.subplots(figsize=(12, 9))

for cluster in clusters:
    mask = (labels == cluster).to_numpy()

    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        color=cluster_colors[cluster],
        s=0.05,
        label=f"Cluster {cluster}",
        rasterized=True
    )

ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")

ax.set_title(
    "StrataMap Hippocampus — Leiden Clusters"
)

# Invert Y-axis to match spatial image coordinates
ax.invert_yaxis()

# Maintain equal spatial scaling
ax.set_aspect("equal", adjustable="box")

# Show ticks on all four sides
ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True
)

# Display grid lines
ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.5
)

# Keep all four borders visible
for spine in ax.spines.values():
    spine.set_visible(True)

ax.legend(
    title="Leiden clusters",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    markerscale=30,
    scatterpoints=1
)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# Step 15: Export cell-bin cluster assignments to CSV
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 13: Exporting cluster assignments")
print("=" * 70)

cluster_df = adata_hvg.obs[["leiden_1.0"]].copy()
cluster_df.index.name = "Cell_Bin_ID"

cluster_df.to_csv(cluster_assignment_file)

print("Cluster assignments saved to:", cluster_assignment_file)

# -----------------------------------------------------------------------
# Step 16: Export cluster sizes to CSV
# -----------------------------------------------------------------------
cluster_summary = (
    adata_hvg.obs["leiden_1.0"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Number_of_Cell_Bins")
)

cluster_summary.to_csv(
    cluster_summary_file,
    index=False
)

print("Cluster summary saved to:", cluster_summary_file)

# -----------------------------------------------------------------------
# Step 17: Calculate marker genes for every Leiden cluster
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 14: Marker gene identification")
print("=" * 70)

# Transfer cluster labels to the normalized object.
# This aligns labels by cell-bin ID rather than row position.
adata_norm.obs["leiden_1.0"] = (
    adata_hvg.obs["leiden_1.0"]
    .reindex(adata_norm.obs_names)
)

if adata_norm.obs["leiden_1.0"].isna().any():
    raise ValueError(
        "Some cell bins do not have matching Leiden cluster labels."
    )

# Calculate marker genes using the normalized, log-transformed expression
sc.tl.rank_genes_groups(
    adata_norm,
    groupby="leiden_1.0",
    method="wilcoxon",
    use_raw=False,
    pts=True
)

print("Marker gene calculation completed.")

# -----------------------------------------------------------------------
# Step 18: Extract all marker genes
# -----------------------------------------------------------------------
markers_df = sc.get.rank_genes_groups_df(
    adata_norm,
    group=None
)

print("Total marker entries:", len(markers_df))
print(
    "Number of clusters represented:",
    markers_df["group"].nunique()
)
print("Columns:", markers_df.columns.tolist())

# -----------------------------------------------------------------------
# Step 19: Save all marker genes to CSV
# -----------------------------------------------------------------------
markers_df.to_csv(
    marker_file,
    index=False
)

print("\nAll marker genes saved to:", marker_file)

# -----------------------------------------------------------------------
# Step 20: Final summary
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("Normalized dataset:", normalized_file)
print("Clustered dataset:", clustered_file)
print("Marker genes CSV:", marker_file)
print("Cell-bin cluster assignments CSV:", cluster_assignment_file)
print("Cluster summary CSV:", cluster_summary_file)

print(f"Total cell bins: {adata_hvg.n_obs:,}")
print(f"Number of HVGs: {adata_hvg.n_vars:,}")
print(f"Number of Leiden clusters: {n_clusters}")
print(f"Total marker entries: {len(markers_df):,}")
