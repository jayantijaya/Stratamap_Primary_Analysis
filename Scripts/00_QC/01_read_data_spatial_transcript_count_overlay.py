# Read StrataMap Data and Generate Spatial Transcript Count Overlay on Registered H&E Image
# ==========================================================================================
# 1. Read 10x data
# ==========================================================================================

import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from tifffile import TiffFile
from ome_types import from_xml
from matplotlib.colors import LogNorm

data_dir = "/home/users/jayanthi/cell_bindata_HB113/"

adata = sc.read_10x_mtx(
    data_dir,
    var_names="gene_symbols",
    make_unique=True
)

print(adata)
print("AnnData shape:", adata.shape)


# ============================================================
# 2. Add spatial coordinates from cell barcodes
# ============================================================

barcode_file = f"{data_dir}/barcodes.tsv.gz"

barcode_df = pd.read_csv(
    barcode_file,
    delimiter=":",
    header=None
)

barcode_df.columns = ["barcode", "Y", "X"]

# Check barcode alignment before assigning coordinates
if not np.array_equal(
    adata.obs_names.astype(str).to_numpy(),
    barcode_df["barcode"].astype(str).to_numpy()
):
    raise ValueError(
        "Barcode order does not match AnnData observation order. "
        "Align the coordinates using barcode identifiers before proceeding."
    )

spatial_coords = barcode_df[["X", "Y"]].to_numpy(
    dtype=np.float32
)

adata.obsm["spatial"] = spatial_coords

print("Spatial coordinates:", adata.obsm["spatial"].shape)


# ============================================================
# 3. Read registered OME-TIFF image
# ============================================================

sample_id = "HB113"

# Confirm that this is the correct registered image for HB113
ome_tiff_path = "/home/users/jayanthi/HCF100_registered.ome.tiff"

with TiffFile(ome_tiff_path) as tif:

    ome = from_xml(tif.ome_metadata)

    annotation = ome.structured_annotations.map_annotations[0]
    value_dict = annotation.value

    Global_top = int(round(float(value_dict["GlobalPos_top"])))
    Global_left = int(round(float(value_dict["GlobalPos_left"])))

    image = tif.pages[2].asarray()

print("Global_top:", Global_top)
print("Global_left:", Global_left)
print("Image shape:", image.shape)


# ============================================================
# 4. Store image and spatial metadata
# ============================================================

adata.uns["spatial"] = {
    sample_id: {
        "images": {
            "hires": image
        },
        "scalefactors": {
            "tissue_hires_scalef": 1,
            "spot_diameter_fullres": 1
        }
    }
}


# ============================================================
# 5. Transform spatial coordinates
# ============================================================

adata.obsm["spatial"][:, 0] -= Global_left
adata.obsm["spatial"][:, 1] -= Global_top


# ============================================================
# 6. Calculate total transcript counts per bin
# ============================================================

if sp.issparse(adata.X):
    transcript_counts = np.asarray(
        adata.X.sum(axis=1)
    ).ravel()
else:
    transcript_counts = np.asarray(
        adata.X.sum(axis=1)
    ).ravel()


# ============================================================
# 7. Identify bins within image boundaries
# ============================================================

coords = adata.obsm["spatial"]

x = coords[:, 0]
y = coords[:, 1]

valid = (
    np.isfinite(x) &
    np.isfinite(y) &
    (x >= 0) &
    (x < image.shape[1]) &
    (y >= 0) &
    (y < image.shape[0])
)

print(
    f"Bins within image: {valid.sum():,} / {adata.n_obs:,}"
)


# ============================================================
# 8. Overlay transcript counts on registered H&E image
# ============================================================

fig, ax = plt.subplots(figsize=(16, 8))

ax.imshow(image, origin="upper")

# Avoid LogNorm errors if no positive counts are present
positive = valid & (transcript_counts > 0)

scatter = ax.scatter(
    x[positive],
    y[positive],
    c=transcript_counts[positive],
    cmap="plasma",
    norm=LogNorm(
        vmin=1,
        vmax=transcript_counts[positive].max()
    ),
    s=1,
    alpha=0.85,
    linewidths=0,
    rasterized=True
)

cbar = fig.colorbar(
    scatter,
    ax=ax,
    fraction=0.025,
    pad=0.02
)

cbar.set_label(
    "Total transcript counts per cell bin (log scale)"
)

ax.set_title(
    "HB113 StrataMap — Spatial Transcript Count Distribution"
)

ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")

ax.set_xlim(0, image.shape[1])
ax.set_ylim(image.shape[0], 0)

plt.tight_layout()

plt.savefig(
    "HB113_transcript_count_heatmap_plasma.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
