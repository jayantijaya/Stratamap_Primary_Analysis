# ============================================================
# Minimal H5AD Preparation for MapMyCells
# Retain expression matrix, cell IDs and Ensembl gene IDs
# ============================================================

library(reticulate)
library(anndata)

scipy <- import("scipy.sparse")

# ------------------------------------------------------------
# 1. Read input H5AD
# ------------------------------------------------------------

input_path <- path.expand(
  "~/Desktop/adata_filtered_upper_cutoff.h5ad"
)

my_adata <- read_h5ad(input_path)

cat("\nOriginal AnnData:\n")
print(my_adata)


# ------------------------------------------------------------
# 2. Extract expression matrix
# ------------------------------------------------------------

count_matrix <- my_adata$X

cat("\nExpression matrix dimensions:\n")
print(dim(count_matrix))


# ------------------------------------------------------------
# 3. Extract original cell-bin IDs
# ------------------------------------------------------------

cell_ids <- as.character(
  unlist(
    py_to_r(my_adata$obs_names$tolist())
  )
)

cat("\nFirst five cell IDs:\n")
print(head(cell_ids))


# ------------------------------------------------------------
# 4. Extract Ensembl gene IDs
# ------------------------------------------------------------

# Reticulate has already converted this column to an R vector.

ensembl_ids <- as.character(
  my_adata$var[["gene_ids"]]
)

# ------------------------------------------------------------
# 5. Validate identifiers
# ------------------------------------------------------------

cat("\nIdentifier validation:\n")

cat("Number of cells:", length(cell_ids), "\n")
cat("Number of genes:", length(ensembl_ids), "\n")

# ------------------------------------------------------------
# 6. Convert expression matrix to CSR format
# ------------------------------------------------------------

count_matrix <- scipy$csr_matrix(count_matrix)

cat("\nExpression matrix converted to CSR format.\n")


# ------------------------------------------------------------
# 7. Create minimal AnnData object
# ------------------------------------------------------------

ad <- anndata$AnnData(
  X = count_matrix
)


# ------------------------------------------------------------
# 8. Assign original cell IDs and Ensembl gene IDs
# ------------------------------------------------------------

ad$obs_names <- r_to_py(cell_ids)

ad$var_names <- r_to_py(ensembl_ids)


# ------------------------------------------------------------
# 9. Verify final AnnData
# ------------------------------------------------------------

cat("\nFinal AnnData object:\n")
print(ad)

cat("\nFirst five cell IDs:\n")
print(head(py_to_r(ad$obs_names$tolist())))

cat("\nFirst five Ensembl IDs:\n")
print(head(py_to_r(ad$var_names$tolist())))


# ------------------------------------------------------------
# 10. Save compressed H5AD
# ------------------------------------------------------------

output_path <- path.expand(
  "~/Desktop/stratamap_hippocampus_EnsemblID_Minimal_MapMyCells.h5ad"
)

ad$write_h5ad(
  output_path,
  compression = "gzip"
)

cat("\nH5AD saved successfully:\n")
cat(output_path, "\n")