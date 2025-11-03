"""
Step 3: Create embeddings using SVD for asymmetric variant-phenotype matrix, visualize with PCA

Unlike symmetric co-occurrence matrices, our matrix is asymmetric:
  - Rows: Variants (rsIDs)
  - Columns: Phenotypes
  - Values: Beta coefficients (effect sizes) where p < threshold (TBD)

SVD decomposition: X ≈ U * Σ * V^T
  - U: Left embeddings (variant embeddings) - obtained from embeddings via svd.fit_transform(X)
  - V^T: Right embeddings (phenotype embeddings) - obtained from embeddings via svd.components_.T

Extract both and see how the PCA visualization differs when using different p-value thresholds.

Comparison of the GitHub Tutorial vs. My Approach
  1. Matrix Type: The tutorial uses a symmetric co-occurrence matrix (words × words), so they only need one set of embeddings.
  Ours is asymmetric (variants × phenotypes), so we need both left (U) and right (V) embeddings.

  2. Visualization: Ours creates both 2D and 3D plots for both variant and phenotype embeddings separately, with subsampling for variants.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
import sys
import os
from pathlib import Path

# Configuration
N_COMPONENTS = 50  # Number of SVD components to extract
VARIANT_SAMPLE_SIZE = 1000  # Subsample variants for readable plots (None = all)

def load_matrix(filepath):
    """Load beta matrix from CSV"""
    df = pd.read_csv(filepath, index_col=0)
    print(f"  Matrix shape: {df.shape} (variants × phenotypes)")

    # Check sparsity
    total_values = df.size
    missing_values = df.isna().sum().sum()
    sparsity = (missing_values / total_values) * 100
    print(f"  Sparsity: {sparsity:.2f}% missing/non-significant")

    return df

def prepare_matrix_for_svd(df):
    """Prepare matrix for SVD using sparse matrix format for memory efficiency"""
    # Fill NaN with 0 and convert to sparse CSR (Compressed Sparse Row) matrix
    # Sparse matrices only store non-zero values and their positions
    df_filled = df.fillna(0)
    X_sparse = csr_matrix(df_filled.values)

    return X_sparse, df.index, df.columns

def perform_svd(X_sparse, n_components):
    """Perform truncated SVD on sparse matrix"""
    print(f"\nPerforming SVD with {n_components} components...")

    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # Left embeddings (U * Σ) - variant embeddings
    left_embeddings = svd.fit_transform(X_sparse)

    # Right embeddings (V^T)^T = V - phenotype embeddings
    right_embeddings = svd.components_.T

    # Explained variance
    cumsum_var = np.cumsum(svd.explained_variance_ratio_)
    print(f"  Total explained variance: {cumsum_var[-1]:.3f}")

    return left_embeddings, right_embeddings, svd

def plot_variance_explained(svd, output_prefix):
    """Plot explained variance ratio"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(range(1, len(svd.explained_variance_ratio_) + 1),
            svd.explained_variance_ratio_)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Variance Explained by Each Component')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumsum = np.cumsum(svd.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, 'b-', linewidth=2)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_variance_explained.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_2d(embeddings, labels, title, output_path, sample_size=None):
    """Plot 2D PCA visualization (first 2 components)"""

    if sample_size and len(embeddings) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                        alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    # Label phenotypes (if not too many)
    if len(embeddings) <= 100:
        for i, label in enumerate(labels):
            ax.annotate(label, (embeddings[i, 0], embeddings[i, 1]),
                       fontsize=8, alpha=0.7)

    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_3d(embeddings, labels, title, output_path, sample_size=None):
    """Plot 3D PCA visualization (first 3 components)"""

    if sample_size and len(embeddings) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                        alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_embeddings(left_emb, right_emb, variant_ids, phenotype_ids, output_prefix):
    """Save embeddings to CSV files"""
    print("\nSaving embeddings...")

    # Save variant embeddings
    left_df = pd.DataFrame(left_emb,
                          index=variant_ids,
                          columns=[f'dim_{i}' for i in range(left_emb.shape[1])])
    left_df.to_csv(f'{output_prefix}_variant_embeddings.csv')

    # Save phenotype embeddings
    right_df = pd.DataFrame(right_emb,
                           index=phenotype_ids,
                           columns=[f'dim_{i}' for i in range(right_emb.shape[1])])
    right_df.to_csv(f'{output_prefix}_phenotype_embeddings.csv')

    return left_df, right_df

def main():
    # Find beta matrix files
    matrix_files = list(Path('.').glob('beta_matrix*.csv'))

    if not matrix_files:
        print("Error: No beta_matrix*.csv files found!")
        print("Run 2_build_matrix.py first!")
        sys.exit(1)

    print(f"Found {len(matrix_files)} matrix file(s) to process:\n")
    for f in matrix_files:
        print(f"  - {f.name}")

    print("\n" + "="*80)

    # Process each matrix
    for matrix_file in matrix_files:
        print(f"\n{'='*80}")
        print(f"Processing: {matrix_file.name}")
        print(f"{'='*80}\n")

        # Extract p-value threshold from filename if present
        # Assuming format like "beta_matrix_1e-5.csv" or "beta_matrix.csv"
        matrix_name = matrix_file.stem

        # Load and prepare matrix
        df = load_matrix(matrix_file)
        X, variant_ids, phenotype_ids = prepare_matrix_for_svd(df)

        # Perform SVD
        left_embeddings, right_embeddings, svd = perform_svd(X, N_COMPONENTS)

        # Create output directory
        output_dir = Path('embeddings_output')
        output_dir.mkdir(exist_ok=True)
        output_prefix = output_dir / matrix_name

        # Save embeddings
        left_df, right_df = save_embeddings(
            left_embeddings, right_embeddings,
            variant_ids, phenotype_ids,
            str(output_prefix)
        )

        # Generate plots
        print("\nCreating visualizations...")
        plot_variance_explained(svd, str(output_prefix))

        # Variant embeddings (LEFT embeddings - U)
        plot_pca_2d(
            left_embeddings, variant_ids,
            f'Variant Embeddings - 2D PCA ({matrix_name})',
            f'{output_prefix}_variants_2d.png',
            sample_size=VARIANT_SAMPLE_SIZE
        )
        plot_pca_3d(
            left_embeddings, variant_ids,
            f'Variant Embeddings - 3D PCA ({matrix_name})',
            f'{output_prefix}_variants_3d.png',
            sample_size=VARIANT_SAMPLE_SIZE
        )

        # Phenotype embeddings (RIGHT embeddings - V^T)
        plot_pca_2d(
            right_embeddings, phenotype_ids,
            f'Phenotype Embeddings - 2D PCA ({matrix_name})',
            f'{output_prefix}_phenotypes_2d.png',
            sample_size=None
        )
        plot_pca_3d(
            right_embeddings, phenotype_ids,
            f'Phenotype Embeddings - 3D PCA ({matrix_name})',
            f'{output_prefix}_phenotypes_3d.png',
            sample_size=None
        )

        print(f"\n{'='*80}")
        print(f"Completed processing: {matrix_file.name}")
        print(f"{'='*80}\n")

    print("\n" + "="*80)
    print("All matrices processed successfully!")
    print(f"Results saved in: {output_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()
