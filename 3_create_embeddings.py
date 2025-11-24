"""
Step 3: Create embeddings using SVD for asymmetric variant-phenotype matrix, visualize with PCA
Configure dataset in config.py before running.

Unlike symmetric co-occurrence matrices, our matrix is asymmetric:
  - Rows: Variants (rsIDs)
  - Columns: Phenotypes
  - Values: Beta coefficients (effect sizes) where p < threshold

SVD decomposition: X = U * S * V^T
  - U: Left embeddings (variant embeddings) - obtained via svd.fit_transform(X)
  - V^T: Right embeddings (phenotype embeddings) - obtained via svd.components_.T
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

# Import configuration
from config import (
    DATASET_NAME, N_COMPONENTS, VARIANT_SAMPLE_SIZE,
    DATASET_DIR, EMBEDDINGS_DIR, get_latest_matrix, setup_directories
)

def load_matrix(filepath):
    """Load beta matrix from CSV"""
    df = pd.read_csv(filepath, index_col=0)
    print(f"  Matrix shape: {df.shape} (variants x phenotypes)")

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

    # Left embeddings (U * S) - variant embeddings
    left_embeddings = svd.fit_transform(X_sparse)

    # Right embeddings (V^T)^T = V - phenotype embeddings
    right_embeddings = svd.components_.T

    # Explained variance
    cumsum_var = np.cumsum(svd.explained_variance_ratio_)
    print(f"  Total explained variance: {cumsum_var[-1]:.3f}")

    return left_embeddings, right_embeddings, svd

def plot_variance_explained(svd, output_prefix):
    """Plot explained variance ratio

    SVD Components (Latent Dimensions):
    - Each component represents a "hidden pattern" in the variant-phenotype relationships
    - Component 1 captures the strongest pattern (most variance), Component 2 the second strongest, etc.
    - These are mathematical constructs that summarize how variants and phenotypes co-vary
    - We use only Components 1-2 (or 1-2-3) for visualization because:
      1. They capture the most information (highest explained variance)
      2. We can only visualize 2D or 3D spaces
      3. Later components often capture noise rather than signal
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(range(1, len(svd.explained_variance_ratio_) + 1),
            svd.explained_variance_ratio_)
    ax1.set_xlabel('SVD Component (Latent Dimension)')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Variance Explained by Each SVD Component\n(Each component captures a hidden pattern in the data)')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumsum = np.cumsum(svd.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, 'b-', linewidth=2)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    ax2.set_xlabel('Number of SVD Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained\n(How much total information is captured)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add annotation explaining why we use first 2-3 components
    first_two = cumsum[1] if len(cumsum) > 1 else cumsum[0]
    ax2.annotate(f'Components 1-2 capture {first_two*100:.1f}%\n(used for 2D visualization)',
                xy=(2, first_two), xytext=(10, first_two - 0.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_variance_explained.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_2d(embeddings, labels, title, output_path, sample_size=None,
                is_variant=False, use_symlog=False):
    """Plot 2D PCA visualization (first 2 components)

    Args:
        embeddings: The embedding vectors
        labels: Labels for each point
        title: Plot title
        output_path: Where to save the plot
        sample_size: Subsample to this many points (for variants)
        is_variant: If True, this is a variant plot (affects labeling/styling)
        use_symlog: If True, use symmetric log scale (handles negative values)
    """

    if sample_size and len(embeddings) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                        alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    # Always add labels (for variants, use smaller font)
    if is_variant:
        # Label all variant dots with smaller font
        for i, label in enumerate(labels):
            ax.annotate(label, (embeddings[i, 0], embeddings[i, 1]),
                       fontsize=4, alpha=0.5, ha='center', va='bottom')
    else:
        # Phenotype labels - larger and more visible
        for i, label in enumerate(labels):
            ax.annotate(label, (embeddings[i, 0], embeddings[i, 1]),
                       fontsize=8, alpha=0.7)

    # Set equal axis ranges (y-axis same as x-axis)
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()

    # Use the larger range for both axes
    data_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # Add 10% padding
    half_range = data_range * 0.55
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)

    # Apply symmetric log scale if requested (handles negative values)
    if use_symlog:
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_yscale('symlog', linthresh=0.01)

    ax.set_xlabel('SVD Component 1 (captures strongest pattern)')
    ax.set_ylabel('SVD Component 2 (captures second strongest pattern)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_3d(embeddings, labels, title, output_path, sample_size=None, is_variant=False):
    """Plot 3D PCA visualization (first 3 components)

    Args:
        embeddings: The embedding vectors
        labels: Labels for each point
        title: Plot title
        output_path: Where to save the plot
        sample_size: Subsample to this many points (for variants)
        is_variant: If True, this is a variant plot (affects labeling/styling)
    """

    if sample_size and len(embeddings) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                        alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    # Always add labels
    if is_variant:
        # Label variant dots with smaller font
        for i, label in enumerate(labels):
            ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2],
                   label, fontsize=4, alpha=0.5)
    else:
        # Phenotype labels - larger and more visible
        for i, label in enumerate(labels):
            ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2],
                   label, fontsize=8, alpha=0.7)

    # Set equal axis ranges for all three axes
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    z_min, z_max = embeddings[:, 2].min(), embeddings[:, 2].max()

    # Use the largest range for all axes
    data_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    # Add 10% padding
    half_range = data_range * 0.55
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_zlim(z_center - half_range, z_center + half_range)

    ax.set_xlabel('SVD Component 1 (strongest pattern)')
    ax.set_ylabel('SVD Component 2 (2nd strongest)')
    ax.set_zlabel('SVD Component 3 (3rd strongest)')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_embeddings(left_emb, right_emb, variant_ids, phenotype_ids, output_dir, matrix_name):
    """Save embeddings to CSV files"""
    print("\nSaving embeddings...")

    # Save variant embeddings
    left_df = pd.DataFrame(left_emb,
                          index=variant_ids,
                          columns=[f'dim_{i}' for i in range(left_emb.shape[1])])
    variant_path = output_dir / f'{matrix_name}_variant_embeddings.csv'
    left_df.to_csv(variant_path)

    # Save phenotype embeddings
    right_df = pd.DataFrame(right_emb,
                           index=phenotype_ids,
                           columns=[f'dim_{i}' for i in range(right_emb.shape[1])])
    pheno_path = output_dir / f'{matrix_name}_phenotype_embeddings.csv'
    right_df.to_csv(pheno_path)

    return left_df, right_df

def main():
    # Setup output directories
    setup_directories()

    print(f"Dataset: {DATASET_NAME}")
    print(f"SVD Components: {N_COMPONENTS}")
    print("=" * 60)

    # Find beta matrix files in dataset directory
    matrix_files = list(DATASET_DIR.glob('beta_matrix_*.csv'))

    if not matrix_files:
        print(f"Error: No beta_matrix_*.csv files found in {DATASET_DIR}")
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

        matrix_name = matrix_file.stem

        # Load and prepare matrix
        df = load_matrix(matrix_file)
        X, variant_ids, phenotype_ids = prepare_matrix_for_svd(df)

        # Perform SVD
        left_embeddings, right_embeddings, svd = perform_svd(X, N_COMPONENTS)

        # Output prefix for this matrix
        output_prefix = EMBEDDINGS_DIR / matrix_name

        # Save embeddings
        left_df, right_df = save_embeddings(
            left_embeddings, right_embeddings,
            variant_ids, phenotype_ids,
            EMBEDDINGS_DIR, matrix_name
        )

        # Generate plots
        print("\nCreating visualizations...")
        plot_variance_explained(svd, str(output_prefix))

        # Variant embeddings (LEFT embeddings - U)
        # Create both regular and symlog versions for comparison
        plot_pca_2d(
            left_embeddings, variant_ids,
            f'Variant Embeddings - 2D ({DATASET_NAME})\nSVD Components 1 & 2 capture the two strongest hidden patterns',
            f'{output_prefix}_variants_2d.png',
            sample_size=VARIANT_SAMPLE_SIZE,
            is_variant=True,
            use_symlog=False
        )
        # Also create symlog version to handle outliers
        plot_pca_2d(
            left_embeddings, variant_ids,
            f'Variant Embeddings - 2D Symmetric Log Scale ({DATASET_NAME})\nLog scale reduces outlier influence while preserving negative values',
            f'{output_prefix}_variants_2d_symlog.png',
            sample_size=VARIANT_SAMPLE_SIZE,
            is_variant=True,
            use_symlog=True
        )
        plot_pca_3d(
            left_embeddings, variant_ids,
            f'Variant Embeddings - 3D ({DATASET_NAME})\nSVD Components 1, 2 & 3',
            f'{output_prefix}_variants_3d.png',
            sample_size=VARIANT_SAMPLE_SIZE,
            is_variant=True
        )

        # Phenotype embeddings (RIGHT embeddings - V^T)
        plot_pca_2d(
            right_embeddings, phenotype_ids,
            f'Phenotype Embeddings - 2D ({DATASET_NAME})\nCloser phenotypes have similar genetic variant associations',
            f'{output_prefix}_phenotypes_2d.png',
            sample_size=None,
            is_variant=False,
            use_symlog=False
        )
        plot_pca_3d(
            right_embeddings, phenotype_ids,
            f'Phenotype Embeddings - 3D ({DATASET_NAME})\nCloser phenotypes have similar genetic variant associations',
            f'{output_prefix}_phenotypes_3d.png',
            sample_size=None,
            is_variant=False
        )

        print(f"\n{'='*80}")
        print(f"Completed processing: {matrix_file.name}")
        print(f"{'='*80}\n")

    print("\n" + "="*80)
    print("All matrices processed successfully!")
    print(f"Results saved in: {EMBEDDINGS_DIR.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()
