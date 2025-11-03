"""
Step 2: Build beta matrix from downloaded phenotype data
Checks allele consistency, applies p-value filtering (p<1e-5), and creates the final matrix
"""

# Explanation of CSV output.
'''
The columns are:
  1. First column (unnamed): Row index (0, 1, 2, 3...)
  2. rsid: The rsID identifier for each variant
  3. __ # of phenotype columns: one column for each phenotype's beta values

For MGI-BioVU, there are 69 phenotypes.
So there are 71 columns total (index + rsid + 69 phenotypes).

Beta matrix structure:
  - Rows: rsIDs
  - Columns: Beta values for each phenotype
  - Values: Effect sizes (log odds ratios) where p < 5e-5, otherwise NaN

Only significant associations (p < 5e-5) are included in the matrix.
This threshold balances signal capture while filtering noise.
'''

import pandas as pd
import pickle
import sys
from datetime import datetime

# Significance threshold (CAN BE ADJUSTED!!!)
PVAL_THRESHOLD = 5e-5

# Load downloaded data
print("Loading downloaded phenotype data from pheno_data.pkl...")
try:
    with open('pheno_data.pkl', 'rb') as f:
        pheno_data = pickle.load(f)
    print(f"Loaded {len(pheno_data)} phenotypes")
except FileNotFoundError:
    print("Error: pheno_data.pkl not found. Run 1_download_phenotypes.py first!")
    sys.exit(1)

# Find common rsIDs across all phenotypes
common_rsids = set(pheno_data[list(pheno_data.keys())[0]]['rsid'])
for pheno in pheno_data:
    common_rsids = common_rsids.intersection(set(pheno_data[pheno]['rsid']))

print(f"Found {len(common_rsids)} common rsIDs")
print(f"Building beta matrix (p<{PVAL_THRESHOLD})...")

# Use the first phenotype as reference for alleles
ref_pheno = list(pheno_data.keys())[0]
ref_df = pheno_data[ref_pheno][pheno_data[ref_pheno]['rsid'].isin(common_rsids)].copy()
ref_df = ref_df.set_index('rsid')[['ref', 'alt', 'beta', 'pval']]
ref_df.columns = ['ref_ref', 'ref_alt', f'{ref_pheno}_beta', f'{ref_pheno}_pval']

# Initialize beta matrix with reference phenotype
beta_matrix = ref_df[[f'{ref_pheno}_beta']].copy()
beta_matrix.columns = [ref_pheno]

# Apply p-value filter to reference phenotype
beta_matrix.loc[ref_df[f'{ref_pheno}_pval'] >= PVAL_THRESHOLD, ref_pheno] = None

allele_flips = {}
sig_counts = {ref_pheno: (ref_df[f'{ref_pheno}_pval'] < PVAL_THRESHOLD).sum()}

# Process each remaining phenotype
for pheno in list(pheno_data.keys())[1:]:
    df = pheno_data[pheno][pheno_data[pheno]['rsid'].isin(common_rsids)].copy()
    df = df.set_index('rsid')

    # Merge with reference to compare alleles
    merged = ref_df[['ref_ref', 'ref_alt']].join(df[['ref', 'alt', 'beta', 'pval']], how='inner')

    # Check for allele flips (vectorized operation)
    alleles_match = (merged['ref'] == merged['ref_ref']) & (merged['alt'] == merged['ref_alt'])
    alleles_flipped = (merged['ref'] == merged['ref_alt']) & (merged['alt'] == merged['ref_ref'])

    # Create beta column with flipped signs where needed
    merged[pheno] = merged['beta']
    merged.loc[alleles_flipped, pheno] = -merged.loc[alleles_flipped, 'beta']

    # Set to None if alleles mismatch OR if p-value doesn't meet threshold
    merged.loc[~(alleles_match | alleles_flipped), pheno] = None
    merged.loc[merged['pval'] >= PVAL_THRESHOLD, pheno] = None

    # Count flips, mismatches, and significant associations
    flipped_count = alleles_flipped.sum()
    mismatch_count = (~(alleles_match | alleles_flipped)).sum()
    sig_count = ((alleles_match | alleles_flipped) & (merged['pval'] < PVAL_THRESHOLD)).sum()
    sig_counts[pheno] = sig_count

    if flipped_count > 0:
        allele_flips[pheno] = flipped_count

    # Add to beta matrix
    beta_matrix[pheno] = merged[pheno]

# Calculate overall statistics
total_values = len(beta_matrix) * len(beta_matrix.columns)
significant_values = beta_matrix.notna().sum().sum()
percent_significant = (significant_values / total_values) * 100

print(f"\nMatrix: {len(beta_matrix)} rsIDs × {len(beta_matrix.columns)} phenotypes")
print(f"Significant associations: {significant_values:,}/{total_values:,} ({percent_significant:.2f}%)")

# Save the matrix
timestamp = datetime.now().strftime("%H-%M-%S")
output_filename = f"beta_matrix_{timestamp}.csv"
beta_matrix.to_csv(output_filename)
print(f"Saved: {output_filename}")
