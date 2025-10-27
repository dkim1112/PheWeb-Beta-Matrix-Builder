"""
Step 2: Build beta matrix from downloaded phenotype data
Checks allele consistency and creates the final matrix
"""

# Explanation of CSV output.
'''
The columns are:
  1. First column (unnamed): Row index (0, 1, 2, 3...)
  2. rsid: The rsID identifier for each variant
  3. __ # of phenotype columns: one column for each phenotype's beta values

For MGI-BioVU, there are 69 phenotypes. 
So there are 71 columns total (index + rsid + 69 phenotypes).

We have a matrix of:
  - Rows: rsIDs
  - Columns: Beta values for each phenotype
Each of the 69 phenotypes (MCH, TIBC, MPV, etc.) represents a different trait that was measured, so you
get the beta coefficient for how each genetic variant (rsID) affects each of those 69 traits.
'''

import pandas as pd
import pickle
import sys

# Load downloaded data
print("Loading downloaded phenotype data from pheno_data.pkl...")
try:
    with open('pheno_data.pkl', 'rb') as f:
        pheno_data = pickle.load(f)
    print(f"Loaded {len(pheno_data)} phenotypes")
except FileNotFoundError:
    print("Error: pheno_data.pkl not found. Run 1_download_phenotypes.py first!")
    sys.exit(1)

print(f"\nFinding common rsIDs across all phenotypes...")

# Find common rsIDs across all phenotypes
common_rsids = set(pheno_data[list(pheno_data.keys())[0]]['rsid'])
for pheno in pheno_data:
    common_rsids = common_rsids.intersection(set(pheno_data[pheno]['rsid']))

print(f"Found {len(common_rsids)} common rsIDs across all phenotypes")

print(f"\nChecking allele consistency and creating beta matrix...")

# Use the first phenotype as reference for alleles
ref_pheno = list(pheno_data.keys())[0]
ref_df = pheno_data[ref_pheno][pheno_data[ref_pheno]['rsid'].isin(common_rsids)].copy()
ref_df = ref_df.set_index('rsid')[['ref', 'alt', 'beta']]
ref_df.columns = ['ref_ref', 'ref_alt', ref_pheno]

# Initialize beta matrix with reference phenotype
beta_matrix = ref_df[[ref_pheno]].copy()

allele_flips = {}

# Process each remaining phenotype
for pheno in list(pheno_data.keys())[1:]:
    df = pheno_data[pheno][pheno_data[pheno]['rsid'].isin(common_rsids)].copy()
    df = df.set_index('rsid')

    # Merge with reference to compare alleles
    merged = ref_df[['ref_ref', 'ref_alt']].join(df[['ref', 'alt', 'beta']], how='inner')

    # Check for allele flips (vectorized operation)
    alleles_match = (merged['ref'] == merged['ref_ref']) & (merged['alt'] == merged['ref_alt'])
    alleles_flipped = (merged['ref'] == merged['ref_alt']) & (merged['alt'] == merged['ref_ref'])

    # Create beta column with flipped signs where needed
    merged[pheno] = merged['beta']
    merged.loc[alleles_flipped, pheno] = -merged.loc[alleles_flipped, 'beta']
    merged.loc[~(alleles_match | alleles_flipped), pheno] = None

    # Count flips and mismatches
    flipped_count = alleles_flipped.sum()
    mismatch_count = (~(alleles_match | alleles_flipped)).sum()

    if flipped_count > 0:
        allele_flips[pheno] = flipped_count
        print(f"  {pheno}: flipped {flipped_count} betas due to ref/alt swap")

    if mismatch_count > 0:
        print(f"  {pheno}: {mismatch_count} variants with mismatched alleles (set to NA)")

    # Add to beta matrix
    beta_matrix[pheno] = merged[pheno]

print(f"\nCreated beta matrix with {len(beta_matrix)} rsIDs and {len(beta_matrix.columns)} phenotypes")

# Save the matrix
print("\nSaving output file...")
beta_matrix.to_csv("beta_matrix.csv")
print(f"  Saved: beta_matrix.csv")

# Print summary
print(f"\nSummary:")
print(f"  Total phenotypes: {len(pheno_data)}")
print(f"  Total rsIDs in matrix: {len(beta_matrix)}")
print(f"  Phenotypes with allele flips: {len(allele_flips)}")
if allele_flips:
    print(f"  Flip details: {allele_flips}")
