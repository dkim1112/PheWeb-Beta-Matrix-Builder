#!/usr/bin/env python3
"""
Step 1: Download phenotype summary statistics from PheWeb
Saves data to pickle files for later processing
"""
import requests
import pandas as pd
import gzip
import io
import sys
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# For record: 69 phenotypes => 69 downloads automation => took about 6 minutes total

# Change this to use different PheWeb instances (dependent on what we want to crawl)
PHEWEB_BASE = "https://pheweb.org/MGI-BioVU"

# Fetch phenotypes from API
print(f"Fetching phenotype list from {PHEWEB_BASE}/api/phenotypes.json...")
try:
    response = requests.get(f"{PHEWEB_BASE}/api/phenotypes.json", timeout=30)
    response.raise_for_status()
    phenotypes = [p['phenocode'] for p in response.json()]
    print(f"Found {len(phenotypes)} phenotypes")
except Exception as e:
    print(f"Error fetching phenotypes: {e}")
    sys.exit(1)

base_url = f"{PHEWEB_BASE}/download/"

def download_phenotype(pheno):
    """Download a single phenotype's summary statistics"""
    url = f"{base_url}{pheno}"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # The file might be gzipped
        try:
            content = gzip.decompress(response.content).decode('utf-8')
        except:
            content = response.content.decode('utf-8')

        # Read into dataframe
        df = pd.read_csv(io.StringIO(content), sep='\t')

        # Filter to only rows with rsID
        rsid_col = 'rsids' if 'rsids' in df.columns else 'rsid' if 'rsid' in df.columns else None
        if rsid_col:
            df = df[df[rsid_col].notna() & (df[rsid_col] != '')]
            df = df.rename(columns={rsid_col: 'rsid'})  # Standardize column name

            # Remove duplicates early
            df = df.drop_duplicates(subset='rsid', keep='first')

            # Keep only necessary columns
            df = df[['rsid', 'ref', 'alt', 'beta']]
            df['phenotype'] = pheno

            return pheno, df, len(df)
        else:
            return pheno, None, 0

    except Exception as e:
        print(f"  -> Error downloading {pheno}: {e}")
        return pheno, None, 0

# Download all phenotypes in parallel
print(f"Downloading summary statistics for {len(phenotypes)} phenotypes in parallel...")
pheno_data = {}
completed = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_phenotype, pheno): pheno for pheno in phenotypes}

    for future in as_completed(futures):
        pheno, df, count = future.result()
        completed += 1

        if df is not None:
            pheno_data[pheno] = df
            print(f"[{completed}/{len(phenotypes)}] {pheno}: {count} variants with rsID")
        else:
            print(f"[{completed}/{len(phenotypes)}] {pheno}: FAILED")

if not pheno_data:
    print("No data downloaded successfully!")
    sys.exit(1)

# Save downloaded data to disk
print(f"\nSaving downloaded data to pheno_data.pkl...")
with open('pheno_data.pkl', 'wb') as f:
    pickle.dump(pheno_data, f)

print(f"Successfully downloaded and saved {len(pheno_data)} phenotypes")
print(f"Run 2_build_matrix.py to create the beta matrix")
