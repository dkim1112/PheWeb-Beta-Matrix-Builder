# PheWeb Beta Matrix Builder Explanation

## What this script is for
It bulk-downloads GWAS summary stats for many phenotypes from a PheWeb site, keeps only variants that have an rsID, and builds a single **beta matrix** (rows = rsIDs, columns = phenotypes). It also fixes sign flips when the reference/alternate alleles are swapped across phenotypes.

---

## Step-by-step walk-through

### 1) Choose which PheWeb to crawl  
```python
PHEWEB_BASE = "https://pheweb.org/MGI-BioVU"
```
- This base URL points to a specific PheWeb instance. All API and download URLs are built from this.

### 2) Get the list of phenotypes from the API  
```python
requests.get(f"{PHEWEB_BASE}/api/phenotypes.json", timeout=30)
phenotypes = [p['phenocode'] for p in response.json()]
```
- Calls the PheWeb API to retrieve metadata for all phenotypes.
- Extracts each phenotype’s **phenocode** (its identifier).
- If anything goes wrong (network error, non-200 status), it logs the error and exits.

### 3) Prepare the per-phenotype download URL pattern  
```python
base_url = f"{PHEWEB_BASE}/download/"
```
- Each phenotype’s summary stats will be at `.../download/<phenocode>` (often returns a gzipped TSV).

### 4) Define how to download and clean **one** phenotype  
```python
def download_phenotype(pheno):
    url = f"{base_url}{pheno}"
    response = requests.get(url, timeout=60)       # fetch file (maybe gzipped)
    content = gzip.decompress(...) or raw bytes    # handle gzip if needed
    df = pd.read_csv(io.StringIO(content), sep='\t')
```
Inside this function it:
- Downloads the file.
- Tries to **gzip-decompress**, and if that fails assumes plain text.
- Reads it into a DataFrame (tab-separated).

Then it:
- Figures out which column carries rsIDs (`'rsids'` or `'rsid'`), and standardizes it to `'rsid'`.
- Drops rows with missing/empty rsIDs.
- Drops duplicate rsIDs (keeps the first occurrence).
- Keeps only the columns it needs: `rsid`, `ref`, `alt`, `beta`.
- Adds a `phenotype` column for bookkeeping.
- Returns `(phenocode, cleaned_dataframe, number_of_rows_kept)`.
- On any error, returns `(phenocode, None, 0)`.

### 5) Download **all** phenotypes in parallel  
```python
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_phenotype, ph): ph for ph in phenotypes}
    for future in as_completed(futures):
        pheno, df, count = future.result()
```
- Spawns up to 10 worker threads to speed things up.
- As each finishes, it logs success (and how many variants with rsIDs were found) or failure.
- Successful DataFrames are collected into `pheno_data = {phenocode: df, ...}`.
- If **none** succeeded, it exits.

### 6) Find the **intersection** of rsIDs across all phenotypes  
```python
common_rsids = set(first_df['rsid'])
for pheno in pheno_data:
    common_rsids &= set(pheno_data[pheno]['rsid'])
```
- Starts with the rsIDs from the first phenotype and intersects with each other phenotype.
- Result: only rsIDs present in **every** phenotype remain.
- This ensures the final matrix has no missing rows for any phenotype (dense matrix).

### 7) Build a reference table for allele orientation  
```python
ref_pheno = first_phenotype
ref_df = pheno_data[ref_pheno][rsid in common_rsids].set_index('rsid')[['ref','alt','beta']]
ref_df.columns = ['ref_ref','ref_alt', ref_pheno]
beta_matrix = ref_df[[ref_pheno]].copy()
```
- Chooses the first phenotype as the **reference** for allele orientation.
- Renames its allele columns to `ref_ref` and `ref_alt` to avoid confusion later.
- Seeds the beta matrix with the reference phenotype’s beta values.

### 8) For each other phenotype: align alleles and fix sign flips  
For each phenotype:
- Select only common rsIDs and set index to `rsid`.
- Join with the reference’s `ref_ref/ref_alt`.
- Compare alleles row-wise:  
  ```python
  alleles_match   = (ref == ref_ref) & (alt == ref_alt)
  alleles_flipped = (ref == ref_alt) & (alt == ref_ref)
  ```
- If **match**, keep `beta` as-is.  
- If **flipped**, **negate** `beta` (because swapping ref/alt flips the effect direction).  
- If neither (mismatch e.g., strand issues, multi-allelic quirks), set to `None` (i.e., NA).
- Count and log how many were flipped and how many were mismatches.
- Append this phenotype’s aligned beta column to `beta_matrix`.

**Why this matters:** GWAS effects depend on which allele you call “reference.” If two files disagree on ref/alt, the sign of beta must be flipped to make them comparable. This block enforces consistent orientation across phenotypes.

### 9) Report size of the final matrix  
```python
print(f"Created beta matrix with {len(beta_matrix)} rsIDs and {len(beta_matrix.columns)} phenotypes")
```
- Rows = number of common rsIDs.
- Columns = number of phenotypes (including the reference).

### 10) Save results and print a summary  
```python
beta_matrix.to_csv("beta_matrix.csv", sep='\t')
print("Summary: ...")
```
- Writes out a tab-separated file you can load later for downstream modeling.
- Prints totals (phenotypes included, row count, and flip statistics).

---

## Pre-reqs
- Python libs: `requests`, `gzip`, `io`, `pandas`, `concurrent.futures`, `sys` (all implied by the code).
- The PheWeb instance must expose `/api/phenotypes.json` and `/download/<phenocode>` endpoints with TSV (plain or gzipped) containing at least: `rsid/rsids`, `ref`, `alt`, `beta`.

---

## Practical notes
- **Intersection can be small:** Requiring presence in *all* phenotypes can drastically shrink the row set. If you want more rows, consider a threshold (e.g., rsIDs present in ≥K phenotypes) and then handle missing values.
- **Allele mismatches:** Rows where alleles are neither match nor flip become NA. That can happen due to strand issues (A/T or C/G SNPs) or differing variant normalization. If you care, add explicit strand handling or liftover/normalization.
- **Duplicates:** The script keeps the first occurrence of a duplicate rsID per phenotype; if there are multiple lines per rsID (e.g., multi-allelic sites), you may want to pre-aggregate or choose by lowest p-value, etc.
- **Performance:** Parallel downloads help. If the instance is rate-limited, reduce `max_workers`.
- **Schema changes:** If a PheWeb instance uses different column names, the `rsid` detection or `['ref','alt','beta']` selection may need tweaking.

---

## Output artifact
- `beta_matrix.csv` — a dense matrix of aligned, sign-consistent betas for the set of rsIDs shared by every successfully downloaded phenotype.
