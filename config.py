"""
Configuration file for PheWeb Matrix Builder
Change DATASET_NAME and PHEWEB_BASE to run on different PheWeb instances.
All outputs will be saved to data/{DATASET_NAME}/
"""

from pathlib import Path

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_NAME = "MGI-BioVU"
PHEWEB_BASE = "https://pheweb.org/MGI-BioVU"

# =============================================================================
# OTHER PHEWEB INSTANCES (uncomment to use)
# =============================================================================

# # UK Biobank - TOPMed
# DATASET_NAME = "UKB-TOPMed"
# PHEWEB_BASE = "https://pheweb.org/UKB-TOPMed/"

# # MGI
# DATASET_NAME = "MGI"
# PHEWEB_BASE = "https://pheweb.org/MGI/"

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# P-value threshold for significance filtering
PVAL_THRESHOLD = 5e-5

# Number of SVD components for embeddings
N_COMPONENTS = 50

# Number of variants to sample for visualization (None = all)
VARIANT_SAMPLE_SIZE = 1000

# Number of parallel download workers
MAX_WORKERS = 10

# =============================================================================
# OUTPUT PATHS
# =============================================================================

# Base data directory
DATA_DIR = Path(__file__).parent / "data"

# Dataset-specific directory
DATASET_DIR = DATA_DIR / DATASET_NAME

# Output file paths
PHENO_DATA_PATH = DATASET_DIR / "pheno_data.pkl"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

def setup_directories():
    """Create necessary directories if they don't exist"""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {DATASET_DIR.absolute()}")

def get_matrix_path(timestamp):
    """Get path for beta matrix with timestamp"""
    return DATASET_DIR / f"beta_matrix_{timestamp}.csv"

def get_latest_matrix():
    """Find the most recent beta matrix file"""
    matrix_files = list(DATASET_DIR.glob("beta_matrix_*.csv"))
    if not matrix_files:
        return None
    return max(matrix_files, key=lambda f: f.stat().st_mtime)
