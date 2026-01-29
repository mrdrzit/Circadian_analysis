from pathlib import Path
import re

# =============================================================================
# Project paths
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
EXPORTS = DATA / "exports"

METADATA_FILE = ROOT / "config" / "experiment_metadata.json"
GROUPS_METADATA_PATH = ROOT / "config" / "group_metadata.json"

# -----------------------------------------------------------------------------
# Derived data directories
# -----------------------------------------------------------------------------
STAGE1_DIR = DATA / "derived" / "stage1_protocols" / "ER4000"
STAGE2_DIR = DATA / "derived" / "stage2_concatenated" / "ER4000"
COSINOR_DIR = DATA / "derived" / "stage2_cosinor" / "ER4000"

# -----------------------------------------------------------------------------
# Figure directories
# -----------------------------------------------------------------------------
ACTOGRAM_FIGURES_DIR = EXPORTS / "figures" / "actograms"
TIME_SERIES_FIGURES_DIR = EXPORTS / "figures" / "time_series"
COSINOR_FIGURES_DIR = EXPORTS / "figures" / "cosinor"

# =============================================================================
# Pipeline toggles (Stage 2)
# =============================================================================
RUN_COSINOR = True
FORCE_COSINOR = False
SAVE_RAW = True
SAVE_NORM = True
NORM_TYPE = "zscore"  # or "minmax"
NORM_PER_DAY = False

# =============================================================================
# Preprocessing (Stage 2)
# =============================================================================
FILTER_WINDOW = 3
RESAMPLE_TO_VALUE = "60T"

# =============================================================================
# Regex / naming conventions
# =============================================================================
ER4000_FOLDER_REGEX = re.compile(r"^[a-zA-Z]{5}_\d{4,8}$")
DATE_REGEX = re.compile(r".*_(\d{2})(\d{2})(\d{4})")

# =============================================================================
# Raw data folders
# =============================================================================
ER4000_DATA_FOLDER = RAW / "ER4000"
INTELLICAGE_DATA_FOLDER = RAW / "INTELLICAGE"

# =============================================================================
# Device registry
# =============================================================================
KNOWN_DEVICES = {
    "ER4000": "ER4000",
    "INTELLICAGE": "INTELLICAGE",
}

# =============================================================================
# Default experiment metadata templates (per device)
# =============================================================================
DEFAULT_METADATA_BY_DEVICE = {
    "ER4000": {
        "zt0_time": 20,
        "labels": {
            "cycle_types": ["LD"],
            "test_labels": ["all"],
            "cycle_days": []
        },
        "files": {
            "pattern": "*.asc",
            "resolved": [],
            "n_files": 0
        },
        "notes": ""
    },
    "INTELLICAGE": {
        "zt0_time": 20,
        "labels": {
            "cycle_types": ["LD"],
            "test_labels": ["all"],
            "cycle_days": []
        },
        "files": {
            "pattern": "*.txt",
            "resolved": [],
            "n_files": 0
        },
        "notes": ""
    }
}

# =============================================================================
# Cosinor configuration
# =============================================================================
COSINOR_CONFIG = {
    "time_shape": "continuous",
    "step": 0.01,
    "start_time": 22,
    "end_time": 26,
    "n_components": [1],
}