"""Project-wide paths and constants for OptiRetain."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Dashboard output directory
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Default data files
TELECO_FILE = RAW_DATA_DIR / "Telco_customer_churn.xlsx"

# Reproducibility
SEED: int = 42
