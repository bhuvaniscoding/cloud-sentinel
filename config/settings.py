# config/settings.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "telemetry.db"
MODEL_DIR = BASE_DIR / "models"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_INTERVAL_SEC = 2.0

THRESHOLDS = {
    "cpu_percent": 85.0,
    "memory_percent": 85.0,
    "disk_percent": 90.0,
}

CONTAMINATION_RATE = 0.05
ROLLING_WINDOW_SIZE = 10
FORECAST_HORIZON_MIN = 30