# config/settings.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "telemetry.db"
MODEL_DIR = BASE_DIR / "models"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Collector Settings
COLLECTION_INTERVAL_SEC = 2.0

# Core raw telemetry metrics collected from OS / Simulator
RAW_METRICS = [
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "disk_read_kbps",
    "disk_write_kbps",
    "net_sent_kbps",
    "net_recv_kbps",
]

# Static Rule Thresholds (Traditional Monitoring Baseline)
THRESHOLDS = {
    "cpu_percent": 85.0,
    "memory_percent": 85.0,
    "disk_percent": 90.0,
    "disk_read_kbps": 5000.0,
    "disk_write_kbps": 5000.0,
    "net_sent_kbps": 2000.0,
    "net_recv_kbps": 5000.0,
}

# Machine Learning & Forecasting Settings
RANDOM_SEED = 42
CONTAMINATION_RATE = 0.06      # Expected anomaly ratio for Isolation Forest
ROLLING_WINDOW_SIZE = 10       # 10 ticks rolling context
PERSISTENCE_WINDOW = 3         # Consecutive anomalies required to trigger Incident Alert
FORECAST_HORIZON_MIN = 30      # Warn if resource exhaustion predicted within 30 mins