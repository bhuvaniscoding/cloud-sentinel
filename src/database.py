# src/database.py
import sqlite3
from typing import Dict, Any
import pandas as pd
from config.settings import DB_PATH

def get_connection() -> sqlite3.Connection:
    """Creates a concurrency-safe SQLite connection using WAL mode."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db() -> None:
    """Creates the telemetry table if it does not already exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                disk_percent REAL NOT NULL,
                net_sent_kbps REAL NOT NULL,
                net_recv_kbps REAL NOT NULL,
                is_labeled_anomaly INTEGER DEFAULT 0
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_source ON metrics(source, timestamp);")

def insert_metric(record: Dict[str, Any]) -> None:
    """Inserts a single metric snapshot into the database."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO metrics (
                timestamp, source, cpu_percent, memory_percent,
                disk_percent, net_sent_kbps, net_recv_kbps, is_labeled_anomaly
            ) VALUES (
                :timestamp, :source, :cpu_percent, :memory_percent,
                :disk_percent, :net_sent_kbps, :net_recv_kbps, :is_labeled_anomaly
            )
        """, record)

def fetch_recent_metrics(source: str = "live", limit: int = 500) -> pd.DataFrame:
    """Fetches the latest N rows in chronological order for ML and graphing."""
    query = """
        SELECT * FROM (
            SELECT * FROM metrics
            WHERE source = ?
            ORDER BY id DESC
            LIMIT ?
        ) ORDER BY id ASC;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=(source, limit))