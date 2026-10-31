# src/database.py
import sqlite3
from contextlib import contextmanager
from typing import Dict, Any, List, Generator
import pandas as pd
from config.settings import DB_PATH

@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that guarantees SQLite WAL concurrency and closes connections."""
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    """Initializes the metrics schema with Disk I/O rates and scenario labels."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                disk_percent REAL NOT NULL,
                disk_read_kbps REAL NOT NULL,
                disk_write_kbps REAL NOT NULL,
                net_sent_kbps REAL NOT NULL,
                net_recv_kbps REAL NOT NULL,
                is_labeled_anomaly INTEGER DEFAULT 0,
                scenario_label TEXT DEFAULT 'normal'
            );
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_timestamp ON metrics(source, timestamp);"
        )

def clear_source_metrics(source: str) -> None:
    """Deletes previous records for a specific source to prevent overlapping runs."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM metrics WHERE source = ?;", (source,))

def insert_metric(record: Dict[str, Any]) -> None:
    """Inserts a single telemetry observation."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO metrics (
                timestamp, source, cpu_percent, memory_percent, disk_percent,
                disk_read_kbps, disk_write_kbps, net_sent_kbps, net_recv_kbps,
                is_labeled_anomaly, scenario_label
            ) VALUES (
                :timestamp, :source, :cpu_percent, :memory_percent, :disk_percent,
                :disk_read_kbps, :disk_write_kbps, :net_sent_kbps, :net_recv_kbps,
                :is_labeled_anomaly, :scenario_label
            )
        """, record)

def insert_metrics_batch(records: List[Dict[str, Any]]) -> None:
    """High-speed batch insertion for synthetic and cloud benchmark datasets."""
    init_db()
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO metrics (
                timestamp, source, cpu_percent, memory_percent, disk_percent,
                disk_read_kbps, disk_write_kbps, net_sent_kbps, net_recv_kbps,
                is_labeled_anomaly, scenario_label
            ) VALUES (
                :timestamp, :source, :cpu_percent, :memory_percent, :disk_percent,
                :disk_read_kbps, :disk_write_kbps, :net_sent_kbps, :net_recv_kbps,
                :is_labeled_anomaly, :scenario_label
            )
        """, records)

def fetch_recent_metrics(source: str = "live", limit: int = 1000) -> pd.DataFrame:
    """Fetches the latest N observations ordered chronologically."""
    init_db()
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