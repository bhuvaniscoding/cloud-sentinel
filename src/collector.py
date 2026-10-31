# src/collector.py
import time
from datetime import datetime, timezone
import psutil
from config.settings import COLLECTION_INTERVAL_SEC
from src.database import init_db, insert_metric

class SystemCollector:
    def __init__(self, source_name: str = "live"):
        self.source_name = source_name
        self.last_monotonic = time.monotonic()

        # Baseline network counters
        net = self._safe_net_counters()
        self.last_bytes_sent = net.bytes_sent if net else 0
        self.last_bytes_recv = net.bytes_recv if net else 0

        # Baseline disk I/O counters
        disk_io = self._safe_disk_io_counters()
        self.last_disk_read = disk_io.read_bytes if disk_io else 0
        self.last_disk_write = disk_io.write_bytes if disk_io else 0

        # Prime CPU measurement
        psutil.cpu_percent(interval=None)

    @staticmethod
    def _safe_net_counters():
        try:
            return psutil.net_io_counters()
        except Exception:
            return None

    @staticmethod
    def _safe_disk_io_counters():
        try:
            return psutil.disk_io_counters()
        except Exception:
            return None

    def collect_once(self) -> dict:
        """Captures fault-isolated OS metrics and converts cumulative counters to KB/s rates."""
        now_mono = time.monotonic()
        dt = max(now_mono - self.last_monotonic, 0.001)
        self.last_monotonic = now_mono

        try:
            cpu = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu = 0.0

        try:
            mem = float(psutil.virtual_memory().percent)
        except Exception:
            mem = 0.0

        try:
            disk_path = "C:\\" if psutil.WINDOWS else "/"
            disk = float(psutil.disk_usage(disk_path).percent)
        except Exception:
            disk = 0.0

        # Network Throughput Rates (KB/s)
        net = self._safe_net_counters()
        if net:
            sent_kbps = max(0.0, (net.bytes_sent - self.last_bytes_sent) / dt / 1024.0)
            recv_kbps = max(0.0, (net.bytes_recv - self.last_bytes_recv) / dt / 1024.0)
            self.last_bytes_sent = net.bytes_sent
            self.last_bytes_recv = net.bytes_recv
        else:
            sent_kbps, recv_kbps = 0.0, 0.0

        # Disk I/O Rates (KB/s)
        disk_io = self._safe_disk_io_counters()
        if disk_io:
            read_kbps = max(0.0, (disk_io.read_bytes - self.last_disk_read) / dt / 1024.0)
            write_kbps = max(0.0, (disk_io.write_bytes - self.last_disk_write) / dt / 1024.0)
            self.last_disk_read = disk_io.read_bytes
            self.last_disk_write = disk_io.write_bytes
        else:
            read_kbps, write_kbps = 0.0, 0.0

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.source_name,
            "cpu_percent": round(cpu, 2),
            "memory_percent": round(mem, 2),
            "disk_percent": round(disk, 2),
            "disk_read_kbps": round(read_kbps, 2),
            "disk_write_kbps": round(write_kbps, 2),
            "net_sent_kbps": round(sent_kbps, 2),
            "net_recv_kbps": round(recv_kbps, 2),
            "is_labeled_anomaly": 0,
            "scenario_label": "normal",
        }

def run_collector_loop(max_iterations: int = None):
    init_db()
    collector = SystemCollector(source_name="live")
    print(f"[Collector] Monitoring started (Interval: {COLLECTION_INTERVAL_SEC}s). Press Ctrl+C to stop.")
    
    count = 0
    try:
        while True:
            time.sleep(COLLECTION_INTERVAL_SEC)
            record = collector.collect_once()
            insert_metric(record)
            count += 1
            print(
                f"[{count}] CPU: {record['cpu_percent']}% | RAM: {record['memory_percent']}% | "
                f"Disk W: {record['disk_write_kbps']} KB/s | Net Down: {record['net_recv_kbps']} KB/s"
            )
            if max_iterations and count >= max_iterations:
                break
    except KeyboardInterrupt:
        print("\n[Collector] Stopped cleanly.")

if __name__ == "__main__":
    run_collector_loop()