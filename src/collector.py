# src/collector.py
import time
from datetime import datetime, timezone
import psutil
from config.settings import COLLECTION_INTERVAL_SEC
from src.database import init_db, insert_metric

class SystemCollector:
    def __init__(self, source_name: str = "live"):
        self.source_name = source_name
        net_init = psutil.net_io_counters()
        self.last_bytes_sent = net_init.bytes_sent if net_init else 0
        self.last_bytes_recv = net_init.bytes_recv if net_init else 0
        self.last_time = time.time()
        psutil.cpu_percent(interval=None)

    def collect_once(self) -> dict:
        """Captures a single snapshot of CPU, RAM, Disk, and Network rates."""
        now = time.time()
        dt = max(now - self.last_time, 0.001)

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        
        disk_path = "C:\\" if psutil.WINDOWS else "/"
        disk = psutil.disk_usage(disk_path).percent

        net = psutil.net_io_counters()
        if net:
            sent_kbps = max(0.0, (net.bytes_sent - self.last_bytes_sent) / dt / 1024.0)
            recv_kbps = max(0.0, (net.bytes_recv - self.last_bytes_recv) / dt / 1024.0)
            self.last_bytes_sent = net.bytes_sent
            self.last_bytes_recv = net.bytes_recv
        else:
            sent_kbps, recv_kbps = 0.0, 0.0

        self.last_time = now

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.source_name,
            "cpu_percent": round(cpu, 2),
            "memory_percent": round(mem, 2),
            "disk_percent": round(disk, 2),
            "net_sent_kbps": round(sent_kbps, 2),
            "net_recv_kbps": round(recv_kbps, 2),
            "is_labeled_anomaly": 0
        }

def run_collector_loop(max_iterations: int = None):
    """Continuously collects and stores system metrics."""
    init_db()
    collector = SystemCollector(source_name="live")
    print(f"[Collector] Started monitoring. Saving to SQLite every {COLLECTION_INTERVAL_SEC}s... (Press Ctrl+C to stop)")
    
    count = 0
    try:
        while True:
            time.sleep(COLLECTION_INTERVAL_SEC)
            record = collector.collect_once()
            insert_metric(record)
            count += 1
            print(
                f"[{count}] CPU: {record['cpu_percent']}% | "
                f"RAM: {record['memory_percent']}% | "
                f"Net Down: {record['net_recv_kbps']} KB/s"
            )
            if max_iterations and count >= max_iterations:
                break
    except KeyboardInterrupt:
        print("\n[Collector] Stopped cleanly by user.")

if __name__ == "__main__":
    run_collector_loop()