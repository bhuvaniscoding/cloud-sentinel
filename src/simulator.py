# src/simulator.py
from datetime import datetime, timezone, timedelta
import numpy as np
from config.settings import RANDOM_SEED, COLLECTION_INTERVAL_SEC
from src.database import clear_source_metrics, insert_metrics_batch

def generate_synthetic_workload(total_steps: int = 600, seed: int = RANDOM_SEED) -> None:
    """
    Generates a reproducible, realistic time-series dataset (default 600 ticks):
    - Steps 0..299: Pure Normal Baseline (Training Split - sinusoidal diurnal load + noise)
    - Steps 300..599: Evaluation Test Split containing 4 distinct operational scenarios:
        1. 'cpu_spike' (Steps 340..355): Sudden compute saturation (>85% threshold)
        2. 'sub_threshold_exfiltration' (Steps 400..420): Multivariate anomaly!
           CPU=62%, Net Sent=480 KB/s, Disk Read=950 KB/s. Below static thresholds, caught by ML!
        3. 'disk_io_thrashing' (Steps 460..475): Runaway log/DB write storm.
        4. 'gradual_memory_leak' (Steps 510..599): Smooth linear ramp from 52% to 94% RAM
           specifically designed to validate both Anomaly Detection and Time-to-Exhaustion Forecasting.
    """
    clear_source_metrics("synthetic")
    rng = np.random.default_rng(seed)
    start_time = datetime.now(timezone.utc) - timedelta(seconds=total_steps * COLLECTION_INTERVAL_SEC)

    records = []
    for step in range(total_steps):
        ts = (start_time + timedelta(seconds=step * COLLECTION_INTERVAL_SEC)).isoformat()
        
        # Smooth realistic wave baseline using sine waves + Gaussian noise
        wave = np.sin(step / 25.0)
        cpu = float(np.clip(24.0 + 6.0 * wave + rng.normal(0, 2.2), 5.0, 100.0))
        mem = float(np.clip(48.0 + 2.0 * np.sin(step / 50.0) + rng.normal(0, 0.6), 20.0, 100.0))
        disk = float(np.clip(42.0 + (step * 0.001), 10.0, 100.0))
        disk_r = float(np.clip(120.0 + 30.0 * wave + rng.normal(0, 15.0), 5.0, 20000.0))
        disk_w = float(np.clip(80.0 + 20.0 * wave + rng.normal(0, 12.0), 5.0, 20000.0))
        net_s = float(np.clip(25.0 + 8.0 * wave + rng.normal(0, 4.0), 1.0, 10000.0))
        net_r = float(np.clip(60.0 + 15.0 * wave + rng.normal(0, 8.0), 2.0, 10000.0))

        is_anomaly = 0
        scenario = "normal"

        # Inject controlled anomalies ONLY in the Test Split (step >= 300)
        if 340 <= step <= 355:
            # Scenario 1: Obvious CPU Saturation Spike
            cpu = float(np.clip(rng.normal(93.0, 2.5), 86.0, 100.0))
            is_anomaly = 1
            scenario = "cpu_spike"

        elif 400 <= step <= 420:
            # Scenario 2: Sub-Threshold Data Exfiltration (Fools Static Thresholds!)
            # Static thresholds are CPU=85%, Net Sent=2000 KB/s, Disk Read=5000 KB/s.
            # Every metric stays well below its static limit, but the combination is abnormal.
            cpu = float(np.clip(rng.normal(64.0, 2.0), 58.0, 72.0))
            disk_r = float(np.clip(rng.normal(1150.0, 80.0), 900.0, 1500.0))
            net_s = float(np.clip(rng.normal(520.0, 40.0), 420.0, 650.0))
            is_anomaly = 1
            scenario = "sub_threshold_exfiltration"

        elif 460 <= step <= 475:
            # Scenario 3: Disk Write Thrashing + Network Ingress Surge
            disk_w = float(np.clip(rng.normal(6200.0, 350.0), 5200.0, 7500.0))
            net_r = float(np.clip(rng.normal(1800.0, 150.0), 1400.0, 2400.0))
            is_anomaly = 1
            scenario = "disk_io_thrashing"

        elif 510 <= step < total_steps:
            # Scenario 4: Gradual Memory Leak Ramp (For Predictive Forecasting!)
            leak_progress = (step - 510) / float(total_steps - 510)
            mem = float(np.clip(50.0 + leak_progress * 44.0 + rng.normal(0, 0.35), 50.0, 98.0))
            # Becomes an active anomaly once it diverges significantly from the 48% baseline
            if mem >= 62.0:
                is_anomaly = 1
                scenario = "gradual_memory_leak"

        records.append({
            "timestamp": ts,
            "source": "synthetic",
            "cpu_percent": round(cpu, 2),
            "memory_percent": round(mem, 2),
            "disk_percent": round(disk, 2),
            "disk_read_kbps": round(disk_r, 2),
            "disk_write_kbps": round(disk_w, 2),
            "net_sent_kbps": round(net_s, 2),
            "net_recv_kbps": round(net_r, 2),
            "is_labeled_anomaly": is_anomaly,
            "scenario_label": scenario,
        })

    insert_metrics_batch(records)
    total_anomalies = sum(r["is_labeled_anomaly"] for r in records)
    print(
        f"[Simulator] Reset & generated {len(records)} rows "
        f"(300 clean training rows + 300 test rows containing {total_anomalies} labeled anomalies)."
    )

if __name__ == "__main__":
    generate_synthetic_workload()