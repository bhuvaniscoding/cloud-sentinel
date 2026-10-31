# Cloud-Sentinel: AI-Powered Cloud Infrastructure Monitoring & Predictive Analytics

An end-to-end **AIOps telemetry, unsupervised anomaly detection, and time-to-exhaustion forecasting platform** built for cloud virtual machines and local hosts.

## 1. Problem Statement & Why Machine Learning?
Traditional cloud monitoring relies on static threshold rules (e.g., `Alert if CPU >= 85%`). Fixed thresholds fail in two critical scenarios:
1. **Sub-Threshold Multivariate Anomalies:** An attacker exfiltrating data or a misconfigured service might hold CPU at `62%`, outbound network at `520 KB/s`, and disk read at `1,150 KB/s`. Every metric remains safely below its static limit, yet the joint pattern is highly abnormal.
2. **Gradual Resource Leaks:** A memory leak slowly growing from `50%` to `94%` triggers zero static alerts until the server is minutes away from an Out-Of-Memory (OOM) crash.

**Cloud-Sentinel** complements traditional rules with:
- **Causal Sliding-Window Feature Engineering (28 features):** Rolling means, rolling volatility, and first-order velocity deltas.
- **Unsupervised Isolation Forest + Z-Score Attribution:** Detects multivariate regime shifts while providing transparent, rule-based Z-score summaries of top deviating metrics.
- **Predictive Resource Forecaster:** Combines Ordinary Least Squares (OLS) linear trend regression ($R^2$ confidence gating) with **Holt's Linear Exponential Smoothing** to estimate exact **Time-to-Exhaustion (TTE)** in minutes.

---

## 2. System Architecture

```text
[ Telemetry Sources ]
  ├── Live OS Kernel Sensor (src/collector.py via psutil monotonic rate deltas)
  ├── Synthetic Chaos Lab (src/simulator.py -> 4 controlled failure modes)
  └── Real AWS CloudWatch EC2 Traces (src/cloud_benchmark.py -> Numenta NAB)
          │
          ▼
[ Concurrent Storage (src/database.py) ]
  └── SQLite in Write-Ahead Logging (WAL) mode for lock-free concurrent reads/writes
          │
          ▼
[ Feature Engineering Pipeline (src/features.py) ]
  └── 7 Raw Metrics × 4 Causal Transforms (Raw, Roll Mean, Roll Std, Delta) = 28 Features
          │
          ├───────────────────────────────────────────┐
          ▼                                           ▼
[ Anomaly Engine (src/detector.py) ]        [ Forecaster (src/forecaster.py) ]
  • StandardScaler + Isolation Forest         • Linear Trend Slope + R² Gating
  • 98.5th Percentile Baseline Calibration    • Holt's Exponential Smoothing
  • Persistence Filter (Incident Alerts)      • Time-to-Exhaustion (TTE) Projection
  • Z-Score Root-Cause Attribution
          │                                           │
          └───────────────────┬───────────────────────┘
                              ▼
[ Scientific Evaluator & Streamlit Command Center (app/dashboard.py) ]
```

---

## 3. Empirical Benchmark Results (300 Train / 300 Unseen Test Split)

| Detection Method | Precision | Recall | F1-Score | Sub-Threshold Attack Recall | Gradual Memory Leak Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Threshold Rules (85%)** | 100.0% | 42.4% | 59.5% | **0.0%** (0/21) | 27.7% (18/65) |
| **Cloud-Sentinel Isolation Forest** | **83.7%** | **100.0%** | **91.1%** | **100.0%** (21/21) | **100.0%** (65/65) |

---

## 4. Quickstart (Local Windows / Linux)

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows: .\venv\Scripts\activate | Linux/macOS: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic & AWS benchmark data and run scientific evaluation
python -m src.simulator
python -m src.cloud_benchmark
python -m src.evaluator

# 4. Run automated test suite
pytest -v

# 5. Launch the interactive dashboard
streamlit run app/dashboard.py
```

---

## 5. Linux Cloud Virtual Machine Deployment (AWS EC2 / DigitalOcean / Azure)

```bash
# On an Ubuntu 22.04 / 24.04 Linux Cloud VM:
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
git clone [https://github.com/bhuvaniscoding/cloud-sentinel.git](https://github.com/bhuvaniscoding/cloud-sentinel.git)
cd cloud-sentinel

# Launch both background collector daemon and web dashboard
sudo docker compose up --build -d
```

Access the dashboard at `http://<YOUR_VM_PUBLIC_IP>:8501`.