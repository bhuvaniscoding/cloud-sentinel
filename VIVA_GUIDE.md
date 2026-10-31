# Cloud-Sentinel: Complete Viva Defense & Architecture Guide

## 1. How Every File Works (In 1 Sentence Each)
- **`config/settings.py`**: Centralizes sampling intervals, static rule limits, rolling window sizes (`10`), and ML hyperparameters so nothing is hardcoded.
- **`src/database.py`**: Manages SQLite connections using `@contextmanager` and `PRAGMA journal_mode=WAL` so the background collector writes every 2 seconds while Streamlit reads concurrently without `Database is locked` errors.
- **`src/collector.py`**: Reads live CPU, RAM, Disk, Disk I/O, and Network counters via `psutil`, converting cumulative OS byte counters into per-second rates (`KB/s`) using `time.monotonic()`.
- **`src/simulator.py`**: Generates 600 deterministic rows (`seed=42`): 300 clean baseline rows and 300 test rows with 4 labeled anomalies (`cpu_spike`, `sub_threshold_exfiltration`, `disk_io_thrashing`, `gradual_memory_leak`).
- **`src/cloud_benchmark.py`**: Fetches real Amazon EC2 CPU traces from the Numenta Anomaly Benchmark (NAB) and includes a multi-core CPU stress generator for live hardware demonstrations.
- **`src/features.py`**: Transforms 7 raw metrics into 28 causal sliding-window features (`raw`, `roll_mean`, `roll_std`, `delta`).
- **`src/detector.py`**: Fits `StandardScaler` + `IsolationForest`, calibrates the anomaly threshold at the 98.5th baseline percentile, applies a persistence filter to stop alert fatigue, and computes transparent Z-score root-cause summaries.
- **`src/forecaster.py`**: Combines Ordinary Least Squares linear regression ($R^2 \ge 0.55$) and Holt's Linear Exponential Smoothing to calculate **Time-to-Exhaustion (TTE)** in minutes.
- **`src/evaluator.py`**: Splits train (`0..299`) and test (`300..599`) chronologically *before* feature extraction to prevent data leakage, outputting Precision, Recall, F1, and Confusion Matrix.

---

## 2. Top 6 Viva Questions & Your Exact Knockout Answers

### Q1: "Why didn't you just use `if cpu > 85%`? Why use Machine Learning?"
**Answer:** "Static thresholds only look at one metric at a time and only trigger after saturation reaches 85%. In our benchmark (`src/evaluator.py`), we tested a `sub_threshold_exfiltration` scenario where CPU is at 64%, outbound network is 520 KB/s, and disk read is 1,150 KB/s. Static rules caught **0%** of those ticks because every individual number was below its fixed limit. Our Isolation Forest caught **100%** because it isolates abnormal multivariate combinations across 28 rolling features."

### Q2: "Isolation Forest is not a time-series model. How did you make it work on time-series data?"
**Answer:** "Raw Isolation Forest treats each row independently. To give it temporal memory without the heavy CPU overhead of deep learning (LSTMs), we engineered causal sliding-window features in `src/features.py`—specifically the 10-tick rolling mean, 10-tick rolling standard deviation (volatility), and 1-tick velocity delta ($\Delta x_t = x_t - x_{t-1}$)."

### Q3: "How does your model explain *why* an anomaly happened? Does Isolation Forest give feature importances?"
**Answer:** "No—standard Isolation Forest in scikit-learn does not provide per-sample feature attribution. Instead of inventing black-box explanations, we compute a transparent, rule-based **Z-score deviation summary** relative to the training baseline ($Z = \frac{x - \mu_{train}}{\sigma_{train}}$) alongside the model prediction, and we explicitly distinguish that summary from the model's internal tree splits."

### Q4: "How did you prevent data leakage during evaluation?"
**Answer:** "In time-series, if you compute rolling averages across the whole dataset before splitting into train and test, rows at the start of the test set leak information from the end of the training set. In `src/evaluator.py`, we slice `raw_train = raw_df.iloc[:300]` and `raw_test = raw_df.iloc[300:]` *first*, and run `extract_features()` independently on each split."

### Q5: "How does your Predictive Analytics (Time-to-Exhaustion) work mathematically?"
**Answer:** "In `src/forecaster.py`, we take the last 75 observations and compute both Ordinary Least Squares linear regression ($y = mx + c$) and Holt's Double Exponential Smoothing (level $\alpha=0.35$, trend $\beta=0.15$). To prevent false alarms on random noise, we gate predictions using the Coefficient of Determination ($R^2 \ge 0.55$). When a sustained positive slope is confirmed, Time-to-Exhaustion is calculated as $\text{TTE} = \frac{\text{Critical Limit (95\%)} - \text{Current Value}}{\text{Growth Rate per Minute}}$."

### Q6: "Why did you convert Network and Disk I/O counters into deltas?"
**Answer:** "`psutil.net_io_counters()` and `disk_io_counters()` return cumulative bytes since system boot, which monotonically increase forever. Feeding cumulative totals into ML causes severe concept drift. We subtract the previous reading from the current reading and divide by elapsed `time.monotonic()` seconds to get stationary throughput rates in KB/s."