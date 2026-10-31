# app/dashboard.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import plotly.graph_objects as go
import streamlit as st

from config.settings import THRESHOLDS, COLLECTION_INTERVAL_SEC
from src.cloud_benchmark import load_aws_cloudwatch_benchmark, run_live_hardware_stress_test
from src.collector import SystemCollector
from src.database import init_db, insert_metric, fetch_recent_metrics
from src.detector import AnomalyDetector
from src.evaluator import run_scientific_evaluation
from src.features import extract_features
from src.forecaster import forecast_resource_exhaustion
from src.simulator import generate_synthetic_workload

st.set_page_config(
    page_title="Cloud-Sentinel | AI Infrastructure Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# --- Sidebar Controls ---
st.sidebar.title("Cloud-Sentinel AIOps")
st.sidebar.caption("Cloud Infrastructure Monitoring & Predictive Analytics")

data_mode = st.sidebar.radio(
    "Select Telemetry Source:",
    options=[
        "Synthetic Evaluation Lab (Controlled)",
        "AWS CloudWatch Benchmark (NAB EC2)",
        "Live Host Telemetry (psutil)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Actions & Controls")

if data_mode.startswith("Synthetic"):
    if st.sidebar.button("Regenerate Synthetic Chaos Dataset", width="stretch"):
        generate_synthetic_workload()
        st.sidebar.success("Generated 600 synthetic rows!")
elif data_mode.startswith("AWS"):
    if st.sidebar.button("Reload AWS CloudWatch EC2 Trace", width="stretch"):
        meta = load_aws_cloudwatch_benchmark()
        st.sidebar.success(f"Loaded {meta['rows_loaded']} AWS EC2 rows ({meta['anomalies_labeled']} anomalies)!")
else:
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Sample 1 Tick", width="stretch"):
        collector = SystemCollector(source_name="live")
        insert_metric(collector.collect_once())
    if col_b.button("Sample 15 Ticks", width="stretch"):
        collector = SystemCollector(source_name="live")
        with st.sidebar.status("Sampling calm live baseline..."):
            for _ in range(15):
                insert_metric(collector.collect_once())
        st.sidebar.success("Captured 15 calm ticks!")

    if st.sidebar.button("1. Train AI on Current Calm Live Data", width="stretch"):
        live_raw = fetch_recent_metrics("live", limit=500)
        if len(live_raw) >= 10:
            det = AnomalyDetector()
            det.train(extract_features(live_raw), save=True)
            st.sidebar.success(f"Trained live baseline on {len(live_raw)} ticks!")
        else:
            st.sidebar.warning("Collect at least 10 live ticks first.")

    if st.sidebar.button("2. Trigger 8s Live CPU Stress Test", width="stretch"):
        with st.sidebar.status("Running multi-core CPU stress workload (8s)..."):
            captured = run_live_hardware_stress_test(duration_sec=8)
        st.sidebar.error(f"Injected {captured} real high-CPU hardware ticks! Check Tab 1.")

st.sidebar.markdown("---")
st.sidebar.subheader("Static Rule Thresholds")
for k, v in THRESHOLDS.items():
    st.sidebar.text(f"{k}: {v}")

# --- Load & Process Active Dataset ---
if data_mode.startswith("Synthetic"):
    raw_df = fetch_recent_metrics("synthetic", limit=600)
    if len(raw_df) < 600:
        generate_synthetic_workload()
    eval_bundle = run_scientific_evaluation(source="synthetic", train_size=300)
    df_analyzed = eval_bundle["analyzed_test_df"]
    source_badge = "SYNTHETIC DEMONSTRATION DATA (Test Split: 300 Ticks)"

elif data_mode.startswith("AWS"):
    raw_df = fetch_recent_metrics("aws_cloudwatch", limit=600)
    if len(raw_df) < 600:
        load_aws_cloudwatch_benchmark()
    eval_bundle = run_scientific_evaluation(source="aws_cloudwatch", train_size=300)
    df_analyzed = eval_bundle["analyzed_test_df"]
    source_badge = "REAL AWS CLOUDWATCH EC2 TRACE (Numenta Benchmark Test Split)"

else:
    raw_df = fetch_recent_metrics("live", limit=300)
    source_badge = "LIVE HOST TELEMETRY (Real-Time OS Hardware Sensor)"
    if len(raw_df) < 5:
        st.info("Need at least 5 live telemetry rows. Click **Sample 15 Ticks** in the sidebar to begin.")
        st.stop()

    feat_df = extract_features(raw_df)
    detector = AnomalyDetector()
    if not detector.load():
        detector.train(feat_df, save=True)
    df_analyzed = detector.analyze(feat_df)
    eval_bundle = None

# --- Header Banner ---
st.title("AI-Powered Cloud Infrastructure Monitoring & Predictive Analytics")
st.markdown(f"**Active Telemetry Stream:** `{source_badge}` | **Sampling Interval:** `{COLLECTION_INTERVAL_SEC}s`")

# --- Section A: System Overview ---
latest = df_analyzed.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CPU Utilization", f"{latest['cpu_percent']:.1f}%", f"{latest['cpu_percent_delta']:+.1f}%")
c2.metric("Memory Utilization", f"{latest['memory_percent']:.1f}%", f"{latest['memory_percent_delta']:+.1f}%")
c3.metric("Disk Utilization", f"{latest['disk_percent']:.1f}%")
c4.metric("Disk Write I/O", f"{latest['disk_write_kbps']:.1f} KB/s")
c5.metric("Network Outbound", f"{latest['net_sent_kbps']:.1f} KB/s")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Telemetry & ML Anomaly Detection",
    "2. Predictive Analytics (Time-to-Exhaustion)",
    "3. Alerts Log (Static Rules vs. ML)",
    "4. Scientific Model Evaluation & Viva Proof",
])

with tab1:
    st.subheader("Time-Series Resource Telemetry with Isolation Forest Overlay")
    metric_choice = st.selectbox(
        "Select Metric to Inspect:",
        options=["cpu_percent", "memory_percent", "disk_read_kbps", "disk_write_kbps", "net_sent_kbps", "net_recv_kbps"],
        index=0,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_analyzed["timestamp"],
        y=df_analyzed[metric_choice],
        mode="lines",
        name=metric_choice,
        line=dict(color="#38bdf8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df_analyzed["timestamp"],
        y=df_analyzed[f"{metric_choice}_roll_mean"],
        mode="lines",
        name="Rolling Mean (Window=10)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
    ))

    if metric_choice in THRESHOLDS:
        fig.add_hline(
            y=THRESHOLDS[metric_choice],
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Static Threshold ({THRESHOLDS[metric_choice]})",
        )

    anoms = df_analyzed[df_analyzed["ml_anomaly"] == 1]
    if not anoms.empty:
        fig.add_trace(go.Scatter(
            x=anoms["timestamp"],
            y=anoms[metric_choice],
            mode="markers",
            name="ML Detected Anomaly",
            marker=dict(color="#ef4444", size=9, symbol="x"),
            text=anoms["root_cause_summary"],
            hovertemplate="<b>ML Anomaly</b><br>Value: %{y}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Isolation Forest Anomaly Score Timeline")
    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=df_analyzed["timestamp"],
        y=df_analyzed["anomaly_score"],
        mode="lines",
        name="Anomaly Score",
        line=dict(color="#a855f7", width=2),
    ))
    fig_score.update_layout(height=240, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_score, width="stretch")

with tab2:
    st.subheader("Resource Exhaustion Forecaster (Linear Regression + Holt Smoothing)")
    target_res = st.radio("Select Resource to Forecast:", ["memory_percent", "disk_percent"], horizontal=True)
    fc = forecast_resource_exhaustion(df_analyzed, metric=target_res, critical_limit=95.0)

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Forecast Status", fc["status"])
    f2.metric("Growth Rate (Slope)", f"{fc['slope_per_min']:+.2f}% / min")
    f3.metric("Trend Confidence (R²)", f"{fc['r_squared']:.3f}")
    tte_label = f"{fc['tte_minutes']} mins" if fc["tte_minutes"] is not None else "No Exhaustion Risk"
    f4.metric("Estimated Time-to-Exhaustion (95%)", tte_label)

    if fc["warning_active"]:
        st.error(
            f"**PREDICTIVE EXHAUSTION WARNING:** `{target_res}` is growing at `{fc['slope_per_min']:+.2f}%/min` "
            f"(Confidence R² = {fc['r_squared']}). Projected to breach `{fc['critical_limit']}%` in **{fc['tte_minutes']} minutes**."
        )
    else:
        st.success(f"`{target_res}` trajectory is stable. No imminent resource exhaustion predicted within horizon.")

    recent_hist = df_analyzed.tail(75)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=list(range(len(recent_hist))),
        y=recent_hist[target_res],
        mode="lines",
        name="Recent Observed Utilization",
        line=dict(color="#22c55e", width=2.5),
    ))
    if fc["forecast_values"]:
        future_x = list(range(len(recent_hist), len(recent_hist) + len(fc["forecast_values"])))
        fig_fc.add_trace(go.Scatter(
            x=future_x,
            y=fc["forecast_values"],
            mode="lines",
            name="Projected Trajectory (Next 60 Ticks)",
            line=dict(color="#ef4444" if fc["warning_active"] else "#38bdf8", width=2.5, dash="dash"),
        ))
    fig_fc.add_hline(y=95.0, line_dash="dot", line_color="#ef4444", annotation_text="Critical Exhaustion Limit (95%)")
    fig_fc.update_layout(height=380, template="plotly_dark", xaxis_title="Relative Time Steps (Ticks)", yaxis_title=f"{target_res} (%)")
    st.plotly_chart(fig_fc, width="stretch")

with tab3:
    st.subheader("Operational Alert Separation: Static Rules vs. ML Anomaly Detection")
    st.caption(
        "Note: 'root_cause_summary' is a transparent, rule-based Z-score summary of deviating metrics surrounding an anomaly, "
        "distinct from internal Isolation Forest tree splits."
    )
    a1, a2, a3 = st.columns(3)
    a1.metric("Static Threshold Violations", int(df_analyzed["rule_alert"].sum()))
    a2.metric("Raw ML Point Anomalies", int(df_analyzed["ml_anomaly"].sum()))
    a3.metric("Persistence-Filtered Incidents", int(df_analyzed["incident_alert"].sum()))

    alerts_df = df_analyzed[(df_analyzed["ml_anomaly"] == 1) | (df_analyzed["rule_alert"] == 1)][[
        "timestamp", "scenario_label", "cpu_percent", "memory_percent",
        "net_sent_kbps", "anomaly_score", "rule_alert", "ml_anomaly",
        "incident_alert", "rule_violation_detail", "root_cause_summary"
    ]].tail(50)
    st.dataframe(alerts_df, width="stretch")

with tab4:
    st.subheader("Defensible Model Evaluation (Chronological Train/Test Split)")
    st.info(
        "**Evaluation Disclosure:** Metrics below are computed on the unseen test split (300 ticks) after training strictly "
        "on the preceding 300 baseline ticks. Performance on benchmark/synthetic scenarios validates pipeline mechanics and "
        "multivariate sensitivity but does not guarantee identical precision on arbitrary production workloads."
    )
    if eval_bundle is not None:
        m_ml = eval_bundle["ml_metrics"]
        m_rule = eval_bundle["rule_metrics"]
        cm = m_ml["confusion_matrix"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Isolation Forest (28 Rolling Features)")
            st.write(f"- **Precision:** `{m_ml['precision']*100:.1f}%`")
            st.write(f"- **Recall:** `{m_ml['recall']*100:.1f}%`")
            st.write(f"- **F1-Score:** `{m_ml['f1_score']*100:.1f}%`")
            st.write(f"- **Confusion Matrix:** `TP={cm['tp']} | FP={cm['fp']} | TN={cm['tn']} | FN={cm['fn']}`")
        with col2:
            st.markdown("#### Traditional Static Threshold Rules")
            st.write(f"- **Precision:** `{m_rule['precision']*100:.1f}%`")
            st.write(f"- **Recall:** `{m_rule['recall']*100:.1f}%`")
            st.write(f"- **F1-Score:** `{m_rule['f1_score']*100:.1f}%`")

        st.markdown("#### Per-Scenario Detection Comparison")
        st.dataframe(eval_bundle["scenario_breakdown"], width="stretch")
    else:
        st.warning("Switch to **Synthetic Evaluation Lab** or **AWS CloudWatch Benchmark** in the sidebar to view ground-truth evaluation metrics.")