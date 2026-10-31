# tests/test_pipeline.py
import pandas as pd
from src.collector import SystemCollector
from src.simulator import generate_synthetic_workload
from src.database import fetch_recent_metrics
from src.features import extract_features, get_feature_columns
from src.evaluator import run_scientific_evaluation
from src.forecaster import forecast_resource_exhaustion


def test_collector_fault_tolerant_snapshot():
    collector = SystemCollector(source_name="test_live")
    snap = collector.collect_once()
    assert 0.0 <= snap["cpu_percent"] <= 100.0
    assert 0.0 <= snap["memory_percent"] <= 100.0
    assert snap["net_sent_kbps"] >= 0.0
    assert snap["disk_write_kbps"] >= 0.0


def test_feature_engineering_no_nans():
    generate_synthetic_workload(total_steps=60)
    raw = fetch_recent_metrics("synthetic", limit=60)
    feat = extract_features(raw)
    feature_cols = get_feature_columns()
    assert len(feature_cols) == 28
    assert not feat[feature_cols].isna().any().any()


def test_end_to_end_ml_beats_static_rules_and_forecasts_leak():
    generate_synthetic_workload(total_steps=600)
    results = run_scientific_evaluation(train_size=300)
    assert results["ml_metrics"]["f1_score"] > results["rule_metrics"]["f1_score"]
    assert results["ml_metrics"]["recall"] >= 0.90

    # Verify sub-threshold anomaly is caught by ML but missed by static rules
    bdf = results["scenario_breakdown"].set_index("scenario")
    assert bdf.loc["sub_threshold_exfiltration", "ml_detection_rate_pct"] >= 85.0
    assert bdf.loc["sub_threshold_exfiltration", "static_rule_rate_pct"] == 0.0

    # Verify forecaster catches the memory leak
    fc = results["memory_forecast"]
    assert fc["warning_active"] is True
    assert fc["r_squared"] >= 0.90