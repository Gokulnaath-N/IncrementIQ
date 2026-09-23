"""
Tiny synthetic-data tests for SMD and A/B test math. Run with:
    pytest tests/test_stage2.py -v
"""
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.ingestion import ensure_windows_hadoop_support
from src.data.transformation import compute_smd_table
from src.stats.ab_test import get_group_counts, two_proportion_ztest, retrospective_mde


@pytest.fixture(scope="module")
def spark():
    ensure_windows_hadoop_support()
    s = SparkSession.builder.appName("test").master("local[2]").getOrCreate()
    yield s
    s.stop()


@pytest.fixture
def synthetic_df(spark):
    # Constructed so f0 is perfectly balanced (mean 5 both groups -> SMD ~ 0)
    # and f1 is deliberately imbalanced (treatment mean >> control mean -> SMD >> 0.1)
    rows = []
    for i in range(200):
        rows.append({
            "f0": 5.0 + (i % 3) * 0.01, "f1": 1.0 + (i % 3) * 0.01,
            "treatment": 1, "visit": 1 if i % 10 == 0 else 0, "conversion": 1 if i % 50 == 0 else 0,
        })
    for i in range(200):
        rows.append({
            "f0": 5.0 + (i % 3) * 0.01, "f1": 10.0 + (i % 3) * 0.01,
            "treatment": 0, "visit": 1 if i % 20 == 0 else 0, "conversion": 1 if i % 100 == 0 else 0,
        })
    return spark.createDataFrame(rows)


def test_smd_balanced_feature_near_zero(synthetic_df, monkeypatch):
    import src.config.config as cfg
    monkeypatch.setattr(cfg, "FEATURE_COLS", ["f0", "f1"])
    import src.data.transformation as tr
    monkeypatch.setattr(tr, "FEATURE_COLS", ["f0", "f1"])

    result = compute_smd_table(synthetic_df)
    smd_by_feature = {r["feature"]: r for r in result}

    assert abs(smd_by_feature["f0"]["smd"]) < 0.1
    assert smd_by_feature["f0"]["balanced"] is True


def test_smd_imbalanced_feature_flagged(synthetic_df, monkeypatch):
    import src.data.transformation as tr
    monkeypatch.setattr(tr, "FEATURE_COLS", ["f0", "f1"])

    result = compute_smd_table(synthetic_df)
    smd_by_feature = {r["feature"]: r for r in result}

    assert abs(smd_by_feature["f1"]["smd"]) > 0.1
    assert smd_by_feature["f1"]["balanced"] is False


def test_group_counts_match_expected(synthetic_df):
    counts = get_group_counts(synthetic_df, "visit")
    assert counts["n_treatment"] == 200
    assert counts["n_control"] == 200
    assert counts["x_treatment"] == 20  # every 10th row, 200/10
    assert counts["x_control"] == 10    # every 20th row, 200/20


def test_ztest_detects_clear_difference(synthetic_df):
    counts = get_group_counts(synthetic_df, "visit")
    result = two_proportion_ztest(counts)
    # treatment visit rate (0.10) vs control (0.05) -- a real, if modest, difference
    assert result["p_treatment"] > result["p_control"]
    assert result["absolute_lift"] > 0
    assert 0.0 <= result["p_value"] <= 1.0


def test_mde_is_positive_and_reasonable(synthetic_df):
    counts = get_group_counts(synthetic_df, "conversion")
    result = retrospective_mde(counts)
    assert result["mde_absolute"] > 0
    assert result["baseline_control_rate"] >= 0
