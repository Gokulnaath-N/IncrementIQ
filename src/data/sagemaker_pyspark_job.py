"""
Stage 3 - SageMaker PySpark Processing Job.

Runs inside AWS SageMaker managed PySpark container (smspark-submit / spark-submit --master yarn).

CRITICAL: Under YARN mode, Spark resolves bare paths as HDFS URIs.
All Spark read/write paths MUST use file:// prefix to hit the local container FS.

Arguments:
    --input-dir:       Directory with raw criteo CSV.gz
    --output-dir:      Directory to write Parquet + reports
    --sample-fraction: Stratified dev sample fraction (default: 0.10)
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Log to stdout so smspark-submit captures to CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("sagemaker-pyspark-job")

FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
LABEL_COLS = ["visit", "conversion"]
LEAK_COL = "exposure"
SMD_BALANCED_THRESHOLD = 0.10

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
    log.info("PySpark imports OK")
except ImportError as exc:
    log.error("PySpark import failed: %s", exc)
    sys.exit(1)

# nullable=True avoids AnalysisException on null rows
SCHEMA = StructType(
    [StructField(c, DoubleType(), True) for c in FEATURE_COLS]
    + [
        StructField(TREATMENT_COL, IntegerType(), True),
        StructField("conversion", IntegerType(), True),
        StructField("visit", IntegerType(), True),
        StructField(LEAK_COL, IntegerType(), True),
    ]
)


def to_file_uri(path) -> str:
    """Prefix an absolute POSIX path with file:// for YARN-mode Spark."""
    return f"file://{path}"


def compute_smd_table(df) -> list:
    agg_exprs = []
    for c in FEATURE_COLS:
        agg_exprs.append(F.mean(c).alias(f"{c}__mean"))
        agg_exprs.append(F.variance(c).alias(f"{c}__var"))
    grouped = df.groupBy(TREATMENT_COL).agg(*agg_exprs).collect()
    stats = {row[TREATMENT_COL]: row.asDict() for row in grouped}
    t, c = stats[1], stats[0]
    rows = []
    for col in FEATURE_COLS:
        mt, mc = t[f"{col}__mean"], c[f"{col}__mean"]
        vt, vc = t[f"{col}__var"], c[f"{col}__var"]
        pooled_sd = math.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({
            "feature": col,
            "mean_treatment": round(mt, 6),
            "mean_control": round(mc, 6),
            "var_treatment": round(vt, 6),
            "var_control": round(vc, 6),
            "smd": round(smd, 6),
            "balanced": abs(smd) < SMD_BALANCED_THRESHOLD,
        })
    return rows


def compute_ab_counts(df, outcome_col: str) -> dict:
    agg = df.groupBy(TREATMENT_COL).agg(
        F.count("*").alias("n"),
        F.sum(outcome_col).alias("x"),
    ).collect()
    by_group = {row[TREATMENT_COL]: row.asDict() for row in agg}
    return {
        "n_treatment": by_group[1]["n"],
        "x_treatment": int(by_group[1]["x"]),
        "n_control": by_group[0]["n"],
        "x_control": int(by_group[0]["x"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    args = parser.parse_args()

    log.info("=== SageMaker PySpark Job Starting ===")
    log.info("input-dir: %s  output-dir: %s  sample: %.2f",
             args.input_dir, args.output_dir, args.sample_fraction)

    # Debug: show all mounted files
    base = Path("/opt/ml/processing/input")
    for f in base.rglob("*"):
        if f.is_file():
            log.info("[MOUNT] %s  %d MB", f, f.stat().st_size // (1024 * 1024))

    # Locate CSV
    csv_files = [
        f for f in base.rglob("*")
        if f.is_file() and (f.name.endswith(".csv.gz") or f.name.endswith(".csv"))
    ]
    if not csv_files:
        log.error("No CSV found under %s", base)
        sys.exit(1)

    # CRITICAL: file:// prefix so YARN resolves to local FS, not HDFS
    raw_uri = to_file_uri(csv_files[0])
    log.info("Raw file URI: %s", raw_uri)

    # SparkSession - defaultFS=file:/// makes bare paths also hit local FS
    log.info("Creating SparkSession...")
    spark = (
        SparkSession.builder
        .appName("sagemaker-criteo-pipeline")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info("SparkSession ready: %s", spark.sparkContext.applicationId)

    out = Path(args.output_dir)

    # 1. Read CSV
    log.info("Reading CSV from %s ...", raw_uri)
    df = (
        spark.read.csv(raw_uri, header=True, schema=SCHEMA)
        .drop(LEAK_COL)
        .select(*(FEATURE_COLS + [TREATMENT_COL] + LABEL_COLS))
        .cache()
    )
    total_rows = df.count()
    log.info("Rows loaded: {:,}".format(total_rows))

    # 2. Full Parquet (file:// URI)
    full_uri = to_file_uri(out / "processed" / "full")
    log.info("Writing full Parquet -> %s", full_uri)
    df.repartition(8).write.mode("overwrite").partitionBy(TREATMENT_COL).parquet(full_uri)
    log.info("Full Parquet done.")

    # 3. 10% sample
    sample_uri = to_file_uri(out / "processed" / "sample_10pct")
    log.info("Writing sample -> %s", sample_uri)
    sample_df = df.sampleBy(
        TREATMENT_COL,
        fractions={0: args.sample_fraction, 1: args.sample_fraction},
        seed=42,
    )
    sample_df.repartition(2).write.mode("overwrite").partitionBy(TREATMENT_COL).parquet(sample_uri)
    log.info("Sample Parquet done.")

    # 4. Stats
    log.info("Computing SMD table...")
    smd_table = compute_smd_table(df)
    log.info("Computing A/B counts...")
    ab_counts = {
        "visit": compute_ab_counts(df, "visit"),
        "conversion": compute_ab_counts(df, "conversion"),
    }

    # 5. JSON reports (plain file writes - not Spark)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "balance_table.json").write_text(
        json.dumps({"smd_threshold": SMD_BALANCED_THRESHOLD, "features": smd_table}, indent=2)
    )
    (reports / "ab_test_counts.json").write_text(
        json.dumps({"total_rows": total_rows, "counts": ab_counts}, indent=2)
    )

    log.info("=== SageMaker PySpark Job COMPLETE -> %s ===", out)
    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()   # full traceback goes to stdout -> CloudWatch
        sys.exit(1)
