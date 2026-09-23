"""
Stage 1 — Ingestion.

Reads the raw Criteo CSV.gz ONCE with an explicit schema (never let Spark infer
schema on a 3+ GB file — it forces a full extra read pass), drops the leak
column, and writes:
  1. data/processed/full/        — full dataset as partitioned Parquet
  2. data/processed/sample_10pct — stratified 10% dev sample (by treatment)

Run:
    python -m src.data.ingestion
"""
import logging
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import (
    FEATURE_COLS,
    FULL_PARQUET,
    LABEL_COLS,
    LEAK_COL,
    RANDOM_SEED,
    RAW_CSV_GZ,
    SAMPLE_FRACTION,
    SAMPLE_PARQUET,
    TREATMENT_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Actual CSV header order in the raw gzip file:
# f0..f11, treatment, conversion, visit, exposure
# We keep the schema aligned to the real file so Spark does not reject the header.
SCHEMA = StructType(
    [StructField(c, DoubleType(), False) for c in FEATURE_COLS]
    + [
        StructField(TREATMENT_COL, IntegerType(), False),
        StructField("conversion", IntegerType(), False),
        StructField("visit", IntegerType(), False),
        StructField(LEAK_COL, IntegerType(), False),
    ]
)


def ensure_windows_hadoop_support() -> None:
    if os.name != "nt":
        return

    hadoop_home = Path(os.environ.get("HADOOP_HOME", "C:/hadoop"))
    winutils_path = hadoop_home / "bin" / "winutils.exe"

    if not winutils_path.exists():
        raise FileNotFoundError(
            "Spark on Windows requires Hadoop winutils.exe for local file writes. "
            "Install it and set HADOOP_HOME to the directory containing bin/winutils.exe, "
            f"or create {winutils_path}."
        )

    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(app_name: str = "criteo-ingestion") -> SparkSession:
    ensure_windows_hadoop_support()
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )


def run():
    if not RAW_CSV_GZ.exists():
        raise FileNotFoundError(
            f"{RAW_CSV_GZ} not found. Run scripts/download_dataset.py first."
        )

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info("Reading raw CSV.gz with explicit schema: %s", RAW_CSV_GZ)
    df = spark.read.csv(str(RAW_CSV_GZ), header=True, schema=SCHEMA)

    df = df.drop(LEAK_COL)

    row_count = df.count()
    log.info("Loaded %s rows, %s columns", f"{row_count:,}", len(df.columns))

    keep_cols = FEATURE_COLS + [TREATMENT_COL] + LABEL_COLS
    df = df.select(*keep_cols)

    log.info("Writing full Parquet to %s", FULL_PARQUET)
    (
        df.repartition(64)
        .write.mode("overwrite")
        .partitionBy(TREATMENT_COL)
        .parquet(str(FULL_PARQUET))
    )

    log.info("Building stratified %.0f%% dev sample (seed=%d)", SAMPLE_FRACTION * 100, RANDOM_SEED)
    fractions = {0: SAMPLE_FRACTION, 1: SAMPLE_FRACTION}
    sample_df = df.sampleBy(TREATMENT_COL, fractions=fractions, seed=RANDOM_SEED)

    sample_count = sample_df.count()
    log.info("Sample rows: %s (%.2f%% of full)", f"{sample_count:,}", 100 * sample_count / row_count)

    (
        sample_df.repartition(8)
        .write.mode("overwrite")
        .partitionBy(TREATMENT_COL)
        .parquet(str(SAMPLE_PARQUET))
    )

    log.info("Ingestion complete.")
    spark.stop()


if __name__ == "__main__":
    run()
