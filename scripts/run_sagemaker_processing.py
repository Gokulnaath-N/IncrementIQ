"""
Stage 3 — AWS SageMaker PySpark Processor Launcher.

Orchestrates Stage 3:
1. Validates AWS credentials & S3 bucket configuration.
2. Uploads raw criteo CSV.gz to s3://<BUCKET>/raw/
3. Launches a SageMaker PySparkProcessor job (fire-and-poll, TCP-drop safe).
4. Polls job status every 30 s; safe to Ctrl-C and re-attach.

Usage:
    python scripts/run_sagemaker_processing.py --dry-run
    python scripts/run_sagemaker_processing.py --skip-upload
    python scripts/run_sagemaker_processing.py --attach <job-name>   # re-attach to running job
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sagemaker-launcher")

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT_DIR / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
JOB_SCRIPT = ROOT_DIR / "src" / "data" / "sagemaker_pyspark_job.py"


def check_aws_credentials(region: str) -> str:
    sts = boto3.client("sts", region_name=region)
    identity = sts.get_caller_identity()
    log.info("Authenticated to AWS as Account: %s, ARN: %s", identity["Account"], identity["Arn"])
    return identity["Account"]


def upload_to_s3(s3_client, local_path: Path, bucket: str, s3_key: str):
    log.info("Checking if %s exists in s3://%s/%s...", local_path.name, bucket, s3_key)
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        log.info("File already exists in S3. Skipping upload.")
    except Exception:
        log.info("Uploading %s (%d MB) to s3://%s/%s...", local_path.name, local_path.stat().st_size // (1024 * 1024), bucket, s3_key)
        s3_client.upload_file(str(local_path), bucket, s3_key)
        log.info("Upload complete.")


def run_pipeline(bucket: str, role_arn: str, region: str, instance_type: str, instance_count: int, dry_run: bool, skip_upload: bool = False):
    log.info("=== Stage 3: Launching SageMaker PySpark Processing Job ===")
    log.info("Region: %s", region)
    log.info("S3 Bucket: s3://%s", bucket)
    log.info("IAM Role: %s", role_arn)
    log.info("Cluster: %d x %s", instance_count, instance_type)

    if dry_run:
        log.info("[DRY RUN] Verification successful. AWS calls simulated without starting cloud resources.")
        log.info("[DRY RUN] Job script target: %s", JOB_SCRIPT)
        log.info("[DRY RUN] S3 Input URI: s3://%s/raw/", bucket)
        log.info("[DRY RUN] S3 Output URI: s3://%s/processed/", bucket)
        return

    try:
        check_aws_credentials(region)
    except Exception as e:
        log.error("Failed to verify AWS credentials: %s", e)
        log.error("Run 'aws configure' or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env.")
        sys.exit(1)

    if not skip_upload:
        s3 = boto3.client("s3", region_name=region)
        s3_raw_key = "raw/criteo-research-uplift-v2.1.csv.gz"
        upload_to_s3(s3, RAW_FILE, bucket, s3_raw_key)
    else:
        log.info("Skipping upload (--skip-upload set). Using existing raw data in s3://%s/raw/", bucket)

    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.spark.processing import PySparkProcessor

    sagemaker_session = sagemaker.Session(
        boto_session=boto3.Session(region_name=region),
        default_bucket=bucket,
    )

    spark_processor = PySparkProcessor(
        base_job_name="criteo-uplift-stage3",
        framework_version="3.5",
        role=role_arn,
        instance_type=instance_type,
        instance_count=instance_count,
        max_runtime_in_seconds=1800,
        sagemaker_session=sagemaker_session,
    )

    s3_input_uri = f"s3://{bucket}/raw/"
    s3_output_uri = f"s3://{bucket}/processed/"

    log.info("Submitting PySparkProcessor job to SageMaker (fire-and-poll mode)...")
    # Use relative POSIX path to avoid Windows drive-letter urlparse bug in SageMaker SDK
    relative_job_script = "src/data/sagemaker_pyspark_job.py"

    # Fire with wait=False so a TCP drop on the local side doesn't kill tracking.
    spark_processor.run(
        submit_app=relative_job_script,
        arguments=[
            "--input-dir", "/opt/ml/processing/input/data",
            "--output-dir", "/opt/ml/processing/output",
            "--sample-fraction", "0.10",
        ],
        inputs=[
            ProcessingInput(
                source=s3_input_uri,
                destination="/opt/ml/processing/input/data",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=s3_output_uri,
                s3_upload_mode="EndOfJob",
            )
        ],
        logs=False,   # avoid streaming log TCP connection (drops on Windows)
        wait=False,   # non-blocking — we poll below
    )

    job_name = spark_processor.latest_job.job_name
    log.info("Job submitted: %s", job_name)
    log.info("Re-attach anytime with: python scripts/run_sagemaker_processing.py --attach %s", job_name)
    _poll_job(region, job_name, s3_output_uri)


def _poll_job(region: str, job_name: str, s3_output_uri: str, poll_interval: int = 30):
    """Poll a SageMaker processing job until it reaches a terminal state."""
    sm = boto3.client("sagemaker", region_name=region)
    terminal_states = {"Completed", "Failed", "Stopped"}
    log.info("Polling job '%s' every %ds (safe to Ctrl-C and re-attach)...", job_name, poll_interval)
    while True:
        resp = sm.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        log.info("Job status: %s", status)
        if status in terminal_states:
            break
        time.sleep(poll_interval)

    if status == "Completed":
        log.info("✅ SageMaker Processing Job completed successfully!")
        log.info("Processed Parquet data at: %s", s3_output_uri)
    elif status == "Failed":
        failure = resp.get("FailureReason", "(no reason returned)")
        log.error("❌ Job FAILED. Reason: %s", failure)
        log.error("Check CloudWatch logs in ap-south-1 for group: /aws/sagemaker/ProcessingJobs")
        sys.exit(1)
    else:
        log.warning("⚠️  Job ended with status: %s", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker PySpark Processing Job")
    parser.add_argument("--bucket", type=str, default=os.getenv("S3_BUCKET", "criteo-uplift-gokulnaath"))
    parser.add_argument("--role", type=str, default=os.getenv("SAGEMAKER_ROLE_ARN", ""))
    parser.add_argument("--region", type=str, default=os.getenv("AWS_REGION", "ap-south-1"))
    parser.add_argument("--instance-type", type=str, default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading raw data (if already in S3)")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without submitting cloud job")
    parser.add_argument("--attach", type=str, default=None, metavar="JOB_NAME",
                        help="Re-attach to an existing in-progress job by name (skips submission)")
    args = parser.parse_args()

    if args.attach:
        # Re-attach mode: just poll the existing job
        region = args.region
        try:
            boto3.client("sts", region_name=region).get_caller_identity()
        except Exception as e:
            log.error("AWS credential check failed: %s", e)
            sys.exit(1)
        s3_output_uri = f"s3://{args.bucket}/processed/"
        _poll_job(region, args.attach, s3_output_uri)
    else:
        run_pipeline(
            bucket=args.bucket,
            role_arn=args.role,
            region=args.region,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            dry_run=args.dry_run,
            skip_upload=args.skip_upload,
        )
