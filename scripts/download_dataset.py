import gzip
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DESTINATION = Path("data/raw/criteo-research-uplift-v2.1.csv.gz")
URLS = [
    "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
]
MAX_RETRIES = 3
CONNECT_TIMEOUT_SECONDS = 30
CHUNK_SIZE = 1024 * 1024


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_gzip_file(path: Path) -> None:
    try:
        with gzip.open(path, "rb") as fh:
            while True:
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    break
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise RuntimeError(f"Downloaded file is corrupt or incomplete: {path}. Error: {exc}") from exc

    logger.info("Integrity check passed for %s", path)


def file_exists_and_valid(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        validate_gzip_file(path)
        return True
    except RuntimeError as exc:
        logger.warning("Existing file failed integrity check: %s", exc)
        path.unlink(missing_ok=True)
        return False


def download_with_resume(url: str, destination: Path) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0"}

    with requests.Session() as session:
        response = session.get(url, stream=True, timeout=(CONNECT_TIMEOUT_SECONDS, CONNECT_TIMEOUT_SECONDS), headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        logger.info("Starting download from %s to %s", url, destination)

        downloaded = 0
        with open(temp_path, "wb") as file_obj, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
            leave=True,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                file_obj.write(chunk)
                downloaded += len(chunk)
                progress_bar.update(len(chunk))

        if total_size and downloaded != total_size:
            raise RuntimeError(
                f"Download size mismatch for {url}: expected {total_size} bytes but got {downloaded} bytes."
            )

    logger.info("Download completed successfully from %s", url)
    os.replace(temp_path, destination)
    validate_gzip_file(destination)


def try_download(urls: Iterable[str], destination: Path) -> None:
    last_error = None

    for attempt_index, url in enumerate(urls, start=1):
        for retry in range(MAX_RETRIES + 1):
            try:
                logger.info("Attempt %d/%d to download from %s", retry + 1, MAX_RETRIES + 1, url)
                download_with_resume(url, destination)
                logger.info("Success: dataset saved to %s", destination)
                return
            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning("Timeout while downloading from %s (attempt %d/%d)", url, retry + 1, MAX_RETRIES + 1)
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                logger.warning("Connection error while downloading from %s (attempt %d/%d)", url, retry + 1, MAX_RETRIES + 1)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                logger.warning("Request failed for %s (attempt %d/%d): %s", url, retry + 1, MAX_RETRIES + 1, exc)
            except RuntimeError as exc:
                last_error = exc
                logger.warning("Download validation failed for %s (attempt %d/%d): %s", url, retry + 1, MAX_RETRIES + 1, exc)

            if retry < MAX_RETRIES:
                sleep_seconds = 2 ** retry
                logger.info("Retrying in %s seconds for %s", sleep_seconds, url)
                time.sleep(sleep_seconds)
            else:
                logger.error("All retry attempts failed for %s", url)

        logger.warning("Falling back to next URL after %s", url)

    raise RuntimeError(f"Could not download the dataset from any configured URL. Last error: {last_error}")


def main() -> None:
    configure_logging()
    logger.info("Starting Criteo uplift dataset download process")
    ensure_parent_directory(DESTINATION)

    if file_exists_and_valid(DESTINATION):
        logger.info("File already exists and passes integrity check: %s", DESTINATION)
        final_size_mb = DESTINATION.stat().st_size / (1024 * 1024)
        logger.info("Final file size: %.2f MB", final_size_mb)
        return

    try_download(URLS, DESTINATION)

    final_size_mb = DESTINATION.stat().st_size / (1024 * 1024)
    logger.info("Final file size: %.2f MB", final_size_mb)
    logger.info("Dataset download process completed successfully")


if __name__ == "__main__":
    main()
