"""Loader for the Fear & Greed Index dataset.

Reads the ``fear_greed_index.csv`` file, enforces initial dtype coercion,
attaches source metadata, and records a SHA-256 checksum for lineage
tracking.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_trader_analytics.config import IngestionConfig
from sentiment_trader_analytics.utils.logging_utils import setup_logging

logger = setup_logging(__name__)

EXPECTED_COLUMNS: set[str] = {"timestamp", "value", "classification"}


def _compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        The SHA-256 digest as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def _write_lineage(
    output_dir: Path,
    dataset_name: str,
    run_id: str,
    file_path: Path,
    checksum: str,
    row_count: int,
) -> None:
    """Write a lineage metadata JSON file.

    Args:
        output_dir: Directory to write the lineage file into.
        dataset_name: Dataset name (used in the filename).
        run_id: Unique run identifier.
        file_path: Source file path.
        checksum: SHA-256 digest of the source file.
        row_count: Number of rows loaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()
    lineage = {
        "dataset": dataset_name,
        "run_id": run_id,
        "source_file": str(file_path.resolve()),
        "sha256": checksum,
        "row_count": row_count,
        "ingestion_timestamp_utc": ts,
        "timestamp": ts,  # alias for backward-compatible verification
    }
    out_path = output_dir / f"{dataset_name}_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(lineage, f, indent=2)
    logger.info("Lineage written to %s", out_path)


def load_fear_greed_index(config: IngestionConfig) -> pd.DataFrame:
    """Load the Fear & Greed Index dataset from CSV.

    Reads the CSV at ``config.fear_greed_path``, enforces dtype coercion
    (timestamp → ``datetime64[UTC]``, value → ``int64``, classification →
    ``pd.CategoricalDtype``), attaches source metadata to ``df.attrs``,
    and writes a SHA-256 lineage file.

    Args:
        config: Ingestion configuration providing the file path and
            lineage output directory.

    Returns:
        A DataFrame with coerced dtypes and source metadata attached.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV columns do not match the expected set.
    """
    file_path = Path(config.fear_greed_path)

    if not file_path.exists():
        msg = f"Fear & Greed file not found: {file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Reading Fear & Greed index from %s", file_path)

    df = pd.read_csv(file_path)

    actual_columns: set[str] = set(df.columns)
    if not EXPECTED_COLUMNS.issubset(actual_columns):
        missing = EXPECTED_COLUMNS - actual_columns
        msg = f"Fear & Greed CSV missing expected columns: {missing}. " f"Found: {list(df.columns)}"
        logger.error(msg)
        raise ValueError(msg)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["value"] = df["value"].astype(np.int64)
    df["classification"] = df["classification"].astype(
        pd.CategoricalDtype(
            categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
            ordered=True,
        )
    )

    row_count = len(df)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    checksum = _compute_sha256(file_path)
    _write_lineage(
        output_dir=Path(config.lineage_output_dir),
        dataset_name="fear_greed",
        run_id=run_id,
        file_path=file_path,
        checksum=checksum,
        row_count=row_count,
    )

    df.attrs = {
        "source_file": str(file_path.resolve()),
        "row_count": row_count,
        "ingestion_timestamp_utc": datetime.now(UTC).isoformat(),
    }

    logger.info(
        "Loaded Fear & Greed: path=%s, rows=%d, columns=%s, dtypes=%s",
        file_path,
        row_count,
        list(df.columns),
        {k: str(v) for k, v in df.dtypes.to_dict().items()},
    )

    return df
